import json
from pathlib import Path

from jsonschema import validate
import polars as pl
import pytest

from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    manifest_is_current,
    quarantine_rows,
    verify_manifest,
    write_authoritative_table,
)
from green_debt.sample import _load_contract, freeze_final_sample


def contract() -> TableContract:
    return TableContract(
        table_id="toy_country_year",
        schema_version="1",
        primary_key=("economy_id", "year"),
        columns={"economy_id": "String", "year": "Int16", "value": "Float64"},
        units={"value": "index"},
        period=(2000, 2001),
        zero_semantics={"value": "valid_index_zero"},
        transformations=("identity",),
    )


def valid_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"economy_id": ["AAA", "AAA"], "year": [2000, 2001], "value": [0.0, 2.0]},
        schema={"economy_id": pl.String, "year": pl.Int16, "value": pl.Float64},
    )


def test_atomic_table_has_schema_manifest_and_hash(tmp_path: Path) -> None:
    source = tmp_path / "raw.bin"
    source.write_bytes(b"raw")
    destination = tmp_path / "05_中间数据" / "normalized" / "toy.parquet"

    manifest = write_authoritative_table(
        valid_frame(),
        contract(),
        destination,
        (InputArtifact.from_path(source),),
        BuildIdentity(command="test", code_commit="abc123"),
    )

    assert not destination.with_name("toy.parquet.partial").exists()
    assert manifest.rows == 2
    assert manifest.duplicate_primary_keys == 0
    assert manifest.null_counts == {"economy_id": 0, "year": 0, "value": 0}
    assert manifest.zero_counts == {"value": 1}
    verified = verify_manifest(destination.with_name("toy.parquet.manifest.json"))
    assert verified.output_sha256
    assert verified.input_artifacts[0].sha256 == InputArtifact.from_path(source).sha256
    assert (tmp_path / "05_中间数据/schemas/toy_country_year.schema.json").is_file()
    assert (tmp_path / "05_中间数据/manifests/toy_country_year.manifest.json").is_file()


def test_duplicate_key_fails_before_rename(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {"economy_id": ["AAA", "AAA"], "year": [2000, 2000], "value": [1.0, 2.0]},
        schema={"economy_id": pl.String, "year": pl.Int16, "value": pl.Float64},
    )
    destination = tmp_path / "bad.parquet"

    with pytest.raises(ValueError, match="duplicate primary key"):
        write_authoritative_table(
            frame,
            contract(),
            destination,
            (),
            BuildIdentity("test", "abc"),
        )

    assert not destination.exists()
    assert not list(tmp_path.rglob("*.partial"))


def test_null_primary_key_fails_before_write(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {"economy_id": ["AAA", None], "year": [2000, 2001], "value": [1.0, 2.0]},
        schema={"economy_id": pl.String, "year": pl.Int16, "value": pl.Float64},
    )

    with pytest.raises(ValueError, match="null primary key"):
        write_authoritative_table(
            frame,
            contract(),
            tmp_path / "null-key.parquet",
            (),
            BuildIdentity("test", "abc"),
        )

    assert not list(tmp_path.iterdir())


def test_nonfinite_or_wrong_schema_never_replaces_prior_output(tmp_path: Path) -> None:
    destination = tmp_path / "toy.parquet"
    write_authoritative_table(
        valid_frame(), contract(), destination, (), BuildIdentity("first", "abc")
    )
    original = destination.read_bytes()
    invalid = valid_frame().with_columns(pl.lit(float("inf")).alias("value"))

    with pytest.raises(ValueError, match="nonfinite"):
        write_authoritative_table(
            invalid, contract(), destination, (), BuildIdentity("second", "def")
        )
    with pytest.raises(ValueError, match="exact columns"):
        write_authoritative_table(
            valid_frame().rename({"value": "other"}),
            contract(),
            destination,
            (),
            BuildIdentity("second", "def"),
        )

    assert destination.read_bytes() == original
    assert not list(tmp_path.rglob("*.partial"))


def test_manifest_verification_detects_output_tampering(tmp_path: Path) -> None:
    destination = tmp_path / "toy.parquet"
    write_authoritative_table(
        valid_frame(), contract(), destination, (), BuildIdentity("test", "abc")
    )
    destination.write_bytes(destination.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="output hash mismatch"):
        verify_manifest(destination.with_name("toy.parquet.manifest.json"))


def test_quarantine_requires_stable_reasons_and_writes_atomically(tmp_path: Path) -> None:
    frame = pl.DataFrame({"value": [1, 2], "reason": ["bad_unit", "bad_code"]})
    destination = tmp_path / "quarantine.parquet"

    assert quarantine_rows(frame, "reason", destination) == 2
    assert pl.read_parquet(destination).to_dicts() == frame.to_dicts()
    assert not list(tmp_path.rglob("*.partial"))

    with pytest.raises(ValueError, match="stable reason code"):
        quarantine_rows(
            pl.DataFrame({"value": [1], "reason": ["Bad reason!"]}),
            "reason",
            tmp_path / "bad.parquet",
        )


def test_manifest_currency_uses_order_independent_parent_hashes() -> None:
    manifest = {"input_hashes": ["bbb", "aaa"]}

    assert manifest_is_current(manifest, ("aaa", "bbb")) is True
    assert manifest_is_current(manifest, ("aaa", "ccc")) is False


def test_sidecar_manifest_is_valid_json(tmp_path: Path) -> None:
    destination = tmp_path / "toy.parquet"
    write_authoritative_table(
        valid_frame(), contract(), destination, (), BuildIdentity("test", "abc")
    )

    payload = json.loads(
        destination.with_name("toy.parquet.manifest.json").read_text(encoding="utf-8")
    )

    assert payload["table_id"] == "toy_country_year"
    assert payload["coverage"]["year"] == {"min": 2000, "max": 2001, "unique": 2}


def test_null_semantics_propagate_to_schema_manifest_and_verification(tmp_path: Path) -> None:
    nullable = TableContract(
        table_id="nullable_toy",
        schema_version="1",
        primary_key=("economy_id", "year"),
        columns={"economy_id": "String", "year": "Int16", "value": "Float64"},
        units={"value": "index"},
        null_semantics={"value": "null_when_source_value_is_structurally_absent"},
    )
    frame = pl.DataFrame(
        {"economy_id": ["AAA"], "year": [2000], "value": [None]},
        schema={"economy_id": pl.String, "year": pl.Int16, "value": pl.Float64},
    )
    destination = tmp_path / "nullable.parquet"
    manifest = write_authoritative_table(
        frame, nullable, destination, (), BuildIdentity("test", "abc")
    )
    schema = json.loads(
        destination.with_name("nullable.parquet.schema.json").read_text()
    )
    assert manifest.null_semantics == nullable.null_semantics
    assert verify_manifest(
        destination.with_name("nullable.parquet.manifest.json")
    ).null_semantics == nullable.null_semantics
    assert schema["properties"]["value"]["x-null-semantics"] == nullable.null_semantics["value"]
    assert schema["properties"]["value"]["anyOf"][-1] == {"type": "null"}
    assert "anyOf" not in schema["properties"]["economy_id"]


@pytest.mark.parametrize(
    "contract_kwargs", ({}, {"null_semantics": {}}), ids=("omitted", "empty")
)
def test_legacy_empty_null_semantics_keeps_schema_nullable(
    tmp_path: Path, contract_kwargs: dict[str, dict[str, str]]
) -> None:
    legacy = TableContract(
        table_id="legacy_nullable_toy",
        schema_version="1",
        primary_key=("economy_id", "year"),
        columns={"economy_id": "String", "year": "Int16", "value": "Float64"},
        units={"value": "index"},
        **contract_kwargs,
    )
    frame = pl.DataFrame(
        {"economy_id": ["AAA"], "year": [2000], "value": [None]},
        schema={"economy_id": pl.String, "year": pl.Int16, "value": pl.Float64},
    )
    destination = tmp_path / f"legacy-{len(contract_kwargs)}.parquet"

    write_authoritative_table(
        frame, legacy, destination, (), BuildIdentity("test", "abc")
    )
    verified = verify_manifest(
        destination.with_name(f"{destination.name}.manifest.json")
    )
    schema = json.loads(
        destination.with_name(f"{destination.name}.schema.json").read_text()
    )

    validate(instance=frame.to_dicts()[0], schema=schema)
    assert verified.null_semantics == {}
    assert all(
        definition["anyOf"][-1] == {"type": "null"}
        for definition in schema["properties"].values()
    )
    assert all(
        "x-null-semantics" not in definition
        for definition in schema["properties"].values()
    )


def test_declared_null_semantics_reject_undeclared_nullable_columns() -> None:
    with pytest.raises(ValueError, match="null-semantics columns"):
        TableContract(
            table_id="bad",
            schema_version="1",
            primary_key=("id",),
            columns={"id": "String"},
            units={},
            null_semantics={"missing": "not_a_column"},
        )

    contract_with_rules = TableContract(
        table_id="partly_nullable",
        schema_version="1",
        primary_key=("id",),
        columns={"id": "String", "value": "Float64", "note": "String"},
        units={},
        null_semantics={"value": "structural_value_gap"},
    )
    frame = pl.DataFrame(
        {"id": ["A"], "value": [None], "note": [None]},
        schema={"id": pl.String, "value": pl.Float64, "note": pl.String},
    )
    with pytest.raises(ValueError, match="undeclared null semantics.*note"):
        write_authoritative_table(
            frame, contract_with_rules, Path("unused.parquet"), (), BuildIdentity("test", "abc")
        )


def test_task16_contract_null_semantics_round_trip_to_schema_and_manifest(
    tmp_path: Path,
) -> None:
    contract_path = Path(__file__).parents[1] / "contracts/final_sample.json"
    task_contract = _load_contract(contract_path)
    assert task_contract.null_semantics == {
        "exclusion_reason": "null_exactly_when_core_eligible"
    }
    frame = freeze_final_sample(pl.DataFrame({
        "economy_id": ["A"],
        "complete_absorption_1996": [True],
        "complete_initialization_1997_1999": [True],
        "valid_main_years": [12],
        "positive_import_baseline_years": [2],
        "baseline_share_identifiable": [True],
        "economy_rule_eligible": [True],
    })).select(*task_contract.columns).cast(
        {name: getattr(pl, dtype) for name, dtype in task_contract.columns.items()}
    )
    destination = tmp_path / "final_sample.parquet"
    manifest = write_authoritative_table(
        frame, task_contract, destination, (InputArtifact.from_path(contract_path),),
        BuildIdentity("round-trip", "abc123"),
    )
    schema = json.loads(destination.with_name("final_sample.parquet.schema.json").read_text())
    verified = verify_manifest(destination.with_name("final_sample.parquet.manifest.json"))
    assert manifest.null_semantics == task_contract.null_semantics
    assert verified.null_semantics == task_contract.null_semantics
    assert schema["properties"]["exclusion_reason"]["x-null-semantics"] == task_contract.null_semantics["exclusion_reason"]


def test_task16_model_contract_round_trips_its_exact_conditional_null_map(
    tmp_path: Path,
) -> None:
    contract_path = Path(__file__).parents[1] / "contracts/model_panel.json"
    task_contract = _load_contract(contract_path)
    declared = {
        "CMZ",
        "CMZ_p01_p99",
        "baseline_outcome",
        "rca_eligible_products",
        "rca_entered_products",
        "renewable_energy_consumption_share",
        "trade_openness_percent_gdp",
        "industry_value_added_share",
        "gdp_per_capita_current_usd",
    }
    assert set(task_contract.null_semantics) == declared
    values: dict[str, list[object]] = {}
    for name, dtype in task_contract.columns.items():
        if dtype == "String":
            values[name] = ["x"]
        elif dtype == "Boolean":
            values[name] = [False]
        else:
            values[name] = [1]
    frame = pl.DataFrame(values).cast(
        {name: getattr(pl, dtype) for name, dtype in task_contract.columns.items()}
    )
    destination = tmp_path / "lp_panel.parquet"
    manifest = write_authoritative_table(
        frame, task_contract, destination, (InputArtifact.from_path(contract_path),),
        BuildIdentity("round-trip", "abc123"),
    )
    schema = json.loads(destination.with_name("lp_panel.parquet.schema.json").read_text())
    verified = verify_manifest(destination.with_name("lp_panel.parquet.manifest.json"))
    assert manifest.null_semantics == task_contract.null_semantics
    assert verified.null_semantics == task_contract.null_semantics
    assert {
        name for name, definition in schema["properties"].items()
        if "x-null-semantics" in definition
    } == declared
