import copy
import json
import math
import os
from pathlib import Path
import subprocess

import polars as pl
import pytest

from green_debt.build import (
    BuildGraph,
    BuildNode,
    BuildStage,
    CHECKPOINT3_CRITERIA,
    ReproductionMismatch,
    canonical_parquet_fingerprint,
    cleanup_reproduction,
    compare_reproduction_parquet_content,
    compare_reproduction_fingerprints,
    create_reproduction_receipt,
    render_delivery_note,
    prepare_reproduction_cleanup,
    run_reproduction_check,
    validate_checkpoint3_bundle,
    validate_checkpoint3_facts,
    validate_checkpoint3_lineage,
)
from green_debt.artifacts import BuildIdentity, TableContract, write_authoritative_table
from green_debt.cli import main
from green_debt.checkpoints import (
    _CHECKPOINT_REQUIREMENTS,
    checkpoint3_facts_from_reports,
    validate_checkpoint3_supporting_evidence,
)
from green_debt.storage import GIB


def _write_partition(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def test_canonical_reproduction_matches_content_rows_and_partition_order(tmp_path: Path) -> None:
    original = tmp_path / "original"
    reproduced = tmp_path / "reproduced"
    for root in (original, reproduced):
        _write_partition(root / "year=2000/part.parquet", [{"id": "A", "value": 1.0}])
        _write_partition(root / "year=2001/part.parquet", [{"id": "B", "value": 2.0}])

    expected = canonical_parquet_fingerprint(original, primary_key=("id",))
    actual = canonical_parquet_fingerprint(reproduced, primary_key=("id",))

    compare_reproduction_fingerprints(expected, actual)
    assert expected.rows == 2
    assert expected.partition_order == (
        "year=2000/part.parquet",
        "year=2001/part.parquet",
    )
    assert expected.content_sha256 == actual.content_sha256


def test_reproduction_detects_a_deliberate_content_mismatch(tmp_path: Path) -> None:
    original = tmp_path / "original"
    reproduced = tmp_path / "reproduced"
    _write_partition(original / "part.parquet", [{"id": "A", "value": 1.0}])
    _write_partition(reproduced / "part.parquet", [{"id": "A", "value": 9.0}])

    with pytest.raises(ReproductionMismatch, match="content hash"):
        compare_reproduction_fingerprints(
            canonical_parquet_fingerprint(original, primary_key=("id",)),
            canonical_parquet_fingerprint(reproduced, primary_key=("id",)),
        )


def test_canonical_reproduction_ignores_only_derived_self_hashes(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    reproduced = tmp_path / "reproduced"
    _write_partition(
        original / "part.parquet",
        [
            {
                "id": "A",
                "value": 1.0,
                "level": 10_000_000_000.0,
                "canonical_hash": "a" * 64,
                "giu_scaler_hash": "b" * 64,
                "regression_bounds_hash": "c" * 64,
            }
        ],
    )
    _write_partition(
        reproduced / "part.parquet",
        [
            {
                "id": "A",
                "value": 1.0,
                "level": 10_000_000_000.0,
                "canonical_hash": "d" * 64,
                "giu_scaler_hash": "e" * 64,
                "regression_bounds_hash": "f" * 64,
            }
        ],
    )

    compare_reproduction_fingerprints(
        canonical_parquet_fingerprint(original, primary_key=("id",)),
        canonical_parquet_fingerprint(reproduced, primary_key=("id",)),
    )


def test_rowwise_reproduction_accepts_only_bounded_float_noise(tmp_path: Path) -> None:
    original = tmp_path / "original"
    reproduced = tmp_path / "reproduced"
    boundary = 1.2345678905
    lower = math.nextafter(boundary, -math.inf)
    upper = math.nextafter(boundary, math.inf)
    _write_partition(
        original / "part.parquet",
        [{"id": "A", "value": lower, "canonical_hash": "a" * 64}],
    )
    _write_partition(
        reproduced / "part.parquet",
        [{"id": "A", "value": upper, "canonical_hash": "b" * 64}],
    )

    expected = canonical_parquet_fingerprint(original, primary_key=("id",))
    actual = canonical_parquet_fingerprint(reproduced, primary_key=("id",))
    assert expected.content_sha256 != actual.content_sha256

    comparison = compare_reproduction_parquet_content(
        original,
        reproduced,
        primary_key=("id",),
        temporary_root=tmp_path / "spill",
    )

    assert comparison.matched is True
    assert comparison.mismatch_rows == 0
    assert 0.0 < comparison.max_scaled_float_error < comparison.numeric_tolerance


def test_rowwise_reproduction_rejects_material_float_change(tmp_path: Path) -> None:
    original = tmp_path / "original"
    reproduced = tmp_path / "reproduced"
    _write_partition(original / "part.parquet", [{"id": "A", "value": 1.0}])
    changed = 1.0 + 5e-11
    _write_partition(reproduced / "part.parquet", [{"id": "A", "value": changed}])

    # This change was hidden by the former 10-significant-digit fast hash, but
    # exceeds the strict rowwise tolerance and therefore must reach the fallback.
    assert format(1.0, ".10g") == format(changed, ".10g")
    assert (
        canonical_parquet_fingerprint(original, primary_key=("id",)).content_sha256
        != canonical_parquet_fingerprint(
            reproduced, primary_key=("id",)
        ).content_sha256
    )

    comparison = compare_reproduction_parquet_content(
        original,
        reproduced,
        primary_key=("id",),
        temporary_root=tmp_path / "spill",
    )

    assert comparison.matched is False
    assert comparison.mismatch_rows == 1
    assert comparison.max_scaled_float_error > comparison.numeric_tolerance


def _scratch(tmp_path: Path, name: str = "reproduce.ABC123") -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    scratch = data_root / "05_中间数据/_tmp" / name
    scratch.mkdir(parents=True)
    return data_root, scratch


def test_cleanup_removes_only_the_exact_receipt_recorded_scratch(tmp_path: Path) -> None:
    data_root, scratch = _scratch(tmp_path)
    sibling = scratch.parent / "reproduce.KEEP"
    sibling.mkdir()
    (scratch / "generated.bin").write_bytes(b"generated")
    receipt = create_reproduction_receipt(
        data_root=data_root,
        scratch_root=scratch,
        artifacts=(),
    )

    removed = cleanup_reproduction(receipt)

    assert removed == scratch.resolve()
    assert not scratch.exists()
    assert sibling.is_dir()


@pytest.mark.parametrize(
    "case",
    (
        "wrong_name",
        "wrong_parent",
        "missing_receipt",
        "mismatched_receipt",
        "data_root",
        "workspace_root",
        "broader_tmp",
    ),
)
def test_cleanup_refuses_every_broader_or_unbound_target(tmp_path: Path, case: str) -> None:
    data_root, scratch = _scratch(tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    receipt = create_reproduction_receipt(
        data_root=data_root,
        scratch_root=scratch,
        artifacts=(),
        workspace_root=workspace_root,
    )
    if case == "wrong_name":
        wrong = scratch.parent / "scratch.ABC123"
        wrong.mkdir()
        payload = json.loads(receipt.read_text())
        payload["scratch_root"] = str(wrong.resolve())
        payload.pop("binding_sha256")
        receipt.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "wrong_parent":
        wrong = data_root / "elsewhere/reproduce.ABC123"
        wrong.mkdir(parents=True)
        payload = json.loads(receipt.read_text())
        payload["scratch_root"] = str(wrong.resolve())
        payload.pop("binding_sha256")
        receipt.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "missing_receipt":
        receipt.unlink()
    elif case == "mismatched_receipt":
        payload = json.loads(receipt.read_text())
        payload["scratch_root"] = str((scratch.parent / "reproduce.OTHER").resolve())
        receipt.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "data_root":
        payload = json.loads(receipt.read_text())
        payload["scratch_root"] = str(data_root.resolve())
        payload.pop("binding_sha256")
        receipt.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "workspace_root":
        payload = json.loads(receipt.read_text())
        payload["scratch_root"] = str(workspace_root.resolve())
        payload.pop("binding_sha256")
        receipt.write_text(json.dumps(payload), encoding="utf-8")
    else:
        payload = json.loads(receipt.read_text())
        payload["scratch_root"] = str(scratch.parent.resolve())
        payload.pop("binding_sha256")
        receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((ValueError, FileNotFoundError), match="receipt|scratch|target|binding"):
        cleanup_reproduction(receipt)


def test_cleanup_refuses_a_symlink_scratch_directory(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    tmp_root = data_root / "05_中间数据/_tmp"
    target = tmp_path / "real"
    target.mkdir()
    tmp_root.mkdir(parents=True)
    scratch = tmp_root / "reproduce.SYMLINK"
    os.symlink(target, scratch)
    payload = {
        "schema_version": "1.0.0",
        "data_root": str(data_root.resolve()),
        "scratch_root": str(scratch.absolute()),
        "workspace_root": None,
        "artifacts": [],
    }
    # create_reproduction_receipt rejects the symlink; write a hostile receipt
    # directly to exercise cleanup's independent guard.
    receipt = tmp_root / "hostile_receipt.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="symlink|receipt"):
        cleanup_reproduction(receipt)
    assert target.is_dir()


@pytest.mark.parametrize("symlink_component", ("data", "intermediate", "tmp"))
def test_cleanup_refuses_a_symlink_in_every_parent_component(
    tmp_path: Path, symlink_component: str
) -> None:
    real_data = tmp_path / "real-data"
    real_tmp = real_data / "05_中间数据/_tmp"
    real_tmp.mkdir(parents=True)
    if symlink_component == "data":
        data_root = tmp_path / "data"
        os.symlink(real_data, data_root)
    else:
        data_root = tmp_path / "data"
        data_root.mkdir()
        if symlink_component == "intermediate":
            os.symlink(real_data / "05_中间数据", data_root / "05_中间数据")
        else:
            (data_root / "05_中间数据").mkdir()
            os.symlink(real_tmp, data_root / "05_中间数据/_tmp")
    scratch = data_root / "05_中间数据/_tmp/reproduce.PARENT"
    scratch.mkdir()
    receipt = scratch / "reproduction_receipt.json"
    payload = {
        "schema_version": "1.0.0",
        "data_root": str(data_root.absolute()),
        "scratch_root": str(scratch.absolute()),
        "workspace_root": None,
        "artifacts": [],
    }
    from green_debt.build import _receipt_binding
    payload["binding_sha256"] = _receipt_binding(payload)
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="symlink"):
        cleanup_reproduction(receipt)
    assert scratch.is_dir()


def test_prepared_cleanup_rejects_replaced_directory_object(tmp_path: Path) -> None:
    data_root, scratch = _scratch(tmp_path)
    receipt = create_reproduction_receipt(
        data_root=data_root, scratch_root=scratch, artifacts=()
    )
    prepared = prepare_reproduction_cleanup(receipt)
    moved = scratch.parent / "reproduce.ORIGINAL"
    scratch.rename(moved)
    scratch.mkdir()
    (scratch / "replacement.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="object changed"):
        prepared.remove()
    assert moved.is_dir()
    assert scratch.is_dir()


def test_prepared_cleanup_rejects_replaced_tmp_parent_object(tmp_path: Path) -> None:
    data_root, scratch = _scratch(tmp_path)
    receipt = create_reproduction_receipt(
        data_root=data_root, scratch_root=scratch, artifacts=()
    )
    prepared = prepare_reproduction_cleanup(receipt)
    original_tmp = scratch.parent
    moved_tmp = original_tmp.with_name("_tmp.original")
    original_tmp.rename(moved_tmp)
    replacement_scratch = original_tmp / scratch.name
    replacement_scratch.mkdir(parents=True)
    (replacement_scratch / "keep.txt").write_text("replacement", encoding="utf-8")

    with pytest.raises(ValueError, match="parent.*changed"):
        prepared.remove()
    assert (moved_tmp / scratch.name).is_dir()
    assert replacement_scratch.is_dir()


def _ten_facts() -> dict[str, object]:
    return {
        "raw_hashes": True,
        "schemas_and_manifests": True,
        "taxonomy_counts": {"cleg": 248, "apec": 126, "overlap": 54},
        "duplicate_authoritative_keys": 0,
        "source_missing_zero_rules": True,
        "single_frozen_gad_scaler": True,
        "timing_overlap_violations": 0,
        "leakage_violations": 0,
        "fresh_full_test_exit_code": 0,
        "capacity": {
            "intermediate_plus_scratch_bytes": 24,
            "project_bytes": 119,
            "absolute_project_bytes": 149,
            "filesystem_reserve_bytes": 31,
            "limits": {
                "intermediate": 25,
                "project": 120,
                "absolute": 150,
                "filesystem_reserve": 30,
            },
        },
    }


def test_checkpoint3_has_exactly_ten_fail_closed_criteria() -> None:
    assert CHECKPOINT3_CRITERIA == (
        "raw_hashes",
        "schemas_and_manifests",
        "taxonomy_counts",
        "duplicate_authoritative_keys",
        "source_missing_zero_rules",
        "single_frozen_gad_scaler",
        "timing_overlap_violations",
        "leakage_violations",
        "fresh_full_test_exit_code",
        "capacity",
    )
    validate_checkpoint3_facts(_ten_facts())
    for criterion in CHECKPOINT3_CRITERIA:
        tampered = copy.deepcopy(_ten_facts())
        if criterion == "taxonomy_counts":
            tampered[criterion]["overlap"] = 53
        elif criterion in {"duplicate_authoritative_keys", "timing_overlap_violations", "leakage_violations", "fresh_full_test_exit_code"}:
            tampered[criterion] = 1
        elif criterion == "capacity":
            tampered[criterion]["project_bytes"] = 120
        else:
            tampered[criterion] = False
        with pytest.raises(ValueError, match=criterion):
            validate_checkpoint3_facts(tampered)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_checkpoint3_receipt_binds_support_hashes_and_evidence_only_lineage(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    code = root / "03_code/src/builder.py"
    code.parent.mkdir(parents=True)
    code.write_text("implementation\n", encoding="utf-8")
    results = root / "06_results"
    results.mkdir()
    unchanged_support = results / "leakage.csv"
    unchanged_support.write_text("leakage.csv\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "implementation")
    implementation = _git(root, "rev-parse", "HEAD")
    support = {"leakage.csv": unchanged_support}
    for name in ("outcome.csv", "iv.csv", "panel.csv", "capacity.json", "tests.json", "delivery.md"):
        path = results / name
        path.write_text(f"{name}\n", encoding="utf-8")
        support[name] = path
    receipt = results / "receipt.json"
    payload = validate_checkpoint3_bundle.create_receipt_payload(
        implementation_commit=implementation,
        facts=_ten_facts(),
        support_files=support,
        code_root=root,
    )
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    _git(root, "add", "06_results")
    _git(root, "commit", "-qm", "evidence")

    validate_checkpoint3_bundle(receipt, root, recomputed_facts=_ten_facts())
    validate_checkpoint3_lineage(
        implementation,
        _git(root, "rev-parse", "HEAD"),
        tuple(f"06_results/{name}" for name in (*support, "receipt.json")),
        is_ancestor=True,
    )

    support["panel.csv"].write_text("forged\n", encoding="utf-8")
    with pytest.raises(ValueError, match="support hash"):
        validate_checkpoint3_bundle(
            receipt,
            root,
            recomputed_facts=_ten_facts(),
            verify_git_state=False,
        )


@pytest.mark.parametrize(
    "field",
    (
        "raw_hashes",
        "taxonomy_counts",
        "single_frozen_gad_scaler",
        "timing_overlap_violations",
        "leakage_violations",
        "fresh_full_test_exit_code",
        "capacity",
    ),
)
def test_checkpoint3_bundle_rejects_tampered_facts(field: str, tmp_path: Path) -> None:
    support = tmp_path / "support.csv"
    support.write_text("support\n", encoding="utf-8")
    payload = validate_checkpoint3_bundle.create_receipt_payload(
        implementation_commit="a" * 40,
        facts=_ten_facts(),
        support_files={"support": support},
        code_root=tmp_path,
    )
    tampered = copy.deepcopy(payload)
    tampered["facts"][field] = "forged"
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="facts"):
        validate_checkpoint3_bundle(
            receipt,
            tmp_path,
            recomputed_facts=_ten_facts(),
            verify_git_state=False,
        )


def test_checkpoint3_bundle_accepts_current_free_space_drift_above_frozen_gate(
    tmp_path: Path,
) -> None:
    support = tmp_path / "support.csv"
    support.write_text("support\n", encoding="utf-8")
    recorded = _ten_facts()
    payload = validate_checkpoint3_bundle.create_receipt_payload(
        implementation_commit="a" * 40,
        facts=recorded,
        support_files={"support": support},
        code_root=tmp_path,
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    current = copy.deepcopy(recorded)
    current["capacity"]["filesystem_reserve_bytes"] = 32

    validate_checkpoint3_bundle(
        receipt,
        tmp_path,
        recomputed_facts=current,
        verify_git_state=False,
    )


def test_delivery_note_inventory_is_complete_and_disclaims_estimation() -> None:
    panels = [
        {
            "table_id": "model_panel",
            "path": "/data/model_panel.parquet",
            "primary_key": ["economy_id", "treatment_year", "horizon", "outcome_id", "gad_version", "sample_version"],
            "period": [2000, 2022],
            "sample_versions": ["core", "lite"],
            "gad_versions": ["gad_core", "gad_lite"],
            "outcome_families": ["environmental", "industrial"],
            "units": {"outcome_value": "source_unit"},
            "rows": 55514,
            "bytes": 1234,
            "manifest_path": "/data/model_panel.parquet.manifest.json",
        },
        {
            "table_id": "regression_bounds",
            "path": "/data/regression_bounds.parquet",
            "primary_key": ["variable"],
            "period": None,
            "sample_versions": ["core", "lite"],
            "gad_versions": ["gad_core", "gad_lite"],
            "outcome_families": ["environmental", "industrial"],
            "units": {"lower": "source_unit", "upper": "source_unit"},
            "rows": 10,
            "bytes": 300,
            "manifest_path": "/data/regression_bounds.parquet.manifest.json",
        },
    ]

    note = render_delivery_note(panels)

    for required in (
        "model_panel",
        "regression_bounds",
        "economy_id",
        "2000–2022",
        "core",
        "gad_core",
        "environmental",
        "source_unit",
        "55,514",
        "1,234",
        "model_panel.parquet.manifest.json",
        "no causal estimate was run",
    ):
        assert required in note


def test_reproduce_check_runs_distinct_staged_builder_and_writes_nonzero_scratch(
    tmp_path: Path,
) -> None:
    data_root, scratch = _scratch(tmp_path)
    raw = data_root / "04_原始数据"
    raw.mkdir(parents=True)
    (raw / "source.bin").write_bytes(b"immutable")
    output = data_root / "05_中间数据/analysis/table.parquet"
    write_authoritative_table(
        pl.DataFrame({"id": ["A", "B"], "value": [1.0, 2.0]}),
        TableContract(
            table_id="fixture_table",
            schema_version="1.0.0",
            primary_key=("id",),
            columns={"id": "String", "value": "Float64"},
            units={"value": "unit"},
        ),
        output,
        (),
        BuildIdentity(command="fixture", code_commit="a" * 40),
    )

    def rebuild(stage: BuildStage) -> None:
        reproduced = stage.path_for(output)
        write_authoritative_table(
            pl.DataFrame({"id": ["A", "B"], "value": [1.0, 2.0]}),
            TableContract(
                table_id="fixture_table",
                schema_version="1.0.0",
                primary_key=("id",),
                columns={"id": "String", "value": "Float64"},
                units={"value": "unit"},
            ),
            reproduced,
            (),
            BuildIdentity(command="fixture", code_commit="a" * 40),
        )

    graph = BuildGraph(
        (
            BuildNode(
                "fixture",
                (),
                lambda: None,
                output_ids=("fixture_table",),
                manifest_paths=(output.with_name(f"{output.name}.manifest.json"),),
                staging_builder=rebuild,
            ),
        )
    )

    receipt = run_reproduction_check(
        code_root=tmp_path / "code",
        data_root=data_root,
        scratch_root=scratch,
        graph=graph,
        implementation_commit="a" * 40,
    )
    payload = json.loads(receipt.read_text())

    assert payload["raw_tree_sha256_before"] == payload["raw_tree_sha256_after"]
    assert payload["artifacts"][0]["table_id"] == "fixture_table"
    assert payload["artifacts"][0]["rows"] == 2
    assert payload["artifacts"][0]["authority_content_sha256"] == payload["artifacts"][0]["reproduced_content_sha256"]
    assert payload["artifacts"][0]["authority_path"] != payload["artifacts"][0]["reproduced_path"]
    assert payload["scratch_bytes_before_receipt"] > 0
    assert payload["rebuild_commands"] == [{"node": "fixture", "kind": "staging_builder"}]
    assert payload["implementation_commit"] == "a" * 40
    assert not list(scratch.rglob("CURRENT.json"))


def test_reproduce_check_records_tolerant_machine_noise_match(tmp_path: Path) -> None:
    data_root, scratch = _scratch(tmp_path)
    raw = data_root / "04_原始数据"
    raw.mkdir(parents=True)
    (raw / "source.bin").write_bytes(b"immutable")
    output = data_root / "05_中间数据/analysis/table.parquet"
    boundary = 1.2345678905
    lower = math.nextafter(boundary, -math.inf)
    upper = math.nextafter(boundary, math.inf)
    contract = TableContract(
        table_id="noisy_fixture_table",
        schema_version="1.0.0",
        primary_key=("id",),
        columns={"id": "String", "value": "Float64", "canonical_hash": "String"},
        units={"value": "unit"},
    )
    write_authoritative_table(
        pl.DataFrame(
            {"id": ["A"], "value": [lower], "canonical_hash": ["a" * 64]}
        ),
        contract,
        output,
        (),
        BuildIdentity(command="fixture", code_commit="a" * 40),
    )

    def rebuild(stage: BuildStage) -> None:
        write_authoritative_table(
            pl.DataFrame(
                {"id": ["A"], "value": [upper], "canonical_hash": ["b" * 64]}
            ),
            contract,
            stage.path_for(output),
            (),
            BuildIdentity(command="fixture", code_commit="a" * 40),
        )

    graph = BuildGraph(
        (
            BuildNode(
                "fixture",
                (),
                lambda: None,
                output_ids=("noisy_fixture_table",),
                manifest_paths=(output.with_name(f"{output.name}.manifest.json"),),
                staging_builder=rebuild,
            ),
        )
    )

    receipt = run_reproduction_check(
        code_root=tmp_path / "code",
        data_root=data_root,
        scratch_root=scratch,
        graph=graph,
        implementation_commit="a" * 40,
    )
    artifact = json.loads(receipt.read_text())["artifacts"][0]

    assert artifact["authority_content_sha256"] != artifact["reproduced_content_sha256"]
    assert artifact["content_hashes_equal"] is False
    assert artifact["matched"] is True
    assert artifact["mismatch_rows"] == 0
    assert 0.0 < artifact["max_scaled_float_error"] < artifact["numeric_tolerance"]


def test_reproduction_preserves_duplicate_table_ids_by_stable_manifest_path(
    tmp_path: Path,
) -> None:
    data_root, scratch = _scratch(tmp_path)
    raw = data_root / "04_原始数据"
    raw.mkdir(parents=True)
    (raw / "source.bin").write_bytes(b"immutable")
    outputs = tuple(
        data_root / f"05_中间数据/normalized/baci/{role}/table.parquet"
        for role in ("exporter_product", "importer_product")
    )
    contract = TableContract(
        table_id="shared_baci_product_id",
        schema_version="1.0.0",
        primary_key=("flow_role", "id"),
        columns={"flow_role": "String", "id": "String", "value": "Float64"},
        units={"value": "unit"},
    )
    for role, output in zip(("exporter", "importer"), outputs, strict=True):
        write_authoritative_table(
            pl.DataFrame({"flow_role": [role], "id": ["A"], "value": [1.0]}),
            contract,
            output,
            (),
            BuildIdentity(command="fixture", code_commit="a" * 40),
        )

    def rebuild(stage: BuildStage) -> None:
        for role, output in zip(("exporter", "importer"), outputs, strict=True):
            write_authoritative_table(
                pl.DataFrame({"flow_role": [role], "id": ["A"], "value": [1.0]}),
                contract,
                stage.path_for(output),
                (),
                BuildIdentity(command="fixture", code_commit="a" * 40),
            )

    graph = BuildGraph(
        (
            BuildNode(
                "fixture",
                (),
                lambda: None,
                output_ids=("shared_baci_product_id",),
                manifest_paths=tuple(
                    output.with_name(f"{output.name}.manifest.json") for output in outputs
                ),
                staging_builder=rebuild,
            ),
        )
    )
    receipt = run_reproduction_check(
        code_root=tmp_path / "code",
        data_root=data_root,
        scratch_root=scratch,
        graph=graph,
        implementation_commit="a" * 40,
    )
    artifacts = json.loads(receipt.read_text())["artifacts"]

    assert len(artifacts) == 2
    assert {item["table_id"] for item in artifacts} == {"shared_baci_product_id"}
    assert {item["manifest_identity"] for item in artifacts} == {
        "05_中间数据/normalized/baci/exporter_product/table.parquet.manifest.json",
        "05_中间数据/normalized/baci/importer_product/table.parquet.manifest.json",
    }


def test_cleanup_cli_reports_and_removes_the_exact_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_root, scratch = _scratch(tmp_path)
    receipt = create_reproduction_receipt(
        data_root=data_root,
        scratch_root=scratch,
        artifacts=(),
    )

    assert main(["cleanup-reproduction", "--receipt", str(receipt)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"removed_path": str(scratch.resolve()), "status": "removed"}
    assert not scratch.exists()


def test_checkpoint3_requires_every_final_outcome_iv_sample_and_panel_authority() -> None:
    assert set(_CHECKPOINT_REQUIREMENTS[3]) == {
        "outcomes_country_year",
        "outcomes_product_year",
        "iv_baseline_shares",
        "iv_partner_shocks",
        "iv_country_year",
        "final_sample",
        "regression_bounds",
        "giu_outcome_scalers",
        "model_panel",
    }


def test_checkpoint3_facts_are_derived_from_parent_based_reports_not_receipt_claims() -> None:
    facts = checkpoint3_facts_from_reports(
        raw_hash_mismatches=0,
        manifest_failures=0,
        taxonomy_counts={"main": 126, "broad": 248, "apec": 54},
        duplicate_keys=0,
        source_failures=0,
        scaler_hashes=("frozen",),
        outcome_report={"gad_value_table_parents": 0},
        instrument_report={
            "destination_exclusion_failures": 0,
            "prohibited_lineage_columns": 0,
            "outcome_value_table_parents": 0,
            "shock_parent_reconstruction_failures": 0,
        },
        panel_report={
            "timing_violations": 0,
            "mapping_overlap_violations": 0,
            "outcome_source_failures": 0,
        },
        test_exit_code=0,
        intermediate_plus_scratch_bytes=24 * GIB,
        project_bytes=119 * GIB,
        filesystem_free_bytes=31 * GIB,
    )

    validate_checkpoint3_facts(facts)
    assert facts["leakage_violations"] == 0
    assert facts["timing_overlap_violations"] == 0


@pytest.mark.parametrize(
    ("report_name", "field"),
    (
        ("outcome_report", "gad_value_table_parents"),
        ("instrument_report", "destination_exclusion_failures"),
        ("instrument_report", "prohibited_lineage_columns"),
        ("instrument_report", "outcome_value_table_parents"),
        ("instrument_report", "shock_parent_reconstruction_failures"),
        ("panel_report", "timing_violations"),
        ("panel_report", "mapping_overlap_violations"),
        ("panel_report", "outcome_source_failures"),
    ),
)
def test_checkpoint3_facts_fail_on_tampered_parent_audit(
    report_name: str, field: str
) -> None:
    reports = {
        "outcome_report": {"gad_value_table_parents": 0},
        "instrument_report": {
            "destination_exclusion_failures": 0,
            "prohibited_lineage_columns": 0,
            "outcome_value_table_parents": 0,
            "shock_parent_reconstruction_failures": 0,
        },
        "panel_report": {
            "timing_violations": 0,
            "mapping_overlap_violations": 0,
            "outcome_source_failures": 0,
        },
    }
    reports[report_name][field] = 1

    facts = checkpoint3_facts_from_reports(
        raw_hash_mismatches=0,
        manifest_failures=0,
        taxonomy_counts={"main": 126, "broad": 248, "apec": 54},
        duplicate_keys=0,
        source_failures=0,
        scaler_hashes=("frozen",),
        test_exit_code=0,
        intermediate_plus_scratch_bytes=1,
        project_bytes=1,
        filesystem_free_bytes=31 * GIB,
        **reports,
    )

    with pytest.raises(ValueError, match="timing_overlap_violations|leakage_violations"):
        validate_checkpoint3_facts(facts)


@pytest.mark.parametrize("tampered", ("results", "leakage", "capacity", "delivery"))
def test_checkpoint3_supporting_evidence_rejects_semantic_tampering(
    tmp_path: Path, tampered: str
) -> None:
    results_dir = tmp_path / "06_结果"
    results_dir.mkdir()
    results = pl.DataFrame(
        {"section": ["outcome", "instrument", "panel"], "metric": ["coverage", "exclusion", "rows"], "value": [0, 0, 55514], "status": ["pass", "pass", "pass"]}
    )
    leakage = pl.DataFrame(
        {"criterion": ["own_destination", "future_shock", "outcome"], "violations": [0, 0, 0], "status": ["pass", "pass", "pass"]}
    )
    capacity = {
        "data_root": "/data",
        "intermediate_bytes": 10,
        "intermediate_plus_scratch_bytes": 10,
        "project_bytes": 20,
        "usage_bytes": {"analysis": 5},
        "limits_bytes": {"filesystem_reserve_at_least": 30},
        "filesystem_free_bytes": 31,
        "passed": True,
    }
    delivery = "delivery\nno causal estimate was run\n"
    results.write_csv(results_dir / "检查点3_结果与IV覆盖_v1.csv")
    leakage.write_csv(results_dir / "检查点3_泄漏审计_v1.csv")
    (results_dir / "检查点3_容量报告_v1.json").write_text(json.dumps(capacity), encoding="utf-8")
    (results_dir / "数据清洗与变量构造交付说明_v1.md").write_text(delivery, encoding="utf-8")
    if tampered == "results":
        results.with_columns(pl.lit(999).alias("value")).write_csv(results_dir / "检查点3_结果与IV覆盖_v1.csv")
    elif tampered == "leakage":
        leakage.with_columns(pl.lit(1).alias("violations")).write_csv(results_dir / "检查点3_泄漏审计_v1.csv")
    elif tampered == "capacity":
        forged = dict(capacity)
        forged["passed"] = False
        (results_dir / "检查点3_容量报告_v1.json").write_text(json.dumps(forged), encoding="utf-8")
    else:
        (results_dir / "数据清洗与变量构造交付说明_v1.md").write_text("forged", encoding="utf-8")

    with pytest.raises(Exception, match="Checkpoint 3 supporting evidence"):
        validate_checkpoint3_supporting_evidence(
            tmp_path,
            results_and_iv=results,
            leakage=leakage,
            capacity_projection=capacity,
            delivery_note=delivery,
        )
