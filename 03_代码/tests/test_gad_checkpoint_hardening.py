import json
from pathlib import Path

import polars as pl
import pytest

from green_debt.checkpoints import (
    CheckpointFailure,
    evaluate_checkpoint,
    parse_git_changed_paths,
    validate_evidence_lineage,
    verify_checkpoint2_evidence_receipt,
)
from green_debt.gad import (
    FROZEN_SCALER_HASH,
    audit_gad_frame,
    build_construction_audit,
    build_gad_variants,
    derive_core_eligibility,
)
from green_debt.storage import GIB, sha256_file


def _sources() -> tuple[pl.DataFrame, pl.DataFrame]:
    years = list(range(1996, 2025))
    values: dict[str, list[object]] = {"economy_id": ["AAA"] * len(years), "year": years}
    for position, column in enumerate(
        (
            "z0_green_import_intensity_raw", "z0_green_import_complexity_raw", "z0_gfvad_raw",
            "z0_gsci_raw", "z0_gud_raw", "z0_grd_raw", "z0_gnir_raw",
        )
    ):
        values[column] = [float(position + 1)] * len(years)
    return (
        pl.DataFrame(values),
        pl.DataFrame(
            {
                "economy_id": ["AAA"],
                "sample_version": ["confirmatory"],
                "provisional_core": [True],
                "positive_green_import_baseline_eligible": [True],
            }
        ),
    )


def _authority() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    scaled, sample = _sources()
    authority, _ = build_gad_variants(scaled, sample, scaler_hash=FROZEN_SCALER_HASH)
    return authority, scaled, sample


def test_audit_recomputes_run_identifiers_not_from_authority_values() -> None:
    authority, scaled, sample = _authority()
    tampered = authority.with_columns(
        pl.when((pl.col("specification_id") == "gad_core") & (pl.col("year") >= 2000))
        .then(pl.col("run_id") + 7)
        .otherwise(pl.col("run_id"))
        .alias("run_id")
    )

    with pytest.raises(ValueError, match="run identifiers"):
        audit_gad_frame(tampered, scaled_components=scaled, provisional_sample=sample)


def test_audit_recomputes_core_eligibility_and_exact_pk_matrix() -> None:
    authority, scaled, sample = _authority()
    ineligible = authority.with_columns(
        pl.when(pl.col("specification_id") == "gad_core").then(pl.lit(False)).otherwise(pl.col("core_economy_eligible")).alias("core_economy_eligible")
    )
    with pytest.raises(ValueError, match="core eligibility"):
        audit_gad_frame(ineligible, scaled_components=scaled, provisional_sample=sample)

    missing_lite_2024 = authority.filter(~((pl.col("specification_id") == "gad_lite") & (pl.col("year") == 2024)))
    with pytest.raises(ValueError, match="primary-key matrix"):
        audit_gad_frame(missing_lite_2024, scaled_components=scaled, provisional_sample=sample)


def test_audit_verifies_lagged_debt_registry_field_and_generic_presence_flags() -> None:
    authority, scaled, sample = _authority()
    assert authority.filter(pl.col("specification_id") == "gad_static")["uses_lagged_debt"].unique().to_list() == [False]
    assert authority.filter(pl.col("specification_id") == "gad_core")["uses_lagged_debt"].unique().to_list() == [True]
    wrong_registry = authority.with_columns(
        pl.when(pl.col("specification_id") == "gad_static").then(pl.lit(True)).otherwise(pl.col("uses_lagged_debt")).alias("uses_lagged_debt")
    )
    with pytest.raises(ValueError, match="registry metadata"):
        audit_gad_frame(wrong_registry, scaled_components=scaled, provisional_sample=sample)
    wrong_presence = authority.with_columns(
        pl.when((pl.col("specification_id") == "gad_core") & (pl.col("year") == 2000)).then(~pl.col("gap_complete")).otherwise(pl.col("gap_complete")).alias("gap_complete")
    )
    with pytest.raises(ValueError, match="presence flags"):
        audit_gad_frame(wrong_presence, scaled_components=scaled, provisional_sample=sample)


def test_audit_rejects_null_boolean_and_descriptive_or_initialization_tampering() -> None:
    authority, scaled, sample = _authority()
    null_debt_rule = authority.with_columns(
        pl.when(pl.col("specification_id") == "gad_static")
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(pl.col("uses_lagged_debt"))
        .alias("uses_lagged_debt")
    )
    with pytest.raises(ValueError, match="never-null"):
        audit_gad_frame(null_debt_rule, scaled_components=scaled, provisional_sample=sample)
    null_presence = authority.with_columns(
        pl.when((pl.col("specification_id") == "gad_core") & (pl.col("year") == 2000))
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(pl.col("gap_complete"))
        .alias("gap_complete")
    )
    with pytest.raises(ValueError, match="never-null"):
        audit_gad_frame(null_presence, scaled_components=scaled, provisional_sample=sample)
    descriptive_2000 = authority.with_columns(
        pl.when((pl.col("specification_id") == "gad_core") & (pl.col("year") == 2000))
        .then(pl.lit(True))
        .otherwise(pl.col("descriptive_only"))
        .alias("descriptive_only")
    )
    with pytest.raises(ValueError, match="descriptive_only"):
        audit_gad_frame(descriptive_2000, scaled_components=scaled, provisional_sample=sample)
    altered_initialization = authority.with_columns(
        pl.when((pl.col("specification_id") == "gad_core") & (pl.col("year") == 1997))
        .then(~pl.col("initialization_run"))
        .otherwise(pl.col("initialization_run"))
        .alias("initialization_run")
    )
    with pytest.raises(ValueError, match="initialization_run"):
        audit_gad_frame(altered_initialization, scaled_components=scaled, provisional_sample=sample)


def test_audit_never_coerces_a_null_registry_boolean_to_false_without_source_inputs() -> None:
    authority, _, _ = _authority()
    null_debt_rule = authority.with_columns(
        pl.when(pl.col("specification_id") == "gad_static")
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(pl.col("uses_lagged_debt"))
        .alias("uses_lagged_debt")
    )
    with pytest.raises(ValueError, match="never-null Boolean"):
        audit_gad_frame(null_debt_rule)


def test_receipt_supporting_evidence_hashes_are_real_and_tamper_fails(tmp_path: Path) -> None:
    evidence = {
        "06_结果/GAD固定缩放器_v1.manifest.json": "anchor\n",
        "06_结果/检查点2_GAD构造审计_v1.csv": "audit\n",
        "06_结果/检查点2_初始化覆盖_v1.csv": "init\n",
        "06_结果/检查点2_容量报告_v1.json": "{}\n",
    }
    checks = []
    for relative, text in evidence.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        checks.append({"check": {"06_结果/GAD固定缩放器_v1.manifest.json": "frozen_scaler_and_anchor", "06_结果/检查点2_GAD构造审计_v1.csv": "component_formula_lag_recursion_registry", "06_结果/检查点2_初始化覆盖_v1.csv": "initialization_and_restart_warmup", "06_结果/检查点2_容量报告_v1.json": "capacity"}[relative], "status": "pass", "evidence": relative, "hash": sha256_file(path), "hash_algorithm": "sha256", "hash_kind": "file_sha256"})
    receipt = tmp_path / "06_结果/检查点2_验收回执_v1.json"
    receipt.write_text(json.dumps({"checks": checks}), encoding="utf-8")

    verify_checkpoint2_evidence_receipt(receipt, tmp_path, verify_git_state=False)
    (tmp_path / "06_结果/检查点2_初始化覆盖_v1.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(CheckpointFailure, match="evidence hash mismatch"):
        verify_checkpoint2_evidence_receipt(receipt, tmp_path, verify_git_state=False)


def test_evidence_only_lineage_allows_only_the_four_checkpoint_files() -> None:
    allowed = {
        "06_结果/检查点2_GAD构造审计_v1.csv",
        "06_结果/检查点2_初始化覆盖_v1.csv",
        "06_结果/检查点2_容量报告_v1.json",
        "06_结果/检查点2_验收回执_v1.json",
    }
    validate_evidence_lineage("a" * 40, "a" * 40, (), is_ancestor=True)
    validate_evidence_lineage("a" * 40, "b" * 40, tuple(sorted(allowed)), is_ancestor=True)
    with pytest.raises(CheckpointFailure, match="evidence-only"):
        validate_evidence_lineage("a" * 40, "b" * 40, ("03_代码/src/green_debt/gad.py",), is_ancestor=True)


def test_git_lineage_parser_decodes_quoted_non_ascii_evidence_paths() -> None:
    raw = '"06_\\347\\273\\223\\346\\236\\234/\\346\\243\\200\\346\\237\\245\\347\\202\\2712_\\345\\256\\271\\351\\207\\217\\346\\212\\245\\345\\221\\212_v1.json"\n'

    assert parse_git_changed_paths(raw) == ("06_结果/检查点2_容量报告_v1.json",)


def test_checkpoint_records_and_enforces_absolute_150_gib_ceiling() -> None:
    with pytest.raises(CheckpointFailure, match="150 GB absolute"):
        evaluate_checkpoint(
            number=2,
            required_manifests=(),
            present_manifests=set(),
            project_bytes=150 * GIB,
            intermediate_bytes=1 * GIB,
            duplicate_keys=0,
            raw_hash_mismatches=0,
        )


def test_construction_evidence_declares_scaled_source_recomputation() -> None:
    authority, _, _ = _authority()

    audit = build_construction_audit(authority, implementation_commit="f" * 40)

    assert audit.get_column("source_recomputed_from_scaled").unique().to_list() == [True]
    assert audit.get_column("implementation_commit").unique().to_list() == ["f" * 40]


def test_initialization_evidence_binds_its_implementation_commit() -> None:
    scaled, sample = _sources()
    components = scaled.with_columns(
        pl.lit(1.0).alias("external_exposure_core"),
        pl.lit(1.0).alias("absorption_core"),
    )

    audit = derive_core_eligibility(components, sample, implementation_commit="e" * 40)

    assert audit.get_column("implementation_commit").unique().to_list() == ["e" * 40]
