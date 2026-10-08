import copy
import json
from pathlib import Path
import subprocess

import polars as pl

import pytest

from green_debt.checkpoints import (
    CheckpointFailure,
    canonical_economy_universe_hash,
    validate_clean_tracked_status,
    validate_frozen_gad_economy_sets,
    load_frozen_gad_universe_anchor,
    pytest_summary,
    validate_head_receipt_bytes,
    validate_evidence_commit_history,
    validate_checkpoint2_receipt_payload,
    validate_checkpoint2_supporting_evidence,
    verify_checkpoint2_evidence_bundle,
)
from green_debt.storage import sha256_file


def _checks() -> list[dict[str, object]]:
    return [
        {"check": name, "status": "pass", "evidence": path, "hash": letter * 64, "hash_algorithm": "sha256", "hash_kind": "file_sha256"}
        for name, path, letter in (
            ("frozen_scaler_and_anchor", "06_结果/GAD固定缩放器_v1.manifest.json", "a"),
            ("component_formula_lag_recursion_registry", "06_结果/检查点2_GAD构造审计_v1.csv", "b"),
            ("initialization_and_restart_warmup", "06_结果/检查点2_初始化覆盖_v1.csv", "c"),
            ("capacity", "06_结果/检查点2_容量报告_v1.json", "d"),
        )
    ]


def _receipt() -> tuple[dict[str, object], dict[str, object], dict[str, object], list[dict[str, object]]]:
    details = {
        "implementation_commit": "a" * 40,
        "scaler_anchor_implementation_commit": "f" * 40,
        "scaler_hash": "b" * 64,
        "gad_output_sha256": "c" * 64,
        "gad_manifest_sha256": "d" * 64,
        "gad_rows": 10,
        "gad_economies": 2,
    }
    facts: dict[str, object] = {
        "number": 2,
        "passed": True,
        "test_command": "python -m pytest -q",
        "test_exit_code": 0,
        "raw_hash_mismatches": 0,
        "stale_input_artifacts": 0,
        "taxonomy_count_mismatches": 0,
        "unresolved_mappings": 0,
        "source_audit_failures": 0,
        "sample_audit_failures": 0,
        "manifest_count": 3,
        "duplicate_keys": 0,
        "intermediate_bytes": 10,
        "project_bytes": 20,
        "layer_byte_counts": {"audits": 5},
    }
    payload: dict[str, object] = {
        **details,
        **facts,
        "git_commit": "a" * 40,
        "passed": True,
        "test_counts": {"summary": "10 passed"},
        "evidence_commit": None,
        "evidence_commit_resolution": "git_head_containing_receipt_verified_by_verify_only",
        "checks": _checks(),
        "checked_at_utc": "stable-format-but-volatile-value",
    }
    return payload, details, facts, _checks()


def _supporting_evidence(tmp_path: Path) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    root = tmp_path / "code"
    results = root / "06_结果"
    results.mkdir(parents=True)
    construction = pl.DataFrame(
        {"specification_id": ["gad_core"], "year": [2000], "rows": [72], "implementation_commit": ["a" * 40]}
    )
    initialization = pl.DataFrame(
        {"economy_id": ["AAA"], "core_eligibility_reason": ["eligible"], "implementation_commit": ["a" * 40]}
    )
    construction.write_csv(results / "检查点2_GAD构造审计_v1.csv")
    initialization.write_csv(results / "检查点2_初始化覆盖_v1.csv")
    capacity: dict[str, object] = {
        "data_root": str(tmp_path / "data"),
        "intermediate_bytes": 10,
        "intermediate_plus_scratch_bytes": 10,
        "project_bytes": 20,
        "usage_bytes": {"raw": 1, "normalized": 2, "harmonized": 3, "measures": 4, "analysis": 0, "scratch": 0, "metadata": 5, "audits": 5},
        "limits_bytes": {"intermediate_plus_scratch_less_than": 25, "project_soft_stop_less_than": 120, "project_absolute_less_than": 150, "filesystem_reserve_at_least": 30},
        "filesystem_free_bytes": 31,
        "passed": True,
    }
    (results / "检查点2_容量报告_v1.json").write_text(json.dumps(capacity), encoding="utf-8")
    return construction, initialization, capacity


def test_frozen_universe_hash_is_order_independent_and_identity_sensitive() -> None:
    assert canonical_economy_universe_hash(("BBB", "AAA")) == canonical_economy_universe_hash(("AAA", "BBB"))
    assert canonical_economy_universe_hash(("AAA", "BBB")) != canonical_economy_universe_hash(("AAA", "CCC"))


def test_frozen_universe_anchor_is_a_separate_compact_contract(tmp_path) -> None:
    anchor = tmp_path / "gad_frozen_universe.json"
    anchor.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "provisional_core_count": 72,
                "economy_ids_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    assert load_frozen_gad_universe_anchor(anchor) == (72, "a" * 64)


@pytest.mark.parametrize("field,value", [
    ("passed", False), ("implementation_commit", "f" * 40),
    ("scaler_anchor_implementation_commit", "e" * 40),
    ("gad_output_sha256", "e" * 64), ("test_command", "other"),
    ("manifest_count", 999), ("layer_byte_counts", {"audits": 999}),
    ("test_counts", {"summary": "999 passed"}),
])
def test_receipt_payload_comparison_rejects_critical_field_tampering(field: str, value: object) -> None:
    payload, details, facts, checks = _receipt()
    tampered = copy.deepcopy(payload)
    tampered[field] = value
    with pytest.raises(CheckpointFailure, match="receipt field mismatch"):
        validate_checkpoint2_receipt_payload(tampered, details, git_commit="a" * 40, verified_head_commit="a" * 40, test_summary="10 passed", receipt_facts=facts, expected_checks=checks)


def test_evidence_commit_history_rejects_an_illegal_intermediate_path() -> None:
    allowed = (
        "06_结果/检查点2_GAD构造审计_v1.csv",
        "06_结果/检查点2_初始化覆盖_v1.csv",
        "06_结果/检查点2_容量报告_v1.json",
        "06_结果/检查点2_验收回执_v1.json",
    )
    validate_evidence_commit_history((("b" * 40, allowed),))
    with pytest.raises(CheckpointFailure, match="evidence-only"):
        validate_evidence_commit_history((("b" * 40, ("03_代码/src/green_debt/gad.py",)),))


def test_receipt_git_commit_is_the_recomputed_implementation_commit_not_evidence_head() -> None:
    payload, details, facts, checks = _receipt()
    validate_checkpoint2_receipt_payload(
        payload, details, git_commit="b" * 40, verified_head_commit="b" * 40, test_summary="10 passed", receipt_facts=facts, expected_checks=checks
    )


def test_receipt_projection_ignores_volatile_check_timestamp() -> None:
    payload, details, facts, checks = _receipt()
    payload["checked_at_utc"] = "old"
    facts["checked_at_utc"] = "new"
    validate_checkpoint2_receipt_payload(
        payload, details, git_commit="a" * 40, verified_head_commit="a" * 40, test_summary="10 passed", receipt_facts=facts, expected_checks=checks
    )


@pytest.mark.parametrize("field,value", [
    ("check", "wrong"), ("status", "fail"), ("evidence", "06_结果/wrong.csv"),
    ("hash", "f" * 64), ("hash_algorithm", "sha1"), ("hash_kind", "wrong"),
])
def test_receipt_projection_rejects_every_check_field_and_order(field: str, value: object) -> None:
    payload, details, facts, checks = _receipt()
    tampered = copy.deepcopy(payload)
    tampered["checks"][0][field] = value
    with pytest.raises(CheckpointFailure, match="receipt"):
        validate_checkpoint2_receipt_payload(tampered, details, git_commit="a" * 40, verified_head_commit="a" * 40, test_summary="10 passed", receipt_facts=facts, expected_checks=checks)
    reordered = copy.deepcopy(payload)
    reordered["checks"] = list(reversed(reordered["checks"]))
    with pytest.raises(CheckpointFailure, match="receipt"):
        validate_checkpoint2_receipt_payload(reordered, details, git_commit="a" * 40, verified_head_commit="a" * 40, test_summary="10 passed", receipt_facts=facts, expected_checks=checks)


def test_receipt_projection_rejects_unverified_head_and_unknown_stable_key() -> None:
    payload, details, facts, checks = _receipt()
    with pytest.raises(CheckpointFailure, match="verified HEAD"):
        validate_checkpoint2_receipt_payload(payload, details, git_commit="f" * 40, verified_head_commit="a" * 40, test_summary="10 passed", receipt_facts=facts, expected_checks=checks)
    payload["unexpected"] = True
    with pytest.raises(CheckpointFailure, match="key set"):
        validate_checkpoint2_receipt_payload(payload, details, git_commit="a" * 40, verified_head_commit="a" * 40, test_summary="10 passed", receipt_facts=facts, expected_checks=checks)


def test_verify_only_receipt_and_tracked_state_helpers_reject_dirty_or_different_bytes() -> None:
    validate_clean_tracked_status("")
    validate_head_receipt_bytes(b"same", b"same")
    with pytest.raises(CheckpointFailure, match="clean tracked"):
        validate_clean_tracked_status(" M 03_代码/src/green_debt/gad.py\\n")
    with pytest.raises(CheckpointFailure, match="HEAD-tracked"):
        validate_head_receipt_bytes(b"receipt", b"tampered")


def test_pytest_summary_compares_the_stable_result_not_wall_clock_duration() -> None:
    assert pytest_summary("..\n216 passed in 4.02s\n") == "216 passed"
    assert pytest_summary("..\n216 passed in 9.91s\n") == "216 passed"


def test_frozen_universe_rejects_a_synchronized_deletion_from_sample_scaled_and_gad() -> None:
    expected = ("AAA", "BBB")
    with pytest.raises(CheckpointFailure, match="frozen 72-economy universe"):
        validate_frozen_gad_economy_sets(
            expected_count=2,
            expected_hash=canonical_economy_universe_hash(expected),
            provisional=("AAA",),
            scaled=("AAA",),
            gad=("AAA",),
        )


@pytest.mark.parametrize("kind", ("construction", "initialization", "capacity"))
def test_supporting_evidence_semantics_rejects_synchronized_hash_style_tampering(tmp_path: Path, kind: str) -> None:
    construction, initialization, capacity = _supporting_evidence(tmp_path)
    root = tmp_path / "code"
    results = root / "06_结果"
    if kind == "construction":
        construction.with_columns(pl.lit(999).alias("rows")).write_csv(results / "检查点2_GAD构造审计_v1.csv")
    elif kind == "initialization":
        initialization.with_columns(pl.lit("tampered").alias("core_eligibility_reason")).write_csv(results / "检查点2_初始化覆盖_v1.csv")
    else:
        altered = dict(capacity)
        altered["passed"] = False
        (results / "检查点2_容量报告_v1.json").write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(CheckpointFailure, match="supporting evidence"):
        validate_checkpoint2_supporting_evidence(
            root,
            construction=construction,
            initialization=initialization,
            capacity_projection=capacity,
        )


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


@pytest.mark.parametrize("kind", ("construction", "initialization", "capacity"))
def test_verify_bundle_rejects_sync_forgery_even_when_receipt_hash_head_and_lineage_are_valid(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    results = root / "06_结果"
    results.mkdir()
    (results / "GAD固定缩放器_v1.manifest.json").write_text("anchor\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "implementation")
    implementation = _git(root, "rev-parse", "HEAD")
    construction, initialization, capacity = _supporting_evidence(tmp_path / "bundle")
    construction.write_csv(results / "检查点2_GAD构造审计_v1.csv")
    initialization.write_csv(results / "检查点2_初始化覆盖_v1.csv")
    (results / "检查点2_容量报告_v1.json").write_text(json.dumps(capacity), encoding="utf-8")
    payload, details, facts, _ = _receipt()
    details["implementation_commit"] = implementation
    payload.update(details)
    payload["git_commit"] = implementation
    checks = []
    for check, path in (
        ("frozen_scaler_and_anchor", "06_结果/GAD固定缩放器_v1.manifest.json"),
        ("component_formula_lag_recursion_registry", "06_结果/检查点2_GAD构造审计_v1.csv"),
        ("initialization_and_restart_warmup", "06_结果/检查点2_初始化覆盖_v1.csv"),
        ("capacity", "06_结果/检查点2_容量报告_v1.json"),
    ):
        checks.append({"check": check, "status": "pass", "evidence": path, "hash": sha256_file(root / path), "hash_algorithm": "sha256", "hash_kind": "file_sha256"})
    payload["checks"] = checks
    receipt = results / "检查点2_验收回执_v1.json"
    receipt.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    _git(root, "add", "06_结果")
    _git(root, "commit", "-qm", "evidence")
    if kind == "construction":
        construction.with_columns(pl.lit(999).alias("rows")).write_csv(results / "检查点2_GAD构造审计_v1.csv")
    elif kind == "initialization":
        initialization.with_columns(pl.lit("forged").alias("core_eligibility_reason")).write_csv(results / "检查点2_初始化覆盖_v1.csv")
    else:
        forged = dict(capacity)
        forged["passed"] = False
        (results / "检查点2_容量报告_v1.json").write_text(json.dumps(forged), encoding="utf-8")
    changed_path = {"construction": "06_结果/检查点2_GAD构造审计_v1.csv", "initialization": "06_结果/检查点2_初始化覆盖_v1.csv", "capacity": "06_结果/检查点2_容量报告_v1.json"}[kind]
    payload["checks"][{"construction": 1, "initialization": 2, "capacity": 3}[kind]]["hash"] = sha256_file(root / changed_path)
    receipt.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    _git(root, "add", changed_path, "06_结果/检查点2_验收回执_v1.json")
    _git(root, "commit", "-qm", "forged evidence")
    with pytest.raises(CheckpointFailure, match="supporting evidence"):
        verify_checkpoint2_evidence_bundle(
            receipt,
            root,
            construction=construction,
            initialization=initialization,
            capacity_projection=capacity,
            details=details,
            receipt_facts=facts,
            test_summary="10 passed",
        )
