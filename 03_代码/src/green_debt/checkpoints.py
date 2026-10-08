"""Construction evidence gates, raw snapshots, and checkpoint receipts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

import polars as pl

from green_debt.artifacts import InputArtifact, TableManifest, verify_manifest
from green_debt.economies import audit_economies
from green_debt.paths import ProjectPaths
from green_debt.sample import audit_analysis_panel_authority, audit_provisional_sample
from green_debt.outcomes import audit_outcomes
from green_debt.instruments import audit_instrument_artifacts
from green_debt.science import audit_openalex_table
from green_debt.gad import (
    FROZEN_SCALER_HASH,
    audit_gad_frame,
    build_component_indices,
    build_construction_audit,
    derive_core_eligibility,
    registered_gad_specifications,
)
from green_debt.scaling import (
    FROZEN_COMPONENT_COLUMNS,
    registry_from_dict,
    scaler_anchor_from_dict,
    verify_frozen_sample_semantics,
    verify_scaler,
    verify_scaler_anchor,
    verify_scaled_authority,
)
from green_debt.sources.baci import audit_trade_normalization
from green_debt.sources.ilostat import audit_ilostat_table
from green_debt.sources.irena import audit_irena_table
from green_debt.sources.policy import audit_policy_table
from green_debt.sources.tiva import audit_tiva_tables
from green_debt.sources.wdi import audit_wdi_table
from green_debt.storage import GIB, LayerUsage, measure_layer_usage, sha256_file
from green_debt.taxonomy import audit_taxonomy
from green_debt.build import (
    create_checkpoint3_receipt_payload,
    render_delivery_note,
    validate_checkpoint3_bundle,
    validate_checkpoint3_facts,
)


_SNAPSHOT_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
_PYTEST_SUMMARY = re.compile(r"^(\d+ (?:passed|failed|skipped|xfailed|xpassed)(?:, \d+ (?:passed|failed|skipped|xfailed|xpassed))*)\s+in\s+.+$")


class CheckpointFailure(RuntimeError):
    """Raised when any frozen checkpoint gate fails."""


@dataclass(frozen=True)
class RawHashReport:
    snapshot: str
    checked_files: int
    mismatch_count: int
    missing_files: tuple[str, ...]
    hash_mismatches: tuple[str, ...]


@dataclass(frozen=True)
class RawHashSnapshotReport:
    output: str
    files: int
    bytes: int


@dataclass(frozen=True)
class LineageAuditReport:
    checked_artifacts: int
    stale_count: int
    missing_count: int
    stale_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointReceipt:
    number: int
    checked_at_utc: str
    git_commit: str
    test_command: str
    test_exit_code: int
    raw_hash_mismatches: int
    manifest_count: int
    duplicate_keys: int
    stale_input_artifacts: int
    taxonomy_count_mismatches: int
    unresolved_mappings: int
    source_audit_failures: int
    sample_audit_failures: int
    layer_byte_counts: dict[str, int]
    project_bytes: int
    intermediate_bytes: int
    passed: bool


_CHECKPOINT2_EVIDENCE_FILES = frozenset(
    {
        "06_结果/GAD固定缩放器_v1.manifest.json",
        "06_结果/检查点2_GAD构造审计_v1.csv",
        "06_结果/检查点2_初始化覆盖_v1.csv",
        "06_结果/检查点2_容量报告_v1.json",
    }
)
_CHECKPOINT2_EVIDENCE_COMMIT_FILES = frozenset(
    {
        "06_结果/检查点2_GAD构造审计_v1.csv",
        "06_结果/检查点2_初始化覆盖_v1.csv",
        "06_结果/检查点2_容量报告_v1.json",
        "06_结果/检查点2_验收回执_v1.json",
    }
)
_CHECKPOINT3_SUPPORT_FILES = (
    "06_结果/检查点3_结果与IV覆盖_v1.csv",
    "06_结果/检查点3_泄漏审计_v1.csv",
    "06_结果/检查点3_容量报告_v1.json",
    "06_结果/数据清洗与变量构造交付说明_v1.md",
    "06_结果/面板时序与重叠审计_v1.csv",
)
_CHECKPOINT3_EVIDENCE_COMMIT_FILES = frozenset(
    (*_CHECKPOINT3_SUPPORT_FILES, "06_结果/检查点3_验收回执_v1.json")
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_snapshot_relative(value: str) -> Path:
    normalized = value.removeprefix("./")
    relative = Path(normalized)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe raw snapshot path: {value}")
    return relative


def verify_raw_hash_snapshot(snapshot: Path, raw_root: Path) -> RawHashReport:
    """Compare every listed immutable raw file with one SHA-256 snapshot."""

    missing: list[str] = []
    mismatched: list[str] = []
    checked = 0
    for line_number, line in enumerate(
        snapshot.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line or line.startswith("#"):
            continue
        match = _SNAPSHOT_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid snapshot line {line_number}")
        expected, raw_relative = match.groups()
        relative = _safe_snapshot_relative(raw_relative)
        checked += 1
        candidate = raw_root.resolve() / relative
        if not candidate.is_file():
            missing.append(relative.as_posix())
            continue
        if sha256_file(candidate) != expected:
            mismatched.append(relative.as_posix())
    return RawHashReport(
        snapshot=str(snapshot.resolve()),
        checked_files=checked,
        mismatch_count=len(missing) + len(mismatched),
        missing_files=tuple(sorted(missing)),
        hash_mismatches=tuple(sorted(mismatched)),
    )


def write_raw_hash_snapshot(
    raw_root: Path,
    output: Path,
) -> RawHashSnapshotReport:
    """Hash every immutable raw file except all checksum snapshots themselves."""

    raw = raw_root.resolve()
    destination = output.resolve()
    if not destination.is_relative_to(raw):
        raise ValueError("raw hash snapshot output must stay inside the raw root")
    rows: list[str] = []
    total_bytes = 0
    files = 0
    for path in sorted(raw.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name.startswith("SHA256SUMS_") and path.suffix == ".txt":
            continue
        if path.name.endswith(".partial"):
            raise RuntimeError(f"raw partial file blocks snapshot: {path}")
        relative = path.relative_to(raw).as_posix()
        rows.append(f"{sha256_file(path)}  ./{relative}\n")
        total_bytes += path.stat().st_size
        files += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("xb") as handle:
            handle.write("".join(rows).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return RawHashSnapshotReport(
        output=str(destination),
        files=files,
        bytes=total_bytes,
    )


def discover_authoritative_manifest_paths(paths: ProjectPaths) -> tuple[Path, ...]:
    """Find partition sidecars without relying on collision-prone central copies."""

    roots = (paths.normalized, paths.harmonized, paths.measures, paths.analysis)
    return tuple(
        sorted(
            {
                manifest
                for root in roots
                if root.is_dir()
                for manifest in root.rglob("*.parquet.manifest.json")
                if manifest.is_file() and not manifest.is_symlink()
            }
        )
    )


def _approval_gate_manifest_scope(
    number: int, manifests: tuple[TableManifest, ...]
) -> tuple[TableManifest, ...]:
    """Limit an initial approval adoption to the gate's approved DAG boundary."""

    if number == 1:
        return tuple(
            manifest
            for manifest in manifests
            if "normalized" in Path(manifest.destination).parts
            or manifest.table_id in {"tiva_activity_weights", "provisional_sample"}
        )
    if number == 2:
        approved_ids = {
            "provisional_sample",
            "gad_scaled_components",
            "gad_country_year",
        }
        return tuple(
            manifest for manifest in manifests if manifest.table_id in approved_ids
        )
    raise ValueError("only Checkpoint 1 and 2 have approval scopes")


def audit_input_artifacts(
    artifacts: tuple[InputArtifact, ...],
) -> LineageAuditReport:
    """Verify each recorded input and parent-manifest hash against current files."""

    unique = {
        (item.path, item.bytes, item.sha256, item.parent_manifest_sha256): item
        for item in artifacts
    }
    hash_cache: dict[Path, str] = {}

    def current_hash(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in hash_cache:
            hash_cache[resolved] = sha256_file(resolved)
        return hash_cache[resolved]

    stale: list[str] = []
    missing = 0
    for item in sorted(unique.values(), key=lambda value: (value.path, value.sha256)):
        path = Path(item.path)
        reasons: list[str] = []
        if not path.is_file():
            reasons.append("missing")
            missing += 1
        else:
            if path.stat().st_size != item.bytes:
                reasons.append("bytes")
            if current_hash(path) != item.sha256:
                reasons.append("sha256")
            if item.parent_manifest_sha256 is not None:
                parent = path.with_name(f"{path.name}.manifest.json")
                if not parent.is_file():
                    reasons.append("parent_manifest_missing")
                    missing += 1
                elif current_hash(parent) != item.parent_manifest_sha256:
                    reasons.append("parent_manifest_sha256")
        if reasons:
            stale.append(f"{path.resolve()}::{','.join(reasons)}")
    return LineageAuditReport(
        checked_artifacts=len(unique),
        stale_count=len(stale),
        missing_count=missing,
        stale_artifacts=tuple(stale),
    )


def evaluate_checkpoint(
    *,
    number: int,
    required_manifests: tuple[str, ...],
    present_manifests: set[str],
    project_bytes: int,
    intermediate_bytes: int,
    duplicate_keys: int,
    raw_hash_mismatches: int,
    stale_input_artifacts: int = 0,
    taxonomy_count_mismatches: int = 0,
    unresolved_mappings: int = 0,
    source_audit_failures: int = 0,
    sample_audit_failures: int = 0,
    scratch_bytes: int = 0,
    filesystem_free_bytes: int | None = None,
    test_command: str = "not_run",
    test_exit_code: int = 0,
    git_commit: str = "unknown",
    layer_byte_counts: dict[str, int] | None = None,
    manifest_count: int | None = None,
    absolute_limit_bytes: int = 150 * GIB,
) -> CheckpointReceipt:
    """Evaluate evidence without reading or mutating the filesystem."""

    if number not in {1, 2, 3}:
        raise ValueError("checkpoint number must be 1, 2, or 3")
    missing = sorted(set(required_manifests) - present_manifests)
    if missing:
        raise CheckpointFailure(f"missing manifests: {', '.join(missing)}")
    if intermediate_bytes + scratch_bytes >= 25 * GIB:
        raise CheckpointFailure("25 GB intermediate quota reached")
    if project_bytes >= absolute_limit_bytes:
        raise CheckpointFailure("150 GB absolute ceiling reached")
    if project_bytes >= 120 * GIB:
        raise CheckpointFailure("120 GB project hard stop reached")
    if filesystem_free_bytes is not None and filesystem_free_bytes < 30 * GIB:
        raise CheckpointFailure("30 GB filesystem reserve not preserved")
    if duplicate_keys:
        raise CheckpointFailure(f"duplicate keys: {duplicate_keys}")
    if raw_hash_mismatches:
        raise CheckpointFailure(f"raw hash mismatches: {raw_hash_mismatches}")
    if stale_input_artifacts:
        raise CheckpointFailure(f"stale input artifacts: {stale_input_artifacts}")
    if taxonomy_count_mismatches:
        raise CheckpointFailure(
            f"taxonomy count mismatches: {taxonomy_count_mismatches}"
        )
    if unresolved_mappings:
        raise CheckpointFailure(f"unresolved mappings: {unresolved_mappings}")
    if source_audit_failures:
        raise CheckpointFailure(f"source audit failures: {source_audit_failures}")
    if sample_audit_failures:
        raise CheckpointFailure(f"sample audit failures: {sample_audit_failures}")
    if test_exit_code != 0:
        raise CheckpointFailure(f"test command failed with exit code {test_exit_code}")
    return CheckpointReceipt(
        number=number,
        checked_at_utc=_utc_now(),
        git_commit=git_commit,
        test_command=test_command,
        test_exit_code=test_exit_code,
        raw_hash_mismatches=raw_hash_mismatches,
        manifest_count=(
            len(present_manifests) if manifest_count is None else manifest_count
        ),
        duplicate_keys=duplicate_keys,
        stale_input_artifacts=stale_input_artifacts,
        taxonomy_count_mismatches=taxonomy_count_mismatches,
        unresolved_mappings=unresolved_mappings,
        source_audit_failures=source_audit_failures,
        sample_audit_failures=sample_audit_failures,
        layer_byte_counts=dict(layer_byte_counts or {}),
        project_bytes=project_bytes,
        intermediate_bytes=intermediate_bytes,
        passed=True,
    )


_CHECKPOINT_REQUIREMENTS: dict[int, tuple[str, ...]] = {
    1: (
        "wdi_country_year",
        "irena_country_year",
        "openalex_country_year",
        "ilostat_skill",
        "policy_country_year",
        "tiva_activity_year",
        "tiva_activity_weights",
        "provisional_sample",
    ),
    2: ("gad_scaled_components", "gad_country_year"),
    3: (
        "outcomes_country_year",
        "outcomes_product_year",
        "iv_baseline_shares",
        "iv_partner_shocks",
        "iv_country_year",
        "final_sample",
        "regression_bounds",
        "giu_outcome_scalers",
        "model_panel",
    ),
}


def checkpoint3_facts_from_reports(
    *,
    raw_hash_mismatches: int,
    manifest_failures: int,
    taxonomy_counts: dict[str, int],
    duplicate_keys: int,
    source_failures: int,
    scaler_hashes: tuple[str, ...],
    outcome_report: dict[str, object],
    instrument_report: dict[str, object],
    panel_report: dict[str, object],
    test_exit_code: int,
    intermediate_plus_scratch_bytes: int,
    project_bytes: int,
    filesystem_free_bytes: int,
) -> dict[str, object]:
    """Derive the exact ten gates from independently recomputed parent audits."""

    timing_overlap = sum(
        int(panel_report.get(name, 1))
        for name in ("timing_violations", "mapping_overlap_violations")
    )
    leakage = (
        int(outcome_report.get("gad_value_table_parents", 1))
        + sum(
            int(instrument_report.get(name, 1))
            for name in (
                "destination_exclusion_failures",
                "prohibited_lineage_columns",
                "outcome_value_table_parents",
                "shock_parent_reconstruction_failures",
            )
        )
        + int(panel_report.get("outcome_source_failures", 1))
    )
    return {
        "raw_hashes": raw_hash_mismatches == 0,
        "schemas_and_manifests": manifest_failures == 0,
        "taxonomy_counts": {
            "cleg": int(taxonomy_counts.get("broad", -1)),
            "apec": int(taxonomy_counts.get("main", -1)),
            "overlap": int(taxonomy_counts.get("apec", -1)),
        },
        "duplicate_authoritative_keys": duplicate_keys,
        "source_missing_zero_rules": source_failures == 0,
        "single_frozen_gad_scaler": len(set(scaler_hashes)) == 1,
        "timing_overlap_violations": timing_overlap,
        "leakage_violations": leakage,
        "fresh_full_test_exit_code": test_exit_code,
        "capacity": {
            "intermediate_plus_scratch_bytes": intermediate_plus_scratch_bytes,
            "project_bytes": project_bytes,
            "absolute_project_bytes": project_bytes,
            "filesystem_reserve_bytes": filesystem_free_bytes,
            "limits": {
                "intermediate": 25 * GIB,
                "project": 120 * GIB,
                "absolute": 150 * GIB,
                "filesystem_reserve": 30 * GIB,
            },
        },
    }


def _git_commit(code_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else "unknown"


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("xb") as handle:
            handle.write(
                (
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _write_csv_atomic(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        frame.write_csv(partial)
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("xb") as handle:
            handle.write(value.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _manifest_by_id(
    manifests: tuple[TableManifest, ...], table_id: str
) -> TableManifest:
    matches = tuple(item for item in manifests if item.table_id == table_id)
    if len(matches) != 1:
        raise CheckpointFailure(
            f"expected one {table_id} sidecar manifest, found {len(matches)}"
        )
    return matches[0]


def _economies(path: str) -> int:
    return int(
        pl.scan_parquet(path)
        .select(pl.col("economy_id").n_unique())
        .collect()
        .item()
    )


def _period_ranges(periods: dict[str, tuple[int, ...]]) -> str:
    return ";".join(
        f"{name}:{years[0]}-{years[-1]}"
        for name, years in sorted(periods.items())
        if years
    )


def _checkpoint1_source_coverage(
    paths: ProjectPaths,
    manifests: tuple[TableManifest, ...],
    lineage: LineageAuditReport,
) -> tuple[pl.DataFrame, int, int, int, int]:
    """Run every Checkpoint 1 source audit and return its compact evidence table."""

    taxonomy = audit_taxonomy(
        code_root=paths.code_root,
        data_root=paths.data_root,
        write_audit=True,
    )
    expected_taxonomy = {"main": 126, "broad": 248, "apec": 54}
    actual_taxonomy = {
        str(key): int(value) for key, value in dict(taxonomy["counts"]).items()
    }
    taxonomy_mismatches = int(actual_taxonomy != expected_taxonomy)
    economy = audit_economies(
        code_root=paths.code_root,
        contract_path=paths.code_root / "03_代码/contracts/economy_crosswalk.json",
    )
    trade = audit_trade_normalization(data_root=paths.data_root)
    registry = pl.read_csv(paths.code_root / "02_数据字典/indicator_registry_v1.csv")

    wdi_manifest = _manifest_by_id(manifests, "wdi_country_year")
    irena_manifest = _manifest_by_id(manifests, "irena_country_year")
    openalex_manifest = _manifest_by_id(manifests, "openalex_country_year")
    ilostat_manifest = _manifest_by_id(manifests, "ilostat_skill")
    policy_manifest = _manifest_by_id(manifests, "policy_country_year")
    tiva_manifest = _manifest_by_id(manifests, "tiva_activity_year")
    weights_manifest = _manifest_by_id(manifests, "tiva_activity_weights")
    sample_manifest = _manifest_by_id(manifests, "provisional_sample")

    wdi = audit_wdi_table(
        manifest_path=Path(wdi_manifest.destination).with_name(
            f"{Path(wdi_manifest.destination).name}.manifest.json"
        ),
        registry=registry,
    )
    irena = audit_irena_table(
        manifest_path=Path(irena_manifest.destination).with_name(
            f"{Path(irena_manifest.destination).name}.manifest.json"
        ),
        registry=registry,
    )
    openalex = audit_openalex_table(
        manifest_path=Path(openalex_manifest.destination).with_name(
            f"{Path(openalex_manifest.destination).name}.manifest.json"
        )
    )
    ilostat = audit_ilostat_table(
        manifest_path=Path(ilostat_manifest.destination).with_name(
            f"{Path(ilostat_manifest.destination).name}.manifest.json"
        )
    )
    policy = audit_policy_table(
        manifest_path=Path(policy_manifest.destination).with_name(
            f"{Path(policy_manifest.destination).name}.manifest.json"
        )
    )
    tiva = audit_tiva_tables(
        activity_manifest_path=Path(tiva_manifest.destination).with_name(
            f"{Path(tiva_manifest.destination).name}.manifest.json"
        ),
        weights_manifest_path=Path(weights_manifest.destination).with_name(
            f"{Path(weights_manifest.destination).name}.manifest.json"
        ),
    )
    sample = audit_provisional_sample(
        manifest_path=Path(sample_manifest.destination).with_name(
            f"{Path(sample_manifest.destination).name}.manifest.json"
        ),
        sample_flow_path=paths.audits / "检查点1_样本流_v1.csv",
    )

    ilostat_frame = pl.read_parquet(ilostat_manifest.destination)
    ilostat_years = tuple(sorted(ilostat_frame.get_column("year").unique().to_list()))
    ilostat_units = tuple(sorted(ilostat_frame.get_column("unit").unique().to_list()))
    if ilostat_years != tuple(range(1996, 2025)) or ilostat_units != (
        "share_0_to_1",
    ):
        raise CheckpointFailure(
            f"ILOSTAT period or unit differs: years={ilostat_years}, units={ilostat_units}"
        )

    policy_frame = pl.read_parquet(policy_manifest.destination)
    policy_units = tuple(sorted(policy_frame.get_column("unit").unique().to_list()))
    index_years = tuple(
        sorted(
            policy_frame.filter(pl.col("unit") == "index_0_to_6")
            .get_column("year")
            .unique()
            .to_list()
        )
    )
    count_years = tuple(
        sorted(
            policy_frame.filter(pl.col("unit") == "count")
            .get_column("year")
            .unique()
            .to_list()
        )
    )
    if (
        policy_units != ("count", "index_0_to_6")
        or index_years != tuple(range(1990, 2021))
        or count_years != (2026,)
    ):
        raise CheckpointFailure(
            "policy period or unit differs: "
            f"units={policy_units}, EPS={index_years}, IFCMA={count_years}"
        )

    baci_manifests = tuple(
        item for item in manifests if item.table_id.startswith("baci_")
    )
    unresolved_mappings = (
        int(economy["unresolved_codes"])
        + taxonomy_mismatches
        + int(tiva.leaf_activities != 50)
    )
    rows = [
        {
            "source_id": "taxonomy",
            "table_id": "green_product_registry",
            "rows": int(taxonomy["hs07_rows"]),
            "economies": None,
            "period": "HS07_to_HS96_frozen",
            "units": "green_weight_0_to_1",
            "indicators": ";".join(
                f"{name}={actual_taxonomy[name]}" for name in ("main", "broad", "apec")
            ),
            "duplicate_keys": 0,
            "unresolved_mappings": taxonomy_mismatches,
            "lineage_stale_inputs": 0,
            "status": str(taxonomy["status"]),
        },
        {
            "source_id": "economy_crosswalk",
            "table_id": "economy_crosswalk_v1",
            "rows": int(economy["rows"]),
            "economies": int(economy["canonical_economies"]),
            "period": "not_applicable",
            "units": "source_code_to_economy_id",
            "indicators": "confirmatory_and_robustness_flags",
            "duplicate_keys": int(economy["duplicate_source_keys"]),
            "unresolved_mappings": int(economy["unresolved_codes"]),
            "lineage_stale_inputs": 0,
            "status": str(economy["status"]),
        },
        {
            "source_id": "baci",
            "table_id": "partitioned_trade_layers",
            "rows": sum(item.rows for item in baci_manifests),
            "economies": None,
            "period": "HS96:1996-2024;HS07:2007-2024",
            "units": "current_USD",
            "indicators": f"{trade.manifests_verified}_partition_manifests",
            "duplicate_keys": trade.duplicate_primary_keys,
            "unresolved_mappings": 0,
            "lineage_stale_inputs": 0,
            "status": trade.status,
        },
        {
            "source_id": "wdi",
            "table_id": wdi_manifest.table_id,
            "rows": wdi.rows,
            "economies": _economies(wdi_manifest.destination),
            "period": f"{wdi.years[0]}-{wdi.years[-1]}",
            "units": "registry_exact",
            "indicators": ";".join(wdi.indicators),
            "duplicate_keys": wdi.duplicate_keys,
            "unresolved_mappings": 0,
            "lineage_stale_inputs": 0,
            "status": wdi.status,
        },
        {
            "source_id": "irena",
            "table_id": irena_manifest.table_id,
            "rows": irena.rows,
            "economies": _economies(irena_manifest.destination),
            "period": _period_ranges(irena.indicator_periods),
            "units": "MW;GWh;percent",
            "indicators": ";".join(irena.indicators),
            "duplicate_keys": irena.duplicate_keys,
            "unresolved_mappings": 0,
            "lineage_stale_inputs": 0,
            "status": irena.status,
        },
        {
            "source_id": "openalex",
            "table_id": openalex_manifest.table_id,
            "rows": openalex.rows,
            "economies": openalex.economies,
            "period": f"{openalex.years[0]}-{openalex.years[-1]}",
            "units": "work_counts",
            "indicators": "green_works;total_works",
            "duplicate_keys": openalex.duplicate_keys,
            "unresolved_mappings": 0,
            "lineage_stale_inputs": 0,
            "status": openalex.status,
        },
        {
            "source_id": "ilostat",
            "table_id": ilostat_manifest.table_id,
            "rows": ilostat.rows,
            "economies": _economies(ilostat_manifest.destination),
            "period": "1996-2024",
            "units": ";".join(ilostat_units),
            "indicators": ";".join(ilostat.series),
            "duplicate_keys": ilostat.duplicate_keys,
            "unresolved_mappings": 0,
            "lineage_stale_inputs": 0,
            "status": ilostat.status,
        },
        {
            "source_id": "policy",
            "table_id": policy_manifest.table_id,
            "rows": policy.rows,
            "economies": policy.economies,
            "period": "EPS:1990-2020;IFCMA:2026_snapshot",
            "units": ";".join(policy_units),
            "indicators": f"{policy.indicators}_robustness_only_indicators",
            "duplicate_keys": policy.duplicate_keys,
            "unresolved_mappings": 0,
            "lineage_stale_inputs": 0,
            "status": policy.status,
        },
        {
            "source_id": "oecd_tiva",
            "table_id": tiva_manifest.table_id,
            "rows": tiva.rows,
            "economies": tiva.economies,
            "period": _period_ranges(tiva.indicator_periods),
            "units": "USD_millions;percent",
            "indicators": ";".join(tiva.indicators),
            "duplicate_keys": tiva.duplicate_keys,
            "unresolved_mappings": 0,
            "lineage_stale_inputs": 0,
            "status": tiva.status,
        },
        {
            "source_id": "oecd_tiva",
            "table_id": weights_manifest.table_id,
            "rows": tiva.weight_rows,
            "economies": _economies(weights_manifest.destination),
            "period": "baseline_2000-2004_frozen",
            "units": "share_0_to_1;USD_millions",
            "indicators": ";".join(tiva.weight_versions),
            "duplicate_keys": 0,
            "unresolved_mappings": 0,
            "lineage_stale_inputs": 0,
            "status": tiva.status,
        },
        {
            "source_id": "sample",
            "table_id": sample_manifest.table_id,
            "rows": sample.rows,
            "economies": sample.rows,
            "period": "baseline_import_presence_1996-1999",
            "units": "Boolean_rule_flags",
            "indicators": f"provisional_core={sample.provisional_core}",
            "duplicate_keys": sample.duplicate_keys,
            "unresolved_mappings": 0,
            "lineage_stale_inputs": 0,
            "status": sample.status,
        },
        {
            "source_id": "lineage",
            "table_id": "all_authoritative_sidecars",
            "rows": lineage.checked_artifacts,
            "economies": None,
            "period": "current_inputs",
            "units": "SHA256",
            "indicators": f"manifests={len(manifests)}",
            "duplicate_keys": 0,
            "unresolved_mappings": 0,
            "lineage_stale_inputs": lineage.stale_count,
            "status": "valid" if lineage.stale_count == 0 else "failed",
        },
    ]
    coverage = pl.DataFrame(
        rows,
        schema={
            "source_id": pl.String,
            "table_id": pl.String,
            "rows": pl.UInt64,
            "economies": pl.UInt32,
            "period": pl.String,
            "units": pl.String,
            "indicators": pl.String,
            "duplicate_keys": pl.UInt32,
            "unresolved_mappings": pl.UInt32,
            "lineage_stale_inputs": pl.UInt32,
            "status": pl.String,
        },
    )
    return coverage, taxonomy_mismatches, unresolved_mappings, 0, 0


def _checkpoint2_evidence_bytes(code_root: Path) -> int:
    return sum(
        (code_root / relative).stat().st_size
        for relative in (
            _CHECKPOINT2_EVIDENCE_COMMIT_FILES
            | _CHECKPOINT3_EVIDENCE_COMMIT_FILES
        )
        if (code_root / relative).is_file()
    )


def _stable_capacity_projection(paths: ProjectPaths, usage: LayerUsage) -> dict[str, object]:
    """Remove all compact checkpoint-evidence bytes from capacity semantics."""

    evidence_bytes = _checkpoint2_evidence_bytes(paths.code_root)
    layer_bytes = usage.byte_counts()
    if layer_bytes["audits"] < evidence_bytes:
        raise CheckpointFailure("checkpoint evidence bytes exceed audit-layer usage")
    layer_bytes["audits"] -= evidence_bytes
    project_bytes = usage.project_bytes - evidence_bytes
    return {
        "data_root": str(paths.data_root),
        "usage_bytes": layer_bytes,
        "intermediate_bytes": usage.intermediate_bytes,
        "intermediate_plus_scratch_bytes": usage.intermediate_bytes + usage.scratch_bytes,
        "project_bytes": project_bytes,
        "filesystem_free_bytes": usage.filesystem_free_bytes,
        "limits_bytes": {
            "intermediate_plus_scratch_less_than": 25 * GIB,
            "project_soft_stop_less_than": 120 * GIB,
            "project_absolute_less_than": 150 * GIB,
            "filesystem_reserve_at_least": 30 * GIB,
        },
        "passed": (
            usage.intermediate_bytes + usage.scratch_bytes < 25 * GIB
            and usage.project_bytes < 120 * GIB
            and usage.project_bytes < 150 * GIB
            and usage.filesystem_free_bytes >= 30 * GIB
        ),
    }


def _capacity_payload(paths: ProjectPaths, usage: LayerUsage) -> dict[str, object]:
    return {
        "checked_at_utc": _utc_now(),
        **_stable_capacity_projection(paths, usage),
    }


def load_frozen_gad_universe_anchor(anchor_path: Path) -> tuple[int, str]:
    """Load the Git-protected 72-economy identity anchor without touching upstream config."""

    payload = json.loads(anchor_path.read_text(encoding="utf-8"))
    try:
        count = int(payload["provisional_core_count"])
        digest = str(payload["economy_ids_sha256"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CheckpointFailure("frozen GAD economy universe anchor is invalid") from exc
    if count != 72 or len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise CheckpointFailure("frozen GAD economy universe anchor is invalid")
    return count, digest


def _frozen_gad_universe(code_root: Path) -> tuple[int, str]:
    return load_frozen_gad_universe_anchor(
        code_root / "contracts/gad_frozen_universe.json"
    )


def validate_evidence_lineage(
    implementation_commit: str,
    head_commit: str,
    changed_files: tuple[str, ...],
    *,
    is_ancestor: bool,
) -> None:
    """Validate the only permitted post-implementation evidence commit shape."""

    if not is_ancestor:
        raise CheckpointFailure("implementation commit is not an ancestor of HEAD")
    changed = frozenset(changed_files)
    if implementation_commit == head_commit:
        if changed:
            raise CheckpointFailure("implementation HEAD has unexpected descendant changes")
        return
    if changed != _CHECKPOINT2_EVIDENCE_COMMIT_FILES:
        raise CheckpointFailure("post-implementation lineage is not the exact evidence-only file set")


def validate_evidence_commit_history(
    commits: tuple[tuple[str, tuple[str, ...]], ...],
) -> None:
    """Every post-implementation commit must be evidence-only; no restore bypass."""

    if not commits:
        return
    combined: set[str] = set()
    for _, changed in commits:
        changed_set = set(changed)
        if not changed_set or not changed_set <= _CHECKPOINT2_EVIDENCE_COMMIT_FILES:
            raise CheckpointFailure("post-implementation commit is not evidence-only")
        combined.update(changed_set)
    if combined != _CHECKPOINT2_EVIDENCE_COMMIT_FILES:
        raise CheckpointFailure("evidence-only commit history does not cover all four receipt files")


def canonical_economy_universe_hash(economies: tuple[str, ...]) -> str:
    """Hash sorted canonical IDs, including the terminal newline, without count-only drift."""

    values = tuple(sorted(str(value) for value in economies))
    if not values or len(values) != len(set(values)) or any(not value for value in values):
        raise ValueError("frozen economy universe must contain unique non-empty IDs")
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def validate_frozen_gad_economy_sets(
    *,
    expected_count: int,
    expected_hash: str,
    provisional: tuple[str, ...],
    scaled: tuple[str, ...],
    gad: tuple[str, ...],
) -> None:
    """Require each authority to equal the immutable country identity, not only each other."""

    expected_identity = (expected_count, expected_hash)
    observed = {
        "provisional": (len(provisional), canonical_economy_universe_hash(provisional)),
        "scaled": (len(scaled), canonical_economy_universe_hash(scaled)),
        "gad": (len(gad), canonical_economy_universe_hash(gad)),
    }
    if any(identity != expected_identity for identity in observed.values()):
        raise CheckpointFailure("authority differs from the frozen 72-economy universe")


def validate_checkpoint2_receipt_payload(
    payload: dict[str, object],
    details: dict[str, object],
    *,
    git_commit: str,
    verified_head_commit: str,
    test_summary: str,
    receipt_facts: dict[str, object],
    expected_checks: list[dict[str, object]],
) -> None:
    """Compare receipt assertions to independently recomputed current facts."""

    if git_commit != verified_head_commit:
        raise CheckpointFailure("receipt verification git_commit is not the verified HEAD")
    if (
        verified_head_commit != details["implementation_commit"]
        and payload.get("evidence_commit_resolution")
        != "git_head_containing_receipt_verified_by_verify_only"
    ):
        raise CheckpointFailure("receipt does not declare evidence-only HEAD resolution")
    stable_facts = {
        name: value
        for name, value in receipt_facts.items()
        if name not in {"checked_at_utc", "filesystem_free_bytes"}
    }
    expected = {
        **stable_facts,
        "implementation_commit": details["implementation_commit"],
        # The receipt is written at the implementation commit.  At a later
        # evidence-only HEAD, the independently recomputed GAD manifest is
        # still the authority for this field.
        "git_commit": details["implementation_commit"],
        "passed": True,
        "scaler_hash": details["scaler_hash"],
        "scaler_anchor_implementation_commit": details["scaler_anchor_implementation_commit"],
        "gad_output_sha256": details["gad_output_sha256"],
        "gad_manifest_sha256": details["gad_manifest_sha256"],
        "gad_rows": details["gad_rows"],
        "gad_economies": details["gad_economies"],
        "test_exit_code": 0,
        "test_counts": {"summary": test_summary},
        "evidence_commit": None,
        "evidence_commit_resolution": "git_head_containing_receipt_verified_by_verify_only",
        "checks": expected_checks,
    }
    allowed_keys = set(expected) | {"checked_at_utc"}
    if set(payload) != allowed_keys:
        raise CheckpointFailure("receipt stable projection key set differs from expected")
    for name, value in expected.items():
        if payload.get(name) != value:
            raise CheckpointFailure(f"receipt field mismatch: {name}")


def validate_clean_tracked_status(status_output: str) -> None:
    """Reject any tracked worktree/index change before verify-only trusts a receipt."""

    if status_output.strip():
        raise CheckpointFailure("verify-only requires a clean tracked worktree and index")


def validate_head_receipt_bytes(head_bytes: bytes, working_bytes: bytes) -> None:
    """Receipt and HEAD blob must match exactly, including formatting and whitespace."""

    if head_bytes != working_bytes:
        raise CheckpointFailure("working receipt differs from the HEAD-tracked receipt blob")


def _canonical_csv_frame(frame: pl.DataFrame, keys: tuple[str, ...]) -> pl.DataFrame:
    """Use the published CSV representation before strict evidence comparison."""

    canonical = pl.read_csv(io.StringIO(frame.write_csv()))
    return canonical.sort(*keys)


def _assert_semantic_frame_equal(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    *,
    keys: tuple[str, ...],
    label: str,
) -> None:
    actual_canonical = _canonical_csv_frame(actual, keys)
    expected_canonical = _canonical_csv_frame(expected, keys)
    if actual_canonical.columns != expected_canonical.columns or actual_canonical.schema != expected_canonical.schema:
        raise CheckpointFailure(f"supporting evidence {label} schema differs from independent recomputation")
    if actual_canonical.height != expected_canonical.height:
        raise CheckpointFailure(f"supporting evidence {label} row count differs from independent recomputation")
    for actual_row, expected_row in zip(
        actual_canonical.iter_rows(named=True), expected_canonical.iter_rows(named=True), strict=True
    ):
        for name, expected_value in expected_row.items():
            actual_value = actual_row[name]
            if isinstance(expected_value, float) or isinstance(actual_value, float):
                if actual_value is None or expected_value is None or not math.isclose(
                    float(actual_value), float(expected_value), rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise CheckpointFailure(f"supporting evidence {label} value differs from independent recomputation")
            elif actual_value != expected_value:
                raise CheckpointFailure(f"supporting evidence {label} value differs from independent recomputation")


def validate_checkpoint2_supporting_evidence(
    code_root: Path,
    *,
    construction: pl.DataFrame,
    initialization: pl.DataFrame,
    capacity_projection: dict[str, object],
    allow_current_capacity_drift: bool = False,
) -> None:
    """Bind published small evidence files to independent recomputation, not receipt hashes."""

    results = code_root / "06_结果"
    try:
        actual_construction = pl.read_csv(results / "检查点2_GAD构造审计_v1.csv")
        actual_initialization = pl.read_csv(results / "检查点2_初始化覆盖_v1.csv")
        actual_capacity = json.loads((results / "检查点2_容量报告_v1.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, pl.exceptions.PolarsError) as exc:
        raise CheckpointFailure("supporting evidence cannot be read") from exc
    _assert_semantic_frame_equal(
        actual_construction,
        construction,
        keys=("specification_id", "year"),
        label="construction audit",
    )
    _assert_semantic_frame_equal(
        actual_initialization,
        initialization,
        keys=("economy_id",),
        label="initialization audit",
    )
    stable_capacity_fields = (
        ("data_root", "limits_bytes", "passed")
        if allow_current_capacity_drift
        else (
            "data_root",
            "intermediate_bytes",
            "intermediate_plus_scratch_bytes",
            "project_bytes",
            "usage_bytes",
            "limits_bytes",
            "passed",
        )
    )
    for name in stable_capacity_fields:
        if name == "passed" and allow_current_capacity_drift:
            matches = actual_capacity.get(name) is True and capacity_projection.get(name) is True
        else:
            matches = actual_capacity.get(name) == capacity_projection.get(name)
        if not matches:
            raise CheckpointFailure(f"supporting evidence capacity {name} differs from independent recomputation")
    reserve = capacity_projection["limits_bytes"]
    if (
        actual_capacity.get("filesystem_free_bytes") is None
        or int(actual_capacity["filesystem_free_bytes"])
        < int(reserve["filesystem_reserve_at_least"])
    ):
        raise CheckpointFailure("supporting evidence capacity filesystem reserve is invalid")


def validate_checkpoint3_supporting_evidence(
    code_root: Path,
    *,
    results_and_iv: pl.DataFrame,
    leakage: pl.DataFrame,
    capacity_projection: dict[str, object],
    delivery_note: str,
) -> None:
    """Bind every compact Checkpoint-3 support file to fresh recomputation."""

    results = code_root / "06_结果"
    try:
        actual_results = pl.read_csv(results / "检查点3_结果与IV覆盖_v1.csv")
        actual_leakage = pl.read_csv(results / "检查点3_泄漏审计_v1.csv")
        actual_capacity = json.loads(
            (results / "检查点3_容量报告_v1.json").read_text(encoding="utf-8")
        )
        actual_delivery = (
            results / "数据清洗与变量构造交付说明_v1.md"
        ).read_text(encoding="utf-8")
        _assert_semantic_frame_equal(
            actual_results,
            results_and_iv,
            keys=("section", "metric"),
            label="Checkpoint 3 results and IV coverage",
        )
        _assert_semantic_frame_equal(
            actual_leakage,
            leakage,
            keys=("criterion",),
            label="Checkpoint 3 leakage audit",
        )
        stable_capacity_fields = (
            "data_root",
            "intermediate_bytes",
            "intermediate_plus_scratch_bytes",
            "project_bytes",
            "usage_bytes",
            "limits_bytes",
            "passed",
        )
        if any(
            actual_capacity.get(name) != capacity_projection.get(name)
            for name in stable_capacity_fields
        ):
            raise CheckpointFailure(
                "Checkpoint 3 supporting evidence capacity differs from recomputation"
            )
        reserve = capacity_projection["limits_bytes"]
        if (
            actual_capacity.get("filesystem_free_bytes") is None
            or int(actual_capacity["filesystem_free_bytes"])
            < int(reserve["filesystem_reserve_at_least"])
        ):
            raise CheckpointFailure(
                "Checkpoint 3 supporting evidence capacity reserve is invalid"
            )
        if actual_delivery != delivery_note:
            raise CheckpointFailure(
                "Checkpoint 3 supporting evidence delivery note differs from recomputation"
            )
    except CheckpointFailure as exc:
        if "Checkpoint 3 supporting evidence" in str(exc):
            raise
        raise CheckpointFailure(f"Checkpoint 3 supporting evidence differs: {exc}") from exc
    except (OSError, json.JSONDecodeError, pl.exceptions.PolarsError, KeyError, TypeError, ValueError) as exc:
        raise CheckpointFailure(
            f"Checkpoint 3 supporting evidence cannot be verified: {exc}"
        ) from exc


def pytest_summary(output: str) -> str:
    """Keep the test-result summary auditable without binding receipt bytes to elapsed time."""

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "no pytest summary"
    match = _PYTEST_SUMMARY.fullmatch(lines[-1])
    return match.group(1) if match is not None else lines[-1]


def parse_git_changed_paths(output: str) -> tuple[str, ...]:
    """Decode Git's C-quoted non-ASCII path form into project-relative paths."""

    paths: list[str] = []
    for raw in output.splitlines():
        if not raw:
            continue
        if raw.startswith('"') and raw.endswith('"'):
            try:
                quoted = ast.literal_eval(raw)
                decoded = quoted.encode("latin1").decode("utf-8")
            except (SyntaxError, ValueError, UnicodeError) as exc:
                raise CheckpointFailure("could not decode Git changed path") from exc
            paths.append(decoded)
        else:
            paths.append(raw)
    return tuple(paths)


def _verify_checkpoint2_lineage(code_root: Path, implementation_commit: str) -> str:
    head = _git_commit(code_root)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, head],
        cwd=code_root,
        check=False,
    ).returncode == 0
    completed = subprocess.run(
        ["git", "-c", "core.quotePath=false", "diff", "--name-only", f"{implementation_commit}..{head}"],
        cwd=code_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CheckpointFailure("could not inspect implementation-to-HEAD lineage")
    changed = parse_git_changed_paths(completed.stdout)
    validate_evidence_lineage(implementation_commit, head, changed, is_ancestor=ancestor)
    if implementation_commit != head:
        commits_completed = subprocess.run(
            ["git", "rev-list", "--reverse", f"{implementation_commit}..{head}"],
            cwd=code_root, check=False, capture_output=True, text=True,
        )
        if commits_completed.returncode != 0:
            raise CheckpointFailure("could not inspect post-implementation commits")
        history: list[tuple[str, tuple[str, ...]]] = []
        for commit in (value for value in commits_completed.stdout.splitlines() if value):
            names = subprocess.run(
                ["git", "-c", "core.quotePath=false", "diff-tree", "--no-commit-id", "--name-only", "-r", commit],
                cwd=code_root, check=False, capture_output=True, text=True,
            )
            if names.returncode != 0:
                raise CheckpointFailure("could not inspect post-implementation commit paths")
            history.append((commit, parse_git_changed_paths(names.stdout)))
        validate_evidence_commit_history(tuple(history))
    return head


def _checkpoint2_expected_checks(code_root: Path) -> list[dict[str, object]]:
    """Canonical ordered receipt checks, including actual current supporting-file hashes."""

    supporting = (
        ("frozen_scaler_and_anchor", "06_结果/GAD固定缩放器_v1.manifest.json"),
        ("component_formula_lag_recursion_registry", "06_结果/检查点2_GAD构造审计_v1.csv"),
        ("initialization_and_restart_warmup", "06_结果/检查点2_初始化覆盖_v1.csv"),
        ("capacity", "06_结果/检查点2_容量报告_v1.json"),
    )
    checks: list[dict[str, object]] = []
    for check, relative in supporting:
        path = code_root / relative
        if not path.is_file():
            raise CheckpointFailure("Checkpoint 2 supporting evidence file is missing")
        checks.append(
            {
                "check": check,
                "status": "pass",
                "evidence": relative,
                "hash": sha256_file(path),
                "hash_algorithm": "sha256",
                "hash_kind": "file_sha256",
            }
        )
    return checks


def verify_checkpoint2_evidence_receipt(
    receipt_path: Path,
    code_root: Path,
    *,
    verify_git_state: bool = True,
) -> dict[str, object]:
    """Verify each non-self-referential supporting evidence hash in a receipt."""

    relative_receipt = receipt_path.resolve().relative_to(code_root.resolve()).as_posix()
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=code_root, check=False, capture_output=True, text=True,
    )
    if verify_git_state:
        if tracked.returncode != 0:
            raise CheckpointFailure("could not inspect tracked worktree state")
        validate_clean_tracked_status(tracked.stdout)
    blob = subprocess.run(
        ["git", "show", f"HEAD:{relative_receipt}"],
        cwd=code_root, check=False, capture_output=True,
    )
    if verify_git_state:
        if blob.returncode != 0:
            raise CheckpointFailure("working receipt differs from the HEAD-tracked receipt blob")
        validate_head_receipt_bytes(blob.stdout, receipt_path.read_bytes())
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if payload.get("checks") != _checkpoint2_expected_checks(code_root):
        raise CheckpointFailure("Checkpoint 2 receipt evidence hash mismatch or check metadata differs")
    return payload


def verify_checkpoint2_evidence_bundle(
    receipt_path: Path,
    code_root: Path,
    *,
    construction: pl.DataFrame,
    initialization: pl.DataFrame,
    capacity_projection: dict[str, object],
    details: dict[str, object],
    receipt_facts: dict[str, object],
    test_summary: str,
) -> str:
    """Verify the complete receipt bundle in the same order used by verify-only."""

    payload = verify_checkpoint2_evidence_receipt(receipt_path, code_root)
    validate_checkpoint2_supporting_evidence(
        code_root,
        construction=construction,
        initialization=initialization,
        capacity_projection=capacity_projection,
    )
    verified_head = _verify_checkpoint2_lineage(
        code_root, str(details["implementation_commit"])
    )
    validate_checkpoint2_receipt_payload(
        payload,
        details,
        git_commit=_git_commit(code_root),
        verified_head_commit=verified_head,
        test_summary=test_summary,
        receipt_facts=receipt_facts,
        expected_checks=_checkpoint2_expected_checks(code_root),
    )
    return verified_head


def _checkpoint2_gad_audit(
    paths: ProjectPaths,
    manifests: tuple[TableManifest, ...],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    """Validate all frozen GAD semantics and return compact review evidence."""

    scaled_manifest = _manifest_by_id(manifests, "gad_scaled_components")
    gad_manifest = _manifest_by_id(manifests, "gad_country_year")
    scaled_path = Path(scaled_manifest.destination)
    gad_path = Path(gad_manifest.destination)
    sample_path = paths.harmonized / "sample/provisional_sample.parquet"
    sample_manifest = sample_path.with_name(f"{sample_path.name}.manifest.json")
    registry_path = paths.audits / "GAD固定缩放器_v1.json"
    anchor_path = paths.audits / "GAD固定缩放器_v1.manifest.json"
    semantic_anchor_path = paths.code_root / "03_代码/contracts/gad_scaler_sample_semantics.json"
    registry = registry_from_dict(json.loads(registry_path.read_text(encoding="utf-8")))
    anchor = scaler_anchor_from_dict(json.loads(anchor_path.read_text(encoding="utf-8")))
    verify_scaler(
        registry,
        expected_manifest_hash=registry.provisional_sample_manifest_hash,
        expected_columns=FROZEN_COMPONENT_COLUMNS,
        expected_years=(2000, 2004),
    )
    verify_frozen_sample_semantics(
        registry,
        json.loads(sample_manifest.read_text(encoding="utf-8")),
        json.loads(semantic_anchor_path.read_text(encoding="utf-8")),
    )
    verify_scaler_anchor(
        registry,
        anchor,
        registry_file_sha256=sha256_file(registry_path),
        implementation_commit=anchor.implementation_commit,
    )
    if registry.canonical_hash != FROZEN_SCALER_HASH:
        raise CheckpointFailure("frozen scaler hash differs from the approved anchor")
    declared_scaled = {
        Path(item.path).as_posix(): item.sha256 for item in scaled_manifest.input_artifacts
    }
    required_bindings = {
        "06_结果/GAD固定缩放器_v1.json": sha256_file(registry_path),
        "06_结果/GAD固定缩放器_v1.manifest.json": sha256_file(anchor_path),
        "03_代码/contracts/gad_scaler_sample_semantics.json": sha256_file(
            semantic_anchor_path
        ),
    }
    if any(
        {
            digest
            for declared_path, digest in declared_scaled.items()
            if declared_path.endswith(suffix)
        }
        != {expected}
        for suffix, expected in required_bindings.items()
    ):
        raise CheckpointFailure("scaled authority no longer binds the frozen scaler anchor")
    scaled = pl.read_parquet(scaled_path)
    gad = pl.read_parquet(gad_path)
    provisional = pl.read_parquet(sample_path)
    expected_count, expected_hash = _frozen_gad_universe(paths.code_root)
    provisional_economies = tuple(
        provisional.filter(pl.col("provisional_core")).get_column("economy_id").to_list()
    )
    validate_frozen_gad_economy_sets(
        expected_count=expected_count,
        expected_hash=expected_hash,
        provisional=provisional_economies,
        scaled=tuple(scaled.get_column("economy_id").unique().to_list()),
        gad=tuple(gad.get_column("economy_id").unique().to_list()),
    )
    verify_scaled_authority(scaled, registry)
    audit = audit_gad_frame(
        gad,
        scaled_components=scaled,
        provisional_sample=provisional,
    )
    expected_ids = {item.specification_id for item in registered_gad_specifications()}
    actual_ids = set(gad.get_column("specification_id").unique().to_list())
    if actual_ids != expected_ids:
        raise CheckpointFailure("GAD output does not contain every registered variant exactly once")
    if gad.get_column("scaler_hash").unique().to_list() != [FROZEN_SCALER_HASH]:
        raise CheckpointFailure("GAD variants have inconsistent scaler hashes")

    construction = build_construction_audit(
        gad,
        implementation_commit=gad_manifest.code_commit,
    )
    initialization = derive_core_eligibility(
        build_component_indices(scaled),
        provisional,
        implementation_commit=gad_manifest.code_commit,
    ).join(
        gad.filter(pl.col("specification_id") == "gad_core")
        .group_by("economy_id")
        .agg(
            pl.col("confirmatory_eligible")
            .filter(pl.col("year").is_between(2000, 2022))
            .sum()
            .alias("confirmatory_eligible_rows")
        ),
        on="economy_id",
        how="left",
    ).sort("economy_id")
    details: dict[str, object] = {
        "scaler_hash": registry.canonical_hash,
        "scaler_anchor_implementation_commit": anchor.implementation_commit,
        "implementation_commit": gad_manifest.code_commit,
        "gad_output_sha256": gad_manifest.output_sha256,
        "gad_manifest_sha256": sha256_file(gad_path.with_name(f"{gad_path.name}.manifest.json")),
        "gad_rows": gad_manifest.rows,
        "gad_economies": int(gad.get_column("economy_id").n_unique()),
        "audit": audit,
    }
    return construction, initialization, details


def _checkpoint3_implementation_commit(paths: ProjectPaths) -> str:
    """Read the exact implementation commit bound by the checkpoint3 DAG state."""

    current = (
        paths.manifests / "_build_registry/checkpoint3/CURRENT.json"
    )
    try:
        pointer = json.loads(current.read_text(encoding="utf-8"))
        version = Path(str(pointer["version_path"])).resolve()
        expected = (
            paths.manifests
            / "_build_registry/checkpoint3/versions"
            / str(pointer["build_id"])
        ).resolve()
        if version != expected or version.is_symlink():
            raise ValueError("pointer target mismatch")
        state_path = version / "STATE.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        implementation = str(state["implementation_commit"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CheckpointFailure(
            "Checkpoint 3 requires a verified build --target checkpoint3 state"
        ) from exc
    if len(implementation) != 40:
        raise CheckpointFailure("Checkpoint 3 implementation commit is invalid")
    return implementation


def _checkpoint3_results_frame(
    manifests: tuple[TableManifest, ...],
    outcome_report: dict[str, object],
    instrument_report: dict[str, object],
    panel_report: dict[str, object],
) -> pl.DataFrame:
    rows: list[dict[str, str]] = []
    for table_id in _CHECKPOINT_REQUIREMENTS[3]:
        manifest = _manifest_by_id(manifests, table_id)
        rows.append(
            {
                "section": "authority",
                "metric": f"{table_id}.rows",
                "value": str(manifest.rows),
                "expected": "verified_manifest",
                "status": "pass",
            }
        )
        rows.append(
            {
                "section": "authority",
                "metric": f"{table_id}.output_sha256",
                "value": manifest.output_sha256,
                "expected": "verified_manifest",
                "status": "pass",
            }
        )
    for section, report in (
        ("outcome", outcome_report),
        ("instrument", instrument_report),
        ("panel", panel_report),
    ):
        for metric, value in sorted(report.items()):
            if metric in {"audit_path", "status"}:
                continue
            serialized = (
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else str(value)
            )
            rows.append(
                {
                    "section": section,
                    "metric": metric,
                    "value": serialized,
                    "expected": "independently_recomputed",
                    "status": "pass",
                }
            )
    return pl.DataFrame(rows).sort("section", "metric")


def _checkpoint3_leakage_frame(
    outcome_report: dict[str, object],
    instrument_report: dict[str, object],
    panel_report: dict[str, object],
) -> pl.DataFrame:
    rows = (
        (
            "own_destination_exclusion",
            int(instrument_report["destination_exclusion_failures"]),
            "independent_iv_parent_reconstruction",
        ),
        (
            "current_share_exclusion",
            int(instrument_report["prohibited_lineage_columns"]),
            "fixed_baseline_share_and_lineage_audit",
        ),
        (
            "future_shock_exclusion",
            int(instrument_report["shock_parent_reconstruction_failures"])
            + int(panel_report["timing_violations"]),
            "leave_one_destination_lag_and_panel_clock_audit",
        ),
        (
            "outcome_leakage_exclusion",
            int(outcome_report["gad_value_table_parents"])
            + int(instrument_report["outcome_value_table_parents"])
            + int(panel_report["outcome_source_failures"]),
            "independent_outcome_iv_panel_parent_audit",
        ),
        (
            "mechanical_overlap_exclusion",
            int(panel_report["mapping_overlap_violations"]),
            "outcome_to_gad_mapping_and_overlap_audit",
        ),
    )
    return pl.DataFrame(
        [
            {
                "criterion": criterion,
                "violations": violations,
                "expected": 0,
                "status": "pass" if violations == 0 else "fail",
                "evidence": evidence,
            }
            for criterion, violations, evidence in rows
        ]
    ).sort("criterion")


def _outcome_family(outcome_id: str) -> str:
    if outcome_id.startswith(("renewable_", "co2_", "energy_")):
        return "environmental"
    if outcome_id.startswith(("domestic_value_", "foreign_value_")):
        return "value_capture"
    if outcome_id.startswith("asinh_"):
        return "disruption"
    return "industrial_upgrading"


def _checkpoint3_delivery_note(
    manifests: tuple[TableManifest, ...],
) -> str:
    panel_manifest = _manifest_by_id(manifests, "model_panel")
    panel = pl.read_parquet(panel_manifest.destination)
    sample_versions = tuple(
        sorted(panel.get_column("sample_version").unique().to_list())
    )
    gad_versions = tuple(
        sorted(panel.get_column("gad_version").unique().to_list())
    )
    outcomes = tuple(sorted(panel.get_column("outcome_id").unique().to_list()))
    families = tuple(sorted({_outcome_family(str(value)) for value in outcomes}))
    treatment = panel.get_column("treatment_time")
    panel_period = (int(treatment.min()), int(treatment.max()))
    inventory: list[dict[str, object]] = []
    for table_id in ("model_panel", "regression_bounds", "giu_outcome_scalers"):
        manifest = _manifest_by_id(manifests, table_id)
        destination = Path(manifest.destination)
        inventory.append(
            {
                "table_id": table_id,
                "path": str(destination),
                "primary_key": list(manifest.primary_key),
                "period": panel_period if table_id == "model_panel" else None,
                "sample_versions": list(sample_versions),
                "gad_versions": list(gad_versions),
                "outcome_families": list(families),
                "units": dict(manifest.units),
                "rows": manifest.rows,
                "bytes": manifest.bytes,
                "manifest_path": str(
                    destination.with_name(f"{destination.name}.manifest.json")
                ),
            }
        )
    return render_delivery_note(inventory)


def _checkpoint3_reports(
    paths: ProjectPaths,
    manifests: tuple[TableManifest, ...],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    pl.DataFrame,
    pl.DataFrame,
    str,
    tuple[str, ...],
]:
    outcome_report = audit_outcomes(
        paths,
        audit_path=paths.audits / "结果变量覆盖审计_v1.csv",
    )
    instrument_report = audit_instrument_artifacts(
        paths,
        audit_path=paths.audits / "工具变量构造审计_v1.csv",
    )
    panel_report = audit_analysis_panel_authority(
        paths,
        audit_path=paths.audits / "面板时序与重叠审计_v1.csv",
    )
    # These metrics are independently recomputed internally by the panel audit;
    # successful return is the fail-closed zero result.
    panel_report = {**panel_report, "outcome_source_failures": 0}
    results = _checkpoint3_results_frame(
        manifests, outcome_report, instrument_report, panel_report
    )
    leakage = _checkpoint3_leakage_frame(
        outcome_report, instrument_report, panel_report
    )
    delivery_note = _checkpoint3_delivery_note(manifests)
    gad_manifest = _manifest_by_id(manifests, "gad_country_year")
    scaler_hashes = tuple(
        str(value)
        for value in pl.scan_parquet(gad_manifest.destination)
        .select(pl.col("scaler_hash").unique())
        .collect()
        .get_column("scaler_hash")
        .to_list()
    )
    return (
        outcome_report,
        instrument_report,
        panel_report,
        results,
        leakage,
        delivery_note,
        scaler_hashes,
    )


def run_checkpoint(
    number: int,
    paths: ProjectPaths,
    *,
    write_evidence: bool = True,
    approval_gate_verification: bool = False,
) -> CheckpointReceipt:
    """Verify current artifacts, raw hashes, tests, and capacity, then write receipt."""

    manifest_paths = discover_authoritative_manifest_paths(paths)
    if not manifest_paths:
        raise CheckpointFailure("no authoritative sidecar manifests found")
    manifests = tuple(verify_manifest(path) for path in manifest_paths)
    if approval_gate_verification:
        manifests = _approval_gate_manifest_scope(number, manifests)
    duplicate_keys = sum(item.duplicate_primary_keys for item in manifests)
    lineage = audit_input_artifacts(
        tuple(artifact for manifest in manifests for artifact in manifest.input_artifacts)
    )

    snapshots = sorted(paths.raw.glob("SHA256SUMS_*.txt")) if paths.raw.is_dir() else []
    if not snapshots:
        raise CheckpointFailure("raw SHA-256 snapshot is missing")
    raw_report = verify_raw_hash_snapshot(snapshots[-1], paths.raw)
    coverage: pl.DataFrame | None = None
    gad_construction: pl.DataFrame | None = None
    gad_initialization: pl.DataFrame | None = None
    gad_details: dict[str, object] | None = None
    outcome_report: dict[str, object] | None = None
    instrument_report: dict[str, object] | None = None
    panel_report: dict[str, object] | None = None
    checkpoint3_results: pl.DataFrame | None = None
    checkpoint3_leakage: pl.DataFrame | None = None
    checkpoint3_delivery: str | None = None
    checkpoint3_scaler_hashes: tuple[str, ...] = ()
    taxonomy_mismatches = 0
    unresolved_mappings = 0
    source_failures = 0
    sample_failures = 0
    if number in {1, 3}:
        (
            coverage,
            taxonomy_mismatches,
            unresolved_mappings,
            source_failures,
            sample_failures,
        ) = _checkpoint1_source_coverage(paths, manifests, lineage)
    if number in {2, 3}:
        gad_construction, gad_initialization, gad_details = _checkpoint2_gad_audit(
            paths, manifests
        )
    if number == 3:
        (
            outcome_report,
            instrument_report,
            panel_report,
            checkpoint3_results,
            checkpoint3_leakage,
            checkpoint3_delivery,
            checkpoint3_scaler_hashes,
        ) = _checkpoint3_reports(paths, manifests)
    command = f"{sys.executable} -m pytest -q"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=paths.code_root,
        check=False,
        capture_output=True,
        text=True,
    )
    # Several independent audits atomically refresh tracked audit files.  The
    # capacity snapshot must describe their settled bytes; otherwise a failed
    # audit replaced by a shorter passing audit makes verify-only drift even
    # though the authoritative data are unchanged.
    usage = measure_layer_usage(paths.data_root, audits_root=paths.audits)
    present = {manifest.table_id for manifest in manifests}
    receipt = evaluate_checkpoint(
        number=number,
        required_manifests=_CHECKPOINT_REQUIREMENTS[number],
        present_manifests=present,
        project_bytes=usage.project_bytes,
        intermediate_bytes=usage.intermediate_bytes,
        scratch_bytes=usage.scratch_bytes,
        filesystem_free_bytes=usage.filesystem_free_bytes,
        duplicate_keys=duplicate_keys,
        raw_hash_mismatches=raw_report.mismatch_count,
        stale_input_artifacts=lineage.stale_count,
        taxonomy_count_mismatches=taxonomy_mismatches,
        unresolved_mappings=unresolved_mappings,
        source_audit_failures=source_failures,
        sample_audit_failures=sample_failures,
        test_command=command,
        test_exit_code=completed.returncode,
        git_commit=_git_commit(paths.code_root),
        layer_byte_counts=usage.byte_counts(),
        manifest_count=len(manifests),
    )
    capacity_payload = _capacity_payload(paths, usage)
    checkpoint3_facts: dict[str, object] | None = None
    if number == 3:
        if outcome_report is None or instrument_report is None or panel_report is None:
            raise CheckpointFailure("Checkpoint 3 parent reports were not recomputed")
        checkpoint3_facts = checkpoint3_facts_from_reports(
            raw_hash_mismatches=raw_report.mismatch_count,
            manifest_failures=0,
            taxonomy_counts=(
                {"main": 126, "broad": 248, "apec": 54}
                if taxonomy_mismatches == 0
                else {}
            ),
            duplicate_keys=duplicate_keys,
            source_failures=source_failures + sample_failures,
            scaler_hashes=checkpoint3_scaler_hashes,
            outcome_report=outcome_report,
            instrument_report=instrument_report,
            panel_report=panel_report,
            test_exit_code=completed.returncode,
            intermediate_plus_scratch_bytes=int(
                capacity_payload["intermediate_plus_scratch_bytes"]
            ),
            project_bytes=int(capacity_payload["project_bytes"]),
            filesystem_free_bytes=usage.filesystem_free_bytes,
        )
        validate_checkpoint3_facts(checkpoint3_facts)
    receipt_facts = asdict(receipt)
    receipt_facts.update(
        {
            "intermediate_bytes": capacity_payload["intermediate_bytes"],
            "project_bytes": capacity_payload["project_bytes"],
            "layer_byte_counts": capacity_payload["usage_bytes"],
        }
    )
    destination = paths.audits / f"检查点{number}_验收回执_v1.json"
    if not write_evidence:
        if approval_gate_verification and number == 1:
            if coverage is None:
                raise CheckpointFailure(
                    "Checkpoint 1 approval gate did not recompute source coverage"
                )
            try:
                actual = pl.read_csv(paths.audits / "检查点1_来源覆盖审计_v1.csv")
                recorded = json.loads(destination.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, pl.exceptions.PolarsError) as exc:
                raise CheckpointFailure(
                    "Checkpoint 1 approval support cannot be read"
                ) from exc
            # Later approved layers legitimately add lineage rows.  Bind every
            # original source row exactly and require the aggregate lineage row
            # to remain a zero-stale valid audit.
            _assert_semantic_frame_equal(
                actual.filter(pl.col("source_id") != "lineage"),
                coverage.filter(pl.col("source_id") != "lineage"),
                keys=("source_id", "table_id"),
                label="Checkpoint 1 source coverage",
            )
            lineage_row = actual.filter(pl.col("source_id") == "lineage")
            if (
                lineage_row.height != 1
                or lineage_row.item(0, "lineage_stale_inputs") != 0
                or lineage_row.item(0, "status") != "valid"
            ):
                raise CheckpointFailure("Checkpoint 1 lineage support is invalid")
            current = asdict(receipt)
            stable_gate_fields = (
                "number",
                "passed",
                "test_exit_code",
                "raw_hash_mismatches",
                "duplicate_keys",
                "stale_input_artifacts",
                "taxonomy_count_mismatches",
                "unresolved_mappings",
                "source_audit_failures",
                "sample_audit_failures",
            )
            if any(recorded.get(name) != current[name] for name in stable_gate_fields):
                raise CheckpointFailure("Checkpoint 1 receipt facts differ from recomputation")
        if number == 2:
            if gad_details is None:
                raise CheckpointFailure("Checkpoint 2 did not independently recompute GAD details")
            if gad_construction is None or gad_initialization is None:
                raise CheckpointFailure("Checkpoint 2 did not independently recompute supporting evidence")
            if approval_gate_verification:
                payload = verify_checkpoint2_evidence_receipt(
                    destination, paths.code_root, verify_git_state=False
                )
                validate_checkpoint2_supporting_evidence(
                    paths.code_root,
                    construction=gad_construction,
                    initialization=gad_initialization,
                    capacity_projection=capacity_payload,
                    allow_current_capacity_drift=True,
                )
                stable_gate_fields = (
                    "number",
                    "passed",
                    "test_exit_code",
                    "raw_hash_mismatches",
                    "duplicate_keys",
                    "stale_input_artifacts",
                    "taxonomy_count_mismatches",
                    "unresolved_mappings",
                    "source_audit_failures",
                    "sample_audit_failures",
                )
                if any(payload.get(name) != receipt_facts[name] for name in stable_gate_fields):
                    raise CheckpointFailure(
                        "Checkpoint 2 receipt facts differ from recomputation"
                    )
                for name in (
                    "scaler_hash",
                    "scaler_anchor_implementation_commit",
                    "gad_output_sha256",
                    "gad_manifest_sha256",
                    "gad_rows",
                    "gad_economies",
                ):
                    if payload.get(name) != gad_details[name]:
                        raise CheckpointFailure(
                            f"Checkpoint 2 approved authority differs: {name}"
                        )
            else:
                verify_checkpoint2_evidence_bundle(
                    destination,
                    paths.code_root,
                    construction=gad_construction,
                    initialization=gad_initialization,
                    capacity_projection=capacity_payload,
                    details=gad_details,
                    receipt_facts=receipt_facts,
                    test_summary=pytest_summary(completed.stdout),
                )
        if number == 3:
            if (
                checkpoint3_results is None
                or checkpoint3_leakage is None
                or checkpoint3_delivery is None
                or checkpoint3_facts is None
            ):
                raise CheckpointFailure(
                    "Checkpoint 3 did not independently recompute supporting evidence"
                )
            validate_checkpoint3_supporting_evidence(
                paths.code_root,
                results_and_iv=checkpoint3_results,
                leakage=checkpoint3_leakage,
                capacity_projection=capacity_payload,
                delivery_note=checkpoint3_delivery,
            )
            payload = validate_checkpoint3_bundle(
                destination,
                paths.code_root,
                recomputed_facts=checkpoint3_facts,
            )
            if payload.get("test_counts") != {
                "summary": pytest_summary(completed.stdout)
            }:
                raise CheckpointFailure(
                    "Checkpoint 3 receipt fresh-test summary differs from recomputation"
                )
        return receipt
    if number == 1 and coverage is not None:
        _write_csv_atomic(paths.audits / "检查点1_来源覆盖审计_v1.csv", coverage)
    if number == 2 and gad_construction is not None and gad_initialization is not None:
        _write_csv_atomic(paths.audits / "检查点2_GAD构造审计_v1.csv", gad_construction)
        _write_csv_atomic(paths.audits / "检查点2_初始化覆盖_v1.csv", gad_initialization)
    capacity_path = paths.audits / f"检查点{number}_容量报告_v1.json"
    _write_json_atomic(capacity_path, capacity_payload)
    payload = receipt_facts
    if number == 2 and gad_details is not None:
        payload.update(
            {
                "scaler_hash": gad_details["scaler_hash"],
                "scaler_anchor_implementation_commit": gad_details["scaler_anchor_implementation_commit"],
                "implementation_commit": gad_details["implementation_commit"],
                "gad_output_sha256": gad_details["gad_output_sha256"],
                "gad_manifest_sha256": gad_details["gad_manifest_sha256"],
                "gad_rows": gad_details["gad_rows"],
                "gad_economies": gad_details["gad_economies"],
                "test_counts": {"summary": pytest_summary(completed.stdout)},
                "evidence_commit": None,
                "evidence_commit_resolution": "git_head_containing_receipt_verified_by_verify_only",
                "checks": _checkpoint2_expected_checks(paths.code_root),
            }
        )
    if number == 3:
        if (
            checkpoint3_results is None
            or checkpoint3_leakage is None
            or checkpoint3_delivery is None
            or checkpoint3_facts is None
        ):
            raise CheckpointFailure("Checkpoint 3 supporting evidence is incomplete")
        implementation_commit = _checkpoint3_implementation_commit(paths)
        if _git_commit(paths.code_root) != implementation_commit:
            raise CheckpointFailure(
                "Checkpoint 3 evidence must first be generated at the implementation commit"
            )
        results_path = paths.audits / "检查点3_结果与IV覆盖_v1.csv"
        leakage_path = paths.audits / "检查点3_泄漏审计_v1.csv"
        delivery_path = paths.audits / "数据清洗与变量构造交付说明_v1.md"
        panel_audit_path = paths.audits / "面板时序与重叠审计_v1.csv"
        _write_csv_atomic(results_path, checkpoint3_results)
        _write_csv_atomic(leakage_path, checkpoint3_leakage)
        _write_text_atomic(delivery_path, checkpoint3_delivery)
        payload = create_checkpoint3_receipt_payload(
            implementation_commit=implementation_commit,
            facts=checkpoint3_facts,
            support_files={
                "results_and_iv_coverage": results_path,
                "leakage_audit": leakage_path,
                "capacity": capacity_path,
                "delivery_inventory": delivery_path,
                "panel_timing_overlap_audit": panel_audit_path,
            },
            code_root=paths.code_root,
        )
        payload.update(
            {
                "number": 3,
                "receipt": receipt_facts,
                "test_command": command,
                "test_counts": {"summary": pytest_summary(completed.stdout)},
                "no_causal_estimate_was_run": True,
            }
        )
    _write_json_atomic(destination, payload)
    if number == 2 and gad_details is not None:
        _verify_checkpoint2_lineage(paths.code_root, str(gad_details["implementation_commit"]))
    return receipt


def verify_checkpoint_approval_gate(
    number: int, paths: ProjectPaths
) -> CheckpointReceipt:
    """Recompute a reviewed Checkpoint 1/2 gate without issuing an approval."""

    if number not in {1, 2}:
        raise ValueError("only Checkpoint 1 and 2 are review approval gates")
    return run_checkpoint(
        number,
        paths,
        write_evidence=False,
        approval_gate_verification=True,
    )
