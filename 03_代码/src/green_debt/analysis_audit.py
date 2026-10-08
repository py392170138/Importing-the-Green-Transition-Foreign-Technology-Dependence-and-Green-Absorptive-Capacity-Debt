"""Fail-closed analysis-output auditing and evidence classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any
import zlib

from lxml import etree
import polars as pl
from pypdf import PdfReader

from green_debt.analysis_io import (
    RunContext,
    analysis_reproduction_audited_at,
    analysis_preflight,
    combined_project_usage_bytes,
    current_analysis_git_commit,
    load_evidence_policy,
    load_stage_a_run_context,
    load_table_contract,
    resolve_authorized_analysis_output,
    verify_threshold_registry_payload,
)
from green_debt.analysis_spec import AnalysisSpec, ModelCell
from green_debt.analysis_spec import load_analysis_spec
from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    verify_manifest,
    write_authoritative_table,
)
from green_debt.config import load_project_config
from green_debt.storage import (
    GIB,
    directory_usage_bytes,
    sha256_file,
)


@dataclass(frozen=True)
class FrameAuditReport:
    status: str
    missing_confirmatory_cells: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class OutputAuditReport:
    status: str
    continuous_cells: int
    ar_cells: int
    threshold_cells: int
    scorecard: pl.DataFrame
    audited_hashes: tuple[tuple[str, str], ...]
    output_bytes: int
    project_bytes: int


@dataclass(frozen=True)
class PublicationAudit:
    hashes: tuple[tuple[str, str], ...]
    reporting_git_commit: str


def grade_evidence(
    *,
    rank_ok: bool,
    clusters: int,
    ar_status: str,
    iv_direction_matches_ar: bool,
    shock_status: str,
    concentration_status: str = "resolved_with_preregistered_cutoff",
) -> str:
    """Map registered inference diagnostics to the frozen evidence vocabulary."""

    if not rank_ok:
        return "not_estimated"
    if concentration_status == "unresolved_no_preregistered_cutoff":
        return "exploratory"
    if concentration_status != "resolved_with_preregistered_cutoff":
        raise ValueError(f"unknown concentration status: {concentration_status}")
    if clusters < 20 or shock_status == "unavailable":
        return "exploratory"
    if ar_status == "bounded" and iv_direction_matches_ar:
        return "conditional_causal"
    return "conditional_association"


def direction_matches(estimate: float, expected_sign: str) -> bool:
    """Return whether a finite point estimate is strictly in the expected direction."""

    if expected_sign == "positive":
        return estimate > 0
    if expected_sign == "negative":
        return estimate < 0
    raise ValueError(f"unknown expected sign: {expected_sign}")


def registered_inference_status(*, rank: int, clusters: int) -> str:
    """Return the one inference branch frozen by the analysis specification."""

    if rank < 2:
        return "fail_rank_deficient"
    if clusters < 20:
        return "exploratory_lt20_clusters"
    if clusters < 30:
        return "wild_bootstrap_required"
    return "cluster_robust"


def ar_directionally_supports(
    *,
    ar_status: str,
    conventional_point_accepted: bool,
    target_term: str,
    expected_sign: str,
    beta_low: float | None,
    beta_high: float | None,
    theta_low: float | None,
    theta_high: float | None,
) -> bool:
    """Apply the frozen bounded-set directional invariant and fail closed."""

    if ar_status != "bounded" or not conventional_point_accepted:
        return False
    if target_term == "gimc_a":
        low, high = beta_low, beta_high
    elif target_term == "gimc_gad_a":
        low, high = theta_low, theta_high
    else:
        raise ValueError(f"unknown target term: {target_term}")
    if low is None or high is None:
        return False
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        return False
    if float(low) > float(high):
        return False
    if expected_sign == "positive":
        return float(low) > 0
    if expected_sign == "negative":
        return float(high) < 0
    raise ValueError(f"unknown expected sign: {expected_sign}")


def _wholly_expected(row: dict[str, object]) -> bool:
    grade = str(row["evidence_grade"])
    if grade not in {"conditional_causal", "conditional_association"}:
        return False
    low = float(row["interval_low"])
    high = float(row["interval_high"])
    sign = str(row["expected_sign"])
    return low > 0 if sign == "positive" else high < 0 if sign == "negative" else False


def _qualifying_outcome_counts(scorecard: pl.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in scorecard.iter_rows(named=True):
        if _wholly_expected(row):
            outcome = str(row["outcome_id"])
            counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def _prefixed_label(scorecard: pl.DataFrame, outcomes: set[str]) -> str:
    grades = [
        str(row["evidence_grade"])
        for row in scorecard.iter_rows(named=True)
        if str(row["outcome_id"]) in outcomes and _wholly_expected(row)
    ]
    if not grades:
        return "mixed_or_inconclusive"
    weakest = (
        "conditional_association"
        if "conditional_association" in grades
        else "conditional_causal"
    )
    return f"{weakest}: broadly_consistent"


def summarize_hypotheses(scorecard: pl.DataFrame) -> dict[str, str]:
    """Apply the frozen H1-H3 breadth rules without binary acceptance language."""

    required = {
        "outcome_id",
        "horizon",
        "expected_sign",
        "interval_low",
        "interval_high",
        "evidence_grade",
    }
    if not required <= set(scorecard.columns):
        raise ValueError("evidence scorecard lacks hypothesis fields")
    counts = _qualifying_outcome_counts(scorecard)
    co2 = "co2_tonnes_per_million_current_usd"
    renewable = "renewable_capacity_additions_mw_per_million"
    energy = "energy_intensity_mj_per_ppp_gdp"
    h1_qualified = {item for item in (co2, renewable, energy) if counts.get(item, 0) >= 2}
    h1 = (
        _prefixed_label(scorecard, h1_qualified)
        if len(h1_qualified) >= 2 and h1_qualified & {co2, renewable}
        else "mixed_or_inconclusive"
    )
    complexity = "green_export_complexity"
    rca = "future_green_rca_entry_rate"
    share = "green_export_share"
    h2_secondary = {item for item in (rca, share) if counts.get(item, 0) >= 4}
    h2_qualified = {complexity, *h2_secondary}
    h2 = (
        _prefixed_label(scorecard, h2_qualified)
        if counts.get(complexity, 0) >= 4 and h2_secondary
        else "mixed_or_inconclusive"
    )
    domestic = "domestic_value_added_share"
    h3_qualified = {complexity, domestic}
    h3 = (
        _prefixed_label(scorecard, h3_qualified)
        if counts.get(complexity, 0) >= 4 and counts.get(domestic, 0) >= 4
        else "mixed_or_inconclusive"
    )
    dependence = "foreign_value_added_dependence"
    h3_corroborating = "mixed_or_inconclusive"
    if counts.get(dependence, 0) >= 4:
        prefix = _prefixed_label(scorecard, {dependence}).split(":", 1)[0]
        h3_corroborating = f"{prefix}: corroborating"
    return {
        "H1": h1,
        "H2": h2,
        "H3": h3,
        "H3_corroborating": h3_corroborating,
    }


def _cell_key(cell: ModelCell) -> tuple[str, str, int, str, str]:
    return (
        cell.analysis_family,
        cell.outcome_id,
        cell.horizon,
        cell.gad_version,
        cell.sample_version,
    )


def _frame_keys(frame: pl.DataFrame, context: RunContext) -> set[tuple[str, str, int, str, str]]:
    required = {
        "run_id",
        "analysis_family",
        "outcome_id",
        "horizon",
        "gad_version",
        "sample_version",
        "status",
    }
    if not required <= set(frame.columns):
        return set()
    valid = frame.filter(
        (pl.col("run_id") == context.run_id)
        & pl.col("status").is_in(["estimated", "explicit_gate"])
    )
    return {
        (
            str(row["analysis_family"]),
            str(row["outcome_id"]),
            int(row["horizon"]),
            str(row["gad_version"]),
            str(row["sample_version"]),
        )
        for row in valid.iter_rows(named=True)
    }


def audit_analysis_frames(
    frames: dict[str, pl.DataFrame], spec: AnalysisSpec, context: RunContext
) -> FrameAuditReport:
    """Check status-frame completeness without inferring missing model results."""

    required_names = {"lp_fe_status", "lp_iv_status", "threshold_status", "ar_status"}
    if set(frames) != required_names:
        missing = tuple(
            sorted((cell.outcome_id, cell.horizon) for cell in spec.confirmatory_cells())
        )
        return FrameAuditReport("failed", missing)
    iv_keys = _frame_keys(frames["lp_iv_status"], context)
    missing = tuple(
        sorted(
            (cell.outcome_id, cell.horizon)
            for cell in spec.confirmatory_cells()
            if _cell_key(cell) not in iv_keys
        )
    )
    return FrameAuditReport("failed" if missing else "passed", missing)


_TABLES = {
    "sample_cells": ("diagnostics/sample_cells.parquet", "sample_cell.json"),
    "missingness": ("diagnostics/missingness.parquet", "missingness_cell.json"),
    "distributions": ("diagnostics/distributions.parquet", "distribution.json"),
    "correlations": ("diagnostics/correlations.parquet", "correlation.json"),
    "descriptive_paths": ("diagnostics/descriptive_paths.parquet", "descriptive_path.json"),
    "exposure_concentration": (
        "diagnostics/exposure_concentration.parquet",
        "exposure_concentration.json",
    ),
    "first_stage_screen": (
        "diagnostics/first_stage_screen.parquet",
        "diagnostic_cell.json",
    ),
    "shift_share_summary": (
        "diagnostics/shift_share_summary.parquet",
        "shift_share_summary.json",
    ),
    "shift_share_weights": (
        "diagnostics/shift_share_weights.parquet",
        "shift_share_weight.json",
    ),
    "lp_fe": ("models/lp_fe.parquet", "model_estimate.json"),
    "lp_iv": ("models/lp_iv.parquet", "model_estimate.json"),
    "model_covariance": (
        "models/model_covariance.parquet",
        "model_covariance.json",
    ),
    "marginal_effects": (
        "models/marginal_effects.parquet",
        "marginal_effect.json",
    ),
    "threshold_estimates": (
        "models/threshold_estimates.parquet",
        "threshold_estimate.json",
    ),
    "weak_iv_sets": ("models/weak_iv_sets.parquet", "weak_iv_set.json"),
}

_IDENTITY_FIELDS = (
    "run_id",
    "spec_id",
    "input_authority_hash",
    "git_commit",
    "renv_lock_sha256",
    "created_at_utc",
)
_CELL_FIELDS = (
    "analysis_family",
    "outcome_id",
    "horizon",
    "gad_version",
    "sample_version",
)


def _context_identity(context: RunContext) -> dict[str, object]:
    identity = asdict(context)
    identity.pop("seed")
    return identity


def _read_validated_tables(
    code_root: Path, output_root: Path, context: RunContext
) -> tuple[dict[str, pl.DataFrame], tuple[tuple[str, str], ...]]:
    frames: dict[str, pl.DataFrame] = {}
    hashes: list[tuple[str, str]] = []
    expected_identity = _context_identity(context)
    contract_root = code_root / "03_代码/contracts/analysis"
    for name, (relative, contract_name) in _TABLES.items():
        path = output_root / relative
        manifest_path = path.with_name(f"{path.name}.manifest.json")
        manifest = verify_manifest(manifest_path)
        contract = load_table_contract(contract_root / contract_name)
        for field in (
            "table_id",
            "schema_version",
            "primary_key",
            "columns",
            "units",
            "period",
            "zero_semantics",
            "null_semantics",
            "transformations",
        ):
            if getattr(manifest, field) != getattr(contract, field):
                raise ValueError(f"contract-manifest mismatch for {name}: {field}")
        frame = pl.read_parquet(path)
        for field, expected in expected_identity.items():
            if field == "evidence_policy_sha256" and field not in frame.columns:
                continue
            if field not in frame.columns or frame.get_column(field).unique().to_list() != [expected]:
                raise ValueError(f"{name} {field} does not match run context")
        frames[name] = frame
        hashes.append((str(path.relative_to(output_root)), manifest.output_sha256))
    return frames, tuple(sorted(hashes))


def _row_key(row: dict[str, Any]) -> tuple[str, str, int, str, str]:
    return tuple(
        int(row[field]) if field == "horizon" else str(row[field])
        for field in _CELL_FIELDS
    )  # type: ignore[return-value]


def _registered_keys(cells: tuple[ModelCell, ...]) -> set[tuple[str, str, int, str, str]]:
    return {_cell_key(cell) for cell in cells}


def _rows_by_cell(frame: pl.DataFrame) -> dict[tuple[str, str, int, str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = {}
    for row in frame.iter_rows(named=True):
        result.setdefault(_row_key(row), []).append(row)
    return result


def _require_model_completeness(frames: dict[str, pl.DataFrame], spec: AnalysisSpec) -> None:
    continuous = (
        *spec.confirmatory_cells(),
        *spec.robustness_cells(),
        *spec.vulnerability_cells(),
    )
    expected = _registered_keys(continuous)
    gates = {_row_key(row): row for row in frames["first_stage_screen"].iter_rows(named=True)}
    if not expected <= set(gates):
        raise ValueError("first-stage screen lacks registered continuous cells")
    estimated_pairs: set[tuple[str, tuple[str, str, int, str, str]]] = set()
    for estimator in ("lp_fe", "lp_iv"):
        grouped = _rows_by_cell(frames[estimator])
        if set(grouped) - expected:
            raise ValueError(f"{estimator} contains unregistered cells")
        for key in expected:
            rows = grouped.get(key, [])
            if not rows:
                if str(gates[key]["gate_status"]) != "fail_rank_deficient":
                    raise ValueError(f"{estimator} lacks an estimate or permitted gate: {key}")
                continue
            if {str(row["term"]) for row in rows} != {"gimc_a", "gimc_gad_a"} or len(rows) != 2:
                raise ValueError(f"{estimator} lacks the exact two model terms: {key}")
            estimated_pairs.add((estimator, key))
            expected_inference = registered_inference_status(
                rank=int(gates[key]["rank"]), clusters=int(gates[key]["clusters"])
            )
            observed = {str(row["inference_status"]) for row in rows}
            permitted = (
                {"wild_bootstrap_required", "wild_bootstrap_unbounded"}
                if expected_inference == "wild_bootstrap_required"
                else {expected_inference}
            )
            if len(observed) != 1 or not observed <= permitted:
                raise ValueError(f"{estimator} violates registered inference rule: {key}")
            for row in rows:
                for field in ("n", "economies", "clusters", "first_stage_status"):
                    if row[field] != gates[key][field]:
                        raise ValueError(f"{estimator} does not match registered {field}: {key}")
                if expected_inference == "wild_bootstrap_required":
                    if row["wild_draws"] != spec.inference.wild_bootstrap_draws or row["wild_seed"] != spec.seed:
                        raise ValueError(f"{estimator} wild-bootstrap identity mismatch: {key}")
                    bounded = str(row["inference_status"]) == "wild_bootstrap_required"
                    if bounded != (row["wild_conf_low"] is not None and row["wild_conf_high"] is not None):
                        raise ValueError(f"{estimator} wild interval semantics mismatch: {key}")
                elif any(row[field] is not None for field in ("wild_conf_low", "wild_conf_high", "wild_p_value", "wild_draws", "wild_seed")):
                    raise ValueError(f"{estimator} contains unregistered wild inference: {key}")
    marginal = frames["marginal_effects"]
    grouped_marginal: dict[tuple[str, tuple[str, str, int, str, str]], list[dict[str, Any]]] = {}
    for row in marginal.iter_rows(named=True):
        grouped_marginal.setdefault((str(row["estimator"]), _row_key(row)), []).append(row)
    if set(grouped_marginal) != estimated_pairs:
        raise ValueError("marginal-effect cells do not match estimated model cells")
    expected_quantiles = set(spec.inference.marginal_gad_quantiles)
    for pair, rows in grouped_marginal.items():
        if len(rows) != 3 or {float(row["gad_quantile"]) for row in rows} != expected_quantiles:
            raise ValueError(f"marginal effects lack p25/p50/p75: {pair}")
        estimator, key = pair
        estimate_status = {
            str(row["inference_status"])
            for row in _rows_by_cell(frames[estimator])[key]
        }
        for row in rows:
            if (
                str(row["inference_status"]) not in estimate_status
                or int(row["clusters"]) != int(gates[key]["clusters"])
                or int(row["n"]) != int(gates[key]["n"])
            ):
                raise ValueError(f"marginal effects do not match registered inference: {pair}")
    for name in ("lp_fe", "lp_iv", "marginal_effects", "model_covariance"):
        bad = frames[name].filter(
            (pl.col("analysis_family") == "confirmatory") & (pl.col("horizon") == 0)
        )
        if bad.height:
            raise ValueError(f"{name} contains a confirmatory h=0 row")


def _require_audit_cells(
    frames: dict[str, pl.DataFrame], spec: AnalysisSpec, registry: dict[str, Any]
) -> None:
    confirmatory = _registered_keys(spec.confirmatory_cells())
    audit_cells = _registered_keys((*spec.confirmatory_cells(), *spec.vulnerability_cells()))
    ar = _rows_by_cell(frames["weak_iv_sets"])
    if set(ar) != audit_cells or any(len(rows) != 1 for rows in ar.values()):
        raise ValueError("AR output does not cover the exact 45 registered IV cells")
    thresholds = _rows_by_cell(frames["threshold_estimates"])
    if set(thresholds) != confirmatory:
        raise ValueError("threshold output does not cover the exact 39 confirmatory cells")
    registry_hash = str(registry["registry_hash"])
    for key, rows in thresholds.items():
        if len(rows) != 2 or {str(row["regime"]) for row in rows} != {"low", "high"}:
            raise ValueError(f"threshold regimes are incomplete: {key}")
        if {str(row["registry_hash"]) for row in rows} != {registry_hash}:
            raise ValueError(f"threshold registry hash mismatch: {key}")
    shocks = _rows_by_cell(frames["shift_share_summary"])
    if set(shocks) != audit_cells or any(len(rows) != 1 for rows in shocks.values()):
        raise ValueError("shock audit does not cover the exact 45 registered IV cells")
    for key, rows in shocks.items():
        row = rows[0]
        status = str(row["shock_inference_status"])
        fields = (row["shock_std_error_gimc"], row["shock_std_error_interaction"])
        if status == "available" and any(value is None for value in fields):
            raise ValueError(f"available shock inference lacks shock-unit standard errors: {key}")
        if status == "unavailable" and any(value is not None for value in fields):
            raise ValueError(f"unavailable shock inference contains fallback standard errors: {key}")
        if status not in {"available", "unavailable"}:
            raise ValueError(f"unknown shock inference status: {key}")


def _registered_interval(
    row: dict[str, Any],
) -> tuple[float | None, float | None]:
    status = str(row["inference_status"])
    if status == "wild_bootstrap_unbounded":
        return None, None
    if status == "wild_bootstrap_required":
        low, high = row["wild_conf_low"], row["wild_conf_high"]
        if low is None or high is None:
            return None, None
        return float(low), float(high)
    return float(row["conf_low"]), float(row["conf_high"])


def _scorecard(
    frames: dict[str, pl.DataFrame],
    spec: AnalysisSpec,
    context: RunContext,
    audit_git_commit: str,
    evidence_policy: dict[str, Any],
) -> pl.DataFrame:
    gates = {_row_key(row): row for row in frames["first_stage_screen"].iter_rows(named=True)}
    fe = _rows_by_cell(frames["lp_fe"])
    iv = _rows_by_cell(frames["lp_iv"])
    ar = {key: rows[0] for key, rows in _rows_by_cell(frames["weak_iv_sets"]).items()}
    shocks = {key: rows[0] for key, rows in _rows_by_cell(frames["shift_share_summary"]).items()}
    outcome_specs = {item.outcome_id: item for item in spec.outcomes}
    identity = _context_identity(context)
    concentration = evidence_policy["concentration"]
    concentration_status = str(concentration["status"])
    concentration_metrics = json.dumps(
        concentration["metrics"], sort_keys=True, separators=(",", ":")
    )
    exposure_hhi_max = float(frames["exposure_concentration"]["hhi"].max())
    exposure_top1_max = float(
        frames["exposure_concentration"]["top1_share"].max()
    )
    rows: list[dict[str, object]] = []
    for cell in spec.confirmatory_cells():
        key = _cell_key(cell)
        outcome = outcome_specs[cell.outcome_id]
        gate = gates[key]
        ar_row = ar[key]
        shock = shocks[key]
        point_membership = bool(ar_row["conventional_point_accepted"])
        supportive = ar_directionally_supports(
            ar_status=str(ar_row["status"]),
            conventional_point_accepted=point_membership,
            target_term=outcome.target_term,
            expected_sign=outcome.expected_sign,
            beta_low=ar_row["beta_low"],
            beta_high=ar_row["beta_high"],
            theta_low=ar_row["theta_low"],
            theta_high=ar_row["theta_high"],
        )
        grade = grade_evidence(
            rank_ok=int(gate["rank"]) >= 2,
            clusters=int(gate["clusters"]),
            ar_status=str(ar_row["status"]),
            iv_direction_matches_ar=supportive,
            shock_status=str(shock["shock_inference_status"]),
            concentration_status=concentration_status,
        )
        candidates = iv.get(key, []) if grade == "conditional_causal" else fe.get(key, [])
        target = next((row for row in candidates if row["term"] == outcome.target_term), None)
        estimate: float | None = None
        interval_low: float | None = None
        interval_high: float | None = None
        interval_source = "none"
        if grade == "conditional_causal":
            estimate = float(target["estimate"]) if target is not None else None
            interval_low = ar_row["beta_low"] if outcome.target_term == "gimc_a" else ar_row["theta_low"]
            interval_high = ar_row["beta_high"] if outcome.target_term == "gimc_a" else ar_row["theta_high"]
            interval_source = "lp_iv_ar"
        elif grade == "conditional_association" and target is not None:
            estimate = float(target["estimate"])
            interval_low, interval_high = _registered_interval(target)
            interval_source = "lp_fe"
        elif grade == "exploratory" and target is not None:
            estimate = float(target["estimate"])
            interval_low, interval_high = _registered_interval(target)
            interval_source = "lp_fe_exploratory"
        match = (
            grade in {"conditional_causal", "conditional_association"}
            and interval_low is not None
            and interval_high is not None
            and (float(interval_low) > 0 if outcome.expected_sign == "positive" else float(interval_high) < 0)
        )
        rows.append(
            {
                **identity,
                "upstream_model_git_commit": context.git_commit,
                "audit_git_commit": audit_git_commit,
                "evidence_policy_sha256": context.evidence_policy_sha256,
                "concentration_status": concentration_status,
                "concentration_metrics": concentration_metrics,
                "concentration_aggregation": str(concentration["aggregation"]),
                "concentration_cutoff_hhi": concentration["cutoffs"]["hhi"],
                "concentration_cutoff_top1_share": concentration["cutoffs"][
                    "top1_share"
                ],
                "exposure_hhi_max": exposure_hhi_max,
                "exposure_top1_share_max": exposure_top1_max,
                "rotemberg_hhi_absolute": shock["hhi_absolute"],
                "rotemberg_top1_absolute_share": shock[
                    "top1_absolute_share"
                ],
                "analysis_family": cell.analysis_family,
                "outcome_id": cell.outcome_id,
                "horizon": cell.horizon,
                "gad_version": cell.gad_version,
                "sample_version": cell.sample_version,
                "role": cell.role,
                "target_term": outcome.target_term,
                "expected_sign": outcome.expected_sign,
                "rank_ok": int(gate["rank"]) >= 2,
                "clusters": int(gate["clusters"]),
                "ar_status": str(ar_row["status"]),
                "shock_status": str(shock["shock_inference_status"]),
                "iv_direction_matches_ar": supportive,
                "conventional_point_accepted": point_membership,
                "evidence_grade": grade,
                "interval_source": interval_source,
                "estimate": estimate,
                "interval_low": interval_low,
                "interval_high": interval_high,
                "direction_matches": match,
            }
        )
    return pl.DataFrame(rows).sort("outcome_id", "horizon")


def audit_analysis_outputs(
    *, code_root: Path, data_root: Path, output_root: Path
) -> OutputAuditReport:
    """Validate every registered output claim without writing success artifacts."""

    code = code_root.resolve()
    data = data_root.resolve()
    preflight = analysis_preflight(code, data, output_root)
    output = resolve_authorized_analysis_output(code, data, output_root)
    spec = load_analysis_spec(code / "config/analysis.yaml")
    context = load_stage_a_run_context(output)
    if context.spec_id != spec.spec_id or context.seed != spec.seed:
        raise ValueError("run context does not match frozen analysis specification")
    evidence_policy, evidence_policy_sha256 = load_evidence_policy(
        code / "config/evidence_policy.json"
    )
    if (
        evidence_policy_sha256 != context.evidence_policy_sha256
        or evidence_policy_sha256 != preflight.evidence_policy_sha256
    ):
        raise ValueError("run context evidence policy authority mismatch")
    frames, hashes = _read_validated_tables(code, output, context)
    _require_model_completeness(frames, spec)
    registry_path = output / "registries/threshold_registry_v1.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid threshold registry: {exc}") from exc
    verify_threshold_registry_payload(registry, spec, context)
    hashes = tuple(
        sorted(
            (
                *hashes,
                (
                    "registries/threshold_registry_v1.json",
                    sha256_file(registry_path),
                ),
            )
        )
    )
    _require_audit_cells(frames, spec, registry)
    config = load_project_config(code / "config/project.yaml")
    output_bytes = directory_usage_bytes(output)
    project_bytes = preflight.project_bytes
    if output_bytes > int(spec.outputs.quota_gb) * GIB:
        raise RuntimeError("10 GB analysis output quota exceeded")
    if config.storage.absolute_limit_gb is None or project_bytes >= int(config.storage.absolute_limit_gb) * GIB:
        raise RuntimeError("150 GB absolute project limit reached")
    scorecard = _scorecard(
        frames,
        spec,
        context,
        current_analysis_git_commit(code),
        evidence_policy,
    )
    return OutputAuditReport(
        status="passed",
        continuous_cells=len(spec.confirmatory_cells()) + len(spec.robustness_cells()) + len(spec.vulnerability_cells()),
        ar_cells=len(spec.confirmatory_cells()) + len(spec.vulnerability_cells()),
        threshold_cells=len(spec.confirmatory_cells()),
        scorecard=scorecard,
        audited_hashes=hashes,
        output_bytes=output_bytes,
        project_bytes=project_bytes,
    )


def _evidence_scorecard_contract() -> TableContract:
    columns = {
        "run_id": "String",
        "spec_id": "String",
        "input_authority_hash": "String",
        "git_commit": "String",
        "upstream_model_git_commit": "String",
        "audit_git_commit": "String",
        "renv_lock_sha256": "String",
        "evidence_policy_sha256": "String",
        "created_at_utc": "String",
        "concentration_status": "String",
        "concentration_metrics": "String",
        "concentration_aggregation": "String",
        "concentration_cutoff_hhi": "Float64",
        "concentration_cutoff_top1_share": "Float64",
        "exposure_hhi_max": "Float64",
        "exposure_top1_share_max": "Float64",
        "rotemberg_hhi_absolute": "Float64",
        "rotemberg_top1_absolute_share": "Float64",
        "analysis_family": "String",
        "outcome_id": "String",
        "horizon": "Int16",
        "gad_version": "String",
        "sample_version": "String",
        "role": "String",
        "target_term": "String",
        "expected_sign": "String",
        "rank_ok": "Boolean",
        "clusters": "UInt16",
        "ar_status": "String",
        "shock_status": "String",
        "iv_direction_matches_ar": "Boolean",
        "conventional_point_accepted": "Boolean",
        "evidence_grade": "String",
        "interval_source": "String",
        "estimate": "Float64",
        "interval_low": "Float64",
        "interval_high": "Float64",
        "direction_matches": "Boolean",
    }
    return TableContract(
        table_id="evidence_scorecard",
        schema_version="1.0.0",
        primary_key=(
            "run_id",
            "analysis_family",
            "outcome_id",
            "horizon",
            "gad_version",
            "sample_version",
        ),
        columns=columns,
        units={
            "horizon": "years_after_treatment",
            "estimate": "registered_target_term_units",
            "interval_low": "registered_target_term_units",
            "interval_high": "registered_target_term_units",
        },
        null_semantics={
            "estimate": "null_only_for_not_estimated_cells",
            "interval_low": "null_for_not_estimated_or_unbounded_registered_interval",
            "interval_high": "null_for_not_estimated_or_unbounded_registered_interval",
            "concentration_cutoff_hhi": "null_because_no_preregistered_cutoff_exists",
            "concentration_cutoff_top1_share": "null_because_no_preregistered_cutoff_exists",
        },
        transformations=(
            "one_grade_for_every_confirmatory_registered_cell",
            "use_lp_iv_ar_only_for_conditional_causal_and_lp_fe_for_conditional_association",
            "never_encode_binary_hypothesis_acceptance_or_rejection",
            "unresolved_concentration_without_a_preregistered_cutoff_forces_exploratory",
            "retain_exploratory_estimates_as_diagnostics_but_exclude_them_from_direction_and_breadth",
        ),
    )


def write_evidence_scorecard(
    *,
    report: OutputAuditReport,
    destination: Path,
    source_root: Path,
    context: RunContext,
) -> Path:
    """Publish the scorecard as a contract-checked, source-bound Parquet table."""

    if report.status != "passed" or report.scorecard.height != 39:
        raise ValueError("only a complete passed audit can publish a scorecard")
    contract = _evidence_scorecard_contract()
    frame = report.scorecard.cast(
        {name: getattr(pl, dtype) for name, dtype in contract.columns.items()}
    ).select(*contract.columns)
    inputs = tuple(
        InputArtifact.from_path(source_root / relative)
        for relative, _ in report.audited_hashes
    )
    observed = {item.sha256 for item in inputs}
    expected = {value for _, value in report.audited_hashes}
    if observed != expected:
        raise ValueError("audited source hashes changed before scorecard publication")
    write_authoritative_table(
        frame,
        contract,
        destination,
        inputs,
        BuildIdentity(
            command="python -m green_debt.cli analysis-output-audit",
            code_commit=str(frame["audit_git_commit"].unique().item()),
            created_at_utc=context.created_at_utc,
        ),
    )
    return destination.resolve()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


_PUBLICATION_FORMATS = ("svg", "pdf", "png")
_PUBLICATION_WIDTH_MM = 183.0
_PUBLICATION_MAX_HEIGHT_MM = 245.0
_PUBLICATION_DPI = 600.0
_SVG_LENGTH = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(mm|cm|in|pt|px)?\s*$",
    flags=re.IGNORECASE,
)


def _svg_length_mm(value: str) -> float:
    match = _SVG_LENGTH.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported SVG length: {value}")
    number = float(match.group(1))
    unit = (match.group(2) or "px").lower()
    factors = {
        "mm": 1.0,
        "cm": 10.0,
        "in": 25.4,
        "pt": 25.4 / 72.0,
        "px": 25.4 / 96.0,
    }
    return number * factors[unit]


def _canvas_mm(metadata: dict[str, Any], figure_id: str) -> tuple[float, float]:
    canvas = metadata.get("target_canvas_mm")
    if not isinstance(canvas, dict):
        raise ValueError(f"figure manifest lacks canvas: {figure_id}")
    try:
        width = float(canvas["width"])
        height = float(canvas["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid figure canvas: {figure_id}") from exc
    if (
        not math.isclose(width, _PUBLICATION_WIDTH_MM, abs_tol=0.01)
        or height <= 0
        or height > _PUBLICATION_MAX_HEIGHT_MM
    ):
        raise ValueError(f"figure canvas exceeds publication bounds: {figure_id}")
    return width, height


def _validate_svg(
    path: Path, *, width_mm: float, height_mm: float, minimum_text_pt: float
) -> None:
    try:
        document = etree.parse(
            str(path), etree.XMLParser(resolve_entities=False, no_network=True)
        )
    except (OSError, etree.XMLSyntaxError) as exc:
        raise ValueError(f"invalid publication SVG: {path.name}") from exc
    root = document.getroot()
    if etree.QName(root).localname != "svg":
        raise ValueError(f"publication SVG root is not svg: {path.name}")
    try:
        observed_width = _svg_length_mm(root.attrib["width"])
        observed_height = _svg_length_mm(root.attrib["height"])
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"publication SVG dimensions are invalid: {path.name}"
        ) from exc
    if not math.isclose(observed_width, width_mm, abs_tol=0.05) or not math.isclose(
        observed_height, height_mm, abs_tol=0.05
    ):
        raise ValueError(f"publication SVG canvas mismatch: {path.name}")
    if document.xpath("//*[local-name()='image']"):
        raise ValueError(f"publication SVG contains raster image content: {path.name}")
    text_nodes = document.xpath("//*[local-name()='text']")
    if not text_nodes:
        raise ValueError(f"publication SVG lacks editable text: {path.name}")
    font_size_pattern = re.compile(r"(?:^|;)\s*font-size\s*:\s*([0-9.]+)")
    font_family_pattern = re.compile(r"(?:^|;)\s*font-family\s*:\s*([^;]+)")
    for node in text_nodes:
        style = node.attrib.get("style", "")
        size = font_size_pattern.search(style)
        family = font_family_pattern.search(style)
        if size is None or float(size.group(1)) + 1e-9 < minimum_text_pt:
            raise ValueError(f"publication SVG text is below minimum size: {path.name}")
        if family is None or "arial" not in family.group(1).lower():
            raise ValueError(f"publication SVG text is not Arial: {path.name}")
    white_background = any(
        "fill:#ffffff" in node.attrib.get("style", "").replace(" ", "").lower()
        and node.attrib.get("width") == "100%"
        and node.attrib.get("height") == "100%"
        for node in document.xpath("//*[local-name()='rect']")
    )
    if not white_background:
        raise ValueError(f"publication SVG lacks a white canvas: {path.name}")


def _validate_pdf(path: Path, *, width_mm: float, height_mm: float) -> None:
    try:
        reader = PdfReader(path)
    except Exception as exc:
        raise ValueError(f"invalid publication PDF: {path.name}") from exc
    if len(reader.pages) != 1:
        raise ValueError(f"publication PDF is not single-page: {path.name}")
    box = reader.pages[0].mediabox
    observed_width = float(box.width)
    observed_height = float(box.height)
    expected_width = math.floor(width_mm / 25.4 * 72.0)
    expected_height = math.floor(height_mm / 25.4 * 72.0)
    width_matches = math.isclose(observed_width, expected_width, abs_tol=0.01)
    height_matches = math.isclose(observed_height, expected_height, abs_tol=0.01)
    if not width_matches or not height_matches:
        raise ValueError(f"publication PDF canvas mismatch: {path.name}")


def _png_chunks(path: Path) -> dict[bytes, list[bytes]]:
    payload = path.read_bytes()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"invalid publication PNG signature: {path.name}")
    chunks: dict[bytes, list[bytes]] = {}
    offset = 8
    saw_end = False
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ValueError(f"truncated publication PNG: {path.name}")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(payload):
            raise ValueError(f"truncated publication PNG chunk: {path.name}")
        data = payload[offset + 8 : offset + 8 + length]
        observed_crc = struct.unpack(">I", payload[offset + 8 + length : end])[0]
        expected_crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        if observed_crc != expected_crc:
            raise ValueError(f"publication PNG CRC mismatch: {path.name}")
        chunks.setdefault(chunk_type, []).append(data)
        offset = end
        if chunk_type == b"IEND":
            saw_end = True
            break
    if not saw_end or offset != len(payload):
        raise ValueError(f"publication PNG has trailing or missing data: {path.name}")
    return chunks


def _validate_png(path: Path, *, width_mm: float, height_mm: float) -> None:
    chunks = _png_chunks(path)
    if len(chunks.get(b"IHDR", [])) != 1:
        raise ValueError(f"publication PNG lacks one IHDR chunk: {path.name}")
    ihdr = chunks[b"IHDR"][0]
    if len(ihdr) != 13:
        raise ValueError(f"publication PNG IHDR is invalid: {path.name}")
    width_px, height_px, bit_depth, colour_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", ihdr)
    )
    expected_width = round(width_mm / 25.4 * _PUBLICATION_DPI)
    expected_height = round(height_mm / 25.4 * _PUBLICATION_DPI)
    if abs(width_px - expected_width) > 1 or abs(height_px - expected_height) > 1:
        raise ValueError(f"publication PNG pixel dimensions mismatch: {path.name}")
    if (bit_depth, colour_type, compression, filtering, interlace) != (8, 2, 0, 0, 0):
        raise ValueError(f"publication PNG is not opaque 8-bit RGB: {path.name}")
    if len(chunks.get(b"pHYs", [])) != 1 or len(chunks[b"pHYs"][0]) != 9:
        raise ValueError(f"publication PNG lacks physical resolution: {path.name}")
    x_ppm, y_ppm, unit = struct.unpack(">IIB", chunks[b"pHYs"][0])
    if unit != 1 or not math.isclose(x_ppm * 0.0254, _PUBLICATION_DPI, abs_tol=0.1):
        raise ValueError(f"publication PNG horizontal DPI mismatch: {path.name}")
    if not math.isclose(y_ppm * 0.0254, _PUBLICATION_DPI, abs_tol=0.1):
        raise ValueError(f"publication PNG vertical DPI mismatch: {path.name}")
    srgb = chunks.get(b"sRGB", [])
    if len(srgb) != 1 or len(srgb[0]) != 1 or srgb[0][0] not in range(4):
        raise ValueError(f"publication PNG is not tagged sRGB: {path.name}")
    if chunks.get(b"bKGD") != [b"\x00\xff\x00\xff\x00\xff"]:
        raise ValueError(f"publication PNG background is not white: {path.name}")


def _publication_manifest(figures_root: Path) -> tuple[dict[str, Any], ...]:
    manifest_path = figures_root / "figure_manifest.json"
    captions_path = figures_root / "figure_captions.md"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid figure manifest") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "1.0"
        or manifest.get("main_figure_count") != 7
        or not isinstance(manifest.get("figures"), list)
        or len(manifest["figures"]) != 7
    ):
        raise ValueError("figure manifest schema mismatch")
    if manifest.get("captions_markdown_sha256") != sha256_file(captions_path):
        raise ValueError("figure captions hash mismatch")
    captions = captions_path.read_text(encoding="utf-8")
    required_metadata = {
        "figure_id",
        "research_question",
        "role",
        "panels",
        "analysis_unit",
        "uncertainty_display",
        "target_canvas_mm",
        "first_citation",
        "caption",
        "output_formats",
        "minimum_text_pt",
        "source_files",
    }
    result: list[dict[str, Any]] = []
    for index, metadata in enumerate(manifest["figures"], start=1):
        figure_id = f"figure_{index}"
        if not isinstance(metadata, dict) or set(metadata) != required_metadata:
            raise ValueError(f"figure manifest metadata is incomplete: {figure_id}")
        if metadata["figure_id"] != figure_id or metadata["role"] != "main":
            raise ValueError(f"figure manifest identity mismatch: {figure_id}")
        if set(metadata["output_formats"]) != set(_PUBLICATION_FORMATS):
            raise ValueError(f"figure manifest formats mismatch: {figure_id}")
        if metadata["first_citation"] != "not_available_no_manuscript":
            raise ValueError(
                f"figure manuscript-insertion status mismatch: {figure_id}"
            )
        source_files = metadata["source_files"]
        if isinstance(source_files, str):
            source_files = [source_files]
        if (
            not isinstance(source_files, list)
            or not source_files
            or any(
                not isinstance(value, str) or not value.strip()
                for value in source_files
            )
        ):
            raise ValueError(f"figure manifest lacks source files: {figure_id}")
        panels = metadata["panels"]
        if isinstance(panels, str):
            panels = [panels]
        if (
            not isinstance(panels, list)
            or not panels
            or any(not isinstance(value, str) or not value.strip() for value in panels)
        ):
            raise ValueError(f"figure manifest lacks panels: {figure_id}")
        for key in (
            "research_question",
            "analysis_unit",
            "uncertainty_display",
            "caption",
        ):
            if not isinstance(metadata[key], str) or not metadata[key].strip():
                raise ValueError(f"figure manifest lacks {key}: {figure_id}")
        if f"## Figure {index}" not in captions or metadata["caption"] not in captions:
            raise ValueError(f"standalone caption mismatch: {figure_id}")
        minimum_text_pt = float(metadata["minimum_text_pt"])
        if minimum_text_pt < 6.5:
            raise ValueError(f"figure minimum text is below 6.5 pt: {figure_id}")
        _canvas_mm(metadata, figure_id)
        result.append(metadata)
    return tuple(result)


def _publication_hashes(
    output_root: Path, source_hashes: set[str]
) -> tuple[tuple[str, str], ...]:
    expected_tables = {f"table_{index}.csv" for index in range(1, 9)}
    expected_figures = {
        *(
            f"figure_{index}.{extension}"
            for index in range(1, 8)
            for extension in _PUBLICATION_FORMATS
        ),
        *(f"figure_{index}.provenance.json" for index in range(1, 8)),
        *(f"data/figure_{index}.parquet" for index in range(1, 8)),
        "figure_manifest.json",
        "figure_captions.md",
    }
    tables_root = output_root / "tables"
    figures_root = output_root / "figures"
    observed_tables = (
        {path.name for path in tables_root.iterdir()} if tables_root.is_dir() else set()
    )
    observed_figures = (
        {
            str(path.relative_to(figures_root))
            for path in figures_root.rglob("*")
            if path.is_file()
        }
        if figures_root.is_dir()
        else set()
    )
    unexpected = sorted(
        (observed_tables - expected_tables) | (observed_figures - expected_figures)
    )
    missing = sorted(
        (expected_tables - observed_tables) | (expected_figures - observed_figures)
    )
    if unexpected or missing:
        raise ValueError(
            "unexpected publication files; "
            f"missing={','.join(missing)}; extra={','.join(unexpected)}"
        )
    metadata = _publication_manifest(figures_root)
    paths: list[Path] = []
    for index in range(1, 9):
        path = output_root / f"tables/table_{index}.csv"
        if not path.is_file():
            raise ValueError(f"publication table is missing: {path.name}")
        paths.append(path)
    for index in range(1, 8):
        figure_id = f"figure_{index}"
        rendered = {
            extension: output_root / f"figures/{figure_id}.{extension}"
            for extension in _PUBLICATION_FORMATS
        }
        data = output_root / f"figures/data/figure_{index}.parquet"
        provenance = output_root / f"figures/figure_{index}.provenance.json"
        for path in (*rendered.values(), data, provenance):
            if not path.is_file():
                raise ValueError(f"publication figure companion is missing: {path}")
        try:
            payload = json.loads(provenance.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid figure provenance: {provenance.name}") from exc
        bound = payload.get("source_table_hashes")
        if not isinstance(bound, dict) or not bound:
            raise ValueError(
                f"figure provenance lacks source hashes: {provenance.name}"
            )
        if not set(str(value) for value in bound.values()) <= source_hashes:
            raise ValueError(
                f"figure provenance uses unaudited inputs: {provenance.name}"
            )
        if payload.get("figure_data_sha256") != sha256_file(data):
            raise ValueError(f"figure data hash mismatch: {data.name}")
        plotted_hash = payload.get("plotted_source_data_hash")
        if not isinstance(plotted_hash, str) or len(plotted_hash) != 64:
            raise ValueError(f"figure plotted-data hash is invalid: {provenance.name}")
        rendered_hashes = payload.get("rendered_files_sha256")
        if not isinstance(rendered_hashes, dict) or set(rendered_hashes) != set(
            _PUBLICATION_FORMATS
        ):
            raise ValueError(
                f"figure rendered hashes are incomplete: {provenance.name}"
            )
        for extension, path in rendered.items():
            if rendered_hashes[extension] != sha256_file(path):
                raise ValueError(f"figure rendered-file hash mismatch: {path.name}")
        width_mm, height_mm = _canvas_mm(metadata[index - 1], figure_id)
        minimum_text_pt = float(metadata[index - 1]["minimum_text_pt"])
        _validate_svg(
            rendered["svg"],
            width_mm=width_mm,
            height_mm=height_mm,
            minimum_text_pt=minimum_text_pt,
        )
        _validate_pdf(rendered["pdf"], width_mm=width_mm, height_mm=height_mm)
        _validate_png(rendered["png"], width_mm=width_mm, height_mm=height_mm)
        paths.extend((*rendered.values(), data, provenance))
    paths.extend(
        (
            figures_root / "figure_manifest.json",
            figures_root / "figure_captions.md",
        )
    )
    return tuple(
        sorted(
            (str(path.relative_to(output_root)), sha256_file(path))
            for path in paths
        )
    )


def _audit_publication_set(
    output_root: Path,
    source_hashes: set[str],
    *,
    upstream_model_git_commit: str,
    expected_reporting_git_commit: str,
    expected_evidence_policy_sha256: str | None = None,
) -> PublicationAudit:
    """Verify the fixed reporting receipt identities before accepting hashes."""

    expected_fields = {
        "figure_id",
        "upstream_model_git_commit",
        "reporting_git_commit",
        "evidence_policy_sha256",
        "source_table_hashes",
        "figure_data_sha256",
        "rendered_files_sha256",
        "plotted_source_data_hash",
    }
    reporting_commits: set[str] = set()
    for index in range(1, 8):
        figure_id = f"figure_{index}"
        path = output_root / f"figures/{figure_id}.provenance.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid figure provenance: {path.name}") from exc
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise ValueError(f"figure provenance schema mismatch: {path.name}")
        if payload["figure_id"] != figure_id:
            raise ValueError(f"figure provenance filename mismatch: {path.name}")
        if payload["upstream_model_git_commit"] != upstream_model_git_commit:
            raise ValueError(f"figure provenance upstream commit mismatch: {path.name}")
        policy_sha256 = payload["evidence_policy_sha256"]
        if (
            not isinstance(policy_sha256, str)
            or len(policy_sha256) != 64
            or any(value not in "0123456789abcdef" for value in policy_sha256)
            or (
                expected_evidence_policy_sha256 is not None
                and policy_sha256 != expected_evidence_policy_sha256
            )
        ):
            raise ValueError(f"figure provenance evidence policy mismatch: {path.name}")
        reporting_commit = payload["reporting_git_commit"]
        if (
            not isinstance(reporting_commit, str)
            or len(reporting_commit) != 40
            or any(value not in "0123456789abcdef" for value in reporting_commit)
        ):
            raise ValueError(f"invalid reporting Git commit: {path.name}")
        reporting_commits.add(reporting_commit)
    if len(reporting_commits) != 1:
        raise ValueError("mixed reporting Git commits in figure provenance")
    reporting_git_commit = next(iter(reporting_commits))
    if reporting_git_commit != expected_reporting_git_commit:
        raise ValueError(
            "stale reporting Git commit: "
            f"{reporting_git_commit} != {expected_reporting_git_commit}"
        )
    return PublicationAudit(
        hashes=_publication_hashes(output_root, source_hashes),
        reporting_git_commit=reporting_git_commit,
    )


def _manifest_git_identities(
    *,
    upstream_model_git_commit: str,
    reporting_git_commit: str,
    audit_git_commit: str,
) -> dict[str, str]:
    return {
        "upstream_model_git_commit": upstream_model_git_commit,
        "reporting_git_commit": reporting_git_commit,
        "audit_git_commit": audit_git_commit,
        "git_commit": audit_git_commit,
    }


def publish_analysis_audit(
    *, code_root: Path, data_root: Path, output_root: Path
) -> dict[str, object]:
    """Audit, publish the scorecard/summary, then atomically install success."""

    output = resolve_authorized_analysis_output(
        code_root, data_root, output_root
    )
    analysis_preflight(code_root, data_root, output)
    manifest_path = output / "run_manifest.json"
    manifest_path.unlink(missing_ok=True)
    report = audit_analysis_outputs(
        code_root=code_root, data_root=data_root, output_root=output
    )
    context = load_stage_a_run_context(output)
    audit_git_commit = current_analysis_git_commit(code_root)
    evidence_policy, evidence_policy_sha256 = load_evidence_policy(
        code_root.resolve() / "config/evidence_policy.json"
    )
    if evidence_policy_sha256 != context.evidence_policy_sha256:
        raise ValueError("publication evidence policy authority mismatch")
    source_hashes = {value for _, value in report.audited_hashes}
    publication = _audit_publication_set(
        output,
        source_hashes,
        upstream_model_git_commit=context.git_commit,
        expected_reporting_git_commit=audit_git_commit,
        expected_evidence_policy_sha256=context.evidence_policy_sha256,
    )
    publication_hashes = publication.hashes
    scorecard_path = write_evidence_scorecard(
        report=report,
        destination=output / "machine_readable/evidence_scorecard.parquet",
        source_root=output,
        context=context,
    )
    labels = summarize_hypotheses(report.scorecard)
    grade_counts = {
        str(row["evidence_grade"]): int(row["len"])
        for row in report.scorecard.group_by("evidence_grade").len().iter_rows(named=True)
    }
    direction_counts = {
        str(row["direction_matches"]).lower(): int(row["len"])
        for row in report.scorecard.group_by("direction_matches").len().iter_rows(named=True)
    }
    summary = (
        "# Analysis evidence summary\n\n"
        "**Evidence gate: exploratory only.** Concentration status is "
        f"`{evidence_policy['concentration']['status']}` because no "
        "preregistered executable cutoff exists; observed HHI/top-1 values are "
        "diagnostic disclosures and were not used to choose a post-results cutoff.\n\n"
        f"Evidence-policy SHA-256: `{evidence_policy_sha256}`. "
        f"Exposure max HHI/top-1: {report.scorecard['exposure_hhi_max'].max():.6g}/"
        f"{report.scorecard['exposure_top1_share_max'].max():.6g}.\n\n"
        f"Run `{context.run_id}` reports evidence grades for all {report.scorecard.height} "
        "confirmatory cells. These grades are conditional evidence labels, not binary "
        "hypothesis acceptance or rejection.\n\n"
        "## Evidence grades\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in sorted(grade_counts.items()))
        + "\n\n## Frozen breadth assessments\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in labels.items())
        + "\n"
    )
    summary_path = output / "analysis_summary.md"
    _atomic_write_bytes(summary_path, summary.encode("utf-8"))
    generated_paths = (
        scorecard_path,
        scorecard_path.with_name(f"{scorecard_path.name}.schema.json"),
        scorecard_path.with_name(f"{scorecard_path.name}.manifest.json"),
        summary_path,
    )
    generated_hashes = tuple(
        (str(path.relative_to(output)), sha256_file(path)) for path in generated_paths
    )
    final_output_bytes = directory_usage_bytes(output)
    final_project_bytes = combined_project_usage_bytes(data_root, output)
    spec = load_analysis_spec(code_root.resolve() / "config/analysis.yaml")
    config = load_project_config(code_root.resolve() / "config/project.yaml")
    if final_output_bytes > spec.outputs.quota_gb * GIB:
        raise RuntimeError("10 GB analysis output quota exceeded after publication")
    if config.storage.absolute_limit_gb is None or final_project_bytes >= config.storage.absolute_limit_gb * GIB:
        raise RuntimeError("150 GB absolute project limit reached after publication")
    registry = json.loads(
        (output / "registries/threshold_registry_v1.json").read_text(encoding="utf-8")
    )
    all_hashes = dict(sorted((*report.audited_hashes, *publication_hashes, *generated_hashes)))
    audited_at = analysis_reproduction_audited_at(
        code_root=code_root,
        data_root=data_root,
        output_root=output,
        default=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        expected_git_commit=audit_git_commit,
    )
    payload: dict[str, object] = {
        **asdict(context),
        **_manifest_git_identities(
            upstream_model_git_commit=context.git_commit,
            reporting_git_commit=publication.reporting_git_commit,
            audit_git_commit=audit_git_commit,
        ),
        "status": "success",
        "audit_kind": "analysis-output-audit",
        "threshold_registry_hash": registry["registry_hash"],
        "evidence_policy_sha256": evidence_policy_sha256,
        "concentration_status": evidence_policy["concentration"]["status"],
        "concentration_authority": evidence_policy["concentration"],
        "concentration_diagnostics": {
            "exposure_hhi_max": report.scorecard["exposure_hhi_max"].max(),
            "exposure_top1_share_max": report.scorecard[
                "exposure_top1_share_max"
            ].max(),
            "rotemberg_hhi_absolute_max": report.scorecard[
                "rotemberg_hhi_absolute"
            ].max(),
            "rotemberg_top1_absolute_share_max": report.scorecard[
                "rotemberg_top1_absolute_share"
            ].max(),
        },
        "audited_output_hashes": all_hashes,
        "evidence_grade_counts": grade_counts,
        "hypothesis_direction_counts": direction_counts,
        "hypothesis_breadth": labels,
        "continuous_cells": report.continuous_cells,
        "ar_cells": report.ar_cells,
        "threshold_cells": report.threshold_cells,
        "output_bytes": final_output_bytes,
        "project_bytes": final_project_bytes,
        "audited_at_utc": audited_at,
    }
    _atomic_write_bytes(
        manifest_path,
        (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return payload
