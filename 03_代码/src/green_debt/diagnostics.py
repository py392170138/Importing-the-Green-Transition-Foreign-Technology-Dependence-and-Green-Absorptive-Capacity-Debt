"""Exact samples and outcome-free diagnostics for stage A."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

import numpy as np
import polars as pl
from scipy import linalg
import statsmodels.api as sm

from green_debt.analysis_io import (
    AnalysisPaths,
    DiagnosticBundle,
    RunContext,
    analysis_preflight,
    load_table_contract,
)
from green_debt.analysis_spec import AnalysisSpec, ModelCell, load_analysis_spec
from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    write_authoritative_table,
)
from green_debt.storage import GIB, directory_usage_bytes, sha256_file


MODEL_ALIASES = {
    "gimc_p01_p99": "gimc_a",
    "gad_lag_p01_p99": "gad_a",
    "Z_p01_p99": "z_a",
    "Z_GAD_p01_p99": "z_gad_a",
    "renewable_energy_consumption_share_analysis_p01_p99": (
        "renewable_energy_consumption_share_a"
    ),
    "trade_openness_percent_gdp_analysis_p01_p99": "trade_openness_a",
    "industry_value_added_share_analysis_p01_p99": (
        "industry_value_added_share_a"
    ),
    "gdp_per_capita_current_usd_analysis_p01_p99": "gdp_per_capita_a",
}
STATE_COLUMNS = tuple(MODEL_ALIASES)
MODEL_COLUMNS = (
    "gimc_a",
    "gimc_gad_a",
    "z_a",
    "z_gad_a",
)
CONTROL_COLUMNS = (
    "renewable_energy_consumption_share_a",
    "trade_openness_a",
    "industry_value_added_share_a",
    "gdp_per_capita_a",
)


@dataclass(frozen=True)
class FirstStageScreen:
    spec_id: str
    outcome_id: str
    horizon: int
    gad_version: str
    n: int
    economies: int
    clusters: int
    instrument_rank: int
    cross_moment_rank: int
    condition_number: float | None
    partial_r2_gimc: float | None
    partial_r2_interaction: float | None
    effective_f_gimc: float | None
    effective_f_interaction: float | None
    gate_status: str


def _cell_predicate(cell: ModelCell, spec: AnalysisSpec) -> pl.Expr:
    predicate = (
        (pl.col("outcome_id") == cell.outcome_id)
        & (pl.col("horizon") == cell.horizon)
        & (pl.col("gad_version") == cell.gad_version)
        & (pl.col("sample_version") == cell.sample_version)
        & pl.col("treatment_time").is_between(spec.period[0], spec.period[1])
    )
    if cell.analysis_family in {"confirmatory", "bounded_controls"}:
        return predicate & (
            pl.col("confirmatory_iv_eligible")
            & ~pl.col("descriptive_only")
            & ~pl.col("threshold_selection_only")
            & ~pl.col("negative_shock_sample")
        )
    if cell.analysis_family == "threshold_selection":
        return predicate & pl.col("threshold_selection_only")
    if cell.analysis_family == "vulnerability":
        return predicate & pl.col("negative_shock_sample")
    raise ValueError(f"unsupported registered analysis family: {cell.analysis_family}")


def _candidate_model_frame(
    panel: pl.LazyFrame, cell: ModelCell, spec: AnalysisSpec
) -> pl.DataFrame:
    spec.require_registered_cell(cell)
    return (
        panel.filter(_cell_predicate(cell, spec))
        .select(
            "economy_id",
            "treatment_time",
            "delta_outcome",
            *(
                pl.col(source).alias(target)
                for source, target in MODEL_ALIASES.items()
            ),
        )
        .sort("economy_id", "treatment_time")
        .collect()
    )


def exact_model_sample(
    panel: pl.LazyFrame, cell: ModelCell, spec: AnalysisSpec
) -> pl.DataFrame:
    """Return the complete-case frame for one and only one registered cell."""

    spec.require_registered_cell(cell)
    selected = (
        panel.filter(_cell_predicate(cell, spec))
        .select(
            "economy_id",
            "treatment_time",
            "delta_outcome",
            *(
                pl.col(source).alias(target)
                for source, target in MODEL_ALIASES.items()
            ),
        )
        .with_columns(
            (pl.col("gimc_a") * pl.col("gad_a")).alias("gimc_gad_a")
        )
        .select(
            "economy_id",
            "treatment_time",
            "delta_outcome",
            "gimc_a",
            "gimc_gad_a",
            "gad_a",
            "z_a",
            "z_gad_a",
            "renewable_energy_consumption_share_a",
            "trade_openness_a",
            "industry_value_added_share_a",
            "gdp_per_capita_a",
        )
        .drop_nulls()
        .sort("economy_id", "treatment_time")
        .collect()
    )
    if selected.is_empty():
        raise ValueError(f"empty exact model sample: {cell}")
    numeric_columns = tuple(
        name
        for name in selected.columns
        if name not in {"economy_id", "treatment_time"}
    )
    nonfinite = selected.select(
        pl.any_horizontal(
            *(~pl.col(name).is_finite() for name in numeric_columns)
        ).sum()
    ).item()
    if nonfinite:
        raise ValueError(f"nonfinite frozen values in exact model sample: {cell}")
    if (
        cell.analysis_family == "vulnerability"
        and selected.filter(pl.col("z_a") >= 0).height
    ):
        raise ValueError("negative-shock sample requires Z < 0 for every row")
    return selected


def replication_safe_state_frame(panel: pl.LazyFrame) -> pl.DataFrame:
    """Remove outcome/horizon replication only after proving state consistency."""

    key = ("economy_id", "treatment_time", "gad_version", "sample_version")
    source = panel.select(*key, *STATE_COLUMNS)
    conflicts = (
        source.group_by(*key)
        .agg(*(pl.col(name).n_unique().alias(name) for name in STATE_COLUMNS))
        .filter(
            pl.any_horizontal(
                *(pl.col(name) > 1 for name in STATE_COLUMNS)
            )
        )
        .collect()
    )
    if conflicts.height:
        raise ValueError("replicated horizon/outcome rows disagree on state variables")
    return (
        source.group_by(*key)
        .agg(*(pl.col(name).first().alias(name) for name in STATE_COLUMNS))
        .sort(*key)
        .collect()
    )


def classify_model_gate(*, rank: int, clusters: int, condition_number: float) -> str:
    if rank < 2 or not math.isfinite(condition_number):
        return "fail_rank_deficient"
    if clusters < 20:
        return "exploratory_lt20_clusters"
    if clusters < 30:
        return "wild_bootstrap_required"
    return "ready_cluster_robust"


def classify_first_stage(
    *, rank: int, effective_f: tuple[float, float]
) -> str:
    if rank < 2:
        return "fail_rank_deficient"
    if min(effective_f) < 10.0:
        return "weak_reference_below_10"
    return "adequate_reference_10"


def _fixed_effect_design(frame: pl.DataFrame) -> np.ndarray:
    n = frame.height
    controls = frame.select("gad_a", *CONTROL_COLUMNS).to_numpy().astype(float)
    economies = frame.get_column("economy_id").to_numpy()
    years = frame.get_column("treatment_time").to_numpy()
    economy_levels = sorted(set(economies.tolist()))
    year_levels = sorted(set(years.tolist()))
    columns: list[np.ndarray] = [np.ones(n), *controls.T]
    columns.extend((economies == value).astype(float) for value in economy_levels[1:])
    columns.extend((years == value).astype(float) for value in year_levels[1:])
    return np.column_stack(columns)


def _qr_residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    q, r, _ = linalg.qr(design, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        return values.copy()
    tolerance = max(design.shape) * np.finfo(float).eps * diagonal.max()
    rank = int((diagonal > tolerance).sum())
    basis = q[:, :rank]
    residualized = values - basis @ (basis.T @ values)
    residualized[np.abs(residualized) < 1e-14] = 0.0
    return residualized


def first_stage_metrics(
    endogenous: np.ndarray,
    instruments: np.ndarray,
    clusters: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Calculate partial R-squared and clustered joint-Wald effective F."""

    partial_r2: list[float] = []
    effective_f: list[float] = []
    restriction = np.eye(instruments.shape[1])
    for column in range(endogenous.shape[1]):
        response = endogenous[:, column]
        restricted_sse = float(response @ response)
        if restricted_sse <= np.finfo(float).eps:
            raise ValueError("first-stage endogenous residual has zero variation")
        fit = sm.OLS(response, instruments).fit(
            cov_type="cluster",
            cov_kwds={"groups": clusters},
        )
        full_sse = float(fit.resid @ fit.resid)
        value = 1.0 - full_sse / restricted_sse
        partial_r2.append(float(min(1.0, max(0.0, value))))
        statistic = float(fit.wald_test(restriction, scalar=True).statistic)
        effective_f.append(statistic / instruments.shape[1])
    return (partial_r2[0], partial_r2[1]), (
        effective_f[0],
        effective_f[1],
    )


_FIRST_STAGE_SCHEMA = {
    "spec_id": pl.String,
    "analysis_family": pl.String,
    "outcome_id": pl.String,
    "horizon": pl.Int16,
    "gad_version": pl.String,
    "sample_version": pl.String,
    "role": pl.String,
    "n": pl.UInt32,
    "economies": pl.UInt16,
    "clusters": pl.UInt16,
    "year_min": pl.Int16,
    "year_max": pl.Int16,
    "instrument_rank": pl.UInt8,
    "cross_moment_rank": pl.UInt8,
    "rank": pl.UInt8,
    "condition_number": pl.Float64,
    "partial_r2_gimc": pl.Float64,
    "partial_r2_interaction": pl.Float64,
    "effective_f_gimc": pl.Float64,
    "effective_f_interaction": pl.Float64,
    "first_stage_status": pl.String,
    "gate_status": pl.String,
}


def first_stage_screen(
    panel: pl.LazyFrame,
    spec: AnalysisSpec,
    cells: Sequence[ModelCell] | None = None,
) -> pl.DataFrame:
    """Run outcome-free identification screens on exact registered samples."""

    selected_cells = tuple(cells) if cells is not None else spec.registered_cells()
    rows: list[dict[str, object]] = []
    for cell in selected_cells:
        sample = exact_model_sample(panel, cell, spec)
        values = sample.select(*MODEL_COLUMNS).to_numpy().astype(float)
        residualized = _qr_residualize(values, _fixed_effect_design(sample))
        endogenous = residualized[:, :2]
        instruments = residualized[:, 2:]
        instrument_rank = int(np.linalg.matrix_rank(instruments))
        cross_moment = instruments.T @ endogenous / sample.height
        cross_moment_rank = int(np.linalg.matrix_rank(cross_moment))
        rank = min(instrument_rank, cross_moment_rank)
        condition = (
            float(np.linalg.cond(cross_moment)) if rank == 2 else float("inf")
        )
        partial: tuple[float | None, float | None] = (None, None)
        effective: tuple[float | None, float | None] = (None, None)
        first_stage_status = "fail_rank_deficient"
        if rank == 2 and math.isfinite(condition):
            cluster_codes = np.unique(
                sample.get_column("economy_id").to_numpy(), return_inverse=True
            )[1]
            try:
                measured_partial, measured_effective = first_stage_metrics(
                    endogenous, instruments, cluster_codes
                )
            except (ValueError, np.linalg.LinAlgError):
                condition = float("inf")
            else:
                if all(
                    math.isfinite(value)
                    for value in (*measured_partial, *measured_effective)
                ):
                    partial = measured_partial
                    effective = measured_effective
                    first_stage_status = classify_first_stage(
                        rank=rank, effective_f=measured_effective
                    )
                else:
                    condition = float("inf")
        clusters = int(sample.get_column("economy_id").n_unique())
        gate_status = classify_model_gate(
            rank=rank,
            clusters=clusters,
            condition_number=condition,
        )
        rows.append(
            {
                "spec_id": spec.spec_id,
                "analysis_family": cell.analysis_family,
                "outcome_id": cell.outcome_id,
                "horizon": cell.horizon,
                "gad_version": cell.gad_version,
                "sample_version": cell.sample_version,
                "role": cell.role,
                "n": sample.height,
                "economies": sample.get_column("economy_id").n_unique(),
                "clusters": clusters,
                "year_min": sample.get_column("treatment_time").min(),
                "year_max": sample.get_column("treatment_time").max(),
                "instrument_rank": instrument_rank,
                "cross_moment_rank": cross_moment_rank,
                "rank": rank,
                "condition_number": condition if math.isfinite(condition) else None,
                "partial_r2_gimc": partial[0],
                "partial_r2_interaction": partial[1],
                "effective_f_gimc": effective[0],
                "effective_f_interaction": effective[1],
                "first_stage_status": first_stage_status,
                "gate_status": gate_status,
            }
        )
    return pl.DataFrame(rows, schema=_FIRST_STAGE_SCHEMA, strict=False).sort(
        "analysis_family",
        "outcome_id",
        "horizon",
        "gad_version",
        "sample_version",
    )


def exposure_concentration(baseline_shares: pl.LazyFrame) -> pl.DataFrame:
    """Summarize frozen importer exposure without changing its support."""

    shares = (
        baseline_shares.filter(
            (pl.col("taxonomy_version") == "main_hs96")
            & (pl.col("share_version") == "main_0.0001")
            & pl.col("retained")
        )
        .select("importer", "baseline_share")
        .collect()
    )
    if shares.is_empty():
        raise ValueError("no retained main baseline shares")
    invalid = shares.filter(
        pl.col("baseline_share").is_null()
        | ~pl.col("baseline_share").is_finite()
        | (pl.col("baseline_share") < 0)
    )
    if invalid.height:
        raise ValueError("retained baseline shares must be finite and nonnegative")
    sums = shares.group_by("importer").agg(pl.col("baseline_share").sum().alias("sum"))
    if sums.filter((pl.col("sum") - 1.0).abs() > 1e-10).height:
        raise ValueError("retained baseline shares must sum to one by importer")

    rows: list[dict[str, object]] = []
    for importer, group in shares.group_by("importer", maintain_order=False):
        values = sorted(group.get_column("baseline_share").to_list(), reverse=True)
        hhi = float(sum(value * value for value in values))
        if hhi <= 0:
            raise ValueError("retained baseline exposure HHI must be positive")
        rows.append(
            {
                "importer": importer[0],
                "exposure_units": len(values),
                "hhi": hhi,
                "effective_units": 1.0 / hhi,
                "top1_share": values[0],
                "top5_share": sum(values[:5]),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "importer": pl.String,
            "exposure_units": pl.UInt32,
            "hhi": pl.Float64,
            "effective_units": pl.Float64,
            "top1_share": pl.Float64,
            "top5_share": pl.Float64,
        },
        strict=False,
    ).sort("importer")


def _provenance(context: RunContext) -> dict[str, object]:
    return {
        "run_id": context.run_id,
        "spec_id": context.spec_id,
        "input_authority_hash": context.input_authority_hash,
        "git_commit": context.git_commit,
        "renv_lock_sha256": context.renv_lock_sha256,
        "evidence_policy_sha256": context.evidence_policy_sha256,
        "created_at_utc": context.created_at_utc,
    }


def _add_provenance(frame: pl.DataFrame, context: RunContext) -> pl.DataFrame:
    return frame.with_columns(
        *(pl.lit(value).alias(name) for name, value in _provenance(context).items())
    )


def _contract_frame(
    rows: list[dict[str, object]], contract: TableContract
) -> pl.DataFrame:
    if not rows:
        raise ValueError(f"cannot construct empty analysis table: {contract.table_id}")
    schema = {
        name: getattr(pl, dtype) for name, dtype in contract.columns.items()
    }
    return pl.DataFrame(rows, schema=schema, strict=False).select(*contract.columns)


def _sample_and_missingness_frames(
    panel: pl.LazyFrame,
    spec: AnalysisSpec,
    context: RunContext,
    sample_contract: TableContract,
    missing_contract: TableContract,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    sample_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    provenance = _provenance(context)
    for cell in spec.registered_cells():
        candidate = _candidate_model_frame(panel, cell, spec)
        if candidate.is_empty():
            raise ValueError(f"empty registered candidate sample: {cell}")
        exact = exact_model_sample(panel, cell, spec)
        loss = candidate.height - exact.height
        cell_values = {
            "analysis_family": cell.analysis_family,
            "outcome_id": cell.outcome_id,
            "horizon": cell.horizon,
            "gad_version": cell.gad_version,
            "sample_version": cell.sample_version,
            "role": cell.role,
        }
        sample_rows.append(
            {
                **provenance,
                **cell_values,
                "candidate_rows": candidate.height,
                "n": exact.height,
                "sample_loss": loss,
                "sample_loss_share": loss / candidate.height,
                "economies": exact.get_column("economy_id").n_unique(),
                "clusters": exact.get_column("economy_id").n_unique(),
                "year_min": exact.get_column("treatment_time").min(),
                "year_max": exact.get_column("treatment_time").max(),
            }
        )
        for variable in candidate.columns[2:]:
            null_count = candidate.get_column(variable).null_count()
            missing_rows.append(
                {
                    **provenance,
                    **cell_values,
                    "variable": variable,
                    "candidate_rows": candidate.height,
                    "null_count": null_count,
                    "null_share": null_count / candidate.height,
                }
            )
    return (
        _contract_frame(sample_rows, sample_contract),
        _contract_frame(missing_rows, missing_contract),
    )


def _registered_state_frame(
    panel: pl.LazyFrame, spec: AnalysisSpec
) -> pl.DataFrame:
    pairs = sorted(
        {
            (cell.gad_version, cell.sample_version)
            for cell in spec.registered_cells()
        }
    )
    pair_predicate = pl.any_horizontal(
        *(
            (pl.col("gad_version") == gad_version)
            & (pl.col("sample_version") == sample_version)
            for gad_version, sample_version in pairs
        )
    )
    return replication_safe_state_frame(
        panel.filter(
            pair_predicate
            & pl.col("treatment_time").is_between(
                spec.period[0], spec.period[1]
            )
        )
    )


def _distribution_and_correlation_frames(
    panel: pl.LazyFrame,
    spec: AnalysisSpec,
    context: RunContext,
    distribution_contract: TableContract,
    correlation_contract: TableContract,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    state = _registered_state_frame(panel, spec)
    provenance = _provenance(context)
    distribution_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    for (gad_version, sample_version), group in state.group_by(
        "gad_version", "sample_version", maintain_order=False
    ):
        for variable in STATE_COLUMNS:
            values = group.get_column(variable).drop_nulls()
            if values.len() < 2:
                raise ValueError(
                    f"too few replication-safe values for {gad_version}/{sample_version}/{variable}"
                )
            if values.filter(~values.is_finite()).len():
                raise ValueError(f"nonfinite replication-safe state variable: {variable}")
            distribution_rows.append(
                {
                    **provenance,
                    "gad_version": gad_version,
                    "sample_version": sample_version,
                    "variable": variable,
                    "n": values.len(),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "min": values.min(),
                    "p01": values.quantile(0.01, interpolation="linear"),
                    "p25": values.quantile(0.25, interpolation="linear"),
                    "p50": values.quantile(0.50, interpolation="linear"),
                    "p75": values.quantile(0.75, interpolation="linear"),
                    "p99": values.quantile(0.99, interpolation="linear"),
                    "max": values.max(),
                }
            )
        for variable_x in STATE_COLUMNS:
            for variable_y in STATE_COLUMNS:
                pair = group.select(
                    pl.col(variable_x).alias("_x"),
                    pl.col(variable_y).alias("_y"),
                ).drop_nulls()
                correlation: float | None = None
                if pair.height >= 2:
                    x = pair.get_column("_x").to_numpy().astype(float)
                    y = pair.get_column("_y").to_numpy().astype(float)
                    if np.std(x) > 0 and np.std(y) > 0:
                        measured = float(np.corrcoef(x, y)[0, 1])
                        correlation = measured if math.isfinite(measured) else None
                correlation_rows.append(
                    {
                        **provenance,
                        "gad_version": gad_version,
                        "sample_version": sample_version,
                        "variable_x": variable_x,
                        "variable_y": variable_y,
                        "n": pair.height,
                        "correlation": correlation,
                    }
                )
    return (
        _contract_frame(distribution_rows, distribution_contract),
        _contract_frame(correlation_rows, correlation_contract),
    )


def _descriptive_path_frame(
    panel: pl.LazyFrame,
    spec: AnalysisSpec,
    context: RunContext,
    contract: TableContract,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    provenance = _provenance(context)
    for outcome in spec.outcomes:
        environmental = outcome.horizons == (1, 2, 3)
        horizons = (0, 1, 2, 3) if environmental else outcome.horizons
        path_family = "environmental" if environmental else "industrial"
        for horizon in horizons:
            flag = (
                pl.col("descriptive_only")
                if horizon == 0
                else ~pl.col("descriptive_only")
            )
            values = (
                panel.filter(
                    (pl.col("outcome_id") == outcome.outcome_id)
                    & (pl.col("horizon") == horizon)
                    & (pl.col("gad_version") == outcome.gad_version)
                    & (pl.col("sample_version") == spec.sample_version)
                    & pl.col("treatment_time").is_between(
                        spec.period[0], spec.period[1]
                    )
                    & flag
                    & ~pl.col("threshold_selection_only")
                    & ~pl.col("negative_shock_sample")
                )
                .select("economy_id", "treatment_time", "delta_outcome")
                .drop_nulls()
                .collect()
            )
            if values.is_empty():
                raise ValueError(
                    f"empty registered descriptive path: {outcome.outcome_id} h={horizon}"
                )
            if values.filter(~pl.col("delta_outcome").is_finite()).height:
                raise ValueError("nonfinite descriptive outcome path")
            rows.append(
                {
                    **provenance,
                    "path_family": path_family,
                    "outcome_id": outcome.outcome_id,
                    "horizon": horizon,
                    "gad_version": outcome.gad_version,
                    "sample_version": spec.sample_version,
                    "n": values.height,
                    "economies": values.get_column("economy_id").n_unique(),
                    "year_min": values.get_column("treatment_time").min(),
                    "year_max": values.get_column("treatment_time").max(),
                    "mean_delta_outcome": values.get_column("delta_outcome").mean(),
                    "interpretation_status": "noncausal_descriptive",
                }
            )
    return _contract_frame(rows, contract)


def _analysis_contracts(code_root: Path) -> dict[str, TableContract]:
    root = code_root / "03_代码/contracts/analysis"
    return {
        name: load_table_contract(root / f"{name}.json")
        for name in (
            "sample_cell",
            "missingness_cell",
            "distribution",
            "correlation",
            "diagnostic_cell",
            "exposure_concentration",
            "descriptive_path",
        )
    }


def _write_json_atomic(payload: dict[str, object], destination: Path) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _enforce_analysis_quota(
    output_root: Path, spec: AnalysisSpec, *, projected_bytes: int = 0
) -> None:
    current = directory_usage_bytes(output_root)
    if current + projected_bytes >= spec.outputs.quota_gb * GIB:
        raise RuntimeError(f"{spec.outputs.quota_gb} GB analysis output quota reached")


def run_stage_a_diagnostics(
    paths: AnalysisPaths, spec: AnalysisSpec, context: RunContext
) -> DiagnosticBundle:
    """Build and atomically publish the complete stage-A diagnostic bundle."""

    code_root = paths.code_root.resolve()
    data_root = paths.data_root.resolve()
    output_root = paths.output_root.resolve()
    if spec != load_analysis_spec(code_root / "config/analysis.yaml"):
        raise ValueError("runtime analysis spec differs from frozen code authority")
    if context.spec_id != spec.spec_id or context.seed != spec.seed:
        raise ValueError("run context does not match the frozen analysis spec")
    preflight = analysis_preflight(code_root, data_root, output_root)
    if preflight.input_authority_hash != context.input_authority_hash:
        raise ValueError("run context input authority hash mismatch")
    if preflight.evidence_policy_sha256 != context.evidence_policy_sha256:
        raise ValueError("run context evidence policy hash mismatch")
    (output_root / "run_manifest.json").unlink(missing_ok=True)

    panel_path = data_root / "05_中间数据/analysis/lp_panel.parquet"
    shares_path = (
        data_root
        / "05_中间数据/measures/instruments/iv_baseline_shares.parquet"
    )
    panel = pl.scan_parquet(panel_path)
    shares = pl.scan_parquet(shares_path)
    contracts = _analysis_contracts(code_root)
    sample_cells, missingness = _sample_and_missingness_frames(
        panel,
        spec,
        context,
        contracts["sample_cell"],
        contracts["missingness_cell"],
    )
    distributions, correlations = _distribution_and_correlation_frames(
        panel,
        spec,
        context,
        contracts["distribution"],
        contracts["correlation"],
    )
    screened = _add_provenance(
        first_stage_screen(panel, spec), context
    ).select(*contracts["diagnostic_cell"].columns)
    concentrated = _add_provenance(
        exposure_concentration(shares), context
    ).with_columns(
        pl.lit("main_hs96").alias("taxonomy_version"),
        pl.lit("main_0.0001").alias("share_version"),
    ).select(*contracts["exposure_concentration"].columns)
    descriptive = _descriptive_path_frame(
        panel, spec, context, contracts["descriptive_path"]
    )

    diagnostics_root = output_root / "diagnostics"
    destinations = (
        (sample_cells, contracts["sample_cell"], diagnostics_root / "sample_cells.parquet"),
        (missingness, contracts["missingness_cell"], diagnostics_root / "missingness.parquet"),
        (distributions, contracts["distribution"], diagnostics_root / "distributions.parquet"),
        (correlations, contracts["correlation"], diagnostics_root / "correlations.parquet"),
        (screened, contracts["diagnostic_cell"], diagnostics_root / "first_stage_screen.parquet"),
        (concentrated, contracts["exposure_concentration"], diagnostics_root / "exposure_concentration.parquet"),
        (descriptive, contracts["descriptive_path"], diagnostics_root / "descriptive_paths.parquet"),
    )
    panel_input = InputArtifact.from_path(panel_path)
    shares_input = InputArtifact.from_path(shares_path)
    table_paths: list[Path] = []
    for frame, contract, destination in destinations:
        projected = max(frame.estimated_size() * 2, 1024 * 1024)
        _enforce_analysis_quota(output_root, spec, projected_bytes=projected)
        source_input = (
            shares_input
            if contract.table_id == "exposure_concentration"
            else panel_input
        )
        contract_path = (
            code_root
            / "03_代码/contracts/analysis"
            / f"{contract.table_id}.json"
        )
        write_authoritative_table(
            frame,
            contract,
            destination,
            (source_input, InputArtifact.from_path(contract_path)),
            BuildIdentity(
                command=(
                    "python -m green_debt.cli analysis-diagnostics "
                    f"--data-root {data_root} --output-root {output_root}"
                ),
                code_commit=context.git_commit,
                created_at_utc=context.created_at_utc,
            ),
        )
        _enforce_analysis_quota(output_root, spec)
        table_paths.append(destination)

    gate_cells = screened.select(
        "analysis_family",
        "outcome_id",
        "horizon",
        "gad_version",
        "sample_version",
        "role",
        "n",
        "economies",
        "clusters",
        "instrument_rank",
        "cross_moment_rank",
        "rank",
        "condition_number",
        "partial_r2_gimc",
        "partial_r2_interaction",
        "effective_f_gimc",
        "effective_f_interaction",
        "first_stage_status",
        "gate_status",
    ).to_dicts()
    summary_path = diagnostics_root / "stage_a_summary.json"
    summary_payload: dict[str, object] = {
        **asdict(context),
        "status": "valid",
        "registered_cells": len(spec.registered_cells()),
        "confirmatory_cells": len(spec.confirmatory_cells()),
        "table_hashes": {
            path.name: sha256_file(path) for path in table_paths
        },
        "output_bytes": directory_usage_bytes(output_root),
    }
    _enforce_analysis_quota(
        output_root,
        spec,
        projected_bytes=len(json.dumps(summary_payload).encode("utf-8")),
    )
    _write_json_atomic(summary_payload, summary_path)

    gate_path = output_root / "registries/analysis_gate_v1.json"
    gate_payload: dict[str, object] = {
        **asdict(context),
        "schema_version": 1,
        "status": "frozen",
        "registered_cells": len(spec.registered_cells()),
        "cells": gate_cells,
        "stage_a_summary_sha256": sha256_file(summary_path),
    }
    _enforce_analysis_quota(
        output_root,
        spec,
        projected_bytes=len(json.dumps(gate_payload).encode("utf-8")),
    )
    _write_json_atomic(gate_payload, gate_path)
    _enforce_analysis_quota(output_root, spec)
    return DiagnosticBundle(
        table_paths=tuple(table_paths),
        gate_path=gate_path,
        cell_count=len(gate_cells),
    )
