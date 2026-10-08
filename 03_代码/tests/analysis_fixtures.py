"""Shared deterministic fixtures for analysis-layer tests."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import polars as pl
from scipy import stats

from green_debt.analysis_io import (
    RunContext,
    canonical_threshold_registry_hash,
)
from green_debt.analysis_spec import AnalysisSpec, load_analysis_spec
from green_debt.artifacts import BuildIdentity, TableContract, write_authoritative_table


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AnalysisProjectFixture:
    code_root: Path
    data_root: Path


AUTHORITY_TABLE_PATHS = {
    "model_panel": Path("05_中间数据/analysis/lp_panel.parquet"),
    "iv_baseline_shares": Path(
        "05_中间数据/measures/instruments/iv_baseline_shares.parquet"
    ),
    "iv_partner_shocks": Path(
        "05_中间数据/measures/instruments/iv_partner_shocks.parquet"
    ),
}


def load_test_contract(code_root: Path, table_id: str) -> TableContract:
    payload = json.loads(
        (code_root / f"03_代码/contracts/{table_id}.json").read_text(
            encoding="utf-8"
        )
    )
    period = payload.get("period")
    return TableContract(
        table_id=payload["table_id"],
        schema_version=payload["schema_version"],
        primary_key=tuple(payload["primary_key"]),
        columns=dict(payload["columns"]),
        units=dict(payload.get("units", {})),
        period=tuple(period) if period else None,
        zero_semantics=dict(payload.get("zero_semantics", {})),
        null_semantics=dict(payload.get("null_semantics", {})),
        transformations=tuple(payload.get("transformations", ())),
    )


def one_row_frame_for_contract(contract: TableContract) -> pl.DataFrame:
    values: dict[str, list[object]] = {}
    for name, dtype in contract.columns.items():
        if dtype == "String":
            values[name] = ["x"]
        elif dtype == "Boolean":
            values[name] = [True]
        elif dtype.startswith("Float"):
            values[name] = [1.0]
        else:
            values[name] = [1]
    for name in ("year", "treatment_time"):
        if name in values:
            values[name] = [2000]
    for name in ("gad_time", "control_time", "baseline_outcome_time"):
        if name in values:
            values[name] = [1999]
    if "outcome_time" in values:
        values["outcome_time"] = [2001]
    if "horizon" in values:
        values["horizon"] = [1]
    return pl.DataFrame(values).cast(
        {name: getattr(pl, dtype) for name, dtype in contract.columns.items()}
    ).select(*contract.columns)


def copy_analysis_contracts_and_configs(source: Path, target: Path) -> None:
    for relative in (
        "03_代码/contracts/model_panel.json",
        "03_代码/contracts/iv_baseline_shares.json",
        "03_代码/contracts/iv_partner_shocks.json",
        "config/project.yaml",
        "config/outcome_gad_map.yaml",
        "config/analysis.yaml",
        "config/evidence_policy.json",
    ):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    for source_contract in sorted(
        (source / "03_代码/contracts/analysis").glob("*.json")
    ):
        destination = target / "03_代码/contracts/analysis" / source_contract.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_contract, destination)


def build_analysis_project_fixture(tmp_path: Path) -> AnalysisProjectFixture:
    code_root = tmp_path / "code"
    data_root = tmp_path / "data"
    copy_analysis_contracts_and_configs(ROOT, code_root)
    for table_id, relative_path in AUTHORITY_TABLE_PATHS.items():
        contract = load_test_contract(code_root, table_id)
        write_authoritative_table(
            one_row_frame_for_contract(contract),
            contract,
            data_root / relative_path,
            (),
            BuildIdentity(command="test", code_commit="test"),
        )
    return AnalysisProjectFixture(code_root=code_root, data_root=data_root)


def rebind_adjacent_manifest_to_bundle(
    project: AnalysisProjectFixture, table_id: str
) -> None:
    table_path = project.data_root / AUTHORITY_TABLE_PATHS[table_id]
    schema_path = table_path.with_name(f"{table_path.name}.schema.json")
    manifest_path = table_path.with_name(f"{table_path.name}.manifest.json")
    bundle = project.data_root / "bundle" / table_id
    bundle.mkdir(parents=True)
    bundle_table = bundle / table_path.name
    bundle_schema = bundle / schema_path.name
    shutil.copy2(table_path, bundle_table)
    shutil.copy2(schema_path, bundle_schema)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["destination"] = str(bundle_table.resolve())
    payload["schema_path"] = str(bundle_schema.resolve())
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def synthetic_python_run_context(spec: AnalysisSpec) -> RunContext:
    return RunContext(
        run_id="synthetic-run",
        spec_id=spec.spec_id,
        input_authority_hash="a" * 64,
        git_commit="b" * 40,
        renv_lock_sha256="c" * 64,
        evidence_policy_sha256=hashlib.sha256(
            (ROOT / "config/evidence_policy.json").read_bytes()
        ).hexdigest(),
        seed=spec.seed,
        created_at_utc="2026-08-29T00:00:00Z",
    )


def complete_result_frames(
    spec: AnalysisSpec, context: RunContext
) -> dict[str, pl.DataFrame]:
    """Return one explicit status row for every registered continuous cell."""

    continuous = (
        *spec.confirmatory_cells(),
        *spec.robustness_cells(),
        *spec.vulnerability_cells(),
    )
    status_rows = [
        {
            "run_id": context.run_id,
            "analysis_family": cell.analysis_family,
            "outcome_id": cell.outcome_id,
            "horizon": cell.horizon,
            "gad_version": cell.gad_version,
            "sample_version": cell.sample_version,
            "status": "estimated",
        }
        for cell in continuous
    ]
    ar_cells = (*spec.confirmatory_cells(), *spec.vulnerability_cells())
    ar_keys = {
        (
            cell.analysis_family,
            cell.outcome_id,
            cell.horizon,
            cell.gad_version,
            cell.sample_version,
        )
        for cell in ar_cells
    }
    ar_rows = [
        row
        for row in status_rows
        if (
            row["analysis_family"],
            row["outcome_id"],
            row["horizon"],
            row["gad_version"],
            row["sample_version"],
        )
        in ar_keys
    ]
    threshold_rows = [
        row
        for row in status_rows
        if row["analysis_family"] == "confirmatory"
    ]
    return {
        "lp_fe_status": pl.DataFrame(status_rows),
        "lp_iv_status": pl.DataFrame(status_rows),
        "threshold_status": pl.DataFrame(threshold_rows),
        "ar_status": pl.DataFrame(ar_rows),
    }


def model_panel_fixture() -> pl.DataFrame:
    n = 40
    gad = np.linspace(0.0, 2.0, n)
    instrument = np.sin(np.arange(n))
    values: dict[str, object] = {
        "economy_id": [f"E{index // 2:03d}" for index in range(n)],
        "treatment_time": [2000 + index % 2 for index in range(n)],
        "horizon": [3] * n,
        "outcome_id": ["green_export_complexity"] * n,
        "gad_version": ["gad_no_supp"] * n,
        "sample_version": ["core_complete_case"] * n,
        "delta_outcome": 1.2 * instrument,
        "gimc_p01_p99": 0.8 * instrument,
        "gad_lag_p01_p99": gad,
        "Z_p01_p99": instrument,
        "Z_GAD_p01_p99": instrument * gad,
        "confirmatory_iv_eligible": [True] * n,
        "descriptive_only": [False] * n,
        "threshold_selection_only": [False] * n,
        "negative_shock_sample": [False] * n,
    }
    for column in (
        "renewable_energy_consumption_share_analysis_p01_p99",
        "trade_openness_percent_gdp_analysis_p01_p99",
        "industry_value_added_share_analysis_p01_p99",
        "gdp_per_capita_current_usd_analysis_p01_p99",
    ):
        values[column] = np.cos(np.arange(n))
    return pl.DataFrame(values)


def baseline_shares_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "importer": ["A", "A", "B", "B"],
            "exporter": ["X", "Y", "X", "Z"],
            "hs6": ["000001", "000002", "000001", "000003"],
            "baseline_share": [0.75, 0.25, 0.5, 0.5],
            "retained": [True] * 4,
            "taxonomy_version": ["main_hs96"] * 4,
            "share_version": ["main_0.0001"] * 4,
        }
    )


def full_registered_contract_panel(
    spec: AnalysisSpec, contract: TableContract, *, rows_per_cell: int = 40
) -> pl.DataFrame:
    template = one_row_frame_for_contract(contract).row(0, named=True)
    registered = [
        (
            cell.outcome_id,
            cell.horizon,
            cell.gad_version,
            cell.sample_version,
            cell.analysis_family,
        )
        for cell in spec.registered_cells()
    ]
    registered.extend(
        (
            outcome.outcome_id,
            0,
            outcome.gad_version,
            spec.sample_version,
            "descriptive",
        )
        for outcome in spec.outcomes
        if outcome.horizons == (1, 2, 3)
    )
    rows: list[dict[str, object]] = []
    for outcome_id, horizon, gad_version, sample_version, family in registered:
        for index in range(rows_per_cell):
            treatment_time = 2000 + index % 2
            instrument = math.sin(index)
            if family == "vulnerability":
                instrument = -abs(instrument) - 0.1
            gad = 2.0 * index / (rows_per_cell - 1)
            control = math.cos(index)
            row = dict(template)
            row.update(
                {
                    "economy_id": f"E{index // 2:03d}",
                    "treatment_time": treatment_time,
                    "horizon": horizon,
                    "outcome_id": outcome_id,
                    "gad_version": gad_version,
                    "sample_version": sample_version,
                    "gad_time": treatment_time - 1,
                    "control_time": treatment_time - 1,
                    "baseline_outcome_time": treatment_time - 1,
                    "outcome_time": treatment_time + horizon,
                    "gimc": 0.8 * instrument,
                    "gimc_p01_p99": 0.8 * instrument,
                    "gad_lag": gad,
                    "gad_lag_p01_p99": gad,
                    "Z": instrument,
                    "Z_p01_p99": instrument,
                    "Z_GAD": instrument * gad,
                    "Z_GAD_p01_p99": instrument * gad,
                    "CMZ": instrument,
                    "CMZ_p01_p99": instrument,
                    "baseline_outcome": instrument,
                    "future_outcome": 2.2 * instrument,
                    "delta_outcome": 1.2 * instrument,
                    "renewable_energy_consumption_share": control,
                    "renewable_energy_consumption_share_analysis": control,
                    "renewable_energy_consumption_share_analysis_p01_p99": control,
                    "trade_openness_percent_gdp": control,
                    "trade_openness_percent_gdp_analysis": control,
                    "trade_openness_percent_gdp_analysis_p01_p99": control,
                    "industry_value_added_share": control,
                    "industry_value_added_share_analysis": control,
                    "industry_value_added_share_analysis_p01_p99": control,
                    "gdp_per_capita_current_usd": control,
                    "gdp_per_capita_current_usd_analysis": control,
                    "gdp_per_capita_current_usd_analysis_p01_p99": control,
                    "renewable_energy_consumption_share_interpolated": False,
                    "trade_openness_percent_gdp_interpolated": False,
                    "industry_value_added_share_interpolated": False,
                    "gdp_per_capita_current_usd_interpolated": False,
                    "interpolated_control": False,
                    "outcome_coverage_eligible": True,
                    "confirmatory_iv_eligible": family
                    in {"confirmatory", "bounded_controls"},
                    "core_eligible": True,
                    "lite_eligible": True,
                    "provisional_core": True,
                    "descriptive_only": family == "descriptive",
                    "threshold_selection_only": family == "threshold_selection",
                    "negative_shock_sample": family == "vulnerability",
                    "gad_interpolated": False,
                    "outcome_interpolated": False,
                    "source_outcome_materialized": True,
                    "gad_scaler_hash": "a" * 64,
                    "regression_bounds_hash": "b" * 64,
                    "giu_scaler_hash": "c" * 64,
                }
            )
            rows.append(row)
    return pl.DataFrame(rows).cast(
        {name: getattr(pl, dtype) for name, dtype in contract.columns.items()}
    ).select(*contract.columns)


def baseline_share_contract_frame(contract: TableContract) -> pl.DataFrame:
    template = one_row_frame_for_contract(contract).row(0, named=True)
    keys = (
        ("A", "X", "000001", 0.75),
        ("A", "Y", "000002", 0.25),
        ("B", "X", "000001", 0.50),
        ("B", "Z", "000003", 0.50),
    )
    rows: list[dict[str, object]] = []
    for importer, exporter, hs6, share in keys:
        row = dict(template)
        row.update(
            {
                "taxonomy_version": "main_hs96",
                "share_version": "main_0.0001",
                "importer": importer,
                "exporter": exporter,
                "hs6": hs6,
                "mean_weighted_import_usd": share * 100.0,
                "baseline_mean_total_weighted_import_usd": 100.0,
                "raw_baseline_share": share,
                "retained": True,
                "retained_coverage": 1.0,
                "baseline_share": share,
                "raw_cell_count": 2,
                "retained_cell_count": 2,
                "coverage_eligible": True,
                "confirmatory_specification": True,
                "confirmatory_baseline_eligible": True,
            }
        )
        rows.append(row)
    return pl.DataFrame(rows).cast(
        {name: getattr(pl, dtype) for name, dtype in contract.columns.items()}
    ).select(*contract.columns)


def replace_authority_table(
    project: AnalysisProjectFixture, table_id: str, frame: pl.DataFrame
) -> None:
    contract = load_test_contract(project.code_root, table_id)
    write_authoritative_table(
        frame,
        contract,
        project.data_root / AUTHORITY_TABLE_PATHS[table_id],
        (),
        BuildIdentity(command="test replacement", code_commit="test"),
    )


def _fixture_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _continuous_cells(spec: AnalysisSpec) -> tuple[object, ...]:
    return tuple(
        cell
        for cell in spec.registered_cells()
        if cell.analysis_family != "threshold_selection"
    )


def _result_metadata(
    context: RunContext,
    clusters: int,
    *,
    inference_kind: str = "cluster_t",
) -> dict[str, object]:
    policy = json.loads(
        (ROOT / "config/evidence_policy.json").read_text(encoding="utf-8")
    )
    if inference_kind == "cluster_t":
        ssc_config = policy["conventional_inference"]["ssc_config"]
        reference_distribution = "cluster_t"
        reference_df = float(clusters - 1)
    elif inference_kind == "cr2_htz_f":
        ssc_config = "clubSandwich:CR2;Wald_test=HTZ"
        reference_distribution = "cr2_htz_f"
        reference_df = float(clusters - 1)
    else:
        ssc_config = "shock_cluster_sandwich:exporter_hs6;reference=not_used"
        reference_distribution = "not_used"
        reference_df = float(max(1, clusters - 1))
    return {
        "evidence_policy_sha256": context.evidence_policy_sha256,
        "year_min": 2000,
        "year_max": 2022,
        "r_version": policy["software"]["r_version"],
        "python_version": policy["software"]["python_version"],
        "julia_version": policy["software"]["julia_version"],
        "package_versions_json": json.dumps(
            policy["software"]["packages"], sort_keys=True, separators=(",", ":")
        ),
        "random_seed": context.seed,
        "ssc_config": ssc_config,
        "reference_distribution": reference_distribution,
        "reference_df": reference_df,
    }


def write_staging_estimates(
    root: Path,
    *,
    omit: set[str] | None = None,
    duplicate_first_row: bool = False,
    covariance_asymmetry: bool = False,
    covariance_negative: bool = False,
    n_offset: int = 0,
    bad_confidence: bool = False,
    clusters: int = 30,
    omit_wild: bool = False,
    wild_unbounded: bool = False,
) -> Path:
    """Write a complete deterministic 84-cell LP staging bundle."""

    spec = load_analysis_spec(ROOT / "config/analysis.yaml")
    context = synthetic_python_run_context(spec)
    staging = root / "_staging" / context.run_id / "lp"
    staging.mkdir(parents=True, exist_ok=True)
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    registries = root / "registries"
    registries.mkdir(parents=True, exist_ok=True)
    diagnostics = root / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    inference = (
        "wild_bootstrap_required"
        if 20 <= clusters < 30
        else "cluster_robust"
    )
    gate_status = (
        "wild_bootstrap_required"
        if inference == "wild_bootstrap_required"
        else "ready_cluster_robust"
    )
    cells = _continuous_cells(spec)
    gate_cells: list[dict[str, object]] = []
    estimate_rows: dict[str, list[dict[str, object]]] = {
        "lp_fe": [],
        "lp_iv": [],
    }
    covariance_rows: list[dict[str, object]] = []
    marginal_rows: list[dict[str, object]] = []
    provenance = asdict(context)
    provenance.pop("seed")
    metadata = _result_metadata(context, clusters)
    critical = float(stats.t.ppf(0.975, clusters - 1))
    terms = ("gimc_a", "gimc_gad_a")
    coefficient = {"gimc_a": 1.0, "gimc_gad_a": -0.2}
    covariance = {
        ("gimc_a", "gimc_a"): 0.01,
        ("gimc_a", "gimc_gad_a"): 0.002,
        ("gimc_gad_a", "gimc_a"): 0.002,
        ("gimc_gad_a", "gimc_gad_a"): 0.04,
    }
    if covariance_asymmetry:
        covariance[("gimc_gad_a", "gimc_a")] = 0.009
    if covariance_negative:
        covariance[("gimc_gad_a", "gimc_gad_a")] = -0.04

    for cell in cells:
        cell_key = {
            "analysis_family": cell.analysis_family,
            "outcome_id": cell.outcome_id,
            "horizon": cell.horizon,
            "gad_version": cell.gad_version,
            "sample_version": cell.sample_version,
        }
        gate_cells.append(
            {
                **cell_key,
                "role": cell.role,
                "n": 40,
                "economies": clusters,
                "clusters": clusters,
                "instrument_rank": 2,
                "cross_moment_rank": 2,
                "rank": 2,
                "condition_number": 2.0,
                "partial_r2_gimc": 0.2,
                "partial_r2_interaction": 0.1,
                "effective_f_gimc": 5.0,
                "effective_f_interaction": 4.0,
                "first_stage_status": "weak_reference_below_10",
                "gate_status": gate_status,
            }
        )
        for estimator in estimate_rows:
            for term in terms:
                estimate = coefficient[term]
                std_error = abs(covariance[(term, term)]) ** 0.5
                conf_low = estimate - critical * std_error
                conf_high = estimate + critical * std_error
                if bad_confidence and not estimate_rows[estimator]:
                    conf_low -= 0.5
                wild = inference == "wild_bootstrap_required" and not omit_wild
                row_inference = (
                    "wild_bootstrap_unbounded"
                    if wild and wild_unbounded
                    else inference
                )
                estimate_rows[estimator].append(
                    {
                        **provenance,
                        **metadata,
                        "estimator": estimator,
                        **cell_key,
                        "term": term,
                        "estimate": estimate,
                        "std_error": std_error,
                        "conf_low": conf_low,
                        "conf_high": conf_high,
                        "p_value": float(
                            2 * stats.t.sf(abs(estimate / std_error), clusters - 1)
                        ),
                        "wild_conf_low": estimate - 0.25
                        if wild and not wild_unbounded
                        else None,
                        "wild_conf_high": estimate + 0.25
                        if wild and not wild_unbounded
                        else None,
                        "wild_p_value": 0.04 if wild else None,
                        "wild_draws": spec.inference.wild_bootstrap_draws
                        if wild
                        else None,
                        "wild_seed": spec.seed if wild else None,
                        "n": 40 + n_offset,
                        "economies": clusters,
                        "clusters": clusters,
                        "first_stage_status": "weak_reference_below_10",
                        "inference_status": row_inference,
                    }
                )
            for term_i in terms:
                for term_j in terms:
                    covariance_rows.append(
                        {
                            **provenance,
                            **metadata,
                            "estimator": estimator,
                            **cell_key,
                            "clusters": clusters,
                            "term_i": term_i,
                            "term_j": term_j,
                            "covariance": covariance[(term_i, term_j)],
                        }
                    )
            for quantile in spec.inference.marginal_gad_quantiles:
                gad_value = quantile
                estimate = coefficient["gimc_a"] + (
                    gad_value * coefficient["gimc_gad_a"]
                )
                variance = (
                    covariance[("gimc_a", "gimc_a")]
                    + gad_value**2
                    * covariance[("gimc_gad_a", "gimc_gad_a")]
                    + 2
                    * gad_value
                    * covariance[("gimc_a", "gimc_gad_a")]
                )
                std_error = math.sqrt(abs(variance))
                wild = inference == "wild_bootstrap_required" and not omit_wild
                row_inference = (
                    "wild_bootstrap_unbounded"
                    if wild and wild_unbounded
                    else inference
                )
                marginal_rows.append(
                    {
                        **provenance,
                        **metadata,
                        "estimator": estimator,
                        **cell_key,
                        "gad_quantile": quantile,
                        "gad_value": gad_value,
                        "estimate": estimate,
                        "std_error": std_error,
                        "conf_low": estimate - critical * std_error,
                        "conf_high": estimate + critical * std_error,
                        "p_value": float(
                            2 * stats.t.sf(abs(estimate / std_error), clusters - 1)
                        ),
                        "wild_conf_low": estimate - 0.25
                        if wild and not wild_unbounded
                        else None,
                        "wild_conf_high": estimate + 0.25
                        if wild and not wild_unbounded
                        else None,
                        "wild_p_value": 0.04 if wild else None,
                        "wild_draws": spec.inference.wild_bootstrap_draws
                        if wild
                        else None,
                        "wild_seed": spec.seed if wild else None,
                        "n": 40 + n_offset,
                        "economies": clusters,
                        "clusters": clusters,
                        "first_stage_status": "weak_reference_below_10",
                        "inference_status": row_inference,
                    }
                )

    if duplicate_first_row:
        estimate_rows["lp_fe"].append(dict(estimate_rows["lp_fe"][0]))
    omitted = omit or set()
    payloads = {
        "lp_fe": pl.DataFrame(estimate_rows["lp_fe"]),
        "lp_iv": pl.DataFrame(estimate_rows["lp_iv"]),
        "model_covariance": pl.DataFrame(covariance_rows),
        "marginal_effects": pl.DataFrame(marginal_rows),
    }
    files: dict[str, dict[str, object]] = {}
    for name, frame in payloads.items():
        for field in omitted:
            if field in frame.columns:
                frame = frame.drop(field)
        path = staging / f"{name}.csv"
        frame.write_csv(path)
        files[name] = {
            "name": path.name,
            "rows": frame.height,
            "sha256": _fixture_sha256(path),
        }

    gate = {
        **provenance,
        "seed": context.seed,
        "status": "frozen",
        "cells": gate_cells,
    }
    gate_path = registries / "analysis_gate_v1.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (diagnostics / "stage_a_summary.json").write_text(
        json.dumps(
            {**asdict(context), "status": "valid"},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = {
        **asdict(context),
        "kind": "lp",
        "gate_sha256": _fixture_sha256(gate_path),
        "files": files,
        "skipped_cells": [],
    }
    (staging / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return staging


def write_threshold_audit_staging(
    root: Path,
    *,
    threshold_q_offset: float = 0.0,
    partial_ar_bounds: bool = False,
    unavailable_shock_covariance: bool = False,
) -> Path:
    """Write a complete deterministic 39/45-cell threshold/audit bundle."""

    write_staging_estimates(root)
    spec = load_analysis_spec(ROOT / "config/analysis.yaml")
    context = synthetic_python_run_context(spec)
    provenance = asdict(context)
    provenance.pop("seed")
    threshold_metadata = _result_metadata(context, 30)
    weak_metadata = _result_metadata(context, 30, inference_kind="cr2_htz_f")
    shift_metadata = _result_metadata(context, 2, inference_kind="not_used")
    gate_path = root / "registries/analysis_gate_v1.json"
    registry: dict[str, object] = {
        "registry_id": "threshold_registry_v1",
        "selection_outcome": spec.threshold.outcome_id,
        "horizon": spec.threshold.horizon,
        "gad_version": spec.threshold.gad_version,
        "sample_version": spec.threshold.sample_version,
        "criterion": spec.threshold.criterion,
        "quantile_type": spec.threshold.quantile_type,
        "tie_break": spec.threshold.tie_break,
        "seed": spec.seed,
        "sample_hash": "d" * 64,
        "input_authority_hash": context.input_authority_hash,
        "q": 0.42,
        "percentile": 50,
        "low_share": 0.5,
        "high_share": 0.5,
        "bootstrap_draws": 999,
        "bootstrap_valid_draws": 999,
        "bootstrap_failed_draws": 0,
        "bootstrap_status": "available",
        "percentile_conf_low": 45.0,
        "percentile_conf_high": 55.0,
        "q_conf_low": 0.3,
        "q_conf_high": 0.54,
        "candidates": [
            {
                "percentile": 50,
                "q": 0.42,
                "low_share": 0.5,
                "high_share": 0.5,
                "eligible": True,
                "ssr": 1.0,
                "n": 40,
                "status": "estimated",
            }
        ],
        "created_at_utc": context.created_at_utc,
    }
    registry["registry_hash"] = canonical_threshold_registry_hash(registry)
    registry_path = root / "registries/threshold_registry_v1.json"
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    confirmatory = spec.confirmatory_cells()
    audit_cells = (*confirmatory, *spec.vulnerability_cells())
    threshold_rows: list[dict[str, object]] = []
    weak_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for cell in confirmatory:
        difference_estimate = -1.2
        low_high_covariance = 0.0
        difference_std_error = math.sqrt(0.02)
        critical = float(stats.t.ppf(0.975, 29))
        for regime, estimate in (("low", 1.0), ("high", -0.2)):
            standard_error = 0.1
            threshold_rows.append(
                {
                    **provenance,
                    **threshold_metadata,
                    "estimator": "threshold_iv",
                    "analysis_family": cell.analysis_family,
                    "selection_outcome": registry["selection_outcome"],
                    "registry_hash": registry["registry_hash"],
                    "registry_sample_hash": registry["sample_hash"],
                    "q": float(registry["q"]) + threshold_q_offset,
                    "outcome_id": cell.outcome_id,
                    "horizon": cell.horizon,
                    "gad_version": cell.gad_version,
                    "sample_version": cell.sample_version,
                    "regime": regime,
                    "estimate": estimate,
                    "std_error": standard_error,
                    "conf_low": estimate - critical * standard_error,
                    "conf_high": estimate + critical * standard_error,
                    "p_value": float(2 * stats.t.sf(abs(estimate / standard_error), 29)),
                    "wild_estimate": None,
                    "wild_std_error": None,
                    "wild_conf_low": None,
                    "wild_conf_high": None,
                    "wild_p_value": None,
                    "wild_inference_status": "not_required",
                    "wild_draws": None,
                    "wild_seed": None,
                    "low_high_covariance": low_high_covariance,
                    "difference_estimate": difference_estimate,
                    "difference_std_error": difference_std_error,
                    "difference_conf_low": difference_estimate
                    - critical * difference_std_error,
                    "difference_conf_high": difference_estimate
                    + critical * difference_std_error,
                    "difference_p_value": float(
                        2 * stats.t.sf(abs(difference_estimate / difference_std_error), 29)
                    ),
                    "difference_wild_estimate": None,
                    "difference_wild_std_error": None,
                    "difference_wild_conf_low": None,
                    "difference_wild_conf_high": None,
                    "difference_wild_p_value": None,
                    "difference_wild_inference_status": "not_required",
                    "difference_wild_draws": None,
                    "difference_wild_seed": None,
                    "regime_n": 20,
                    "regime_share": 0.5,
                    "n": 40,
                    "economies": 30,
                    "clusters": 30,
                    "first_stage_status": "weak_reference_below_10",
                    "inference_status": "cluster_robust",
                }
            )
    for cell in audit_cells:
        identity = {
            "analysis_family": cell.analysis_family,
            "outcome_id": cell.outcome_id,
            "horizon": cell.horizon,
            "gad_version": cell.gad_version,
            "sample_version": cell.sample_version,
        }
        weak_rows.append(
            {
                **provenance,
                **weak_metadata,
                "estimator": "lp_iv_ar",
                **identity,
                "beta_low": None,
                "beta_high": -0.1 if partial_ar_bounds else None,
                "theta_low": -0.4 if partial_ar_bounds else None,
                "theta_high": 0.2 if partial_ar_bounds else None,
                "status": "unbounded",
                "expansions": 4,
                "accepted_points": 10,
                "accepted_hash": "e" * 64,
                "conventional_point_accepted": True,
                "n": 40,
                "economies": 30,
                "clusters": 30,
                "inference_status": "available",
            }
        )
        summary_rows.append(
            {
                **provenance,
                **shift_metadata,
                "estimator": "lp_iv_shift_share",
                **identity,
                "n": 40,
                "economies": 30,
                "shock_observations": (
                    1 if unavailable_shock_covariance else 2
                ),
                "shock_clusters": 1 if unavailable_shock_covariance else 2,
                "signed_weight_sum": 1.0,
                "absolute_weight_sum": 1.0,
                "hhi_absolute": 1.0,
                "top1_absolute_share": 1.0,
                "top5_absolute_share": 1.0,
                "negative_weight_share": 0.0,
                "z_reconstruction_error": 0.0,
                "z_gad_reconstruction_error": 0.0,
                "shock_std_error_gimc": (
                    None if unavailable_shock_covariance else 0.2
                ),
                "shock_std_error_interaction": (
                    None if unavailable_shock_covariance else 0.3
                ),
                "cross_moment_rank": 2,
                "shock_inference_status": (
                    "unavailable" if unavailable_shock_covariance else "available"
                ),
            }
        )
        weight_rows.append(
            {
                **provenance,
                **shift_metadata,
                **identity,
                "shock_id": "X|000001|2001",
                "exporter": "X",
                "hs6": "000001",
                "year": 2001,
                "shock_cluster_id": "X|000001",
                "signed_weight": 1.0,
                "absolute_weight": 1.0,
                "absolute_rank": 1,
            }
        )
    staging = root / "_staging" / context.run_id / "threshold-and-iv-audit"
    staging.mkdir(parents=True, exist_ok=True)
    payloads = {
        "threshold_estimates": pl.DataFrame(threshold_rows),
        "weak_iv_sets": pl.DataFrame(weak_rows),
        "shift_share_summary": pl.DataFrame(summary_rows),
        "shift_share_weights": pl.DataFrame(weight_rows),
    }
    files: dict[str, dict[str, object]] = {}
    for name, frame in payloads.items():
        path = staging / f"{name}.csv"
        frame.write_csv(path)
        files[name] = {
            "name": path.name,
            "rows": frame.height,
            "sha256": _fixture_sha256(path),
        }
    receipt = {
        **asdict(context),
        "kind": "threshold-and-iv-audit",
        "gate_sha256": _fixture_sha256(gate_path),
        "completed_stages": ["shift-share", "threshold", "weak-iv"],
        "files": files,
        "registry_sha256": _fixture_sha256(registry_path),
        "registry_hash": registry["registry_hash"],
        "registry_sample_hash": registry["sample_hash"],
        "registry_q": registry["q"],
        "selection_outcome": registry["selection_outcome"],
    }
    (staging / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return staging
