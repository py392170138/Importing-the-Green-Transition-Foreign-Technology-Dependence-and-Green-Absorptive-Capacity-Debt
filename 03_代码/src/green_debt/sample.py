"""Outcome-free provisional sample construction and sample-flow evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import polars as pl

from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    verify_manifest,
    write_authoritative_table,
)
from green_debt.gad import FROZEN_SCALER_HASH
from green_debt.outcomes import (
    GIUComponentScaler,
    GIUHorizonScaler,
    GIUScalerRegistry,
    OutcomeSpec,
    build_rca_entry_rate,
    choose_gad_variant,
    fit_giu_scalers,
    outcome_specs,
)
from green_debt.paths import ProjectPaths


OUTPUT_SCHEMA = {
    "economy_id": pl.String,
    "sample_version": pl.String,
    "economy_rule_eligible": pl.Boolean,
    "population_2000": pl.Float64,
    "population_at_least_1m": pl.Boolean,
    "wdi_available": pl.Boolean,
    "baci_available": pl.Boolean,
    "openalex_available": pl.Boolean,
    "required_sources_available": pl.Boolean,
    "tiva_member": pl.Boolean,
    "positive_green_import_baseline_years": pl.UInt8,
    "positive_green_import_baseline_eligible": pl.Boolean,
    "provisional_core": pl.Boolean,
    "exclusion_reason": pl.String,
}
FLOW_STAGES = (
    ("economy_rule_eligible", "economy_rule_eligible"),
    ("population_at_least_1m", "population_at_least_1m"),
    ("required_sources_available", "required_sources_available"),
    ("tiva_member", "tiva_member"),
    (
        "positive_green_import_baseline_eligible",
        "positive_green_import_baseline_eligible",
    ),
)

MAIN_CONTROL_COLUMNS = (
    "renewable_energy_consumption_share",
    "trade_openness_percent_gdp",
    "industry_value_added_share",
    "gdp_per_capita_current_usd",
)
MAPPED_GAD_VARIANTS = (
    "gad_core",
    "gad_no_gfvad",
    "gad_no_gsci",
    "gad_no_supp",
)


@dataclass(frozen=True)
class ProvisionalSampleBuildReport:
    universe_economies: int
    economy_rule_eligible: int
    population_eligible: int
    required_sources_available: int
    tiva_members: int
    positive_baseline_eligible: int
    provisional_core: int
    duplicate_keys: int
    outcome_tables_read: int
    sample_flow_identity_failures: int
    output_path: str
    output_bytes: int
    sample_flow_path: str


@dataclass(frozen=True)
class ProvisionalSampleAuditReport:
    rows: int
    provisional_core: int
    duplicate_keys: int
    rule_mismatches: int
    outcome_columns: int
    outcome_lineage_inputs: int
    sample_flow_identity_failures: int
    status: str


@dataclass(frozen=True)
class RegressionBound:
    lower: float
    upper: float
    nonnull_fit_rows: int


@dataclass(frozen=True)
class RegressionBoundsRegistry:
    bounds: dict[str, RegressionBound]
    percentiles: tuple[int, int]
    fit_years: tuple[int, int]
    fit_row_count: int
    canonical_hash: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "registry_id": "regression_bounds",
            "registry_version": "1.0.0",
            "percentiles": list(self.percentiles),
            "fit_years": list(self.fit_years),
            "fit_row_count": self.fit_row_count,
            "bounds": {
                name: {
                    "lower": bound.lower,
                    "upper": bound.upper,
                    "nonnull_fit_rows": bound.nonnull_fit_rows,
                }
                for name, bound in sorted(self.bounds.items())
            },
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.canonical_payload()
        payload["canonical_hash"] = self.canonical_hash
        return payload


@dataclass(frozen=True)
class FinalSampleBuildReport:
    rows: int
    core_economies: int
    lite_economies: int
    taiwan_robustness_economies: int
    micro_robustness_economies: int
    output_path: str
    output_sha256: str
    sample_flow_path: str


@dataclass(frozen=True)
class AnalysisPanelBuildReport:
    rows: int
    complete_case_rows: int
    bounded_control_rows: int
    negative_shock_rows: int
    descriptive_rows: int
    outcomes: int
    economies: int
    regression_bounds_hash: str
    giu_scaler_hash: str
    gad_scaler_hash: str
    output_path: str
    output_sha256: str
    regression_bounds_path: str
    giu_scalers_path: str


def _bool_value(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"provisional sample {field} must be Boolean")


def build_provisional_sample(
    coverage: pl.DataFrame,
    *,
    minimum_population: int = 1_000_000,
    minimum_positive_baseline_years: int = 2,
) -> pl.DataFrame:
    """Apply ordered source/rule gates without reading or referencing outcomes."""

    required = {
        "economy_id",
        "population_2000",
        "tiva_member",
        "positive_green_import_baseline_years",
        "economy_rule_eligible",
    }
    missing = sorted(required - set(coverage.columns))
    if missing:
        raise ValueError(f"provisional coverage lacks columns: {missing}")
    if any("outcome" in name.lower() for name in coverage.columns):
        raise ValueError("provisional sample may not consume outcome columns")
    if minimum_population <= 0 or not 1 <= minimum_positive_baseline_years <= 4:
        raise ValueError("invalid provisional sample thresholds")
    if coverage.group_by("economy_id").len().filter(pl.col("len") > 1).height:
        raise ValueError("provisional coverage has duplicate economies")

    rows: list[dict[str, Any]] = []
    for raw in coverage.sort("economy_id").iter_rows(named=True):
        economy_id = str(raw.get("economy_id") or "").strip()
        if not economy_id:
            raise ValueError("provisional coverage has an empty economy_id")
        population_raw = raw.get("population_2000")
        if population_raw is None:
            population = None
        elif isinstance(population_raw, bool):
            raise ValueError("population_2000 must be numeric")
        else:
            population = float(population_raw)
            if not math.isfinite(population) or population < 0.0:
                raise ValueError("population_2000 must be finite and nonnegative")
        positive_raw = raw.get("positive_green_import_baseline_years")
        if isinstance(positive_raw, bool):
            raise ValueError("positive baseline years must be an integer")
        try:
            positive_years = int(positive_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("positive baseline years must be an integer") from exc
        if positive_years < 0 or positive_years > 4:
            raise ValueError("positive baseline years must be between zero and four")
        economy_rule = _bool_value(
            raw.get("economy_rule_eligible"), field="economy_rule_eligible"
        )
        population_rule = population is not None and population >= minimum_population
        wdi = _bool_value(raw.get("wdi_available", True), field="wdi_available")
        baci = _bool_value(raw.get("baci_available", True), field="baci_available")
        openalex = _bool_value(
            raw.get("openalex_available", True), field="openalex_available"
        )
        required_sources = wdi and baci and openalex
        tiva = _bool_value(raw.get("tiva_member"), field="tiva_member")
        positive_rule = positive_years >= minimum_positive_baseline_years
        provisional = (
            economy_rule
            and population_rule
            and required_sources
            and tiva
            and positive_rule
        )
        if not economy_rule:
            reason = "economy_rule_ineligible"
        elif not population_rule:
            reason = "population_below_1m_or_missing"
        elif not required_sources:
            reason = "required_source_unavailable"
        elif not tiva:
            reason = "not_tiva_member"
        elif not positive_rule:
            reason = "fewer_than_two_positive_baseline_import_years"
        else:
            reason = None
        rows.append(
            {
                "economy_id": economy_id,
                "sample_version": str(raw.get("sample_version") or "confirmatory"),
                "economy_rule_eligible": economy_rule,
                "population_2000": population,
                "population_at_least_1m": population_rule,
                "wdi_available": wdi,
                "baci_available": baci,
                "openalex_available": openalex,
                "required_sources_available": required_sources,
                "tiva_member": tiva,
                "positive_green_import_baseline_years": positive_years,
                "positive_green_import_baseline_eligible": positive_rule,
                "provisional_core": provisional,
                "exclusion_reason": reason,
            }
        )
    return pl.DataFrame(rows, schema=OUTPUT_SCHEMA).sort("economy_id")


def build_source_coverage(
    *,
    crosswalk: pl.DataFrame,
    wdi: pl.DataFrame,
    baci_totals: pl.DataFrame,
    green_imports: pl.DataFrame,
    openalex: pl.DataFrame,
    tiva: pl.DataFrame,
) -> pl.DataFrame:
    """Assemble only source availability and pre-outcome eligibility evidence."""

    needed_crosswalk = {
        "economy_id",
        "confirmatory_eligible",
        "sample_version",
    }
    if not needed_crosswalk <= set(crosswalk.columns):
        raise ValueError("economy crosswalk lacks provisional-sample fields")
    canonical = crosswalk.filter(pl.col("economy_id").is_not_null())
    consistency = canonical.group_by("economy_id").agg(
        pl.col("confirmatory_eligible").n_unique().alias("eligibility_versions"),
        pl.col("sample_version").n_unique().alias("sample_versions"),
    )
    if consistency.filter(
        (pl.col("eligibility_versions") > 1) | (pl.col("sample_versions") > 1)
    ).height:
        raise ValueError("economy rule flags differ across source mappings")
    universe = canonical.select(
        "economy_id",
        pl.col("confirmatory_eligible").alias("economy_rule_eligible"),
        "sample_version",
    ).unique("economy_id", keep="first")

    population = wdi.filter(
        (pl.col("indicator_id") == "SP.POP.TOTL") & (pl.col("year") == 2000)
    ).select(
        "economy_id",
        pl.col("value").alias("population_2000"),
        pl.lit(True).alias("wdi_available"),
    )
    if population.group_by("economy_id").len().filter(pl.col("len") > 1).height:
        raise ValueError("WDI population has duplicate economy-year rows")

    baseline_totals = baci_totals.filter(pl.col("year").is_between(1996, 1999))
    baci = baseline_totals.group_by("economy_id").agg(
        (
            (pl.col("import_source_rows") > 0)
            | pl.col("reported_as_importer")
        ).any().alias("baci_available")
    )
    positive = (
        green_imports.filter(
            pl.col("year").is_between(1996, 1999)
            & (pl.col("flow_role") == "importer")
        )
        .group_by("economy_id", "year")
        .agg(pl.col("weighted_green_trade_usd").sum().alias("green_imports"))
        .filter(pl.col("green_imports") > 0.0)
        .group_by("economy_id")
        .agg(
            pl.col("year")
            .n_unique()
            .cast(pl.UInt8)
            .alias("positive_green_import_baseline_years")
        )
    )
    openalex_coverage = (
        openalex.filter(
            pl.col("year").is_between(1996, 2022) & (pl.col("total_works") > 0)
        )
        .select("economy_id")
        .unique()
        .with_columns(pl.lit(True).alias("openalex_available"))
    )
    tiva_members = (
        tiva.filter(pl.col("indicator_id") == "prod_level")
        .select("economy_id")
        .unique()
        .with_columns(pl.lit(True).alias("tiva_member"))
    )
    return (
        universe.join(population, on="economy_id", how="left")
        .join(baci, on="economy_id", how="left")
        .join(positive, on="economy_id", how="left")
        .join(openalex_coverage, on="economy_id", how="left")
        .join(tiva_members, on="economy_id", how="left")
        .with_columns(
            pl.col("wdi_available").fill_null(False),
            pl.col("baci_available").fill_null(False),
            pl.col("openalex_available").fill_null(False),
            pl.col("tiva_member").fill_null(False),
            pl.col("positive_green_import_baseline_years")
            .fill_null(0)
            .cast(pl.UInt8),
        )
        .sort("economy_id")
    )


def sample_flow(frame: pl.DataFrame) -> pl.DataFrame:
    active = pl.Series("active", [True] * frame.height, dtype=pl.Boolean)
    rows: list[dict[str, Any]] = [
        {
            "stage_order": 0,
            "stage": "starting_universe",
            "entered": frame.height,
            "retained": frame.height,
            "excluded_at_stage": 0,
            "identity_valid": True,
        }
    ]
    previous = frame.height
    for order, (stage, column) in enumerate(FLOW_STAGES, start=1):
        active = active & frame.get_column(column)
        retained = int(active.sum())
        excluded = previous - retained
        rows.append(
            {
                "stage_order": order,
                "stage": stage,
                "entered": previous,
                "retained": retained,
                "excluded_at_stage": excluded,
                "identity_valid": previous == retained + excluded,
            }
        )
        previous = retained
    return pl.DataFrame(
        rows,
        schema={
            "stage_order": pl.UInt8,
            "stage": pl.String,
            "entered": pl.UInt32,
            "retained": pl.UInt32,
            "excluded_at_stage": pl.UInt32,
            "identity_valid": pl.Boolean,
        },
    )


def _write_csv_atomic(frame: pl.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        frame.write_csv(partial)
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _load_contract(path: Path) -> TableContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    period = payload.get("period")
    return TableContract(
        table_id=str(payload["table_id"]),
        schema_version=str(payload["schema_version"]),
        primary_key=tuple(payload["primary_key"]),
        columns={str(key): str(value) for key, value in payload["columns"].items()},
        units={str(key): str(value) for key, value in payload["units"].items()},
        period=(int(period[0]), int(period[1])) if period else None,
        zero_semantics={
            str(key): str(value)
            for key, value in payload.get("zero_semantics", {}).items()
        },
        null_semantics={
            str(key): str(value)
            for key, value in payload.get("null_semantics", {}).items()
        },
        transformations=tuple(payload.get("transformations", [])),
    )


def build_provisional_sample_table(
    *,
    coverage: pl.DataFrame,
    destination: Path,
    contract_path: Path,
    sample_flow_path: Path,
    inputs: tuple[InputArtifact, ...],
    build: BuildIdentity,
    minimum_population: int = 1_000_000,
    minimum_positive_baseline_years: int = 2,
) -> ProvisionalSampleBuildReport:
    output = build_provisional_sample(
        coverage,
        minimum_population=minimum_population,
        minimum_positive_baseline_years=minimum_positive_baseline_years,
    )
    manifest = write_authoritative_table(
        output, _load_contract(contract_path), destination, inputs, build
    )
    flow = sample_flow(output)
    _write_csv_atomic(flow, sample_flow_path)
    return ProvisionalSampleBuildReport(
        universe_economies=output.height,
        economy_rule_eligible=output.filter(pl.col("economy_rule_eligible")).height,
        population_eligible=output.filter(pl.col("population_at_least_1m")).height,
        required_sources_available=output.filter(
            pl.col("required_sources_available")
        ).height,
        tiva_members=output.filter(pl.col("tiva_member")).height,
        positive_baseline_eligible=output.filter(
            pl.col("positive_green_import_baseline_eligible")
        ).height,
        provisional_core=output.filter(pl.col("provisional_core")).height,
        duplicate_keys=0,
        outcome_tables_read=0,
        sample_flow_identity_failures=flow.filter(~pl.col("identity_valid")).height,
        output_path=str(destination.resolve()),
        output_bytes=manifest.bytes,
        sample_flow_path=str(sample_flow_path.resolve()),
    )


def audit_provisional_sample(
    *, manifest_path: Path, sample_flow_path: Path
) -> ProvisionalSampleAuditReport:
    manifest = verify_manifest(manifest_path)
    frame = pl.read_parquet(manifest.destination)
    duplicates = frame.group_by("economy_id").len().filter(pl.col("len") > 1).height
    recomputed = build_provisional_sample(
        frame.select(
            "economy_id",
            "sample_version",
            "population_2000",
            "tiva_member",
            "positive_green_import_baseline_years",
            "economy_rule_eligible",
            "wdi_available",
            "baci_available",
            "openalex_available",
        )
    )
    rule_columns = [
        name
        for name in OUTPUT_SCHEMA
        if name not in {"economy_id", "sample_version", "population_2000"}
    ]
    comparison = frame.select("economy_id", *rule_columns).join(
        recomputed.select(
            "economy_id", *(pl.col(name).alias(f"expected__{name}") for name in rule_columns)
        ),
        on="economy_id",
    )
    mismatch_expression = pl.any_horizontal(
        *(
            pl.col(name).fill_null("__NULL__")
            != pl.col(f"expected__{name}").fill_null("__NULL__")
            if OUTPUT_SCHEMA[name] == pl.String
            else pl.col(name) != pl.col(f"expected__{name}")
            for name in rule_columns
        )
    )
    rule_mismatches = comparison.filter(mismatch_expression).height
    outcome_columns = sum("outcome" in name.lower() for name in frame.columns)
    outcome_lineage = sum(
        "outcome" in artifact.path.lower() for artifact in manifest.input_artifacts
    )
    flow = pl.read_csv(sample_flow_path)
    flow_failures = flow.filter(
        ~pl.col("identity_valid")
        | (pl.col("entered") != pl.col("retained") + pl.col("excluded_at_stage"))
    ).height
    if duplicates or rule_mismatches or outcome_columns or outcome_lineage or flow_failures:
        raise RuntimeError(
            "provisional sample audit failed: "
            f"duplicates={duplicates}, rules={rule_mismatches}, "
            f"outcome_columns={outcome_columns}, outcome_lineage={outcome_lineage}, "
            f"flow={flow_failures}"
        )
    return ProvisionalSampleAuditReport(
        rows=frame.height,
        provisional_core=frame.filter(pl.col("provisional_core")).height,
        duplicate_keys=duplicates,
        rule_mismatches=rule_mismatches,
        outcome_columns=outcome_columns,
        outcome_lineage_inputs=outcome_lineage,
        sample_flow_identity_failures=flow_failures,
        status="valid",
    )


def freeze_final_sample(
    coverage: pl.DataFrame,
    *,
    minimum_valid_main_years: int = 12,
    minimum_positive_baseline_years: int = 2,
) -> pl.DataFrame:
    """Freeze outcome-independent Core/Lite structural economy eligibility."""

    required = {
        "economy_id",
        "complete_absorption_1996",
        "complete_initialization_1997_1999",
        "valid_main_years",
        "positive_import_baseline_years",
        "baseline_share_identifiable",
        "economy_rule_eligible",
    }
    missing = sorted(required - set(coverage.columns))
    if missing:
        raise ValueError(f"final sample coverage lacks columns: {missing}")
    if coverage.group_by("economy_id").len().filter(pl.col("len") > 1).height:
        raise ValueError("final sample coverage has duplicate economy ids")
    if minimum_valid_main_years <= 0 or minimum_positive_baseline_years <= 0:
        raise ValueError("final sample minima must be positive")
    rows: list[dict[str, object]] = []
    for source in coverage.sort("economy_id").iter_rows(named=True):
        economy = str(source["economy_id"])
        economy_rule = _bool_value(
            source["economy_rule_eligible"], field="economy_rule_eligible"
        )
        absorption = _bool_value(
            source["complete_absorption_1996"], field="complete_absorption_1996"
        )
        initialization = _bool_value(
            source["complete_initialization_1997_1999"],
            field="complete_initialization_1997_1999",
        )
        try:
            valid_main_years = int(source["valid_main_years"])
            positive_years = int(source["positive_import_baseline_years"])
        except (TypeError, ValueError) as exc:
            raise ValueError("final sample year counts must be integers") from exc
        if valid_main_years < 0 or positive_years < 0:
            raise ValueError("final sample year counts cannot be negative")
        baseline_share = _bool_value(
            source["baseline_share_identifiable"],
            field="baseline_share_identifiable",
        )
        positive_rule = positive_years >= minimum_positive_baseline_years
        main_year_rule = valid_main_years >= minimum_valid_main_years
        core = (
            economy_rule
            and positive_rule
            and baseline_share
            and absorption
            and initialization
            and main_year_rule
        )
        lite_absorption = _bool_value(
            source.get("complete_lite_absorption_1996", absorption),
            field="complete_lite_absorption_1996",
        )
        lite_initialization = _bool_value(
            source.get(
                "complete_lite_initialization_1997_1999", initialization
            ),
            field="complete_lite_initialization_1997_1999",
        )
        lite_years = int(source.get("valid_lite_years", valid_main_years))
        if lite_years < 0:
            raise ValueError("valid_lite_years cannot be negative")
        lite_year_rule = lite_years >= minimum_valid_main_years
        lite = (
            economy_rule
            and positive_rule
            and baseline_share
            and lite_absorption
            and lite_initialization
            and lite_year_rule
        )
        if not economy_rule:
            reason = "economy_rule_ineligible"
        elif not positive_rule:
            reason = "fewer_than_two_positive_baseline_import_years"
        elif not baseline_share:
            reason = "baseline_share_unidentifiable"
        elif not absorption:
            reason = "incomplete_absorption_1996"
        elif not initialization:
            reason = "incomplete_gad_initialization"
        elif not main_year_rule:
            reason = "fewer_than_twelve_valid_main_years"
        else:
            reason = None
        source_sample_version = str(source.get("source_sample_version", "confirmatory"))
        robustness_structural = (
            positive_rule
            and baseline_share
            and lite_absorption
            and lite_initialization
            and lite_year_rule
        )
        rows.append(
            {
                "economy_id": economy,
                "source_sample_version": source_sample_version,
                "rule_economy_eligible": economy_rule,
                "rule_positive_import_baseline": positive_rule,
                "rule_baseline_share_identifiable": baseline_share,
                "rule_complete_absorption_1996": absorption,
                "rule_complete_initialization_1997_1999": initialization,
                "rule_minimum_valid_main_years": main_year_rule,
                "rule_complete_lite_absorption_1996": lite_absorption,
                "rule_complete_lite_initialization_1997_1999": lite_initialization,
                "rule_minimum_valid_lite_years": lite_year_rule,
                "valid_main_years": valid_main_years,
                "valid_lite_years": lite_years,
                "positive_import_baseline_years": positive_years,
                "core_eligible": core,
                "lite_eligible": lite,
                "descriptive_only_2023": lite,
                "descriptive_only_2024": lite,
                "taiwan_robustness_eligible": (
                    source_sample_version == "taiwan_robustness"
                    and robustness_structural
                ),
                "micro_robustness_eligible": (
                    source_sample_version == "micro_robustness"
                    and robustness_structural
                ),
                "exclusion_reason": reason,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("valid_main_years").cast(pl.Int16),
        pl.col("valid_lite_years").cast(pl.Int16),
        pl.col("positive_import_baseline_years").cast(pl.UInt8),
    )


def validate_final_sample(frame: pl.DataFrame) -> dict[str, int]:
    """Recompute every frozen structural flag and its ordered exclusion reason."""

    required = {
        "economy_id",
        "rule_economy_eligible",
        "rule_positive_import_baseline",
        "rule_baseline_share_identifiable",
        "rule_complete_absorption_1996",
        "rule_complete_initialization_1997_1999",
        "rule_minimum_valid_main_years",
        "rule_complete_lite_absorption_1996",
        "rule_complete_lite_initialization_1997_1999",
        "rule_minimum_valid_lite_years",
        "core_eligible",
        "lite_eligible",
        "descriptive_only_2023",
        "descriptive_only_2024",
        "exclusion_reason",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"final sample lacks validation columns: {missing}")
    duplicates = frame.group_by("economy_id").len().filter(pl.col("len") > 1).height
    if duplicates:
        raise ValueError(f"final sample duplicate economy ids: {duplicates}")
    core_rules = (
        "rule_economy_eligible",
        "rule_positive_import_baseline",
        "rule_baseline_share_identifiable",
        "rule_complete_absorption_1996",
        "rule_complete_initialization_1997_1999",
        "rule_minimum_valid_main_years",
    )
    lite_rules = (
        "rule_economy_eligible",
        "rule_positive_import_baseline",
        "rule_baseline_share_identifiable",
        "rule_complete_lite_absorption_1996",
        "rule_complete_lite_initialization_1997_1999",
        "rule_minimum_valid_lite_years",
    )
    reason_order = (
        ("rule_economy_eligible", "economy_rule_ineligible"),
        ("rule_positive_import_baseline", "fewer_than_two_positive_baseline_import_years"),
        ("rule_baseline_share_identifiable", "baseline_share_unidentifiable"),
        ("rule_complete_absorption_1996", "incomplete_absorption_1996"),
        ("rule_complete_initialization_1997_1999", "incomplete_gad_initialization"),
        ("rule_minimum_valid_main_years", "fewer_than_twelve_valid_main_years"),
    )
    failures = 0
    for row in frame.iter_rows(named=True):
        expected_core = all(bool(row[name]) for name in core_rules)
        expected_lite = all(bool(row[name]) for name in lite_rules)
        expected_reason = next(
            (reason for name, reason in reason_order if not bool(row[name])), None
        )
        failures += int(bool(row["core_eligible"]) != expected_core)
        failures += int(bool(row["lite_eligible"]) != expected_lite)
        failures += int(bool(row["descriptive_only_2023"]) != expected_lite)
        failures += int(bool(row["descriptive_only_2024"]) != expected_lite)
        failures += int(row["exclusion_reason"] != expected_reason)
    if failures:
        raise ValueError(
            f"final sample exclusion_reason or eligibility conditional violations: {failures}"
        )
    return {"rows": frame.height, "conditional_rule_violations": 0}


def interpolate_single_year_controls(
    frame: pl.DataFrame,
    controls: tuple[str, ...],
) -> pl.DataFrame:
    """Append bounded one-year interpolation copies for declared controls only."""

    controls = tuple(controls)
    if not controls or len(controls) != len(set(controls)):
        raise ValueError("control columns must be non-empty and unique")
    required = {"economy_id", "year", *controls}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"control frame lacks columns: {missing}")
    if frame.group_by("economy_id", "year").len().filter(pl.col("len") > 1).height:
        raise ValueError("control frame has duplicate economy-year keys")
    output = frame.with_row_index("__row_order").sort("economy_id", "year")
    interpolation_flags: list[str] = []
    for control in controls:
        prior = pl.col(control).shift(1).over("economy_id")
        future = pl.col(control).shift(-1).over("economy_id")
        prior_year = pl.col("year").shift(1).over("economy_id")
        future_year = pl.col("year").shift(-1).over("economy_id")
        flag_name = f"{control}_interpolated"
        flag = (
            pl.col(control).is_null()
            & prior.is_not_null()
            & prior.is_finite()
            & future.is_not_null()
            & future.is_finite()
            & (prior_year == pl.col("year") - 1)
            & (future_year == pl.col("year") + 1)
        )
        output = output.with_columns(
            flag.alias(flag_name),
            pl.when(flag)
            .then((prior + future) / 2.0)
            .otherwise(pl.col(control))
            .alias(f"{control}_analysis"),
        )
        interpolation_flags.append(flag_name)
    return (
        output.with_columns(
            pl.any_horizontal([pl.col(name) for name in interpolation_flags])
            .alias("interpolated_control")
        )
        .sort("__row_order")
        .drop("__row_order")
    )


def build_wdi_control_frame(
    wdi: pl.DataFrame,
    registry: pl.DataFrame,
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Select only frozen WDI rows approved as main controls and pivot them wide."""

    registry_required = {
        "source_id",
        "source_field",
        "project_field",
        "role",
        "status",
    }
    missing_registry = sorted(registry_required - set(registry.columns))
    if missing_registry:
        raise ValueError(f"indicator registry lacks columns: {missing_registry}")
    controls = registry.filter(
        (pl.col("source_id") == "wdi")
        & (pl.col("role") == "control")
        & (pl.col("status") == "approved_control")
    ).select("source_field", "project_field")
    if controls.is_empty():
        raise ValueError("indicator registry has no approved WDI controls")
    if controls.get_column("source_field").n_unique() != controls.height:
        raise ValueError("approved WDI control source fields are duplicated")
    if controls.get_column("project_field").n_unique() != controls.height:
        raise ValueError("approved WDI control project fields are duplicated")
    wdi_required = {"economy_id", "year", "indicator_id", "value"}
    missing_wdi = sorted(wdi_required - set(wdi.columns))
    if missing_wdi:
        raise ValueError(f"WDI table lacks columns: {missing_wdi}")
    source_fields = controls.get_column("source_field").to_list()
    project_fields = tuple(str(value) for value in controls.get_column("project_field"))
    selected = wdi.filter(pl.col("indicator_id").is_in(source_fields)).select(
        pl.col("economy_id").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("indicator_id").cast(pl.String),
        pl.col("value").cast(pl.Float64),
    )
    if selected.group_by("economy_id", "year", "indicator_id").len().filter(
        pl.col("len") > 1
    ).height:
        raise ValueError("WDI controls have duplicate economy-year-indicator keys")
    rename = {
        str(row["source_field"]): str(row["project_field"])
        for row in controls.iter_rows(named=True)
    }
    wide = selected.pivot(
        on="indicator_id",
        index=["economy_id", "year"],
        values="value",
    )
    for source in source_fields:
        if source not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(source))
    return (
        wide.rename(rename)
        .select("economy_id", "year", *project_fields)
        .sort("economy_id", "year"),
        project_fields,
    )


def _registry_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_regression_copies(
    frame: pl.DataFrame,
    *,
    columns: tuple[str, ...],
    lower: int = 1,
    upper: int = 99,
    fit_years: tuple[int, int] = (2000, 2022),
) -> tuple[pl.DataFrame, RegressionBoundsRegistry]:
    """Fit one pooled pre-horizon bounds registry and append clipped copies."""

    columns = tuple(columns)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("regression columns must be non-empty and unique")
    if not 0 < lower < upper < 100:
        raise ValueError("regression percentiles must lie inside zero and 100")
    if fit_years[0] > fit_years[1]:
        raise ValueError("regression fit years must be ascending")
    missing = sorted({"economy_id", "year", *columns} - set(frame.columns))
    if missing:
        raise ValueError(f"regression frame lacks columns: {missing}")
    if frame.group_by("economy_id", "year").len().filter(pl.col("len") > 1).height:
        raise ValueError("regression bounds require one underlying economy-year row")
    fit = frame.filter(pl.col("year").is_between(*fit_years))
    if fit.is_empty():
        raise ValueError("regression bounds have no main-period rows")
    bounds: dict[str, RegressionBound] = {}
    output = frame
    for column in columns:
        values = fit.filter(
            pl.col(column).is_not_null() & pl.col(column).is_finite()
        ).get_column(column)
        if values.is_empty():
            raise ValueError(f"regression bound has no finite values: {column}")
        low = float(values.quantile(lower / 100.0, interpolation="linear"))
        high = float(values.quantile(upper / 100.0, interpolation="linear"))
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"invalid regression bounds for {column}")
        bounds[column] = RegressionBound(
            lower=low,
            upper=high,
            nonnull_fit_rows=len(values),
        )
        output = output.with_columns(
            pl.col(column).clip(low, high).alias(f"{column}_p{lower:02d}_p{upper:02d}")
        )
    partial = RegressionBoundsRegistry(
        bounds=bounds,
        percentiles=(int(lower), int(upper)),
        fit_years=(int(fit_years[0]), int(fit_years[1])),
        fit_row_count=fit.height,
        canonical_hash="",
    )
    registry = RegressionBoundsRegistry(
        bounds=partial.bounds,
        percentiles=partial.percentiles,
        fit_years=partial.fit_years,
        fit_row_count=partial.fit_row_count,
        canonical_hash=_registry_hash(partial.canonical_payload()),
    )
    return output, registry


_GAD_COMPONENT_MEMBERSHIP = {
    "gad_core": frozenset({"gfvad", "supp", "gsci"}),
    "gad_no_gfvad": frozenset({"supp", "gsci"}),
    "gad_no_supp": frozenset({"gfvad", "gsci"}),
    "gad_no_gsci": frozenset({"gfvad", "supp"}),
    "gad_lite": frozenset({"supp", "gsci"}),
}


def validate_outcome_gad_pair(outcome: str, gad_version: str) -> None:
    """Reject component-overlapping or non-frozen outcome/GAD pairings."""

    required_component = None
    if outcome in {"domestic_value_added_share", "foreign_value_added_dependence"}:
        required_component = "gfvad"
    elif outcome in {
        "green_export_complexity",
        "green_export_share",
        "future_green_rca_entry_rate",
        "green_industrial_upgrading_index",
        "supplier_deepening",
    }:
        required_component = "supp"
    elif outcome == "green_science_output":
        required_component = "gsci"
    if required_component in _GAD_COMPONENT_MEMBERSHIP.get(gad_version, frozenset()):
        raise ValueError(
            f"mechanical overlap: {outcome} cannot use {gad_version} containing {required_component}"
        )
    expected = choose_gad_variant(outcome)
    if gad_version != expected:
        raise ValueError(
            f"outcome/GAD pair differs from frozen mapping: {outcome} requires {expected}"
        )


def _finite(value: object) -> bool:
    return value is not None and not isinstance(value, bool) and math.isfinite(float(value))


def build_lp_panel(
    frame: pl.DataFrame,
    spec: OutcomeSpec,
    *,
    control_columns: tuple[str, ...] | None = None,
    sample_version: str = "complete_case",
    eligibility_column: str | None = None,
    horizon_outcomes: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build timing-valid LP rows from current treatment and strict lag/future states."""

    validate_outcome_gad_pair(spec.outcome_id, spec.gad_variant)
    if not spec.horizons or any(horizon < 0 for horizon in spec.horizons):
        raise ValueError("outcome horizons must be non-empty and nonnegative")
    prohibited = [
        name
        for name in frame.columns
        if "current_import_share" in name.lower() or "future_shock" in name.lower()
    ]
    if prohibited:
        raise ValueError(f"prohibited panel input column: {prohibited[0]}")
    required = {"economy_id", "year", "gimc", spec.gad_variant}
    if horizon_outcomes is None:
        required.add(spec.outcome_id)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"LP frame lacks columns: {missing}")
    if frame.group_by("economy_id", "year").len().filter(pl.col("len") > 1).height:
        raise ValueError("LP frame has duplicate economy-year keys")
    controls = (
        tuple(control_columns)
        if control_columns is not None
        else (("control",) if "control" in frame.columns else ())
    )
    if len(controls) != len(set(controls)) or any(name not in frame.columns for name in controls):
        raise ValueError("LP control columns are missing or duplicated")
    if eligibility_column is not None and eligibility_column not in frame.columns:
        raise ValueError(f"LP eligibility column is missing: {eligibility_column}")
    if "iv_share_version" in frame.columns:
        invalid_share = frame.filter(
            pl.col("iv_share_version").is_not_null()
            & (pl.col("iv_share_version") != "main_0.0001")
        )
        if invalid_share.height:
            raise ValueError("confirmatory L4 requires iv_share_version=main_0.0001")
    if "iv_gad_version" in frame.columns:
        invalid_gad = frame.filter(
            pl.col("iv_gad_version").is_not_null()
            & (pl.col("iv_gad_version") != spec.gad_variant)
        )
        if invalid_gad.height:
            raise ValueError("IV gad_version differs from the frozen outcome mapping")
    annual = {
        (str(row["economy_id"]), int(row["year"])): row
        for row in frame.sort("economy_id", "year").iter_rows(named=True)
    }
    horizon_values: dict[tuple[str, int, int], dict[str, object]] = {}
    if horizon_outcomes is not None:
        needed = {"economy_id", "treatment_time", "horizon", spec.outcome_id}
        absent = sorted(needed - set(horizon_outcomes.columns))
        if absent:
            raise ValueError(f"horizon outcome frame lacks columns: {absent}")
        if horizon_outcomes.group_by("economy_id", "treatment_time", "horizon").len().filter(
            pl.col("len") > 1
        ).height:
            raise ValueError("horizon outcome frame has duplicate keys")
        horizon_values = {
            (str(row["economy_id"]), int(row["treatment_time"]), int(row["horizon"])): row
            for row in horizon_outcomes.iter_rows(named=True)
        }
    has_iv = "iv_share_version" in frame.columns
    rows: list[dict[str, object]] = []
    for (economy, treatment_time), treatment in sorted(annual.items()):
        if eligibility_column is not None and not bool(treatment[eligibility_column]):
            continue
        if not _finite(treatment.get("gimc")):
            continue
        if has_iv:
            if treatment.get("iv_share_version") != "main_0.0001":
                continue
            if treatment.get("iv_gad_version") != spec.gad_variant:
                continue
            if not bool(treatment.get("confirmatory_iv_eligible")):
                continue
        state = annual.get((economy, treatment_time - 1))
        if state is None or not _finite(state.get(spec.gad_variant)):
            continue
        if any(not _finite(state.get(control)) for control in controls):
            continue
        gad_lag = float(state[spec.gad_variant])
        if "iv_gad_lag" in treatment:
            if not _finite(treatment.get("iv_gad_lag")) or not math.isclose(
                gad_lag,
                float(treatment["iv_gad_lag"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("IV lagged GAD does not equal the exact t-1 mapped GAD")
        for horizon in spec.horizons:
            baseline = annual.get((economy, treatment_time - 1))
            future_time = treatment_time + int(horizon)
            if baseline is None:
                continue
            if horizon_outcomes is None:
                future = annual.get((economy, future_time))
                if (
                    future is None
                    or not _finite(baseline.get(spec.outcome_id))
                    or not _finite(future.get(spec.outcome_id))
                ):
                    continue
                future_value = float(future[spec.outcome_id])
                baseline_value = float(baseline[spec.outcome_id])
            else:
                outcome_row = horizon_values.get((economy, treatment_time, int(horizon)))
                if outcome_row is None or not _finite(outcome_row.get(spec.outcome_id)):
                    continue
                future_value = float(outcome_row[spec.outcome_id])
                baseline_value = (
                    float(outcome_row["baseline_outcome"])
                    if _finite(outcome_row.get("baseline_outcome"))
                    else None
                )
                if "outcome_time" in outcome_row and int(outcome_row["outcome_time"]) != future_time:
                    raise ValueError("horizon outcome time is not t+h")
            if spec.outcome_id in {
                "future_green_rca_entry_rate",
                "green_industrial_upgrading_index",
            }:
                delta = future_value
            else:
                if baseline_value is None:
                    continue
                delta = future_value - baseline_value
            result: dict[str, object] = {
                "economy_id": economy,
                "treatment_time": treatment_time,
                "horizon": int(horizon),
                "outcome_id": spec.outcome_id,
                "gad_version": spec.gad_variant,
                "sample_version": sample_version,
                "gad_time": treatment_time - 1,
                "control_time": treatment_time - 1,
                "baseline_outcome_time": treatment_time - 1,
                "outcome_time": future_time,
                "gimc": float(treatment["gimc"]),
                "gad_lag": gad_lag,
                "Z": float(treatment["Z"]) if _finite(treatment.get("Z")) else None,
                "Z_GAD": (
                    float(treatment["Z_GAD"])
                    if _finite(treatment.get("Z_GAD"))
                    else None
                ),
                "CMZ": float(treatment["CMZ"]) if _finite(treatment.get("CMZ")) else None,
                "baseline_outcome": baseline_value,
                "future_outcome": future_value,
                "delta_outcome": delta,
                "rca_eligible_products": (
                    int(outcome_row["eligible_products"])
                    if horizon_outcomes is not None
                    and spec.outcome_id == "future_green_rca_entry_rate"
                    and outcome_row.get("eligible_products") is not None
                    else None
                ),
                "rca_entered_products": (
                    int(outcome_row["entered_products"])
                    if horizon_outcomes is not None
                    and spec.outcome_id == "future_green_rca_entry_rate"
                    and outcome_row.get("entered_products") is not None
                    else None
                ),
                "outcome_coverage_eligible": True,
                "confirmatory_iv_eligible": bool(
                    treatment.get("confirmatory_iv_eligible", False)
                ),
                "core_eligible": bool(treatment.get("core_eligible", False)),
                "lite_eligible": bool(treatment.get("lite_eligible", False)),
                "provisional_core": bool(treatment.get("provisional_core", False)),
                "descriptive_only": bool(horizon == 0 or treatment_time >= 2023),
                "threshold_selection_only": spec.outcome_id
                == "green_industrial_upgrading_index",
                "negative_shock_sample": False,
                "gad_interpolated": False,
                "outcome_interpolated": False,
            }
            for control in controls:
                result[control] = float(state[control])
                flag_name = f"{control.removesuffix('_analysis')}_interpolated"
                result[flag_name] = bool(state.get(flag_name, False))
            result["interpolated_control"] = any(
                bool(state.get(f"{control.removesuffix('_analysis')}_interpolated", False))
                for control in controls
            )
            rows.append(result)
    if not rows:
        return pl.DataFrame(
            schema={
                "economy_id": pl.String,
                "treatment_time": pl.Int16,
                "horizon": pl.Int16,
                "outcome_id": pl.String,
                "gad_version": pl.String,
                "sample_version": pl.String,
            }
        )
    output = pl.DataFrame(rows).with_columns(
        pl.col("treatment_time").cast(pl.Int16),
        pl.col("horizon").cast(pl.Int16),
        pl.col("gad_time").cast(pl.Int16),
        pl.col("control_time").cast(pl.Int16),
        pl.col("baseline_outcome_time").cast(pl.Int16),
        pl.col("outcome_time").cast(pl.Int16),
    )
    key = [
        "economy_id",
        "treatment_time",
        "horizon",
        "outcome_id",
        "gad_version",
        "sample_version",
    ]
    if output.group_by(key).len().filter(pl.col("len") > 1).height:
        raise ValueError("LP panel has duplicate frozen keys")
    return output.sort(key)


def build_vulnerability_panel(panel: pl.DataFrame) -> pl.DataFrame:
    """Select negative-shock rows while retaining the unmultiplied outcome change."""

    required = {"outcome_id", "Z", "delta_outcome"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"vulnerability panel lacks columns: {missing}")
    allowed = {
        "asinh_weighted_green_imports",
        "renewable_capacity_additions_mw_per_million",
    }
    return panel.filter(
        pl.col("outcome_id").is_in(allowed)
        & pl.col("Z").is_not_null()
        & pl.col("Z").is_finite()
        & (pl.col("Z") < 0.0)
    ).with_columns(pl.lit(True).alias("negative_shock_sample"))


def validate_model_panel(panel: pl.DataFrame) -> dict[str, int]:
    """Independently recompute L4 timing, mapping, overlap, and leakage guarantees."""

    required = {
        "economy_id",
        "treatment_time",
        "horizon",
        "outcome_id",
        "gad_version",
        "sample_version",
        "gad_time",
        "control_time",
        "baseline_outcome_time",
        "outcome_time",
        "gimc",
        "gad_lag",
        "Z",
        "Z_GAD",
        "CMZ",
        "delta_outcome",
        "confirmatory_iv_eligible",
        "gad_interpolated",
        "outcome_interpolated",
        "negative_shock_sample",
        "core_eligible",
        "lite_eligible",
        "descriptive_only",
        "threshold_selection_only",
        "source_outcome_materialized",
        "baseline_outcome",
        "future_outcome",
        "outcome_coverage_eligible",
        "rca_eligible_products",
        "rca_entered_products",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"model panel lacks required columns: {missing}")
    prohibited = [
        name
        for name in panel.columns
        if "current_import_share" in name.lower() or "future_shock" in name.lower()
    ]
    if prohibited:
        raise ValueError(f"model panel contains prohibited leakage column: {prohibited[0]}")
    key = [
        "economy_id",
        "treatment_time",
        "horizon",
        "outcome_id",
        "gad_version",
        "sample_version",
    ]
    duplicates = panel.group_by(key).len().filter(pl.col("len") > 1).height
    if duplicates:
        raise ValueError(f"model panel duplicate frozen keys: {duplicates}")
    timing_checks = {
        "gad_time": pl.col("gad_time") != pl.col("treatment_time") - 1,
        "control_time": pl.col("control_time") != pl.col("treatment_time") - 1,
        "baseline_outcome_time": pl.col("baseline_outcome_time")
        != pl.col("treatment_time") - 1,
        "outcome_time": pl.col("outcome_time")
        != pl.col("treatment_time") + pl.col("horizon"),
    }
    for name, expression in timing_checks.items():
        count = panel.filter(expression).height
        if count:
            raise ValueError(f"model panel {name} timing violations: {count}")
    invalid_finite = panel.filter(
        pl.any_horizontal(
            [
                pl.col(name).is_null() | ~pl.col(name).is_finite()
                for name in ("gimc", "gad_lag", "Z", "Z_GAD", "delta_outcome")
            ]
        )
    ).height
    if invalid_finite:
        raise ValueError(f"model panel incomplete confirmatory values: {invalid_finite}")
    ineligible_iv = panel.filter(~pl.col("confirmatory_iv_eligible")).height
    if ineligible_iv:
        raise ValueError(f"model panel includes nonconfirmatory IV rows: {ineligible_iv}")
    interpolation = panel.filter(
        pl.col("gad_interpolated") | pl.col("outcome_interpolated")
    ).height
    if interpolation:
        raise ValueError(f"model panel outcome/GAD interpolation violations: {interpolation}")
    mapping_violations = 0
    for outcome, gad_version in panel.select("outcome_id", "gad_version").unique().iter_rows():
        try:
            validate_outcome_gad_pair(str(outcome), str(gad_version))
        except (KeyError, ValueError):
            mapping_violations += 1
    if mapping_violations:
        raise ValueError(f"model panel mapping or mechanical overlap violations: {mapping_violations}")
    horizons = {spec.outcome_id: frozenset(spec.horizons) for spec in outcome_specs()}
    invalid_horizons = sum(
        int(row["horizon"]) not in horizons.get(str(row["outcome_id"]), frozenset())
        for row in panel.select("outcome_id", "horizon").unique().iter_rows(named=True)
    )
    if invalid_horizons:
        raise ValueError(f"model panel frozen horizon violations: {invalid_horizons}")
    allowed_samples = {
        f"{branch}_{mode}{suffix}"
        for branch in ("core", "lite")
        for mode in ("complete_case", "bounded_controls")
        for suffix in ("", "_negative_shock")
    }
    unknown_samples = panel.filter(~pl.col("sample_version").is_in(allowed_samples)).height
    if unknown_samples:
        raise ValueError(f"model panel sample_version vocabulary violations: {unknown_samples}")
    core_lite = panel.filter(
        (
            pl.col("sample_version").str.starts_with("core_")
            & (
                ~pl.col("treatment_time").is_between(2000, 2022)
                | ~pl.col("core_eligible")
            )
        )
        | (
            pl.col("sample_version").str.starts_with("lite_")
            & (
                ~pl.col("treatment_time").is_between(2023, 2024)
                | ~pl.col("lite_eligible")
                | ~pl.col("descriptive_only")
            )
        )
    ).height
    if core_lite:
        raise ValueError(f"model panel Core/Lite separation violations: {core_lite}")
    materialized_ids = {
        spec.outcome_id
        for spec in outcome_specs()
        if spec.materialized and spec.authority_table == "outcomes_country_year"
    }
    expected_negative = pl.col("sample_version").str.ends_with("_negative_shock")
    discriminator = panel.filter(
        (pl.col("source_outcome_materialized") != pl.col("outcome_id").is_in(materialized_ids))
        | (
            pl.col("threshold_selection_only")
            != (pl.col("outcome_id") == "green_industrial_upgrading_index")
        )
        | (
            pl.col("descriptive_only")
            != ((pl.col("horizon") == 0) | pl.col("sample_version").str.starts_with("lite_"))
        )
        | (pl.col("negative_shock_sample") != expected_negative)
        | ~pl.col("outcome_coverage_eligible")
    ).height
    if discriminator:
        raise ValueError(f"model panel discriminator flag violations: {discriminator}")
    negative = panel.filter(
        pl.col("negative_shock_sample")
        & (
            pl.col("Z").is_null()
            | ~pl.col("Z").is_finite()
            | (pl.col("Z") >= 0.0)
            | ~pl.col("outcome_id").is_in(
                [
                    "asinh_weighted_green_imports",
                    "renewable_capacity_additions_mw_per_million",
                ]
            )
        )
    ).height
    if negative:
        raise ValueError(f"model panel negative shock sample violations: {negative}")
    derived_ids = {"future_green_rca_entry_rate", "green_industrial_upgrading_index"}
    derived_nulls = panel.filter(
        pl.col("outcome_id").is_in(derived_ids)
        & (
            pl.col("baseline_outcome").is_not_null()
            | pl.col("future_outcome").is_null()
            | _numeric_mismatch("future_outcome", "delta_outcome")
        )
    ).height
    if derived_nulls:
        raise ValueError(f"model panel derived outcome conditional-null violations: {derived_nulls}")
    materialized_nulls = panel.filter(
        pl.col("outcome_id").is_in(materialized_ids)
        & (
            pl.col("baseline_outcome").is_null()
            | pl.col("future_outcome").is_null()
            | (((pl.col("future_outcome") - pl.col("baseline_outcome")) - pl.col("delta_outcome")).abs() > 1e-12)
        )
    ).height
    if materialized_nulls:
        raise ValueError(f"model panel source outcome conditional-null violations: {materialized_nulls}")
    rca_conditions = panel.filter(
        (
            (pl.col("outcome_id") == "future_green_rca_entry_rate")
            & (
                pl.col("rca_eligible_products").is_null()
                | pl.col("rca_entered_products").is_null()
                | (pl.col("rca_eligible_products") == 0)
                | (pl.col("rca_entered_products") > pl.col("rca_eligible_products"))
                | (
                    (
                        (
                            pl.col("rca_entered_products")
                            / pl.col("rca_eligible_products")
                        )
                        - pl.col("future_outcome")
                    ).abs()
                    > 1e-12
                )
            )
        )
        | (
            (pl.col("outcome_id") != "future_green_rca_entry_rate")
            & (
                pl.col("rca_eligible_products").is_not_null()
                | pl.col("rca_entered_products").is_not_null()
            )
        )
    ).height
    if rca_conditions:
        raise ValueError(f"model panel RCA conditional value/null violations: {rca_conditions}")
    control_null_violations = 0
    control_contract_columns = {
        name
        for control in MAIN_CONTROL_COLUMNS
        for name in (
            control,
            f"{control}_analysis",
            f"{control}_interpolated",
        )
    } | {"interpolated_control"}
    present_control_columns = control_contract_columns & set(panel.columns)
    if present_control_columns and present_control_columns != control_contract_columns:
        missing_controls = sorted(control_contract_columns - set(panel.columns))
        raise ValueError(f"model panel lacks complete control-mode columns: {missing_controls}")
    if present_control_columns:
        control_flag_columns = [
            *(f"{control}_interpolated" for control in MAIN_CONTROL_COLUMNS),
            "interpolated_control",
        ]
        non_boolean_flags = [
            name for name in control_flag_columns if panel.schema[name] != pl.Boolean
        ]
        if non_boolean_flags:
            raise ValueError(
                f"model panel control interpolation flags must be Boolean: {non_boolean_flags}"
            )
        null_flag_rows = panel.filter(
            pl.any_horizontal(
                [pl.col(name).is_null() for name in control_flag_columns]
            ).fill_null(True)
        ).height
        if null_flag_rows:
            raise ValueError(
                f"model panel control interpolation flag null violations: {null_flag_rows}"
            )
        complete_mode = pl.col("sample_version").str.contains("complete_case")
        bounded_mode = pl.col("sample_version").str.contains("bounded_controls")
        for control in MAIN_CONTROL_COLUMNS:
            analysis = f"{control}_analysis"
            flag = f"{control}_interpolated"
            raw_finite = (
                pl.col(control).is_not_null() & pl.col(control).is_finite()
            ).fill_null(False)
            analysis_finite = (
                pl.col(analysis).is_not_null() & pl.col(analysis).is_finite()
            ).fill_null(False)
            control_violation = (
                (
                    complete_mode
                    & (
                        ~raw_finite
                        | ~analysis_finite
                        | pl.col(flag)
                        | _numeric_mismatch(control, analysis)
                    )
                )
                | (
                    bounded_mode
                    & (
                        (
                            pl.col(control).is_not_null()
                            & (
                                ~raw_finite
                                | ~analysis_finite
                                | pl.col(flag)
                                | _numeric_mismatch(control, analysis)
                            )
                        )
                        | (
                            pl.col(control).is_null()
                            & (
                                (
                                    pl.col(flag)
                                    & ~analysis_finite
                                )
                                | (
                                    ~pl.col(flag)
                                    & pl.col(analysis).is_not_null()
                                )
                            )
                        )
                    )
                )
            )
            control_null_violations += panel.filter(
                control_violation.fill_null(True)
            ).height
        control_null_violations += panel.filter(
            (
                pl.col("interpolated_control")
                != pl.any_horizontal(
                    [
                        pl.col(f"{control}_interpolated")
                        for control in MAIN_CONTROL_COLUMNS
                    ]
                )
            ).fill_null(True)
        ).height
    if "CMZ_p01_p99" in panel.columns:
        control_null_violations += panel.filter(
            pl.col("CMZ").is_null() != pl.col("CMZ_p01_p99").is_null()
        ).height
    if control_null_violations:
        raise ValueError(
            f"model panel control/CMZ conditional-null violations: {control_null_violations}"
        )
    return {
        "rows": panel.height,
        "duplicate_keys": duplicates,
        "timing_violations": 0,
        "mapping_overlap_violations": 0,
        "outcome_gad_interpolation_count": interpolation,
        "current_share_future_shock_columns": len(prohibited),
        "negative_shock_violations": negative,
        "discriminator_flag_violations": discriminator,
        "conditional_null_violations": (
            derived_nulls + materialized_nulls + rca_conditions + control_null_violations
        ),
    }


def _final_sample_flow(frame: pl.DataFrame, implementation_commit: str) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    branches = {
        "core": (
            "rule_economy_eligible",
            "rule_positive_import_baseline",
            "rule_baseline_share_identifiable",
            "rule_complete_absorption_1996",
            "rule_complete_initialization_1997_1999",
            "rule_minimum_valid_main_years",
        ),
        "lite": (
            "rule_economy_eligible",
            "rule_positive_import_baseline",
            "rule_baseline_share_identifiable",
            "rule_complete_lite_absorption_1996",
            "rule_complete_lite_initialization_1997_1999",
            "rule_minimum_valid_lite_years",
        ),
    }
    for branch, rules in branches.items():
        active = pl.Series([True] * frame.height, dtype=pl.Boolean)
        previous = frame.height
        rows.append(
            {
                "sample_definition": branch,
                "stage_order": 0,
                "stage": "starting_universe",
                "entered": previous,
                "retained": previous,
                "excluded_at_stage": 0,
                "identity_valid": True,
                "implementation_commit": implementation_commit,
            }
        )
        for order, rule in enumerate(rules, start=1):
            active = active & frame.get_column(rule)
            retained = int(active.sum())
            excluded = previous - retained
            rows.append(
                {
                    "sample_definition": branch,
                    "stage_order": order,
                    "stage": rule,
                    "entered": previous,
                    "retained": retained,
                    "excluded_at_stage": excluded,
                    "identity_valid": previous == retained + excluded,
                    "implementation_commit": implementation_commit,
                }
            )
            previous = retained
    return pl.DataFrame(rows).with_columns(
        pl.col("stage_order").cast(pl.UInt8),
        pl.col("entered", "retained", "excluded_at_stage").cast(pl.UInt32),
    )


def freeze_final_sample_authority(
    paths: ProjectPaths,
    *,
    build: BuildIdentity,
) -> FinalSampleBuildReport:
    """Build and publish the authoritative structural final-sample flags."""

    provisional_path = paths.harmonized / "sample/provisional_sample.parquet"
    gad_path = paths.measures / "gad/gad_country_year.parquet"
    baseline_path = paths.measures / "instruments/iv_baseline_shares.parquet"
    provisional = pl.read_parquet(provisional_path)
    gad = pl.read_parquet(gad_path)
    baseline = pl.read_parquet(baseline_path)

    def gad_coverage(specification: str, prefix: str, end_year: int) -> pl.DataFrame:
        selected = gad.filter(pl.col("specification_id") == specification)
        return selected.group_by("economy_id").agg(
            pl.col("absorption_present")
            .filter(pl.col("year") == 1996)
            .first()
            .fill_null(False)
            .alias(f"complete_{prefix}_absorption_1996"),
            (
                pl.col("gap_valid")
                .filter(pl.col("year").is_between(1997, 1999))
                .sum()
                == 3
            ).alias(f"complete_{prefix}_initialization_1997_1999"),
            pl.col("gap_valid")
            .filter(pl.col("year").is_between(2000, end_year))
            .sum()
            .cast(pl.Int16)
            .alias(f"valid_{prefix}_years"),
        )

    core = gad_coverage("gad_core", "main", 2022).rename(
        {
            "complete_main_absorption_1996": "complete_absorption_1996",
            "complete_main_initialization_1997_1999": "complete_initialization_1997_1999",
        }
    )
    lite = gad_coverage("gad_lite", "lite", 2024)
    shares = (
        baseline.filter(
            (pl.col("taxonomy_version") == "main_hs96")
            & (pl.col("share_version") == "main_0.0001")
        )
        .group_by("importer")
        .agg(
            pl.col("confirmatory_baseline_eligible")
            .any()
            .alias("baseline_share_identifiable")
        )
        .rename({"importer": "economy_id"})
    )
    coverage = (
        provisional.select(
            "economy_id",
            pl.col("sample_version").alias("source_sample_version"),
            "economy_rule_eligible",
            pl.col("positive_green_import_baseline_years").alias(
                "positive_import_baseline_years"
            ),
        )
        .join(core, on="economy_id", how="left")
        .join(lite, on="economy_id", how="left")
        .join(shares, on="economy_id", how="left")
        .with_columns(
            pl.col("complete_absorption_1996").fill_null(False),
            pl.col("complete_initialization_1997_1999").fill_null(False),
            pl.col("valid_main_years").fill_null(0),
            pl.col("complete_lite_absorption_1996").fill_null(False),
            pl.col("complete_lite_initialization_1997_1999").fill_null(False),
            pl.col("valid_lite_years").fill_null(0),
            pl.col("baseline_share_identifiable").fill_null(False),
        )
    )
    final = freeze_final_sample(coverage)
    validate_final_sample(final)
    contract_path = paths.code_root / "03_代码/contracts/final_sample.json"
    destination = paths.harmonized / "sample/final_sample.parquet"
    manifest = write_authoritative_table(
        final,
        _load_contract(contract_path),
        destination,
        tuple(
            InputArtifact.from_path(path)
            for path in (provisional_path, gad_path, baseline_path, contract_path)
        ),
        build,
    )
    flow_path = paths.audits / "最终样本流_v1.csv"
    flow = _final_sample_flow(final, build.code_commit)
    _write_csv_atomic(flow, flow_path)
    if flow.filter(~pl.col("identity_valid")).height:
        raise RuntimeError("final sample flow identity failed")
    return FinalSampleBuildReport(
        rows=final.height,
        core_economies=final.filter(pl.col("core_eligible")).height,
        lite_economies=final.filter(pl.col("lite_eligible")).height,
        taiwan_robustness_economies=final.filter(
            pl.col("taiwan_robustness_eligible")
        ).height,
        micro_robustness_economies=final.filter(
            pl.col("micro_robustness_eligible")
        ).height,
        output_path=str(destination),
        output_sha256=manifest.output_sha256,
        sample_flow_path=str(flow_path),
    )


def _bounds_registry_frame(registry: RegressionBoundsRegistry) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "source_column": name,
                "lower_percentile": registry.percentiles[0],
                "upper_percentile": registry.percentiles[1],
                "fit_start_year": registry.fit_years[0],
                "fit_end_year": registry.fit_years[1],
                "lower_bound": bound.lower,
                "upper_bound": bound.upper,
                "nonnull_fit_rows": bound.nonnull_fit_rows,
                "underlying_fit_rows": registry.fit_row_count,
                "canonical_hash": registry.canonical_hash,
            }
            for name, bound in sorted(registry.bounds.items())
        ]
    ).with_columns(
        pl.col("lower_percentile", "upper_percentile").cast(pl.UInt8),
        pl.col("fit_start_year", "fit_end_year").cast(pl.Int16),
        pl.col("nonnull_fit_rows", "underlying_fit_rows").cast(pl.UInt32),
    )


def _giu_registry_frame(registry: GIUScalerRegistry) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon, record in sorted(registry.by_horizon.items()):
        for source_column, scaler in sorted(record.components.items()):
            rows.append(
                {
                    "horizon": horizon,
                    "source_column": source_column,
                    "fit_start_year": registry.fit_years[0],
                    "fit_end_year": registry.fit_years[1],
                    "center": scaler.center,
                    "scale": scaler.scale,
                    "scale_method": scaler.scale_method,
                    "row_count": scaler.row_count,
                    "economy_count": scaler.economy_count,
                    "canonical_hash": registry.canonical_hash,
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("horizon", "fit_start_year", "fit_end_year").cast(pl.Int16),
        pl.col("row_count", "economy_count").cast(pl.UInt32),
    )


def _future_rca_horizon_frame(product: pl.DataFrame) -> pl.DataFrame:
    rca = product.select(
        "economy_id",
        "year",
        "hs6",
        pl.col("green_product_rca").alias("rca"),
    )
    frames: list[pl.DataFrame] = []
    for horizon in range(3, 9):
        for treatment_time in range(2000, 2025 - horizon):
            result = build_rca_entry_rate(
                rca,
                treatment_year=treatment_time,
                horizon=horizon,
            )
            if not result.is_empty():
                frames.append(
                    result.rename({"treatment_year": "treatment_time"}).with_columns(
                        (pl.col("treatment_time") + pl.col("horizon"))
                        .cast(pl.Int16)
                        .alias("outcome_time")
                    )
                )
    if not frames:
        raise ValueError("future RCA entry construction produced no rows")
    return pl.concat(frames).sort("economy_id", "treatment_time", "horizon")


def _giu_component_frame(
    country: pl.DataFrame,
    entry: pl.DataFrame,
    provisional: pl.DataFrame,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for horizon in range(3, 9):
        baseline = country.select(
            "economy_id",
            (pl.col("year") + 1).cast(pl.Int16).alias("treatment_time"),
            pl.col("green_export_complexity").alias("baseline_complexity"),
            pl.col("green_export_share").alias("baseline_share"),
        )
        future = country.select(
            "economy_id",
            (pl.col("year") - horizon).cast(pl.Int16).alias("treatment_time"),
            pl.col("green_export_complexity").alias("future_complexity"),
            pl.col("green_export_share").alias("future_share"),
        )
        changes = baseline.join(
            future, on=["economy_id", "treatment_time"], how="inner"
        ).with_columns(
            pl.lit(horizon, dtype=pl.Int16).alias("horizon"),
            (pl.col("future_complexity") - pl.col("baseline_complexity")).alias(
                "green_export_complexity_change"
            ),
            (pl.col("future_share") - pl.col("baseline_share")).alias(
                "green_export_share_change"
            ),
        )
        frames.append(
            entry.filter(pl.col("horizon") == horizon)
            .select(
                "economy_id",
                "treatment_time",
                "horizon",
                "future_green_rca_entry_rate",
            )
            .join(
                changes.select(
                    "economy_id",
                    "treatment_time",
                    "horizon",
                    "green_export_complexity_change",
                    "green_export_share_change",
                ),
                on=["economy_id", "treatment_time", "horizon"],
                how="left",
            )
        )
    return (
        pl.concat(frames)
        .join(
            provisional.select("economy_id", "provisional_core"),
            on="economy_id",
            how="left",
        )
        .with_columns(pl.col("provisional_core").fill_null(False))
        .sort("economy_id", "treatment_time", "horizon")
    )


def _annual_panel_authority(
    *,
    gad: pl.DataFrame,
    country: pl.DataFrame,
    controls: pl.DataFrame,
    final_sample: pl.DataFrame,
) -> tuple[pl.DataFrame, str]:
    mapped = gad.filter(pl.col("specification_id").is_in(MAPPED_GAD_VARIANTS))
    gad_wide = mapped.select(
        "economy_id", "year", "specification_id", "gad"
    ).pivot(
        on="specification_id",
        index=["economy_id", "year"],
        values="gad",
    )
    for variant in MAPPED_GAD_VARIANTS:
        if variant not in gad_wide.columns:
            gad_wide = gad_wide.with_columns(
                pl.lit(None, dtype=pl.Float64).alias(variant)
            )
    core_common = gad.filter(pl.col("specification_id") == "gad_core").select(
        "economy_id",
        "year",
        "gimc",
        "provisional_core",
        "scaler_hash",
    )
    lite_extension = gad.filter(
        (pl.col("specification_id") == "gad_lite") & (pl.col("year") > 2022)
    ).select(
        "economy_id",
        "year",
        "gimc",
        "provisional_core",
        "scaler_hash",
    )
    common = pl.concat([core_common, lite_extension]).sort("economy_id", "year")
    if common.group_by("economy_id", "year").len().filter(pl.col("len") > 1).height:
        raise ValueError("annual panel GIMC authority has duplicate economy-year keys")
    scaler_hashes = common.get_column("scaler_hash").drop_nulls().unique().to_list()
    if len(scaler_hashes) != 1:
        raise ValueError("annual panel authority has multiple GAD scaler hashes")
    annual = (
        common.join(gad_wide, on=["economy_id", "year"], how="left")
        .join(country, on=["economy_id", "year"], how="left")
        .join(controls, on=["economy_id", "year"], how="left")
        .join(
            final_sample.select(
                "economy_id", "core_eligible", "lite_eligible"
            ),
            on="economy_id",
            how="left",
        )
        .with_columns(
            pl.col("core_eligible").fill_null(False),
            pl.col("lite_eligible").fill_null(False),
        )
        .with_columns(
            (
                pl.col("core_eligible")
                & pl.col("year").is_between(2000, 2022)
            ).alias(
                "core_row_eligible"
            ),
            (pl.col("lite_eligible") & (pl.col("year") >= 2023)).alias(
                "lite_row_eligible"
            ),
        )
    )
    return interpolate_single_year_controls(annual, MAIN_CONTROL_COLUMNS), str(
        scaler_hashes[0]
    )


def _regression_authority(
    annual: pl.DataFrame,
    iv: pl.DataFrame,
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    authority = annual.select(
        "economy_id", "year", "core_eligible", "gimc", *MAIN_CONTROL_COLUMNS
    )
    columns: list[str] = ["gimc", *MAIN_CONTROL_COLUMNS]
    for variant in MAPPED_GAD_VARIANTS:
        subset = iv.filter(
            (pl.col("taxonomy_version") == "main_hs96")
            & (pl.col("share_version") == "main_0.0001")
            & (pl.col("gad_version") == variant)
        ).select(
            pl.col("importer").alias("economy_id"),
            "year",
            pl.col("z").alias("Z") if variant == "gad_core" else pl.col("z").alias(f"Z__{variant}"),
            pl.col("gad_lag").alias(f"gad_lag__{variant}"),
            pl.col("z_gad").alias(f"Z_GAD__{variant}"),
            pl.col("cmz").alias(f"CMZ__{variant}"),
        )
        authority = authority.join(subset, on=["economy_id", "year"], how="left")
        if variant == "gad_core":
            columns.append("Z")
        columns.extend(
            [f"gad_lag__{variant}", f"Z_GAD__{variant}", f"CMZ__{variant}"]
        )
    return authority.filter(pl.col("core_eligible")).drop("core_eligible"), tuple(
        columns
    )


def _attach_iv(
    annual: pl.DataFrame,
    iv: pl.DataFrame,
    gad_version: str,
) -> pl.DataFrame:
    source = iv.filter(
        (pl.col("taxonomy_version") == "main_hs96")
        & (pl.col("share_version") == "main_0.0001")
        & (pl.col("gad_version") == gad_version)
    ).select(
        pl.col("importer").alias("economy_id"),
        "year",
        pl.col("share_version").alias("iv_share_version"),
        pl.col("gad_version").alias("iv_gad_version"),
        pl.col("confirmatory_iv_eligible"),
        pl.col("gad_lag").alias("iv_gad_lag"),
        pl.col("z").alias("Z"),
        pl.col("z_gad").alias("Z_GAD"),
        pl.col("cmz").alias("CMZ"),
    )
    if source.group_by("economy_id", "year").len().filter(pl.col("len") > 1).height:
        raise ValueError("main IV source has duplicate economy-year-mapping rows")
    return annual.join(source, on=["economy_id", "year"], how="left")


def _normalize_panel_controls(
    panel: pl.DataFrame,
    annual: pl.DataFrame,
    *,
    used_analysis_controls: bool,
) -> pl.DataFrame:
    if panel.is_empty():
        return panel
    output = panel
    if not used_analysis_controls:
        output = output.rename(
            {name: f"{name}_analysis" for name in MAIN_CONTROL_COLUMNS}
        )
    raw = annual.select(
        "economy_id",
        pl.col("year").alias("control_time"),
        *MAIN_CONTROL_COLUMNS,
    )
    return output.join(raw, on=["economy_id", "control_time"], how="left")


def _apply_panel_bounds(
    panel: pl.DataFrame,
    registry: RegressionBoundsRegistry,
) -> pl.DataFrame:
    bounds = registry.bounds
    output = panel.with_columns(
        pl.col("gimc")
        .clip(bounds["gimc"].lower, bounds["gimc"].upper)
        .alias("gimc_p01_p99"),
        pl.col("Z").clip(bounds["Z"].lower, bounds["Z"].upper).alias("Z_p01_p99"),
    )
    gad_expression = pl.lit(None, dtype=pl.Float64)
    zgad_expression = pl.lit(None, dtype=pl.Float64)
    cmz_expression = pl.lit(None, dtype=pl.Float64)
    for variant in reversed(MAPPED_GAD_VARIANTS):
        gad_bound = bounds[f"gad_lag__{variant}"]
        zgad_bound = bounds[f"Z_GAD__{variant}"]
        cmz_bound = bounds[f"CMZ__{variant}"]
        gad_expression = pl.when(pl.col("gad_version") == variant).then(
            pl.col("gad_lag").clip(gad_bound.lower, gad_bound.upper)
        ).otherwise(gad_expression)
        zgad_expression = pl.when(pl.col("gad_version") == variant).then(
            pl.col("Z_GAD").clip(zgad_bound.lower, zgad_bound.upper)
        ).otherwise(zgad_expression)
        cmz_expression = pl.when(pl.col("gad_version") == variant).then(
            pl.col("CMZ").clip(cmz_bound.lower, cmz_bound.upper)
        ).otherwise(cmz_expression)
    expressions: list[pl.Expr] = [
        gad_expression.alias("gad_lag_p01_p99"),
        zgad_expression.alias("Z_GAD_p01_p99"),
        cmz_expression.alias("CMZ_p01_p99"),
    ]
    for control in MAIN_CONTROL_COLUMNS:
        bound = bounds[control]
        expressions.append(
            pl.col(f"{control}_analysis")
            .clip(bound.lower, bound.upper)
            .alias(f"{control}_analysis_p01_p99")
        )
    return output.with_columns(expressions)


def build_analysis_panel_authority(
    paths: ProjectPaths,
    *,
    build: BuildIdentity,
) -> AnalysisPanelBuildReport:
    """Build and publish timing-safe Core/Lite L4 panels and frozen registries."""

    gad_path = paths.measures / "gad/gad_country_year.parquet"
    country_path = paths.measures / "outcomes/outcomes_country_year.parquet"
    product_path = paths.measures / "outcomes/outcomes_product_year.parquet"
    iv_path = paths.measures / "instruments/iv_country_year.parquet"
    wdi_path = paths.normalized / "wdi/wdi_country_year.parquet"
    provisional_path = paths.harmonized / "sample/provisional_sample.parquet"
    final_path = paths.harmonized / "sample/final_sample.parquet"
    registry_path = paths.code_root / "02_数据字典/indicator_registry_v1.csv"
    map_path = paths.code_root / "config/outcome_gad_map.yaml"
    model_contract_path = paths.code_root / "03_代码/contracts/model_panel.json"
    bounds_contract_path = paths.code_root / "03_代码/contracts/regression_bounds.json"
    giu_contract_path = paths.code_root / "03_代码/contracts/giu_outcome_scalers.json"

    gad = pl.read_parquet(gad_path)
    country = pl.read_parquet(country_path)
    product = pl.read_parquet(product_path)
    iv = pl.read_parquet(iv_path)
    provisional = pl.read_parquet(provisional_path)
    final_sample = pl.read_parquet(final_path)
    controls, control_columns = build_wdi_control_frame(
        pl.read_parquet(wdi_path), pl.read_csv(registry_path)
    )
    if control_columns != MAIN_CONTROL_COLUMNS:
        raise ValueError(
            f"approved main control order differs from frozen model contract: {control_columns}"
        )
    annual, gad_scaler_hash = _annual_panel_authority(
        gad=gad,
        country=country,
        controls=controls,
        final_sample=final_sample,
    )
    regression_authority, regression_columns = _regression_authority(annual, iv)
    _, bounds_registry = build_regression_copies(
        regression_authority,
        columns=regression_columns,
        lower=1,
        upper=99,
        fit_years=(2000, 2022),
    )
    if bounds_registry.canonical_hash == gad_scaler_hash:
        raise ValueError("regression bounds hash must be independent of the GAD scaler")
    bounds_path = paths.analysis / "regression_bounds.parquet"
    bounds_manifest = write_authoritative_table(
        _bounds_registry_frame(bounds_registry),
        _load_contract(bounds_contract_path),
        bounds_path,
        tuple(
            InputArtifact.from_path(path)
            for path in (gad_path, iv_path, wdi_path, final_path, bounds_contract_path)
        ),
        build,
    )

    entry = _future_rca_horizon_frame(product)
    giu_components = _giu_component_frame(country, entry, provisional)
    giu_values, giu_registry = fit_giu_scalers(giu_components)
    if giu_registry.canonical_hash in {
        gad_scaler_hash,
        bounds_registry.canonical_hash,
    }:
        raise ValueError("GIU scaler hash must be independent of GAD and regression bounds")
    giu_scalers_path = paths.analysis / "giu_outcome_scalers.parquet"
    giu_manifest = write_authoritative_table(
        _giu_registry_frame(giu_registry),
        _load_contract(giu_contract_path),
        giu_scalers_path,
        tuple(
            InputArtifact.from_path(path)
            for path in (
                country_path,
                product_path,
                provisional_path,
                giu_contract_path,
            )
        ),
        build,
    )
    horizon_authorities = {
        "future_green_rca_entry_rate": entry.select(
            "economy_id",
            "treatment_time",
            "horizon",
            "outcome_time",
            "future_green_rca_entry_rate",
            "eligible_products",
            "entered_products",
        ),
        "green_industrial_upgrading_index": giu_values.select(
            "economy_id",
            "treatment_time",
            "horizon",
            (pl.col("treatment_time") + pl.col("horizon"))
            .cast(pl.Int16)
            .alias("outcome_time"),
            "green_industrial_upgrading_index",
        ),
    }
    registered = outcome_specs()
    materialized_country = [
        spec
        for spec in registered
        if spec.materialized and spec.authority_table == "outcomes_country_year"
    ]
    derived = [
        next(spec for spec in registered if spec.outcome_id == outcome)
        for outcome in (
            "future_green_rca_entry_rate",
            "green_industrial_upgrading_index",
        )
    ]
    panels: list[pl.DataFrame] = []
    for spec in (*materialized_country, *derived):
        source = _attach_iv(annual, iv, spec.gad_variant)
        horizon_source = horizon_authorities.get(spec.outcome_id)
        for eligibility, prefix in (
            ("core_row_eligible", "core"),
            ("lite_row_eligible", "lite"),
        ):
            complete = build_lp_panel(
                source,
                spec,
                control_columns=MAIN_CONTROL_COLUMNS,
                sample_version=f"{prefix}_complete_case",
                eligibility_column=eligibility,
                horizon_outcomes=horizon_source,
            )
            if not complete.is_empty():
                panels.append(
                    _normalize_panel_controls(
                        complete, annual, used_analysis_controls=False
                    )
                )
            analysis_controls = tuple(
                f"{name}_analysis" for name in MAIN_CONTROL_COLUMNS
            )
            bounded = build_lp_panel(
                source,
                spec,
                control_columns=analysis_controls,
                sample_version=f"{prefix}_bounded_controls",
                eligibility_column=eligibility,
                horizon_outcomes=horizon_source,
            )
            if not bounded.is_empty():
                panels.append(
                    _normalize_panel_controls(
                        bounded, annual, used_analysis_controls=True
                    )
                )
    if not panels:
        raise ValueError("analysis panel construction produced no timing-valid rows")
    base_panel = pl.concat(panels, how="diagonal_relaxed")
    vulnerability = build_vulnerability_panel(base_panel).with_columns(
        (pl.col("sample_version") + pl.lit("_negative_shock")).alias(
            "sample_version"
        )
    )
    panel = pl.concat([base_panel, vulnerability], how="diagonal_relaxed")
    panel = _apply_panel_bounds(panel, bounds_registry).with_columns(
        pl.col("outcome_id")
        .is_in([spec.outcome_id for spec in materialized_country])
        .alias("source_outcome_materialized"),
        pl.lit(gad_scaler_hash).alias("gad_scaler_hash"),
        pl.lit(bounds_registry.canonical_hash).alias("regression_bounds_hash"),
        pl.lit(giu_registry.canonical_hash).alias("giu_scaler_hash"),
    )
    contract = _load_contract(model_contract_path)
    panel = panel.select(*contract.columns).cast(
        {name: getattr(pl, dtype) for name, dtype in contract.columns.items()}
    ).sort(list(contract.primary_key))
    validate_model_panel(panel)
    destination = paths.analysis / "lp_panel.parquet"
    manifest = write_authoritative_table(
        panel,
        contract,
        destination,
        tuple(
            InputArtifact.from_path(path)
            for path in (
                gad_path,
                country_path,
                product_path,
                iv_path,
                wdi_path,
                provisional_path,
                final_path,
                registry_path,
                map_path,
                bounds_path,
                giu_scalers_path,
                model_contract_path,
            )
        ),
        build,
    )
    complete_case_rows = panel.filter(
        pl.col("sample_version").str.contains("complete_case")
        & ~pl.col("negative_shock_sample")
    ).height
    bounded_rows = panel.filter(
        pl.col("sample_version").str.contains("bounded_controls")
        & ~pl.col("negative_shock_sample")
    ).height
    return AnalysisPanelBuildReport(
        rows=panel.height,
        complete_case_rows=complete_case_rows,
        bounded_control_rows=bounded_rows,
        negative_shock_rows=panel.filter(pl.col("negative_shock_sample")).height,
        descriptive_rows=panel.filter(pl.col("descriptive_only")).height,
        outcomes=panel.get_column("outcome_id").n_unique(),
        economies=panel.get_column("economy_id").n_unique(),
        regression_bounds_hash=bounds_registry.canonical_hash,
        giu_scaler_hash=giu_registry.canonical_hash,
        gad_scaler_hash=gad_scaler_hash,
        output_path=str(destination),
        output_sha256=manifest.output_sha256,
        regression_bounds_path=str(bounds_manifest.destination),
        giu_scalers_path=str(giu_manifest.destination),
    )


def _numeric_mismatch(left: str, right: str, *, tolerance: float = 1e-12) -> pl.Expr:
    return (
        pl.col(left).is_null() != pl.col(right).is_null()
    ) | (
        pl.col(left).is_not_null()
        & pl.col(right).is_not_null()
        & ((pl.col(left) - pl.col(right)).abs() > tolerance)
    )


def _independent_rca_expectations(product: pl.DataFrame) -> pl.DataFrame:
    """Rebuild all h=3..8 RCA entries directly from product-year parent rows."""

    required = {"economy_id", "year", "hs6", "green_product_rca"}
    missing = sorted(required - set(product.columns))
    if missing:
        raise ValueError(f"RCA parent lacks columns: {missing}")
    source = product.select(
        pl.col("economy_id").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("hs6").cast(pl.String),
        pl.col("green_product_rca").cast(pl.Float64).alias("rca"),
    )
    if source.group_by("economy_id", "year", "hs6").len().filter(pl.col("len") > 1).height:
        raise ValueError("RCA parent has duplicate economy-year-product rows")
    frames: list[pl.DataFrame] = []
    baseline = source.filter(pl.col("rca").is_not_null() & (pl.col("rca") < 1.0))
    for horizon in range(3, 9):
        eligible = baseline.select(
            "economy_id",
            "hs6",
            (pl.col("year") + 1).cast(pl.Int16).alias("treatment_time"),
        )
        future = source.select(
            "economy_id",
            "hs6",
            (pl.col("year") - horizon).cast(pl.Int16).alias("treatment_time"),
            pl.col("rca").alias("future_rca"),
        )
        frames.append(
            eligible.join(
                future, on=["economy_id", "hs6", "treatment_time"], how="left"
            )
            .group_by("economy_id", "treatment_time")
            .agg(
                pl.len().cast(pl.UInt32).alias("expected_rca_eligible_products"),
                pl.col("future_rca").is_not_null().sum().cast(pl.UInt32).alias("__observed"),
                (pl.col("future_rca") >= 1.0).sum().cast(pl.UInt32).alias(
                    "expected_rca_entered_products"
                ),
            )
            .with_columns(
                pl.lit(horizon, dtype=pl.Int16).alias("horizon"),
                pl.when(pl.col("__observed") == pl.col("expected_rca_eligible_products"))
                .then(
                    pl.col("expected_rca_entered_products")
                    / pl.col("expected_rca_eligible_products")
                )
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias("expected_rca_rate"),
            )
            .drop("__observed")
        )
    return pl.concat(frames).sort("economy_id", "treatment_time", "horizon")


def _independent_giu_expectations(
    country: pl.DataFrame,
    product: pl.DataFrame,
    provisional: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Independently refit the frozen GIU registry and rebuild every parent value."""

    entry = _independent_rca_expectations(product).rename(
        {"expected_rca_rate": "future_green_rca_entry_rate"}
    )
    provisional_core = provisional.select("economy_id", "provisional_core").unique()
    components: list[pl.DataFrame] = []
    for horizon in range(3, 9):
        baseline = country.select(
            "economy_id",
            (pl.col("year") + 1).cast(pl.Int16).alias("treatment_time"),
            pl.col("green_export_complexity").alias("baseline_complexity"),
            pl.col("green_export_share").alias("baseline_share"),
        )
        future = country.select(
            "economy_id",
            (pl.col("year") - horizon).cast(pl.Int16).alias("treatment_time"),
            pl.col("green_export_complexity").alias("future_complexity"),
            pl.col("green_export_share").alias("future_share"),
        )
        changes = baseline.join(
            future, on=["economy_id", "treatment_time"], how="inner"
        ).with_columns(
            (pl.col("future_complexity") - pl.col("baseline_complexity")).alias(
                "green_export_complexity_change"
            ),
            (pl.col("future_share") - pl.col("baseline_share")).alias(
                "green_export_share_change"
            ),
        )
        components.append(
            entry.filter(pl.col("horizon") == horizon)
            .join(changes, on=["economy_id", "treatment_time"], how="left")
            .join(provisional_core, on="economy_id", how="left")
            .with_columns(pl.col("provisional_core").fill_null(False))
        )
    values = pl.concat(components, how="diagonal_relaxed")
    component_names = (
        "future_green_rca_entry_rate",
        "green_export_complexity_change",
        "green_export_share_change",
    )
    registry_records: dict[int, dict[str, dict[str, object]]] = {}
    for horizon in range(3, 9):
        baseline = values.filter(
            (pl.col("horizon") == horizon)
            & pl.col("treatment_time").is_between(2000, 2004)
            & pl.col("provisional_core")
            & pl.all_horizontal(
                [
                    pl.col(name).is_not_null() & pl.col(name).is_finite()
                    for name in component_names
                ]
            )
        )
        if baseline.is_empty():
            raise ValueError(f"independent GIU horizon {horizon} has no complete fit triplets")
        registry_records[horizon] = {}
        z_names: list[str] = []
        for name in component_names:
            series = baseline.get_column(name)
            center = float(series.median())
            mad = float((series - center).abs().median())
            if math.isfinite(mad) and mad > 0.0:
                scale = 1.4826 * mad
                method = "mad_1_4826"
            else:
                q25 = float(series.quantile(0.25, interpolation="linear"))
                q75 = float(series.quantile(0.75, interpolation="linear"))
                scale = (q75 - q25) / 1.349
                method = "iqr_div_1_349"
            if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"independent GIU scaler is unidentified: h={horizon}, {name}")
            registry_records[horizon][name] = {
                "center": center,
                "scale": scale,
                "scale_method": method,
                "row_count": baseline.height,
                "economy_count": baseline.get_column("economy_id").n_unique(),
            }
            z_name = f"__independent_z_{name}"
            z_names.append(z_name)
            values = values.with_columns(
                pl.when(pl.col("horizon") == horizon)
                .then((pl.col(name) - center) / scale)
                .otherwise(
                    pl.col(z_name)
                    if z_name in values.columns
                    else pl.lit(None, dtype=pl.Float64)
                )
                .alias(z_name)
            )
    payload = {
        "registry_id": "green_industrial_upgrading_outcome_scaler",
        "registry_version": "1.0.0",
        "fit_years": [2000, 2004],
        "horizons": [
            {"horizon": horizon, "components": registry_records[horizon]}
            for horizon in range(3, 9)
        ],
    }
    canonical_hash = _registry_hash(payload)
    scaler_rows = [
        {
            "horizon": horizon,
            "source_column": name,
            "fit_start_year": 2000,
            "fit_end_year": 2004,
            **record,
            "canonical_hash": canonical_hash,
        }
        for horizon, records in registry_records.items()
        for name, record in records.items()
    ]
    complete = pl.all_horizontal(
        [pl.col(name).is_not_null() & pl.col(name).is_finite() for name in component_names]
    )
    values = values.with_columns(
        pl.when(complete)
        .then(
            pl.mean_horizontal(
                [pl.col(f"__independent_z_{name}") for name in component_names]
            )
        )
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("expected_giu")
    )
    return (
        values.select("economy_id", "treatment_time", "horizon", "expected_giu"),
        pl.DataFrame(scaler_rows).with_columns(
            pl.col("horizon", "fit_start_year", "fit_end_year").cast(pl.Int16),
            pl.col("row_count", "economy_count").cast(pl.UInt32),
        ),
    )


def _audit_parent_derivations(
    panel: pl.DataFrame,
    giu_registry: pl.DataFrame,
    gad: pl.DataFrame,
    country: pl.DataFrame,
    product: pl.DataFrame,
    provisional: pl.DataFrame,
) -> dict[str, int]:
    """Compare published L4 derivations with independently rebuilt parent authorities."""

    entry = _independent_rca_expectations(product)
    rca = panel.filter(pl.col("outcome_id") == "future_green_rca_entry_rate").join(
        entry, on=["economy_id", "treatment_time", "horizon"], how="left"
    )
    rca_failures = rca.filter(
        pl.col("expected_rca_rate").is_null()
        | (pl.col("rca_eligible_products") != pl.col("expected_rca_eligible_products"))
        | (pl.col("rca_entered_products") != pl.col("expected_rca_entered_products"))
        | _numeric_mismatch("future_outcome", "expected_rca_rate")
        | _numeric_mismatch("delta_outcome", "expected_rca_rate")
    ).height

    expected_giu, expected_registry = _independent_giu_expectations(
        country, product, provisional
    )
    giu_panel = panel.filter(
        pl.col("outcome_id") == "green_industrial_upgrading_index"
    ).join(expected_giu, on=["economy_id", "treatment_time", "horizon"], how="left")
    giu_panel_failures = giu_panel.filter(
        pl.col("expected_giu").is_null()
        | _numeric_mismatch("future_outcome", "expected_giu")
        | _numeric_mismatch("delta_outcome", "expected_giu")
    ).height

    published = giu_registry.with_columns(pl.lit(True).alias("__published"))
    expected = expected_registry.rename(
        {
            name: f"expected__{name}"
            for name in expected_registry.columns
            if name not in {"horizon", "source_column"}
        }
    ).with_columns(pl.lit(True).alias("__expected"))
    scaler_check = published.join(
        expected, on=["horizon", "source_column"], how="full", coalesce=True
    )
    scaler_failures = scaler_check.filter(
        pl.col("__published").is_null()
        | pl.col("__expected").is_null()
        | _numeric_mismatch("center", "expected__center")
        | _numeric_mismatch("scale", "expected__scale")
        | (pl.col("scale_method") != pl.col("expected__scale_method"))
        | (pl.col("row_count") != pl.col("expected__row_count"))
        | (pl.col("economy_count") != pl.col("expected__economy_count"))
        | (pl.col("fit_start_year") != pl.col("expected__fit_start_year"))
        | (pl.col("fit_end_year") != pl.col("expected__fit_end_year"))
        | (pl.col("canonical_hash") != pl.col("expected__canonical_hash"))
    ).height
    expected_giu_hash = str(expected_registry.get_column("canonical_hash").first())
    giu_panel_failures += panel.filter(
        pl.col("giu_scaler_hash") != expected_giu_hash
    ).height

    current = gad.filter(pl.col("specification_id").is_in(["gad_core", "gad_lite"])).select(
        "economy_id",
        pl.col("year").alias("treatment_time"),
        pl.col("specification_id").alias("expected_current_specification"),
        pl.col("gimc").alias("expected_gimc"),
        pl.col("scaler_hash").alias("expected_gad_scaler_hash"),
    )
    lag = gad.select(
        "economy_id",
        pl.col("year").alias("gad_time"),
        pl.col("specification_id").alias("gad_version"),
        pl.col("gad").alias("expected_gad_lag"),
    )
    source_check = (
        panel.with_columns(
            pl.when(pl.col("sample_version").str.starts_with("lite_"))
            .then(pl.lit("gad_lite"))
            .otherwise(pl.lit("gad_core"))
            .alias("expected_current_specification")
        )
        .join(
            current,
            on=["economy_id", "treatment_time", "expected_current_specification"],
            how="left",
        )
        .join(lag, on=["economy_id", "gad_time", "gad_version"], how="left")
    )
    gimc_failures = source_check.filter(
        _numeric_mismatch("gimc", "expected_gimc")
        | _numeric_mismatch("gad_lag", "expected_gad_lag")
    ).height
    gad_scaler_failures = source_check.filter(
        pl.col("expected_gad_scaler_hash").is_null()
        | (pl.col("expected_gad_scaler_hash") != FROZEN_SCALER_HASH)
        | (pl.col("gad_scaler_hash") != pl.col("expected_gad_scaler_hash"))
    ).height
    return {
        "rca_parent_failures": rca_failures,
        "giu_scaler_parent_failures": scaler_failures,
        "giu_panel_parent_failures": giu_panel_failures,
        "gimc_source_failures": gimc_failures,
        "gad_scaler_source_failures": gad_scaler_failures,
    }


def audit_analysis_panel_authority(
    paths: ProjectPaths,
    *,
    audit_path: Path,
) -> dict[str, int | str]:
    """Audit L4 against source authorities rather than self-reported panel fields."""

    panel_path = paths.analysis / "lp_panel.parquet"
    bounds_path = paths.analysis / "regression_bounds.parquet"
    giu_path = paths.analysis / "giu_outcome_scalers.parquet"
    final_path = paths.harmonized / "sample/final_sample.parquet"
    panel_manifest = verify_manifest(panel_path.with_name(f"{panel_path.name}.manifest.json"))
    bounds_manifest = verify_manifest(bounds_path.with_name(f"{bounds_path.name}.manifest.json"))
    giu_manifest = verify_manifest(giu_path.with_name(f"{giu_path.name}.manifest.json"))
    verify_manifest(final_path.with_name(f"{final_path.name}.manifest.json"))
    panel = pl.read_parquet(panel_path)
    validation = validate_model_panel(panel)
    bounds = pl.read_parquet(bounds_path)
    giu = pl.read_parquet(giu_path)
    final_sample = pl.read_parquet(final_path)
    validate_final_sample(final_sample)
    gad_parent = pl.read_parquet(paths.measures / "gad/gad_country_year.parquet")
    country = pl.read_parquet(paths.measures / "outcomes/outcomes_country_year.parquet")
    product = pl.read_parquet(paths.measures / "outcomes/outcomes_product_year.parquet")
    provisional = pl.read_parquet(paths.harmonized / "sample/provisional_sample.parquet")
    parent_metrics = _audit_parent_derivations(
        panel, giu, gad_parent, country, product, provisional
    )
    bounds_hashes = bounds.get_column("canonical_hash").unique().to_list()
    giu_hashes = giu.get_column("canonical_hash").unique().to_list()
    gad_hashes = panel.get_column("gad_scaler_hash").unique().to_list()
    if len(bounds_hashes) != 1 or len(giu_hashes) != 1 or len(gad_hashes) != 1:
        raise RuntimeError("analysis panel registry hashes are not unique")
    bounds_hash = str(bounds_hashes[0])
    giu_hash = str(giu_hashes[0])
    gad_hash = str(gad_hashes[0])
    reconstructed_bounds = RegressionBoundsRegistry(
        bounds={
            str(row["source_column"]): RegressionBound(
                lower=float(row["lower_bound"]),
                upper=float(row["upper_bound"]),
                nonnull_fit_rows=int(row["nonnull_fit_rows"]),
            )
            for row in bounds.iter_rows(named=True)
        },
        percentiles=(
            int(bounds.get_column("lower_percentile").unique().item()),
            int(bounds.get_column("upper_percentile").unique().item()),
        ),
        fit_years=(
            int(bounds.get_column("fit_start_year").unique().item()),
            int(bounds.get_column("fit_end_year").unique().item()),
        ),
        fit_row_count=int(bounds.get_column("underlying_fit_rows").unique().item()),
        canonical_hash=bounds_hash,
    )
    giu_records: dict[int, GIUHorizonScaler] = {}
    for horizon in sorted(int(value) for value in giu.get_column("horizon").unique()):
        selected = giu.filter(pl.col("horizon") == horizon)
        giu_records[horizon] = GIUHorizonScaler(
            horizon=horizon,
            components={
                str(row["source_column"]): GIUComponentScaler(
                    center=float(row["center"]),
                    scale=float(row["scale"]),
                    scale_method=str(row["scale_method"]),
                    row_count=int(row["row_count"]),
                    economy_count=int(row["economy_count"]),
                )
                for row in selected.iter_rows(named=True)
            },
        )
    reconstructed_giu = GIUScalerRegistry(
        by_horizon=giu_records,
        fit_years=(
            int(giu.get_column("fit_start_year").unique().item()),
            int(giu.get_column("fit_end_year").unique().item()),
        ),
        canonical_hash=giu_hash,
    )
    hash_binding_failures = (
        panel.filter(pl.col("regression_bounds_hash") != bounds_hash).height
        + panel.filter(pl.col("giu_scaler_hash") != giu_hash).height
        + int(len({bounds_hash, giu_hash, gad_hash}) != 3)
        + int(gad_hash != FROZEN_SCALER_HASH)
        + int(_registry_hash(reconstructed_bounds.canonical_payload()) != bounds_hash)
        + int(_registry_hash(reconstructed_giu.canonical_payload()) != giu_hash)
        + int(set(giu_records) != set(range(3, 9)))
        + sum(len(record.components) != 3 for record in giu_records.values())
    )

    bound_records = {
        str(row["source_column"]): row for row in bounds.iter_rows(named=True)
    }
    clip_failures = 0
    common = {
        "gimc": "gimc_p01_p99",
        "Z": "Z_p01_p99",
    }
    for source, copy in common.items():
        record = bound_records[source]
        checked = panel.with_columns(
            pl.col(source)
            .clip(float(record["lower_bound"]), float(record["upper_bound"]))
            .alias("__expected")
        )
        clip_failures += checked.filter(_numeric_mismatch(copy, "__expected")).height
    for control in MAIN_CONTROL_COLUMNS:
        record = bound_records[control]
        copy = f"{control}_analysis_p01_p99"
        checked = panel.with_columns(
            pl.col(f"{control}_analysis")
            .clip(float(record["lower_bound"]), float(record["upper_bound"]))
            .alias("__expected")
        )
        clip_failures += checked.filter(_numeric_mismatch(copy, "__expected")).height
    for variant in MAPPED_GAD_VARIANTS:
        selected = panel.filter(pl.col("gad_version") == variant)
        for source, copy, registry_name in (
            ("gad_lag", "gad_lag_p01_p99", f"gad_lag__{variant}"),
            ("Z_GAD", "Z_GAD_p01_p99", f"Z_GAD__{variant}"),
            ("CMZ", "CMZ_p01_p99", f"CMZ__{variant}"),
        ):
            record = bound_records[registry_name]
            checked = selected.with_columns(
                pl.col(source)
                .clip(float(record["lower_bound"]), float(record["upper_bound"]))
                .alias("__expected")
            )
            clip_failures += checked.filter(_numeric_mismatch(copy, "__expected")).height

    gad_source = gad_parent.select(
        "economy_id",
        pl.col("year").alias("gad_time"),
        pl.col("specification_id").alias("gad_version"),
        pl.col("gad").alias("source_gad_lag"),
    )
    gad_check = panel.join(
        gad_source,
        on=["economy_id", "gad_time", "gad_version"],
        how="left",
    )
    gad_source_failures = gad_check.filter(
        _numeric_mismatch("gad_lag", "source_gad_lag")
    ).height

    iv_parent = pl.read_parquet(paths.measures / "instruments/iv_country_year.parquet")
    iv_source = iv_parent.filter(
        (pl.col("taxonomy_version") == "main_hs96")
        & (pl.col("share_version") == "main_0.0001")
    ).select(
        pl.col("importer").alias("economy_id"),
        pl.col("year").alias("treatment_time"),
        pl.col("gad_version"),
        pl.col("z").alias("source_Z"),
        pl.col("z_gad").alias("source_Z_GAD"),
        pl.col("cmz").alias("source_CMZ"),
        pl.col("confirmatory_iv_eligible").alias("source_iv_eligible"),
    )
    iv_check = panel.join(
        iv_source,
        on=["economy_id", "treatment_time", "gad_version"],
        how="left",
    )
    iv_source_failures = iv_check.filter(
        _numeric_mismatch("Z", "source_Z")
        | _numeric_mismatch("Z_GAD", "source_Z_GAD")
        | _numeric_mismatch("CMZ", "source_CMZ")
        | ~pl.col("source_iv_eligible").fill_null(False)
    ).height

    registry = pl.read_csv(paths.code_root / "02_数据字典/indicator_registry_v1.csv")
    approved = registry.filter(
        (pl.col("source_id") == "wdi")
        & (pl.col("role") == "control")
        & (pl.col("status") == "approved_control")
    ).select("source_field", "project_field")
    audit_columns_order = tuple(str(value) for value in approved["project_field"])
    if audit_columns_order != MAIN_CONTROL_COLUMNS:
        raise RuntimeError("audit found a changed approved-control registry")
    controls_raw = (
        pl.read_parquet(paths.normalized / "wdi/wdi_country_year.parquet")
        .join(approved, left_on="indicator_id", right_on="source_field", how="inner")
        .select("economy_id", "year", "project_field", "value")
        .pivot(on="project_field", index=["economy_id", "year"], values="value")
        .select("economy_id", "year", *MAIN_CONTROL_COLUMNS)
        .sort("economy_id", "year")
    )
    audit_regression = (
        gad_parent.filter(pl.col("specification_id") == "gad_core")
        .select("economy_id", "year", "gimc")
        .join(controls_raw, on=["economy_id", "year"], how="left")
        .join(final_sample.select("economy_id", "core_eligible"), on="economy_id", how="left")
        .filter(pl.col("core_eligible").fill_null(False))
        .drop("core_eligible")
    )
    audit_columns: list[str] = ["gimc", *MAIN_CONTROL_COLUMNS]
    for variant in MAPPED_GAD_VARIANTS:
        selected_iv = iv_parent.filter(
            (pl.col("taxonomy_version") == "main_hs96")
            & (pl.col("share_version") == "main_0.0001")
            & (pl.col("gad_version") == variant)
        ).select(
            pl.col("importer").alias("economy_id"),
            "year",
            pl.col("z").alias("Z") if variant == "gad_core" else pl.col("z").alias(f"Z__{variant}"),
            pl.col("gad_lag").alias(f"gad_lag__{variant}"),
            pl.col("z_gad").alias(f"Z_GAD__{variant}"),
            pl.col("cmz").alias(f"CMZ__{variant}"),
        )
        audit_regression = audit_regression.join(
            selected_iv, on=["economy_id", "year"], how="left"
        )
        if variant == "gad_core":
            audit_columns.append("Z")
        audit_columns.extend(
            [f"gad_lag__{variant}", f"Z_GAD__{variant}", f"CMZ__{variant}"]
        )
    fit = audit_regression.filter(pl.col("year").is_between(2000, 2022))
    bounds_fit_failures = int(fit.height != reconstructed_bounds.fit_row_count)
    for source in audit_columns:
        record = reconstructed_bounds.bounds[source]
        values = fit.filter(
            pl.col(source).is_not_null() & pl.col(source).is_finite()
        ).get_column(source)
        expected_lower = float(values.quantile(0.01, interpolation="linear"))
        expected_upper = float(values.quantile(0.99, interpolation="linear"))
        bounds_fit_failures += int(len(values) != record.nonnull_fit_rows)
        bounds_fit_failures += int(
            not math.isclose(expected_lower, record.lower, rel_tol=0.0, abs_tol=1e-12)
        )
        bounds_fit_failures += int(
            not math.isclose(expected_upper, record.upper, rel_tol=0.0, abs_tol=1e-12)
        )
    independently_interpolated = controls_raw.with_row_index("__row").sort(
        "economy_id", "year"
    )
    audit_flags: list[str] = []
    for control in MAIN_CONTROL_COLUMNS:
        prior = pl.col(control).shift(1).over("economy_id")
        following = pl.col(control).shift(-1).over("economy_id")
        prior_year = pl.col("year").shift(1).over("economy_id")
        following_year = pl.col("year").shift(-1).over("economy_id")
        flag_name = f"{control}_interpolated"
        flag = (
            pl.col(control).is_null()
            & prior.is_not_null()
            & prior.is_finite()
            & following.is_not_null()
            & following.is_finite()
            & (prior_year == pl.col("year") - 1)
            & (following_year == pl.col("year") + 1)
        )
        independently_interpolated = independently_interpolated.with_columns(
            flag.alias(flag_name),
            pl.when(flag)
            .then((prior + following) / 2.0)
            .otherwise(pl.col(control))
            .alias(f"{control}_analysis"),
        )
        audit_flags.append(flag_name)
    independently_interpolated = independently_interpolated.with_columns(
        pl.any_horizontal([pl.col(name) for name in audit_flags]).alias(
            "interpolated_control"
        )
    ).sort("__row").drop("__row")
    rebuilt_controls = independently_interpolated.select(
        "economy_id",
        pl.col("year").alias("control_time"),
        *MAIN_CONTROL_COLUMNS,
        *(f"{name}_analysis" for name in MAIN_CONTROL_COLUMNS),
        *(f"{name}_interpolated" for name in MAIN_CONTROL_COLUMNS),
        "interpolated_control",
    ).rename(
        {
            name: f"source_raw__{name}" for name in MAIN_CONTROL_COLUMNS
        }
        | {
            f"{name}_analysis": f"source_analysis__{name}"
            for name in MAIN_CONTROL_COLUMNS
        }
        | {
            f"{name}_interpolated": f"source_flag__{name}"
            for name in MAIN_CONTROL_COLUMNS
        }
        | {"interpolated_control": "source_interpolated_control"}
    )
    control_check = panel.join(
        rebuilt_controls, on=["economy_id", "control_time"], how="left"
    )
    interpolation_failures = 0
    for control in MAIN_CONTROL_COLUMNS:
        checked = control_check.with_columns(
            pl.when(pl.col("sample_version").str.contains("bounded_controls"))
            .then(pl.col(f"source_analysis__{control}"))
            .otherwise(pl.col(f"source_raw__{control}"))
            .alias("__expected_control")
        )
        interpolation_failures += checked.filter(
            _numeric_mismatch(control, f"source_raw__{control}")
            | _numeric_mismatch(f"{control}_analysis", "__expected_control")
        ).height
    # Compare the row/per-column flags separately; complete-case rows must never use a fill.
    for control in MAIN_CONTROL_COLUMNS:
        interpolation_failures += control_check.filter(
            pl.col(f"{control}_interpolated")
            != (
                pl.col(f"source_flag__{control}")
                & pl.col("sample_version").str.contains("bounded_controls")
            )
        ).height
    interpolation_failures += control_check.filter(
        pl.col("interpolated_control")
        != (
            pl.col("source_interpolated_control")
            & pl.col("sample_version").str.contains("bounded_controls")
        )
    ).height

    materialized_ids = {
        spec.outcome_id
        for spec in outcome_specs()
        if spec.materialized and spec.authority_table == "outcomes_country_year"
    }
    materialized = panel.filter(pl.col("outcome_id").is_in(materialized_ids))
    outcome_source_failures = 0
    for outcome in materialized.get_column("outcome_id").unique().to_list():
        selected = materialized.filter(pl.col("outcome_id") == outcome)
        baseline = country.select(
            "economy_id",
            pl.col("year").alias("baseline_outcome_time"),
            pl.col(str(outcome)).alias("source_baseline"),
        )
        future = country.select(
            "economy_id",
            pl.col("year").alias("outcome_time"),
            pl.col(str(outcome)).alias("source_future"),
        )
        checked = selected.join(
            baseline, on=["economy_id", "baseline_outcome_time"], how="left"
        ).join(future, on=["economy_id", "outcome_time"], how="left")
        outcome_source_failures += checked.filter(
            _numeric_mismatch("baseline_outcome", "source_baseline")
            | _numeric_mismatch("future_outcome", "source_future")
            | ((
                (
                    pl.col("future_outcome") - pl.col("baseline_outcome")
                )
                - pl.col("delta_outcome")
            ).abs() > 1e-12)
        ).height

    rca_double_difference_failures = panel.filter(
        (pl.col("outcome_id") == "future_green_rca_entry_rate")
        & (
            pl.col("baseline_outcome").is_not_null()
            | ((pl.col("future_outcome") - pl.col("delta_outcome")).abs() > 1e-12)
        )
    ).height
    vulnerability = panel.filter(pl.col("negative_shock_sample")).with_columns(
        pl.col("sample_version")
        .str.strip_suffix("_negative_shock")
        .alias("base_sample_version")
    )
    base = panel.filter(~pl.col("negative_shock_sample")).select(
        "economy_id",
        "treatment_time",
        "horizon",
        "outcome_id",
        "gad_version",
        pl.col("sample_version").alias("base_sample_version"),
        pl.col("delta_outcome").alias("base_delta_outcome"),
    )
    vulnerability_check = vulnerability.join(
        base,
        on=[
            "economy_id",
            "treatment_time",
            "horizon",
            "outcome_id",
            "gad_version",
            "base_sample_version",
        ],
        how="left",
    )
    vulnerability_failures = vulnerability_check.filter(
        (pl.col("Z") >= 0.0)
        | _numeric_mismatch("delta_outcome", "base_delta_outcome")
    ).height

    sample_flags = (
        panel.join(
            final_sample.select(
                "economy_id",
                pl.col("core_eligible").alias("source_core_eligible"),
                pl.col("lite_eligible").alias("source_lite_eligible"),
            ),
            on="economy_id",
            how="left",
        )
        .join(
            provisional.select(
                "economy_id",
                pl.col("provisional_core").alias("source_provisional_core"),
            ),
            on="economy_id",
            how="left",
        )
    )
    discriminator_authority_failures = sample_flags.filter(
        pl.col("source_core_eligible").is_null()
        | pl.col("source_lite_eligible").is_null()
        | pl.col("source_provisional_core").is_null()
        | (pl.col("core_eligible") != pl.col("source_core_eligible"))
        | (pl.col("lite_eligible") != pl.col("source_lite_eligible"))
        | (pl.col("provisional_core") != pl.col("source_provisional_core"))
    ).height

    flow = pl.read_csv(paths.audits / "最终样本流_v1.csv")
    flow_failures = flow.filter(
        ~pl.col("identity_valid")
        | (pl.col("entered") != pl.col("retained") + pl.col("excluded_at_stage"))
    ).height
    metrics = {
        **validation,
        "hash_binding_failures": hash_binding_failures,
        "bounds_fit_failures": bounds_fit_failures,
        "clip_failures": clip_failures,
        "gad_source_failures": gad_source_failures,
        "iv_source_failures": iv_source_failures,
        "control_interpolation_failures": interpolation_failures,
        "outcome_source_failures": outcome_source_failures,
        "rca_double_difference_failures": rca_double_difference_failures,
        "vulnerability_failures": vulnerability_failures,
        "sample_flow_failures": flow_failures,
        "discriminator_authority_failures": discriminator_authority_failures,
        **parent_metrics,
    }
    failures = sum(
        int(value)
        for key, value in metrics.items()
        if key != "rows" and key.endswith(("failures", "violations", "count", "columns", "keys"))
    )
    audit = pl.DataFrame(
        [
            {
                "metric": key,
                "value": int(value),
                "status": "pass" if key == "rows" or int(value) == 0 else "fail",
                "implementation_commit": panel_manifest.code_commit,
            }
            for key, value in sorted(metrics.items())
        ]
        + [
            {
                "metric": "panel_rows",
                "value": panel.height,
                "status": "pass",
                "implementation_commit": panel_manifest.code_commit,
            },
            {
                "metric": "complete_case_rows",
                "value": panel.filter(
                    pl.col("sample_version").str.contains("complete_case")
                    & ~pl.col("negative_shock_sample")
                ).height,
                "status": "pass",
                "implementation_commit": panel_manifest.code_commit,
            },
            {
                "metric": "bounded_control_rows",
                "value": panel.filter(
                    pl.col("sample_version").str.contains("bounded_controls")
                    & ~pl.col("negative_shock_sample")
                ).height,
                "status": "pass",
                "implementation_commit": panel_manifest.code_commit,
            },
        ]
    ).sort("metric")
    _write_csv_atomic(audit, audit_path)
    if failures:
        raise RuntimeError(f"analysis panel audit failed with {failures} violations")
    return {
        "status": "valid",
        "rows": panel.height,
        "duplicate_keys": validation["duplicate_keys"],
        "timing_violations": validation["timing_violations"],
        "mapping_overlap_violations": validation["mapping_overlap_violations"],
        "clip_failures": clip_failures,
        "control_interpolation_failures": interpolation_failures,
        "negative_shock_violations": vulnerability_failures,
        "regression_bounds_hash": bounds_hash,
        "giu_scaler_hash": giu_hash,
        "gad_scaler_hash": gad_hash,
        "panel_sha256": panel_manifest.output_sha256,
        "bounds_sha256": bounds_manifest.output_sha256,
        "giu_scalers_sha256": giu_manifest.output_sha256,
        "audit_path": str(audit_path),
    }
