"""Leakage-safe fixed-share instruments and cumulative mismatch shocks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    verify_manifest,
    write_authoritative_table,
)
from green_debt.config import load_construction_config, load_outcome_gad_map
from green_debt.paths import ProjectPaths
from green_debt.storage import measure_layer_usage, sha256_file


_PROHIBITED_IV_COLUMNS = (
    "current_import_share",
    "future_shock",
    "outcome_value",
)


@dataclass(frozen=True)
class BaselineShareResult:
    """Raw shares, retained normalized shares, and importer coverage."""

    raw: pl.DataFrame
    retained: pl.DataFrame
    coverage: pl.DataFrame


@dataclass(frozen=True)
class ShiftShareResult:
    """Long fixed-share contributions and their country-year aggregation."""

    contributions: pl.DataFrame
    country_year: pl.DataFrame


@dataclass(frozen=True)
class InstrumentBuildReport:
    """Published Task-15 artifact counts and locations."""

    baseline_rows: int
    shock_contribution_rows: int
    country_year_rows: int
    baseline_bytes: int
    shock_contribution_bytes: int
    country_year_bytes: int
    baseline_output_path: str
    shock_output_path: str
    country_year_output_path: str


def _normalize_partner_aliases(frame: pl.DataFrame) -> pl.DataFrame:
    aliases = {
        "importer_id": "importer",
        "exporter_id": "exporter",
        "economy_id": "importer",
    }
    renames = {
        source: target
        for source, target in aliases.items()
        if source in frame.columns and target not in frame.columns
    }
    return frame.rename(renames)


def _require_columns(frame: pl.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks columns: {missing}")


def validate_iv_input_columns(columns: Sequence[str]) -> None:
    """Fail closed on columns that can make fixed-share IV inputs endogenous."""

    lowered = {column.lower(): column for column in columns}
    for prohibited in _PROHIBITED_IV_COLUMNS:
        if prohibited in lowered:
            raise ValueError(f"prohibited IV lineage column: {lowered[prohibited]}")
    for column in columns:
        lower = column.lower()
        if lower.startswith("outcome_") or lower.endswith("_outcome"):
            raise ValueError(f"prohibited IV lineage column: {column}")


def _finite_nonnegative(frame: pl.DataFrame, column: str, label: str) -> None:
    invalid = frame.filter(
        pl.col(column).is_null()
        | ~pl.col(column).is_finite()
        | (pl.col(column) < 0.0)
    )
    if invalid.height:
        raise ValueError(f"{label} must be finite and nonnegative")


def build_baseline_shares(
    baseline: pl.DataFrame,
    *,
    minimum_share: float,
    minimum_coverage: float = 0.95,
    baseline_years: tuple[int, int] = (1996, 1999),
) -> BaselineShareResult:
    """Build time-invariant importer-supplier-product shares and coverage."""

    if not math.isfinite(minimum_share) or not 0.0 <= minimum_share <= 1.0:
        raise ValueError("minimum_share must lie in [0,1]")
    if not math.isfinite(minimum_coverage) or not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must lie in [0,1]")
    validate_iv_input_columns(baseline.columns)
    source = _normalize_partner_aliases(baseline)
    _require_columns(source, ("importer", "exporter", "hs6"), "baseline")

    if "mean_weighted_import_usd" in source.columns:
        _finite_nonnegative(source, "mean_weighted_import_usd", "baseline means")
        cells = source.group_by("importer", "exporter", "hs6").agg(
            pl.col("mean_weighted_import_usd").sum()
        )
    else:
        _require_columns(
            source,
            ("year", "weighted_green_trade_usd"),
            "annual baseline",
        )
        _finite_nonnegative(source, "weighted_green_trade_usd", "baseline trade")
        start, end = baseline_years
        expected = set(range(start, end + 1))
        observed = set(
            source.filter(pl.col("year").is_between(start, end))["year"].unique()
        )
        missing_years = sorted(expected - observed)
        if missing_years:
            raise ValueError(f"baseline annual partitions are missing: {missing_years}")
        count = end - start + 1
        cells = (
            source.filter(pl.col("year").is_between(start, end))
            .group_by("importer", "exporter", "hs6")
            .agg(
                (pl.col("weighted_green_trade_usd").sum() / count).alias(
                    "mean_weighted_import_usd"
                )
            )
        )

    cells = cells.filter(pl.col("mean_weighted_import_usd") > 0.0)
    totals = cells.group_by("importer").agg(
        pl.col("mean_weighted_import_usd")
        .sum()
        .alias("baseline_mean_total_weighted_import_usd")
    )
    raw = (
        cells.join(totals, on="importer", how="left", validate="m:1")
        .with_columns(
            (
                pl.col("mean_weighted_import_usd")
                / pl.col("baseline_mean_total_weighted_import_usd")
            ).alias("raw_baseline_share")
        )
        .with_columns((pl.col("raw_baseline_share") > minimum_share).alias("retained"))
        .sort("importer", "exporter", "hs6")
    )
    if raw.is_empty():
        raise ValueError("baseline has no positive importer-supplier-product cells")
    coverage = (
        raw.group_by("importer")
        .agg(
            pl.col("baseline_mean_total_weighted_import_usd").first(),
            pl.col("raw_baseline_share")
            .filter(pl.col("retained"))
            .sum()
            .alias("retained_coverage"),
            pl.len().alias("raw_cell_count"),
            pl.col("retained").sum().alias("retained_cell_count"),
        )
        .with_columns(
            (pl.col("retained_coverage") >= minimum_coverage).alias(
                "confirmatory_iv_eligible"
            )
        )
        .sort("importer")
    )
    retained = (
        raw.filter(pl.col("retained"))
        .join(
            coverage.select(
                "importer", "retained_coverage", "confirmatory_iv_eligible"
            ),
            on="importer",
            how="left",
            validate="m:1",
        )
        .with_columns(
            (pl.col("raw_baseline_share") / pl.col("retained_coverage")).alias(
                "baseline_share"
            )
        )
        .drop("retained")
        .sort("importer", "exporter", "hs6")
    )
    return BaselineShareResult(raw=raw, retained=retained, coverage=coverage)


def symmetric_growth(previous: float, current: float) -> float | None:
    """Return bounded symmetric growth, with double zero explicitly missing."""

    values = (float(previous), float(current))
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("symmetric growth levels must be finite and nonnegative")
    if values == (0.0, 0.0):
        return None
    return 2.0 * (values[1] - values[0]) / (values[1] + values[0])


def _growth_expr(current: str, previous: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(previous).is_null())
        .then(None)
        .when((pl.col(current) == 0.0) & (pl.col(previous) == 0.0))
        .then(None)
        .otherwise(
            2.0
            * (pl.col(current) - pl.col(previous))
            / (pl.col(current) + pl.col(previous))
        )
        .cast(pl.Float64)
        .alias(alias)
    )


def build_partner_shocks(
    trade: pl.DataFrame,
    *,
    cells: pl.DataFrame | None = None,
    expected_years: Sequence[int] | None = None,
    exporter_product_totals: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Construct exporter-product shocks after excluding the receiving destination."""

    validate_iv_input_columns(trade.columns)
    source = _normalize_partner_aliases(trade)
    _require_columns(
        source,
        ("year", "exporter", "importer", "hs6", "weighted_green_trade_usd"),
        "trade",
    )
    _finite_nonnegative(source, "weighted_green_trade_usd", "trade values")
    years = (
        tuple(int(year) for year in expected_years)
        if expected_years is not None
        else tuple(sorted(int(year) for year in source["year"].unique()))
    )
    if not years or len(years) != len(set(years)):
        raise ValueError("expected_years must be nonempty and unique")
    observed_years = set(int(year) for year in source["year"].unique())
    missing = sorted(set(years) - observed_years)
    if missing:
        raise ValueError(f"annual partitions are missing: {missing}")

    annual = source.group_by("year", "exporter", "importer", "hs6").agg(
        pl.col("weighted_green_trade_usd").sum()
    )
    if exporter_product_totals is None:
        totals = annual.group_by("year", "exporter", "hs6").agg(
            pl.col("weighted_green_trade_usd")
            .sum()
            .alias("all_destination_exports")
        )
    else:
        validate_iv_input_columns(exporter_product_totals.columns)
        totals_source = exporter_product_totals
        _require_columns(
            totals_source,
            ("year", "exporter", "hs6", "weighted_green_trade_usd"),
            "exporter-product totals",
        )
        _finite_nonnegative(
            totals_source,
            "weighted_green_trade_usd",
            "exporter-product totals",
        )
        totals_missing = sorted(
            set(years) - set(int(year) for year in totals_source["year"].unique())
        )
        if totals_missing:
            raise ValueError(
                f"exporter-product annual partitions are missing: {totals_missing}"
            )
        totals = totals_source.group_by("year", "exporter", "hs6").agg(
            pl.col("weighted_green_trade_usd")
            .sum()
            .alias("all_destination_exports")
        )
    if cells is None:
        destinations = annual.select("importer").unique()
        exporter_products = annual.select("exporter", "hs6").unique()
        requested = destinations.join(exporter_products, how="cross")
    else:
        validate_iv_input_columns(cells.columns)
        requested = _normalize_partner_aliases(cells)
        _require_columns(requested, ("importer", "exporter", "hs6"), "shock cells")
        requested = requested.select("importer", "exporter", "hs6").unique()
        available_exporters = set(totals["exporter"].unique())
        available_products = set(totals["hs6"].unique())
        unmapped_exporters = sorted(set(requested["exporter"].unique()) - available_exporters)
        unmapped_products = sorted(set(requested["hs6"].unique()) - available_products)
        if unmapped_exporters or unmapped_products:
            raise ValueError(
                "shock cells contain unmapped keys: "
                f"exporters={unmapped_exporters}, products={unmapped_products}"
            )

    year_frame = pl.DataFrame({"year": years}, schema={"year": pl.Int64})
    requested = requested.with_columns(
        pl.col("importer").alias("destination_excluded")
    ).drop("importer")
    grid = requested.join(year_frame, how="cross")
    all_exporter_product = totals
    own = annual.select(
        "year",
        "exporter",
        pl.col("importer").alias("destination_excluded"),
        "hs6",
        pl.col("weighted_green_trade_usd").alias("exports_to_destination"),
    )
    all_product = totals.group_by("year", "hs6").agg(
        pl.col("all_destination_exports").sum().alias("all_exporter_product_exports")
    )
    product_to_destination = annual.group_by("year", "importer", "hs6").agg(
        pl.col("weighted_green_trade_usd")
        .sum()
        .alias("product_exports_to_destination")
    ).rename({"importer": "destination_excluded"})
    levels = (
        grid.join(
            all_exporter_product, on=("year", "exporter", "hs6"), how="left"
        )
        .join(
            own,
            on=("year", "exporter", "destination_excluded", "hs6"),
            how="left",
        )
        .join(all_product, on=("year", "hs6"), how="left")
        .join(
            product_to_destination,
            on=("year", "destination_excluded", "hs6"),
            how="left",
        )
        .with_columns(
            pl.col("all_destination_exports").fill_null(0.0),
            pl.col("exports_to_destination").fill_null(0.0),
            pl.col("all_exporter_product_exports").fill_null(0.0),
            pl.col("product_exports_to_destination").fill_null(0.0),
        )
        .with_columns(
            (
                pl.col("all_destination_exports")
                - pl.col("exports_to_destination")
            ).alias("exports_excluding_destination"),
            (
                pl.col("all_exporter_product_exports")
                - pl.col("product_exports_to_destination")
            ).alias("global_product_exports_excluding_destination"),
        )
    )
    lagged = levels.select(
        "destination_excluded",
        "exporter",
        "hs6",
        (pl.col("year") + 1).alias("year"),
        pl.col("exports_excluding_destination").alias(
            "exports_excluding_destination_lag"
        ),
        pl.col("global_product_exports_excluding_destination").alias(
            "global_product_exports_excluding_destination_lag"
        ),
    )
    out = (
        levels.join(
            lagged,
            on=("destination_excluded", "exporter", "hs6", "year"),
            how="left",
        )
        .with_columns(
            _growth_expr(
                "exports_excluding_destination",
                "exports_excluding_destination_lag",
                "exporter_growth_excluding_destination",
            ),
            _growth_expr(
                "global_product_exports_excluding_destination",
                "global_product_exports_excluding_destination_lag",
                "global_product_growth_excluding_destination",
            ),
        )
        .with_columns(
            (
                pl.col("exporter_growth_excluding_destination")
                - pl.col("global_product_growth_excluding_destination")
            ).alias("partner_shock"),
            pl.when(pl.col("exports_excluding_destination_lag").is_null())
            .then(pl.lit("prior_year_not_available"))
            .when(
                (pl.col("exports_excluding_destination") == 0.0)
                & (pl.col("exports_excluding_destination_lag") == 0.0)
            )
            .then(pl.lit("both_adjacent_exporter_levels_zero"))
            .when(
                (pl.col("global_product_exports_excluding_destination") == 0.0)
                & (
                    pl.col("global_product_exports_excluding_destination_lag")
                    == 0.0
                )
            )
            .then(pl.lit("both_adjacent_global_product_levels_zero"))
            .otherwise(None)
            .alias("shock_missing_reason"),
        )
        .select(
            "destination_excluded",
            "exporter",
            "hs6",
            "year",
            "exports_excluding_destination_lag",
            "exports_excluding_destination",
            "global_product_exports_excluding_destination_lag",
            "global_product_exports_excluding_destination",
            "exporter_growth_excluding_destination",
            "global_product_growth_excluding_destination",
            "partner_shock",
            "shock_missing_reason",
        )
        .sort("destination_excluded", "exporter", "hs6", "year")
    )
    return out


def aggregate_shift_share(
    retained_weights: pl.DataFrame,
    shocks: pl.DataFrame,
    *,
    expected_years: Sequence[int],
    weight_tolerance: float = 1e-12,
) -> ShiftShareResult:
    """Apply every fixed retained weight, preserving missing shocks without reweighting."""

    validate_iv_input_columns(retained_weights.columns)
    validate_iv_input_columns(shocks.columns)
    weights = _normalize_partner_aliases(retained_weights)
    _require_columns(
        weights,
        ("importer", "exporter", "hs6", "baseline_share"),
        "retained weights",
    )
    _require_columns(
        shocks,
        (
            "destination_excluded",
            "exporter",
            "hs6",
            "year",
            "partner_shock",
        ),
        "partner shocks",
    )
    if "share_version" not in weights.columns:
        weights = weights.with_columns(pl.lit("main_0.0001").alias("share_version"))
    if "confirmatory_iv_eligible" not in weights.columns:
        weights = weights.with_columns(
            pl.lit(True).alias("confirmatory_iv_eligible")
        )
    sums = weights.group_by("importer", "share_version").agg(
        pl.col("baseline_share").sum().alias("weight_sum")
    )
    bad = sums.filter((pl.col("weight_sum") - 1.0).abs() > weight_tolerance)
    if bad.height:
        raise ValueError("retained baseline shares are not normalized to one")
    if weights.group_by("importer", "exporter", "hs6", "share_version").len().filter(
        pl.col("len") > 1
    ).height:
        raise ValueError("retained weights contain duplicate cells")

    years = pl.DataFrame({"year": [int(year) for year in expected_years]})
    grid = weights.join(years, how="cross")
    shock_keys = shocks.select(
        "destination_excluded",
        "exporter",
        "hs6",
        "year",
        "partner_shock",
        *(["shock_missing_reason"] if "shock_missing_reason" in shocks.columns else []),
    )
    if "shock_missing_reason" not in shock_keys.columns:
        shock_keys = shock_keys.with_columns(
            pl.lit(None, dtype=pl.String).alias("shock_missing_reason")
        )
    if shock_keys.group_by(
        "destination_excluded", "exporter", "hs6", "year"
    ).len().filter(pl.col("len") > 1).height:
        raise ValueError("partner shocks contain duplicate keys")
    contributions = (
        grid.join(
            shock_keys,
            left_on=("importer", "exporter", "hs6", "year"),
            right_on=("destination_excluded", "exporter", "hs6", "year"),
            how="left",
        )
        .with_columns(pl.col("importer").alias("destination_excluded"))
        .with_columns(
            pl.when(pl.col("partner_shock").is_not_null())
            .then(pl.col("baseline_share") * pl.col("partner_shock"))
            .otherwise(None)
            .alias("contribution"),
            pl.when(pl.col("partner_shock").is_null())
            .then(pl.col("shock_missing_reason").fill_null("shock_row_missing"))
            .otherwise(None)
            .alias("contribution_missing_reason"),
        )
        .sort("importer", "share_version", "year", "exporter", "hs6")
    )
    grouped = contributions.group_by("importer", "share_version", "year").agg(
        pl.len().alias("retained_cell_count"),
        pl.col("partner_shock").is_not_null().sum().alias("observed_shock_count"),
        pl.col("baseline_share")
        .filter(pl.col("partner_shock").is_not_null())
        .sum()
        .alias("observed_baseline_share"),
        pl.col("contribution").sum().alias("available_contribution_sum"),
        pl.col("confirmatory_iv_eligible").all().alias("baseline_confirmatory_eligible"),
    )
    country_year = (
        grouped.with_columns(
            pl.when(pl.col("observed_shock_count") == pl.col("retained_cell_count"))
            .then(pl.col("available_contribution_sum"))
            .otherwise(None)
            .alias("z"),
            pl.when(pl.col("observed_shock_count") < pl.col("retained_cell_count"))
            .then(pl.lit("retained_shock_missing"))
            .otherwise(None)
            .alias("z_missing_reason"),
        )
        .with_columns(
            (
                pl.col("baseline_confirmatory_eligible")
                & pl.col("z").is_not_null()
            ).alias("confirmatory_iv_eligible")
        )
        .drop("available_contribution_sum")
        .sort("importer", "share_version", "year")
    )
    return ShiftShareResult(contributions=contributions, country_year=country_year)


def _z_with_versions(z: pl.DataFrame, versions: Sequence[str]) -> pl.DataFrame:
    source = _normalize_partner_aliases(z)
    _require_columns(source, ("importer", "year", "z"), "Z")
    if "share_version" not in source.columns:
        source = source.with_columns(pl.lit("main_0.0001").alias("share_version"))
    return source.join(
        pl.DataFrame({"gad_version": list(dict.fromkeys(versions))}), how="cross"
    )


def build_z_gad_interactions(
    z: pl.DataFrame,
    gad: pl.DataFrame,
    *,
    gad_versions: Sequence[str],
) -> pl.DataFrame:
    """Interact current Z with the requested version's strictly lagged GAD."""

    validate_iv_input_columns(z.columns)
    validate_iv_input_columns(gad.columns)
    _require_columns(gad, ("economy_id", "year", "specification_id", "gad"), "GAD")
    requested = tuple(dict.fromkeys(gad_versions))
    available = set(gad["specification_id"].unique())
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"requested GAD versions are absent: {missing}")
    grid = _z_with_versions(z, requested)
    lagged = gad.filter(pl.col("specification_id").is_in(requested)).select(
        pl.col("economy_id").alias("importer"),
        (pl.col("year") + 1).alias("year"),
        pl.col("year").alias("gad_time"),
        pl.col("specification_id").alias("gad_version"),
        pl.col("gad").alias("gad_lag"),
    )
    return (
        grid.join(
            lagged,
            on=("importer", "year", "gad_version"),
            how="left",
            validate="m:1",
        )
        .with_columns(
            (pl.col("z") * pl.col("gad_lag")).alias("z_gad"),
            pl.when(pl.col("z").is_null())
            .then(pl.lit("z_missing"))
            .when(pl.col("gad_lag").is_null())
            .then(pl.lit("lagged_gad_missing"))
            .otherwise(None)
            .alias("z_gad_missing_reason"),
        )
        .sort("importer", "share_version", "year", "gad_version")
    )


def _midranks(values: list[tuple[str, float]]) -> dict[str, float] | None:
    if len(values) < 2:
        return None
    ordered = sorted(values, key=lambda item: (item[1], item[0]))
    result: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        scaled = (average_rank - 1.0) / (len(ordered) - 1.0)
        for position in range(index, end):
            result[ordered[position][0]] = scaled
        index = end
    return result


def build_cmz(
    z: pl.DataFrame,
    gad: pl.DataFrame,
    *,
    gad_versions: Sequence[str],
    lags: int = 5,
) -> pl.DataFrame:
    """Build complete-window CMZ using lagged absorption ranks for each GAD version."""

    if lags != 5:
        raise ValueError("confirmatory CMZ requires exactly five terms")
    validate_iv_input_columns(z.columns)
    validate_iv_input_columns(gad.columns)
    source_z = _normalize_partner_aliases(z)
    _require_columns(source_z, ("importer", "year", "z"), "Z")
    _require_columns(
        gad,
        ("economy_id", "year", "specification_id", "absorption", "rho"),
        "GAD absorption",
    )
    if "share_version" not in source_z.columns:
        source_z = source_z.with_columns(
            pl.lit("main_0.0001").alias("share_version")
        )
    versions = tuple(dict.fromkeys(gad_versions))
    available = set(gad["specification_id"].unique())
    missing = sorted(set(versions) - available)
    if missing:
        raise ValueError(f"requested GAD versions are absent: {missing}")

    eligible_column = (
        "confirmatory_eligible" if "confirmatory_eligible" in gad.columns else None
    )
    gad_rows: dict[tuple[str, int, str], dict[str, object]] = {}
    rank_inputs: dict[tuple[int, str], list[tuple[str, float]]] = {}
    for row in gad.filter(pl.col("specification_id").is_in(versions)).iter_rows(
        named=True
    ):
        economy = str(row["economy_id"])
        year = int(row["year"])
        version = str(row["specification_id"])
        gad_rows[(economy, year, version)] = row
        absorption = row["absorption"]
        eligible = bool(row[eligible_column]) if eligible_column else True
        if (
            eligible
            and absorption is not None
            and math.isfinite(float(absorption))
        ):
            rank_inputs.setdefault((year, version), []).append(
                (economy, float(absorption))
            )
    ranks: dict[tuple[str, int, str], float] = {}
    for (year, version), values in rank_inputs.items():
        year_ranks = _midranks(values)
        if year_ranks is not None:
            for economy, rank in year_ranks.items():
                ranks[(economy, year, version)] = rank

    z_values = {
        (
            str(row["importer"]),
            str(row["share_version"]),
            int(row["year"]),
        ): row["z"]
        for row in source_z.iter_rows(named=True)
    }
    output: list[dict[str, object]] = []
    keys = source_z.select("importer", "share_version", "year").unique().sort(
        "importer", "share_version", "year"
    )
    for target in keys.iter_rows(named=True):
        economy = str(target["importer"])
        share_version = str(target["share_version"])
        year = int(target["year"])
        for version in versions:
            terms: list[float] = []
            for lag in range(lags):
                shock = z_values.get((economy, share_version, year - lag))
                absorption_year = year - lag - 1
                rank = ranks.get((economy, absorption_year, version))
                gad_row = gad_rows.get((economy, absorption_year, version))
                rho = gad_row.get("rho") if gad_row is not None else None
                if (
                    shock is None
                    or rank is None
                    or rho is None
                    or not math.isfinite(float(shock))
                    or not math.isfinite(float(rho))
                ):
                    continue
                terms.append(
                    float(rho) ** lag * float(shock) * (1.0 - float(rank))
                )
            complete = len(terms)
            output.append(
                {
                    "importer": economy,
                    "share_version": share_version,
                    "year": year,
                    "gad_version": version,
                    "cmz": sum(terms) if complete == lags else None,
                    "cmz_complete_terms": complete,
                    "absorption_latest_time": year - 1,
                    "cmz_missing_reason": (
                        None
                        if complete == lags
                        else "fewer_than_five_complete_terms"
                    ),
                }
            )
    return pl.DataFrame(output, schema_overrides={"cmz": pl.Float64}).sort(
        "importer", "share_version", "year", "gad_version"
    )


_CONTRACT_ROOT = Path(__file__).resolve().parents[2] / "contracts"
_ROOT = Path(__file__).resolve().parents[3]
_CONSTRUCTION_CONFIG = _ROOT / "config/construction.yaml"
_OUTCOME_GAD_MAP = _ROOT / "config/outcome_gad_map.yaml"
_YEARS = tuple(range(1996, 2025))
_SHARE_VERSIONS = (
    ("all_cells", 0.0),
    ("main_0.0001", 0.0001),
    ("robustness_0.0005", 0.0005),
)


def _contract(name: str) -> TableContract:
    payload = json.loads((_CONTRACT_ROOT / name).read_text(encoding="utf-8"))
    period = payload.get("period")
    return TableContract(
        table_id=str(payload["table_id"]),
        schema_version=str(payload["schema_version"]),
        primary_key=tuple(str(value) for value in payload["primary_key"]),
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
        transformations=tuple(
            str(value) for value in payload.get("transformations", [])
        ),
    )


def _annual_paths(root: Path, layer: str, taxonomy: str) -> tuple[Path, ...]:
    paths = tuple(
        root
        / layer
        / f"year={year}"
        / f"taxonomy_version={taxonomy}.parquet"
        for year in _YEARS
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required annual partitions are missing: {missing}")
    for path in paths:
        verify_manifest(path.with_name(f"{path.name}.manifest.json"))
    return paths


def _baseline_authority(
    trade: pl.DataFrame,
    approved_importers: pl.Series,
) -> pl.DataFrame:
    return trade.filter(
        pl.col("year").is_between(1996, 1999)
        & pl.col("importer").is_in(approved_importers)
    )


def _baseline_version_table(
    result: BaselineShareResult,
    *,
    taxonomy: str,
    share_version: str,
) -> pl.DataFrame:
    normalized = result.retained.select(
        "importer", "exporter", "hs6", "baseline_share"
    )
    return (
        result.raw.join(
            result.coverage.rename(
                {"confirmatory_iv_eligible": "coverage_eligible"}
            ),
            on="importer",
            how="left",
            validate="m:1",
        )
        .join(
            normalized,
            on=("importer", "exporter", "hs6"),
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.lit(taxonomy).alias("taxonomy_version"),
            pl.lit(share_version).alias("share_version"),
            pl.lit(share_version == "main_0.0001").alias(
                "confirmatory_specification"
            ),
        )
        .with_columns(
            (
                pl.col("confirmatory_specification")
                & pl.col("coverage_eligible")
            ).alias("confirmatory_baseline_eligible")
        )
        .select(
            "taxonomy_version",
            "share_version",
            "importer",
            "exporter",
            "hs6",
            "mean_weighted_import_usd",
            "baseline_mean_total_weighted_import_usd",
            "raw_baseline_share",
            "retained",
            "retained_coverage",
            "baseline_share",
            pl.col("raw_cell_count").cast(pl.UInt32),
            pl.col("retained_cell_count").cast(pl.UInt32),
            "coverage_eligible",
            "confirmatory_specification",
            "confirmatory_baseline_eligible",
        )
        .sort("taxonomy_version", "share_version", "importer", "exporter", "hs6")
    )


def _retained_for_aggregation(table: pl.DataFrame) -> pl.DataFrame:
    return (
        table.filter(pl.col("retained"))
        .select(
            "share_version",
            "importer",
            "exporter",
            "hs6",
            "baseline_share",
            pl.col("confirmatory_baseline_eligible").alias(
                "confirmatory_iv_eligible"
            ),
        )
        .sort("importer", "share_version", "exporter", "hs6")
    )


def _build_long_contributions(
    retained: pl.DataFrame,
    shocks: pl.DataFrame,
    *,
    taxonomy: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    aggregate = aggregate_shift_share(retained, shocks, expected_years=_YEARS)
    detail = shocks.select(
        "destination_excluded",
        "exporter",
        "hs6",
        "year",
        "exports_excluding_destination_lag",
        "exports_excluding_destination",
        "global_product_exports_excluding_destination_lag",
        "global_product_exports_excluding_destination",
        "exporter_growth_excluding_destination",
        "global_product_growth_excluding_destination",
    )
    long = (
        aggregate.contributions.join(
            detail,
            on=("destination_excluded", "exporter", "hs6", "year"),
            how="left",
            validate="m:1",
        )
        .with_columns(pl.lit(taxonomy).alias("taxonomy_version"))
        .select(
            "taxonomy_version",
            "share_version",
            "importer",
            "destination_excluded",
            "exporter",
            "hs6",
            pl.col("year").cast(pl.Int16),
            "baseline_share",
            "exports_excluding_destination_lag",
            "exports_excluding_destination",
            "global_product_exports_excluding_destination_lag",
            "global_product_exports_excluding_destination",
            "exporter_growth_excluding_destination",
            "global_product_growth_excluding_destination",
            "partner_shock",
            "shock_missing_reason",
            "contribution",
            "contribution_missing_reason",
        )
        .sort(
            "taxonomy_version",
            "share_version",
            "importer",
            "exporter",
            "hs6",
            "year",
        )
    )
    return long, aggregate.country_year


def _build_country_year(
    z: pl.DataFrame,
    gad: pl.DataFrame,
    *,
    taxonomy: str,
    gad_versions: Sequence[str],
) -> pl.DataFrame:
    interaction = build_z_gad_interactions(z, gad, gad_versions=gad_versions)
    cmz = build_cmz(z, gad, gad_versions=gad_versions, lags=5)
    return (
        interaction.join(
            cmz,
            on=("importer", "share_version", "year", "gad_version"),
            how="left",
            validate="1:1",
        )
        .with_columns(pl.lit(taxonomy).alias("taxonomy_version"))
        .select(
            "taxonomy_version",
            "share_version",
            "importer",
            pl.col("year").cast(pl.Int16),
            "gad_version",
            pl.col("retained_cell_count").cast(pl.UInt32),
            pl.col("observed_shock_count").cast(pl.UInt32),
            "observed_baseline_share",
            "baseline_confirmatory_eligible",
            "z",
            "z_missing_reason",
            "confirmatory_iv_eligible",
            pl.col("gad_time").cast(pl.Int16),
            "gad_lag",
            "z_gad",
            "z_gad_missing_reason",
            "cmz",
            pl.col("cmz_complete_terms").cast(pl.UInt8),
            pl.col("absorption_latest_time").cast(pl.Int16),
            "cmz_missing_reason",
        )
        .sort(
            "taxonomy_version", "share_version", "importer", "year", "gad_version"
        )
    )


def build_instrument_artifacts(
    paths: ProjectPaths,
    *,
    taxonomy: str,
    build: BuildIdentity,
) -> InstrumentBuildReport:
    """Build and publish fixed shares, long shock contributions, and versioned IVs."""

    if taxonomy != "main_hs96":
        raise ValueError("Task-15 instruments require taxonomy main_hs96")
    construction = load_construction_config(_CONSTRUCTION_CONFIG)
    if (
        construction.iv.baseline_years != (1996, 1999)
        or construction.iv.minimum_cell_share != 0.0001
        or construction.iv.robustness_cell_share != 0.0005
        or construction.iv.minimum_retained_coverage != 0.95
        or construction.iv.cmz_lags != 5
    ):
        raise ValueError("IV construction config differs from the frozen Task-15 design")

    bilateral_paths = _annual_paths(
        paths.normalized / "baci", "green_bilateral", taxonomy
    )
    total_paths = _annual_paths(
        paths.normalized / "baci", "green_economy_product", taxonomy
    )
    sample_path = paths.harmonized / "sample/provisional_sample.parquet"
    gad_path = paths.measures / "gad/gad_country_year.parquet"
    verify_manifest(sample_path.with_name(f"{sample_path.name}.manifest.json"))
    verify_manifest(gad_path.with_name(f"{gad_path.name}.manifest.json"))
    sample = pl.read_parquet(sample_path)
    approved = sample.filter(pl.col("provisional_core")).get_column("economy_id")
    if approved.is_empty():
        raise ValueError("approved provisional-core baseline is empty")

    trade = (
        pl.scan_parquet(bilateral_paths)
        .select(
            pl.col("year").cast(pl.Int64),
            pl.col("exporter_id").alias("exporter"),
            pl.col("importer_id").alias("importer"),
            "hs6",
            "weighted_green_trade_usd",
        )
        .collect(engine="streaming")
    )
    totals = (
        pl.scan_parquet(total_paths)
        .filter(pl.col("flow_role") == "exporter")
        .select(
            pl.col("year").cast(pl.Int64),
            pl.col("economy_id").alias("exporter"),
            "hs6",
            "weighted_green_trade_usd",
        )
        .collect(engine="streaming")
    )
    baseline_input = _baseline_authority(trade, approved)
    versions: list[pl.DataFrame] = []
    for share_version, threshold in _SHARE_VERSIONS:
        result = build_baseline_shares(
            baseline_input,
            minimum_share=threshold,
            minimum_coverage=construction.iv.minimum_retained_coverage,
            baseline_years=construction.iv.baseline_years,
        )
        versions.append(
            _baseline_version_table(
                result, taxonomy=taxonomy, share_version=share_version
            )
        )
    baseline = pl.concat(versions).sort(
        "taxonomy_version", "share_version", "importer", "exporter", "hs6"
    )
    retained = _retained_for_aggregation(baseline)
    all_cells = retained.filter(pl.col("share_version") == "all_cells").select(
        "importer", "exporter", "hs6"
    )
    shocks = build_partner_shocks(
        trade,
        cells=all_cells,
        expected_years=_YEARS,
        exporter_product_totals=totals,
    )
    long, z = _build_long_contributions(retained, shocks, taxonomy=taxonomy)
    gad = pl.read_parquet(gad_path)
    outcome_map = load_outcome_gad_map(_OUTCOME_GAD_MAP)
    gad_versions = tuple(sorted(set(outcome_map.outcomes.values())))
    country_year = _build_country_year(
        z,
        gad,
        taxonomy=taxonomy,
        gad_versions=gad_versions,
    )

    output_root = paths.measures / "instruments"
    baseline_path = output_root / "iv_baseline_shares.parquet"
    shock_path = output_root / "iv_partner_shocks.parquet"
    country_path = output_root / "iv_country_year.parquet"
    baseline_inputs = (
        *(InputArtifact.from_path(path) for path in bilateral_paths),
        InputArtifact.from_path(sample_path),
        InputArtifact.from_path(_CONSTRUCTION_CONFIG),
    )
    baseline_manifest = write_authoritative_table(
        baseline,
        _contract("iv_baseline_shares.json"),
        baseline_path,
        (
            *baseline_inputs,
            InputArtifact.from_path(_CONTRACT_ROOT / "iv_baseline_shares.json"),
        ),
        build,
    )
    shock_manifest = write_authoritative_table(
        long,
        _contract("iv_partner_shocks.json"),
        shock_path,
        (
            InputArtifact.from_path(baseline_path),
            *(InputArtifact.from_path(path) for path in bilateral_paths),
            *(InputArtifact.from_path(path) for path in total_paths),
            InputArtifact.from_path(_CONSTRUCTION_CONFIG),
            InputArtifact.from_path(_CONTRACT_ROOT / "iv_partner_shocks.json"),
        ),
        build,
    )
    country_manifest = write_authoritative_table(
        country_year,
        _contract("iv_country_year.json"),
        country_path,
        (
            InputArtifact.from_path(baseline_path),
            InputArtifact.from_path(shock_path),
            InputArtifact.from_path(gad_path),
            InputArtifact.from_path(_CONSTRUCTION_CONFIG),
            InputArtifact.from_path(_OUTCOME_GAD_MAP),
            InputArtifact.from_path(_CONTRACT_ROOT / "iv_country_year.json"),
        ),
        build,
    )
    return InstrumentBuildReport(
        baseline_rows=baseline_manifest.rows,
        shock_contribution_rows=shock_manifest.rows,
        country_year_rows=country_manifest.rows,
        baseline_bytes=baseline_manifest.bytes,
        shock_contribution_bytes=shock_manifest.bytes,
        country_year_bytes=country_manifest.bytes,
        baseline_output_path=str(baseline_path),
        shock_output_path=str(shock_path),
        country_year_output_path=str(country_path),
    )


def _nullable_float_failures(
    frame: pl.DataFrame, left: str, right: str, *, tolerance: float = 1e-12
) -> int:
    return frame.filter(
        (pl.col(left).is_null() != pl.col(right).is_null())
        | (
            pl.col(left).is_not_null()
            & pl.col(right).is_not_null()
            & ((pl.col(left) - pl.col(right)).abs() > tolerance)
        )
    ).height


def _float_matches(left: object, right: object, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= tolerance


def _independent_baseline_metrics(baseline: pl.DataFrame) -> dict[str, int]:
    _require_columns(
        baseline,
        (
            "share_version",
            "importer",
            "mean_weighted_import_usd",
            "baseline_mean_total_weighted_import_usd",
            "raw_baseline_share",
            "retained",
            "retained_coverage",
            "baseline_share",
            "raw_cell_count",
            "retained_cell_count",
            "coverage_eligible",
            "confirmatory_specification",
            "confirmatory_baseline_eligible",
        ),
        "baseline audit",
    )
    thresholds = pl.DataFrame(
        {
            "share_version": [version for version, _ in _SHARE_VERSIONS],
            "minimum_share_expected": [threshold for _, threshold in _SHARE_VERSIONS],
        }
    )
    monetary_totals = baseline.group_by("share_version", "importer").agg(
        pl.col("mean_weighted_import_usd")
        .sum()
        .alias("baseline_denominator_expected"),
        pl.len().alias("raw_cell_count_expected"),
    )
    checked = (
        baseline.join(
            monetary_totals,
            on=("share_version", "importer"),
            how="left",
            validate="m:1",
        )
        .join(thresholds, on="share_version", how="left", validate="m:1")
        .with_columns(
            (
                pl.col("mean_weighted_import_usd")
                / pl.col("baseline_denominator_expected")
            ).alias("raw_baseline_share_expected")
        )
        .with_columns(
            (
                pl.col("raw_baseline_share_expected")
                > pl.col("minimum_share_expected")
            ).alias("retained_expected")
        )
    ).with_columns(
        pl.col("raw_baseline_share").sum().over("share_version", "importer").alias(
            "published_raw_share_sum"
        ),
        pl.col("raw_baseline_share_expected")
        .sum()
        .over("share_version", "importer")
        .alias("expected_raw_share_sum"),
        pl.col("raw_baseline_share_expected")
        .filter(pl.col("retained_expected"))
        .sum()
        .over("share_version", "importer")
        .alias("coverage_expected"),
        pl.col("retained_expected")
        .sum()
        .over("share_version", "importer")
        .alias("retained_cell_count_expected"),
    ).with_columns(
        (pl.col("coverage_expected") >= 0.95).alias("coverage_eligible_expected"),
        (pl.col("share_version") == "main_0.0001").alias(
            "confirmatory_specification_expected"
        ),
    ).with_columns(
        (
            pl.col("coverage_eligible_expected")
            & pl.col("confirmatory_specification_expected")
        ).alias("confirmatory_baseline_eligible_expected"),
        pl.when(pl.col("retained_expected"))
        .then(pl.col("raw_baseline_share_expected") / pl.col("coverage_expected"))
        .otherwise(None)
        .alias("baseline_share_expected"),
    )
    coverage_recomputation_failures = checked.filter(
        pl.col("minimum_share_expected").is_null()
        | pl.col("mean_weighted_import_usd").is_null()
        | ~pl.col("mean_weighted_import_usd").is_finite()
        | (pl.col("mean_weighted_import_usd") <= 0.0)
        | pl.col("baseline_denominator_expected").is_null()
        | ~pl.col("baseline_denominator_expected").is_finite()
        | (pl.col("baseline_denominator_expected") <= 0.0)
        | (
            (
                pl.col("baseline_mean_total_weighted_import_usd")
                - pl.col("baseline_denominator_expected")
            ).abs()
            > 1e-12
        )
        | (
            (pl.col("raw_baseline_share") - pl.col("raw_baseline_share_expected"))
            .abs()
            > 1e-12
        )
        | ((pl.col("published_raw_share_sum") - 1.0).abs() > 1e-12)
        | ((pl.col("expected_raw_share_sum") - 1.0).abs() > 1e-12)
        | (pl.col("retained") != pl.col("retained_expected"))
        | (
            (pl.col("retained_coverage") - pl.col("coverage_expected")).abs()
            > 1e-12
        )
        | (pl.col("raw_cell_count") != pl.col("raw_cell_count_expected"))
        | (pl.col("retained_cell_count") != pl.col("retained_cell_count_expected"))
        | (
            pl.col("baseline_share").is_null()
            != pl.col("baseline_share_expected").is_null()
        )
        | (
            pl.col("baseline_share").is_not_null()
            & (
                (pl.col("baseline_share") - pl.col("baseline_share_expected")).abs()
                > 1e-12
            )
        )
    ).height
    threshold_failures = checked.filter(
        pl.col("minimum_share_expected").is_null()
        | (pl.col("retained") != pl.col("retained_expected"))
    ).height
    coverage_label_failures = checked.filter(
        (pl.col("coverage_eligible") != pl.col("coverage_eligible_expected"))
        | (
            pl.col("confirmatory_specification")
            != pl.col("confirmatory_specification_expected")
        )
        | (
            pl.col("confirmatory_baseline_eligible")
            != pl.col("confirmatory_baseline_eligible_expected")
        )
    ).height
    return {
        "coverage_recomputation_failures": coverage_recomputation_failures,
        "coverage_label_failures": coverage_label_failures,
        "threshold_failures": threshold_failures,
    }


def independent_baseline_audit(baseline: pl.DataFrame) -> dict[str, int]:
    """Reconstruct baseline shares and labels from published monetary amounts."""

    metrics = _independent_baseline_metrics(baseline)
    return {
        "coverage_recomputation_failures": metrics[
            "coverage_recomputation_failures"
        ],
        "coverage_label_failures": metrics["coverage_label_failures"],
    }


def _independent_growth(previous: float, current: float) -> float | None:
    if previous == 0.0 and current == 0.0:
        return None
    return 2.0 * (current - previous) / (current + previous)


def independent_partner_shock_audit(
    published: pl.DataFrame,
    bilateral: pl.DataFrame,
    exporter_totals: pl.DataFrame,
) -> dict[str, int]:
    """Reconstruct every published shock field directly from authoritative parents."""

    trade = _normalize_partner_aliases(bilateral)
    totals = exporter_totals
    _require_columns(
        trade,
        ("year", "exporter", "importer", "hs6", "weighted_green_trade_usd"),
        "bilateral shock parent",
    )
    _require_columns(
        totals,
        ("year", "exporter", "hs6", "weighted_green_trade_usd"),
        "export-total shock parent",
    )
    parent_years = sorted(int(value) for value in totals["year"].unique())
    cells = published.select("destination_excluded", "exporter", "hs6").unique()
    required_years = sorted(
        set(int(value) for value in published["year"].unique())
        | {int(value) - 1 for value in published["year"].unique() if int(value) - 1 in parent_years}
    )
    grid = cells.join(pl.DataFrame({"year": required_years}), how="cross")
    all_exports = totals.group_by("year", "exporter", "hs6").agg(
        pl.col("weighted_green_trade_usd").sum().alias("all_exports")
    )
    own = trade.group_by("year", "exporter", "importer", "hs6").agg(
        pl.col("weighted_green_trade_usd").sum().alias("own_exports")
    ).rename({"importer": "destination_excluded"})
    global_product = totals.group_by("year", "hs6").agg(
        pl.col("weighted_green_trade_usd").sum().alias("global_exports")
    )
    product_to_destination = trade.group_by("year", "importer", "hs6").agg(
        pl.col("weighted_green_trade_usd").sum().alias("destination_product_exports")
    ).rename({"importer": "destination_excluded"})
    levels = (
        grid.join(all_exports, on=("year", "exporter", "hs6"), how="left")
        .join(own, on=("year", "exporter", "destination_excluded", "hs6"), how="left")
        .join(global_product, on=("year", "hs6"), how="left")
        .join(product_to_destination, on=("year", "destination_excluded", "hs6"), how="left")
        .with_columns(
            pl.col("all_exports").fill_null(0.0),
            pl.col("own_exports").fill_null(0.0),
            pl.col("global_exports").fill_null(0.0),
            pl.col("destination_product_exports").fill_null(0.0),
        )
        .with_columns(
            (pl.col("all_exports") - pl.col("own_exports")).alias("expected_exporter_level"),
            (pl.col("global_exports") - pl.col("destination_product_exports")).alias("expected_global_level"),
        )
    )
    lagged = levels.select(
        "destination_excluded",
        "exporter",
        "hs6",
        (pl.col("year") + 1).alias("year"),
        pl.col("expected_exporter_level").alias("expected_exporter_lag"),
        pl.col("expected_global_level").alias("expected_global_lag"),
    )
    expected = (
        levels.join(lagged, on=("destination_excluded", "exporter", "hs6", "year"), how="left")
        .with_columns(
            pl.when(pl.col("expected_exporter_lag").is_null())
            .then(None)
            .when((pl.col("expected_exporter_lag") == 0.0) & (pl.col("expected_exporter_level") == 0.0))
            .then(None)
            .otherwise(2.0 * (pl.col("expected_exporter_level") - pl.col("expected_exporter_lag")) / (pl.col("expected_exporter_level") + pl.col("expected_exporter_lag")))
            .alias("expected_exporter_growth"),
            pl.when(pl.col("expected_global_lag").is_null())
            .then(None)
            .when((pl.col("expected_global_lag") == 0.0) & (pl.col("expected_global_level") == 0.0))
            .then(None)
            .otherwise(2.0 * (pl.col("expected_global_level") - pl.col("expected_global_lag")) / (pl.col("expected_global_level") + pl.col("expected_global_lag")))
            .alias("expected_global_growth"),
        )
        .with_columns(
            (pl.col("expected_exporter_growth") - pl.col("expected_global_growth")).alias("expected_shock"),
            pl.when(pl.col("expected_exporter_lag").is_null()).then(pl.lit("prior_year_not_available"))
            .when((pl.col("expected_exporter_lag") == 0.0) & (pl.col("expected_exporter_level") == 0.0)).then(pl.lit("both_adjacent_exporter_levels_zero"))
            .when((pl.col("expected_global_lag") == 0.0) & (pl.col("expected_global_level") == 0.0)).then(pl.lit("both_adjacent_global_product_levels_zero"))
            .otherwise(None).alias("expected_reason"),
        )
        .filter(pl.col("year").is_in(published["year"].unique().to_list()))
    )
    checked = published.join(
        expected,
        on=("destination_excluded", "exporter", "hs6", "year"),
        how="left",
        validate="m:1",
    ).with_columns(
        (pl.col("baseline_share") * pl.col("expected_shock")).alias("expected_contribution")
    )
    fields = (
        ("exports_excluding_destination_lag", "expected_exporter_lag"),
        ("exports_excluding_destination", "expected_exporter_level"),
        ("global_product_exports_excluding_destination_lag", "expected_global_lag"),
        ("global_product_exports_excluding_destination", "expected_global_level"),
        ("exporter_growth_excluding_destination", "expected_exporter_growth"),
        ("global_product_growth_excluding_destination", "expected_global_growth"),
        ("partner_shock", "expected_shock"),
        ("contribution", "expected_contribution"),
    )
    mismatch = pl.lit(False)
    for actual, wanted in fields:
        scale = pl.max_horizontal(
            pl.col(actual).abs(), pl.col(wanted).abs(), pl.lit(1.0)
        )
        mismatch = mismatch | (pl.col(actual).is_null() != pl.col(wanted).is_null()) | (
            pl.col(actual).is_not_null()
            & pl.col(wanted).is_not_null()
            & ((pl.col(actual) - pl.col(wanted)).abs() > 1e-12 * scale)
        )
    failures = checked.filter(
        mismatch
        | (pl.col("shock_missing_reason") != pl.col("expected_reason")).fill_null(False)
        | (pl.col("shock_missing_reason").is_null() != pl.col("expected_reason").is_null())
        | (pl.col("contribution_missing_reason") != pl.col("expected_reason")).fill_null(False)
        | (pl.col("contribution_missing_reason").is_null() != pl.col("expected_reason").is_null())
        | (pl.col("importer") != pl.col("destination_excluded"))
    ).height
    return {"shock_parent_reconstruction_failures": failures}


def _independent_midranks(values: list[tuple[str, float]]) -> dict[str, float]:
    if len(values) < 2:
        return {}
    ordered = sorted(values, key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        midrank = (((position + 1) + end) / 2.0 - 1.0) / (len(ordered) - 1.0)
        for index in range(position, end):
            ranks[ordered[index][0]] = midrank
        position = end
    return ranks


def independent_country_instrument_audit(
    country: pl.DataFrame, gad: pl.DataFrame
) -> dict[str, int]:
    """Independently derive lagged GAD and every CMZ rank/term from GAD parents."""

    gad_rows = {
        (str(row["economy_id"]), int(row["year"]), str(row["specification_id"])): row
        for row in gad.iter_rows(named=True)
    }
    rank_inputs: dict[tuple[int, str], list[tuple[str, float]]] = {}
    for row in gad.iter_rows(named=True):
        absorption = row.get("absorption")
        eligible = bool(row.get("confirmatory_eligible", True))
        if absorption is not None and eligible and math.isfinite(float(absorption)):
            rank_inputs.setdefault(
                (int(row["year"]), str(row["specification_id"])), []
            ).append((str(row["economy_id"]), float(absorption)))
    ranks = {
        (economy, year, version): rank
        for (year, version), values in rank_inputs.items()
        for economy, rank in _independent_midranks(values).items()
    }
    z_values: dict[tuple[str, str, int], object] = {}
    for row in country.iter_rows(named=True):
        key = (str(row["importer"]), str(row["share_version"]), int(row["year"]))
        if key in z_values and not _float_matches(z_values[key], row["z"]):
            z_values[key] = float("nan")
        else:
            z_values[key] = row["z"]
    failures = {
        "gad_parent_reconstruction_failures": 0,
        "cmz_value_reconstruction_failures": 0,
        "cmz_term_reason_reconstruction_failures": 0,
        "cmz_time_reconstruction_failures": 0,
    }
    for row in country.iter_rows(named=True):
        economy = str(row["importer"])
        share_version = str(row["share_version"])
        year = int(row["year"])
        version = str(row["gad_version"])
        lagged = gad_rows.get((economy, year - 1, version))
        gad_time = year - 1 if lagged is not None else None
        gad_lag = lagged.get("gad") if lagged is not None else None
        z = row["z"]
        z_gad = float(z) * float(gad_lag) if z is not None and gad_lag is not None else None
        z_gad_reason = "z_missing" if z is None else ("lagged_gad_missing" if gad_lag is None else None)
        if (
            row.get("gad_time") != gad_time
            or not _float_matches(row.get("gad_lag"), gad_lag)
            or not _float_matches(row.get("z_gad"), z_gad)
            or row.get("z_gad_missing_reason") != z_gad_reason
        ):
            failures["gad_parent_reconstruction_failures"] += 1
        terms: list[float] = []
        for lag in range(5):
            shock = z_values.get((economy, share_version, year - lag))
            absorption_year = year - lag - 1
            rank = ranks.get((economy, absorption_year, version))
            parent = gad_rows.get((economy, absorption_year, version))
            rho = parent.get("rho") if parent is not None else None
            if (
                shock is None
                or rank is None
                or rho is None
                or not math.isfinite(float(shock))
                or not math.isfinite(float(rho))
            ):
                continue
            terms.append(float(rho) ** lag * float(shock) * (1.0 - rank))
        count = len(terms)
        cmz = sum(terms) if count == 5 else None
        reason = None if count == 5 else "fewer_than_five_complete_terms"
        if not _float_matches(row.get("cmz"), cmz):
            failures["cmz_value_reconstruction_failures"] += 1
        if row.get("cmz_complete_terms") != count or row.get("cmz_missing_reason") != reason:
            failures["cmz_term_reason_reconstruction_failures"] += 1
        if row.get("absorption_latest_time") != year - 1:
            failures["cmz_time_reconstruction_failures"] += 1
    return failures


def _manifest_input_binding_failures(manifest: object) -> int:
    failures = 0
    actual_parent_hashes: list[str] = []
    for artifact in manifest.input_artifacts:  # type: ignore[attr-defined]
        path = Path(artifact.path)
        if (
            not path.is_file()
            or path.stat().st_size != artifact.bytes
            or sha256_file(path) != artifact.sha256
        ):
            failures += 1
            continue
        if artifact.parent_manifest_sha256 is not None:
            sidecar = path.with_name(f"{path.name}.manifest.json")
            if (
                not sidecar.is_file()
                or sha256_file(sidecar) != artifact.parent_manifest_sha256
            ):
                failures += 1
            else:
                actual_parent_hashes.append(artifact.parent_manifest_sha256)
    if tuple(sorted(actual_parent_hashes)) != manifest.parent_manifest_hashes:  # type: ignore[attr-defined]
        failures += 1
    return failures


def _null_semantics_audit_failures(
    manifests: Sequence[object], contracts: Sequence[TableContract]
) -> int:
    failures = 0
    for manifest, contract in zip(manifests, contracts, strict=True):
        if manifest.null_semantics != contract.null_semantics:  # type: ignore[attr-defined]
            failures += 1
        schema = json.loads(Path(manifest.schema_path).read_text(encoding="utf-8"))  # type: ignore[attr-defined]
        declared = {
            name: definition.get("x-null-semantics")
            for name, definition in schema["properties"].items()
            if "x-null-semantics" in definition
        }
        if declared != contract.null_semantics:
            failures += 1
        for name, definition in schema["properties"].items():
            nullable = any(
                option.get("type") == "null" for option in definition.get("anyOf", [])
            )
            if nullable != (name in contract.null_semantics):
                failures += 1
        actual_nullable = {
            name for name, count in manifest.null_counts.items() if count  # type: ignore[attr-defined]
        }
        if not actual_nullable <= set(contract.null_semantics):
            failures += 1
    return failures


def _authoritative_parent_path(path: Path) -> str:
    sidecar = path.with_name(f"{path.name}.manifest.json")
    if not sidecar.is_file():
        return str(path.resolve())
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        destination = payload["destination"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid parent manifest: {sidecar}") from exc
    if not isinstance(destination, str):
        raise ValueError(f"invalid parent manifest destination: {sidecar}")
    authority = Path(destination)
    if not authority.is_file():
        raise ValueError(f"parent manifest destination is missing: {authority}")
    return str(authority.resolve())


def _direct_parent_lineage_failures(
    manifests: Sequence[object], expected_paths: Sequence[Sequence[Path]]
) -> int:
    if len(manifests) != len(expected_paths):
        raise ValueError("instrument lineage groups do not match manifests")
    declared_inputs = [
        {artifact.path for artifact in manifest.input_artifacts}  # type: ignore[attr-defined]
        for manifest in manifests
    ]
    expected_authorities = [
        {_authoritative_parent_path(path) for path in group}
        for group in expected_paths
    ]
    return sum(
        not expected <= actual
        for expected, actual in zip(
            expected_authorities, declared_inputs, strict=True
        )
    )


def audit_instrument_artifacts(
    paths: ProjectPaths,
    *,
    audit_path: Path,
) -> dict[str, int | float | str]:
    """Audit weights, exclusion lineage, exact recomputation, versions, and capacity."""

    output_root = paths.measures / "instruments"
    baseline_path = output_root / "iv_baseline_shares.parquet"
    shock_path = output_root / "iv_partner_shocks.parquet"
    country_path = output_root / "iv_country_year.parquet"
    manifests = tuple(
        verify_manifest(path.with_name(f"{path.name}.manifest.json"))
        for path in (baseline_path, shock_path, country_path)
    )
    contracts = (
        _contract("iv_baseline_shares.json"),
        _contract("iv_partner_shocks.json"),
        _contract("iv_country_year.json"),
    )
    baseline = pl.read_parquet(baseline_path)
    long = pl.read_parquet(shock_path)
    country = pl.read_parquet(country_path)
    gad_path = paths.measures / "gad/gad_country_year.parquet"
    gad = pl.read_parquet(gad_path)
    taxonomy_values = baseline["taxonomy_version"].unique().to_list()
    if taxonomy_values != ["main_hs96"]:
        raise RuntimeError(f"instrument taxonomy authority is invalid: {taxonomy_values}")
    bilateral_paths = _annual_paths(
        paths.normalized / "baci", "green_bilateral", "main_hs96"
    )
    total_paths = _annual_paths(
        paths.normalized / "baci", "green_economy_product", "main_hs96"
    )
    bilateral = (
        pl.scan_parquet(bilateral_paths)
        .select(
            pl.col("year").cast(pl.Int64),
            pl.col("exporter_id").alias("exporter"),
            pl.col("importer_id").alias("importer"),
            "hs6",
            "weighted_green_trade_usd",
        )
        .collect(engine="streaming")
    )
    exporter_totals = (
        pl.scan_parquet(total_paths)
        .filter(pl.col("flow_role") == "exporter")
        .select(
            pl.col("year").cast(pl.Int64),
            pl.col("economy_id").alias("exporter"),
            "hs6",
            "weighted_green_trade_usd",
        )
        .collect(engine="streaming")
    )

    retained_sums = baseline.filter(pl.col("retained")).group_by(
        "taxonomy_version", "share_version", "importer"
    ).agg(pl.col("baseline_share").sum().alias("weight_sum"))
    max_weight_error = float(
        retained_sums.select((pl.col("weight_sum") - 1.0).abs().max()).item()
    )
    weight_sum_failures = retained_sums.filter(
        (pl.col("weight_sum") - 1.0).abs() > 1e-12
    ).height
    baseline_independent = _independent_baseline_metrics(baseline)
    coverage_recomputation_failures = baseline_independent[
        "coverage_recomputation_failures"
    ]
    coverage_label_failures = baseline_independent["coverage_label_failures"]
    threshold_failures = baseline_independent["threshold_failures"]
    destination_exclusion_failures = long.filter(
        pl.col("destination_excluded") != pl.col("importer")
    ).height
    bounded_growth_failures = long.filter(
        (pl.col("exporter_growth_excluding_destination").abs() > 2.0 + 1e-12)
        | (
            pl.col("global_product_growth_excluding_destination").abs()
            > 2.0 + 1e-12
        )
    ).height
    adjacent_zero_failures = long.filter(
        (
            (pl.col("exports_excluding_destination") == 0.0)
            & (pl.col("exports_excluding_destination_lag") == 0.0)
            & pl.col("exporter_growth_excluding_destination").is_not_null()
        )
        | (
            (
                pl.col("global_product_exports_excluding_destination")
                == 0.0
            )
            & (
                pl.col("global_product_exports_excluding_destination_lag")
                == 0.0
            )
            & pl.col("global_product_growth_excluding_destination").is_not_null()
        )
    ).height
    shock_parent_reconstruction_failures = independent_partner_shock_audit(
        long, bilateral, exporter_totals
    )["shock_parent_reconstruction_failures"]

    recomputed = (
        long.group_by("taxonomy_version", "share_version", "importer", "year")
        .agg(
            pl.len().alias("required"),
            pl.col("partner_shock").is_not_null().sum().alias("observed"),
            pl.col("contribution").sum().alias("sum_contribution"),
        )
        .with_columns(
            pl.when(pl.col("required") == pl.col("observed"))
            .then(pl.col("sum_contribution"))
            .otherwise(None)
            .alias("z_recomputed")
        )
    )
    unique_country = country.select(
        "taxonomy_version",
        "share_version",
        "importer",
        "year",
        "z",
        "z_missing_reason",
        "confirmatory_iv_eligible",
    ).unique()
    if unique_country.height * country["gad_version"].n_unique() != country.height:
        country_duplication_failures = 1
    else:
        country_duplication_failures = 0
    z_check = unique_country.join(
        recomputed,
        on=("taxonomy_version", "share_version", "importer", "year"),
        how="left",
        validate="1:1",
    )
    z_recomputation_failures = _nullable_float_failures(
        z_check, "z", "z_recomputed"
    )
    confirmatory_missing_z_failures = country.filter(
        pl.col("confirmatory_iv_eligible") & pl.col("z").is_null()
    ).height
    missing_z_reason_failures = country.filter(
        pl.col("z").is_null() & pl.col("z_missing_reason").is_null()
    ).height
    expected_versions = set(
        load_outcome_gad_map(_OUTCOME_GAD_MAP).outcomes.values()
    )
    gad_version_failures = int(set(country["gad_version"].unique()) != expected_versions)
    version_key_failures = country.group_by(
        "taxonomy_version", "share_version", "importer", "year", "gad_version"
    ).len().filter(pl.col("len") > 1).height

    country_independent = independent_country_instrument_audit(country, gad)
    gad_parent_reconstruction_failures = country_independent[
        "gad_parent_reconstruction_failures"
    ]
    cmz_value_reconstruction_failures = country_independent[
        "cmz_value_reconstruction_failures"
    ]
    cmz_term_reason_reconstruction_failures = country_independent[
        "cmz_term_reason_reconstruction_failures"
    ]
    cmz_time_reconstruction_failures = country_independent[
        "cmz_time_reconstruction_failures"
    ]
    incomplete_cmz_value_failures = country.filter(
        pl.col("cmz").is_not_null() & (pl.col("cmz_complete_terms") != 5)
    ).height
    cmz_timing_failures = country.filter(
        pl.col("absorption_latest_time") != pl.col("year") - 1
    ).height
    null_semantics_failures = _null_semantics_audit_failures(manifests, contracts)
    parent_hash_binding_failures = sum(
        _manifest_input_binding_failures(manifest) for manifest in manifests
    )
    expected_direct_parents = (
        (*bilateral_paths, paths.harmonized / "sample/provisional_sample.parquet"),
        (baseline_path, *bilateral_paths, *total_paths),
        (baseline_path, shock_path, gad_path),
    )
    parent_lineage_failures = _direct_parent_lineage_failures(
        manifests, expected_direct_parents
    )
    prohibited_lineage_columns = sum(
        any(
            column in _PROHIBITED_IV_COLUMNS
            or column.lower().startswith("outcome_")
            for column in frame.columns
        )
        for frame in (baseline, long, country)
    )
    outcome_value_table_parents = sum(
        "/measures/outcomes/" in artifact.path.replace("\\", "/")
        for manifest in manifests
        for artifact in manifest.input_artifacts
    )
    usage = measure_layer_usage(paths.data_root, audits_root=paths.audits)
    capacity_failures = int(
        usage.intermediate_bytes + usage.scratch_bytes >= 25 * 1024**3
        or usage.project_bytes >= 120 * 1024**3
        or usage.filesystem_free_bytes < 30 * 1024**3
    )
    checks = {
        "weight_sum_failures": weight_sum_failures,
        "coverage_recomputation_failures": coverage_recomputation_failures,
        "coverage_label_failures": coverage_label_failures,
        "threshold_failures": threshold_failures,
        "destination_exclusion_failures": destination_exclusion_failures,
        "bounded_growth_failures": bounded_growth_failures,
        "adjacent_zero_failures": adjacent_zero_failures,
        "shock_parent_reconstruction_failures": shock_parent_reconstruction_failures,
        "z_recomputation_failures": z_recomputation_failures,
        "country_version_projection_failures": country_duplication_failures,
        "confirmatory_missing_z_failures": confirmatory_missing_z_failures,
        "missing_z_reason_failures": missing_z_reason_failures,
        "gad_parent_reconstruction_failures": gad_parent_reconstruction_failures,
        "gad_version_failures": gad_version_failures,
        "version_key_failures": version_key_failures,
        "cmz_value_reconstruction_failures": cmz_value_reconstruction_failures,
        "cmz_term_reason_reconstruction_failures": cmz_term_reason_reconstruction_failures,
        "cmz_time_reconstruction_failures": cmz_time_reconstruction_failures,
        "incomplete_cmz_value_failures": incomplete_cmz_value_failures,
        "cmz_timing_failures": cmz_timing_failures,
        "null_semantics_failures": null_semantics_failures,
        "parent_hash_binding_failures": parent_hash_binding_failures,
        "parent_lineage_failures": parent_lineage_failures,
        "prohibited_lineage_columns": prohibited_lineage_columns,
        "outcome_value_table_parents": outcome_value_table_parents,
        "capacity_failures": capacity_failures,
    }
    audit = pl.DataFrame(
        {
            "metric": [*checks, "max_retained_weight_sum_error"],
            "value": [*(float(value) for value in checks.values()), max_weight_error],
            "maximum_allowed": [0.0] * len(checks) + [1e-12],
        }
    ).with_columns(
        (pl.col("value") <= pl.col("maximum_allowed")).alias("passed")
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.write_csv(audit_path)
    if any(value != 0 for value in checks.values()) or max_weight_error > 1e-12:
        raise RuntimeError(
            f"instrument audit failed: checks={checks}, max_weight_error={max_weight_error}"
        )
    return {
        **checks,
        "max_retained_weight_sum_error": max_weight_error,
        "baseline_rows": baseline.height,
        "shock_contribution_rows": long.height,
        "country_year_rows": country.height,
        "intermediate_bytes": usage.intermediate_bytes,
        "scratch_bytes": usage.scratch_bytes,
        "status": "valid",
        "audit_path": str(audit_path),
    }
