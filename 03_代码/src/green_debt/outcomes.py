"""Independent annual outcome bases and product-entry primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import polars as pl

from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    verify_manifest,
    write_authoritative_table,
)
from green_debt.config import load_outcome_gad_map
from green_debt.gad import registered_gad_specifications
from green_debt.paths import ProjectPaths
from green_debt.trade import compute_rca


_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT_ROOT = Path(__file__).resolve().parents[2] / "contracts"
_OUTCOME_GAD_MAP = _ROOT / "config" / "outcome_gad_map.yaml"
_YEARS = tuple(range(1996, 2025))
_WDI_CO2 = "EN.GHG.CO2.MT.CE.AR5"
_WDI_GDP = "NY.GDP.MKTP.CD"
_WDI_POPULATION = "SP.POP.TOTL"
_WDI_ENERGY_INTENSITY = "EG.EGY.PRIM.PP.KD"
_IRENA_ADDITIONS = "renewable_capacity_additions_mw"
_GAD_COMPONENT_COLUMNS = frozenset(
    {
        "gimc",
        "supp",
        "external_exposure",
        "absorption",
        "lagged_absorption",
        "gap",
        "gad",
    }
)


@dataclass(frozen=True)
class OutcomeSpec:
    """A frozen outcome-to-leave-one-GAD pairing and its physical authority."""

    outcome_id: str
    gad_variant: str
    horizons: tuple[int, ...] = ()
    authority_table: str = ""
    unit: str = ""
    orientation: str = ""
    formula_inputs: tuple[str, ...] = ()
    transformation_flags: frozenset[str] = frozenset()
    materialized: bool = False


@dataclass(frozen=True)
class OutcomeBuildReport:
    country_rows: int
    product_rows: int
    country_output_path: str
    product_output_path: str


@dataclass(frozen=True)
class GIUComponentScaler:
    center: float
    scale: float
    scale_method: str
    row_count: int
    economy_count: int


@dataclass(frozen=True)
class GIUHorizonScaler:
    horizon: int
    components: dict[str, GIUComponentScaler]


@dataclass(frozen=True)
class GIUScalerRegistry:
    """Independent, horizon-specific outcome-scaler authority."""

    by_horizon: dict[int, GIUHorizonScaler]
    fit_years: tuple[int, int]
    canonical_hash: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "registry_id": "green_industrial_upgrading_outcome_scaler",
            "registry_version": "1.0.0",
            "fit_years": list(self.fit_years),
            "horizons": [
                {
                    "horizon": horizon,
                    "components": {
                        name: asdict(scaler)
                        for name, scaler in sorted(record.components.items())
                    },
                }
                for horizon, record in sorted(self.by_horizon.items())
            ],
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.canonical_payload()
        payload["canonical_hash"] = self.canonical_hash
        return payload


_REQUIRED_RAW_FLAGS = frozenset(
    {"raw", "unwinsorized", "uninterpolated", "not_shock_multiplied"}
)
_VALID_ORIENTATIONS = frozenset(
    {"higher_is_better", "higher_is_worse", "higher_is_exposure"}
)
_OUTCOME_METADATA = {
    "domestic_value_added_share": (
        "outcomes_country_year", "percent_of_gross_exports", "higher_is_better",
        ("dvashare_raw",), _REQUIRED_RAW_FLAGS, True,
    ),
    "foreign_value_added_dependence": (
        "outcomes_country_year", "fraction_of_final_demand_value_added", "higher_is_worse",
        ("gfvad_raw",), _REQUIRED_RAW_FLAGS, True,
    ),
    "green_export_complexity": (
        "outcomes_country_year", "population_standard_deviations", "higher_is_better",
        ("weighted_green_export_usd", "gpci"), _REQUIRED_RAW_FLAGS, True,
    ),
    "green_export_share": (
        "outcomes_country_year", "fraction_of_all_goods_exports", "higher_is_better",
        ("weighted_green_export_usd", "total_export_usd"), _REQUIRED_RAW_FLAGS, True,
    ),
    "future_green_rca_entry_rate": (
        "outcomes_product_year", "fraction_of_baseline_non_rca_green_products", "higher_is_better",
        ("baseline_rca", "future_rca"), _REQUIRED_RAW_FLAGS, False,
    ),
    "green_industrial_upgrading_index": (
        "outcomes_country_year", "horizon_specific_provisional_core_standardized_mean", "higher_is_better",
        ("future_green_rca_entry_rate", "green_export_complexity", "green_export_share"), _REQUIRED_RAW_FLAGS, False,
    ),
    "supplier_deepening": (
        "outcomes_country_year", "supplier_capability_change", "higher_is_better",
        ("supplier_capability",), _REQUIRED_RAW_FLAGS, False,
    ),
    "green_science_output": (
        "outcomes_country_year", "green_science_output_change", "higher_is_better",
        ("green_science_output",), _REQUIRED_RAW_FLAGS, False,
    ),
    "renewable_capacity_additions_mw_per_million": (
        "outcomes_country_year", "MW_per_million_persons", "higher_is_better",
        ("renewable_capacity_additions_mw", "population"), _REQUIRED_RAW_FLAGS, True,
    ),
    "co2_tonnes_per_million_current_usd": (
        "outcomes_country_year", "tonnes_CO2e_per_million_current_USD", "higher_is_worse",
        ("co2_mt", "gdp_current_usd"), _REQUIRED_RAW_FLAGS, True,
    ),
    "energy_intensity_mj_per_ppp_gdp": (
        "outcomes_country_year", "MJ_per_2021_PPP_GDP", "higher_is_worse",
        ("energy_intensity_mj_per_ppp_gdp",), _REQUIRED_RAW_FLAGS, True,
    ),
    "asinh_weighted_green_imports": (
        "outcomes_country_year", "asinh_current_USD", "higher_is_exposure",
        ("green_imports_usd",), _REQUIRED_RAW_FLAGS | {"asinh"}, True,
    ),
}

_ENVIRONMENTAL_HORIZONS = (0, 1, 2, 3)
_VULNERABILITY_HORIZONS = (1, 2, 3)
_INDUSTRIAL_HORIZONS = (3, 4, 5, 6, 7, 8)
_OUTCOME_HORIZONS = {
    "domestic_value_added_share": _INDUSTRIAL_HORIZONS,
    "foreign_value_added_dependence": _INDUSTRIAL_HORIZONS,
    "green_export_complexity": _INDUSTRIAL_HORIZONS,
    "green_export_share": _INDUSTRIAL_HORIZONS,
    "future_green_rca_entry_rate": _INDUSTRIAL_HORIZONS,
    "green_industrial_upgrading_index": _INDUSTRIAL_HORIZONS,
    "supplier_deepening": _INDUSTRIAL_HORIZONS,
    "green_science_output": _INDUSTRIAL_HORIZONS,
    "renewable_capacity_additions_mw_per_million": _ENVIRONMENTAL_HORIZONS,
    "co2_tonnes_per_million_current_usd": _ENVIRONMENTAL_HORIZONS,
    "energy_intensity_mj_per_ppp_gdp": _ENVIRONMENTAL_HORIZONS,
    "asinh_weighted_green_imports": _VULNERABILITY_HORIZONS,
}

_GIU_COMPONENTS = (
    "future_green_rca_entry_rate",
    "green_export_complexity_change",
    "green_export_share_change",
)


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fit_robust_outcome_scaler(values: pl.Series, economies: pl.Series) -> GIUComponentScaler:
    center = float(values.median())
    mad = float((values - center).abs().median())
    if math.isfinite(mad) and mad > 0.0:
        scale = 1.4826 * mad
        method = "mad_1_4826"
    else:
        q25 = float(values.quantile(0.25, interpolation="linear"))
        q75 = float(values.quantile(0.75, interpolation="linear"))
        scale = (q75 - q25) / 1.349
        method = "iqr_div_1_349"
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("GIU outcome scaler is not identifiable")
    return GIUComponentScaler(
        center=center,
        scale=scale,
        scale_method=method,
        row_count=len(values),
        economy_count=economies.n_unique(),
    )


def fit_giu_scalers(
    frame: pl.DataFrame,
    *,
    fit_years: tuple[int, int] = (2000, 2004),
) -> tuple[pl.DataFrame, GIUScalerRegistry]:
    """Fit separate provisional-core outcome scalers for every industrial horizon."""

    required = {
        "economy_id",
        "treatment_time",
        "horizon",
        "provisional_core",
        *_GIU_COMPONENTS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"GIU scaler frame lacks columns: {missing}")
    if frame.group_by("economy_id", "treatment_time", "horizon").len().filter(
        pl.col("len") > 1
    ).height:
        raise ValueError("GIU scaler frame has duplicate economy-treatment-horizon rows")
    horizons = sorted(int(value) for value in frame.get_column("horizon").unique())
    if not horizons:
        raise ValueError("GIU scaler frame has no horizons")
    records: dict[int, GIUHorizonScaler] = {}
    output = frame
    z_columns: list[str] = []
    for horizon in horizons:
        baseline = frame.filter(
            (pl.col("horizon") == horizon)
            & pl.col("treatment_time").is_between(*fit_years)
            & pl.col("provisional_core")
            & pl.all_horizontal(
                [
                    pl.col(column).is_not_null() & pl.col(column).is_finite()
                    for column in _GIU_COMPONENTS
                ]
            )
        )
        components: dict[str, GIUComponentScaler] = {}
        for column in _GIU_COMPONENTS:
            if baseline.is_empty():
                raise ValueError(f"GIU horizon {horizon} has no baseline values for {column}")
            scaler = _fit_robust_outcome_scaler(
                baseline.get_column(column), baseline.get_column("economy_id")
            )
            components[column] = scaler
            z_name = f"giu_z_{column}"
            if z_name not in z_columns:
                z_columns.append(z_name)
                output = output.with_columns(pl.lit(None, dtype=pl.Float64).alias(z_name))
            output = output.with_columns(
                pl.when(pl.col("horizon") == horizon)
                .then((pl.col(column) - scaler.center) / scaler.scale)
                .otherwise(pl.col(z_name))
                .alias(z_name)
            )
        records[horizon] = GIUHorizonScaler(horizon=horizon, components=components)
    partial = GIUScalerRegistry(
        by_horizon=records,
        fit_years=(int(fit_years[0]), int(fit_years[1])),
        canonical_hash="",
    )
    registry = GIUScalerRegistry(
        by_horizon=partial.by_horizon,
        fit_years=partial.fit_years,
        canonical_hash=_canonical_hash(partial.canonical_payload()),
    )
    complete = pl.all_horizontal(
        [pl.col(column).is_not_null() & pl.col(column).is_finite() for column in _GIU_COMPONENTS]
    )
    output = output.with_columns(
        pl.when(complete)
        .then(pl.mean_horizontal([pl.col(name) for name in z_columns]))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("green_industrial_upgrading_index"),
        pl.lit(True).alias("threshold_selection_only"),
    )
    return output, registry


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
            str(key): str(value) for key, value in payload.get("zero_semantics", {}).items()
        },
        null_semantics={
            str(key): str(value)
            for key, value in payload.get("null_semantics", {}).items()
        },
        transformations=tuple(str(value) for value in payload.get("transformations", [])),
    )


def _require_columns(frame: pl.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _unique_keys(frame: pl.DataFrame, keys: list[str], *, name: str) -> None:
    duplicates = frame.group_by(keys).len().filter(pl.col("len") > 1).height
    if duplicates:
        raise ValueError(f"{name} has duplicate keys: {duplicates}")


def _indicator_values(wdi: pl.DataFrame, indicator: str, name: str) -> pl.DataFrame:
    _require_columns(wdi, ("economy_id", "year", "indicator_id", "value"), name="WDI")
    result = wdi.filter(pl.col("indicator_id") == indicator).select(
        pl.col("economy_id").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("value").cast(pl.Float64).alias(name),
    )
    _unique_keys(result, ["economy_id", "year"], name=f"WDI {indicator}")
    return result


def _normalized_additions(irena: pl.DataFrame) -> pl.DataFrame:
    _require_columns(irena, ("economy_id", "year"), name="IRENA")
    if {"indicator_id", "value"} <= set(irena.columns):
        result = irena.filter(pl.col("indicator_id") == _IRENA_ADDITIONS).select(
            pl.col("economy_id").cast(pl.String),
            pl.col("year").cast(pl.Int16),
            pl.col("value").cast(pl.Float64).alias("renewable_capacity_additions_mw"),
            (
                pl.col("retirement_or_revision").cast(pl.Boolean)
                if "retirement_or_revision" in irena.columns
                else pl.lit(False, dtype=pl.Boolean)
            ).alias("renewable_capacity_additions_retirement_or_revision"),
        )
    else:
        _require_columns(irena, ("renewable_capacity_additions_mw",), name="IRENA")
        result = irena.select(
            pl.col("economy_id").cast(pl.String),
            pl.col("year").cast(pl.Int16),
            pl.col("renewable_capacity_additions_mw").cast(pl.Float64),
            (
                pl.col("retirement_or_revision").cast(pl.Boolean)
                if "retirement_or_revision" in irena.columns
                else pl.lit(False, dtype=pl.Boolean)
            ).alias("renewable_capacity_additions_retirement_or_revision"),
        )
    _unique_keys(result, ["economy_id", "year"], name="IRENA additions")
    return result


def build_environmental_outcomes(
    wdi: pl.DataFrame,
    irena: pl.DataFrame,
    population: pl.DataFrame,
) -> pl.DataFrame:
    """Construct raw annual environmental outcomes without interpolation or clipping."""

    _require_columns(population, ("economy_id", "year", "population"), name="population")
    people = population.select(
        pl.col("economy_id").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("population").cast(pl.Float64),
    )
    _unique_keys(people, ["economy_id", "year"], name="population")
    co2 = _indicator_values(wdi, _WDI_CO2, "co2_mt")
    gdp = _indicator_values(wdi, _WDI_GDP, "gdp_current_usd")
    energy = _indicator_values(wdi, _WDI_ENERGY_INTENSITY, "energy_intensity_mj_per_ppp_gdp")
    additions = _normalized_additions(irena)
    output = (
        people.join(co2, on=["economy_id", "year"], how="left")
        .join(gdp, on=["economy_id", "year"], how="left")
        .join(energy, on=["economy_id", "year"], how="left")
        .join(additions, on=["economy_id", "year"], how="left")
        .with_columns(
            pl.when(
                pl.col("population").is_finite().fill_null(False)
                & (pl.col("population") > 0.0).fill_null(False)
                & pl.col("renewable_capacity_additions_mw").is_finite().fill_null(False)
            )
            .then(pl.col("renewable_capacity_additions_mw") / (pl.col("population") / 1_000_000.0))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("renewable_capacity_additions_mw_per_million"),
            pl.when(
                pl.col("gdp_current_usd").is_finite().fill_null(False)
                & (pl.col("gdp_current_usd") > 0.0).fill_null(False)
                & pl.col("co2_mt").is_finite().fill_null(False)
            )
            .then(pl.col("co2_mt") * 1_000_000_000_000.0 / pl.col("gdp_current_usd"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("co2_tonnes_per_million_current_usd"),
            pl.when(pl.col("energy_intensity_mj_per_ppp_gdp").is_finite().fill_null(False))
            .then(pl.col("energy_intensity_mj_per_ppp_gdp"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("energy_intensity_mj_per_ppp_gdp"),
            pl.col("renewable_capacity_additions_retirement_or_revision")
            .fill_null(False)
            .cast(pl.Boolean),
        )
        .select(
            "economy_id",
            "year",
            "renewable_capacity_additions_mw_per_million",
            "renewable_capacity_additions_retirement_or_revision",
            "co2_tonnes_per_million_current_usd",
            "energy_intensity_mj_per_ppp_gdp",
        )
        .sort(["year", "economy_id"])
    )
    return output.with_columns(
        pl.col("year").cast(pl.Int16),
        pl.col("renewable_capacity_additions_mw_per_million").cast(pl.Float64),
        pl.col("co2_tonnes_per_million_current_usd").cast(pl.Float64),
        pl.col("energy_intensity_mj_per_ppp_gdp").cast(pl.Float64),
    )


def build_green_export_outcomes(
    exports: pl.DataFrame,
    totals: pl.DataFrame,
    gpci: pl.DataFrame,
) -> pl.DataFrame:
    """Use same-year GPCI for raw green export complexity and shares."""

    _require_columns(
        exports,
        ("economy_id", "year", "hs6", "weighted_green_export_usd"),
        name="green exports",
    )
    _require_columns(totals, ("economy_id", "year", "total_export_usd"), name="export totals")
    _require_columns(gpci, ("year", "hs6", "gpci"), name="GPCI")
    total = totals.select(
        pl.col("economy_id").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("total_export_usd").cast(pl.Float64),
    )
    _unique_keys(total, ["economy_id", "year"], name="export totals")
    complexity = gpci.select(
        pl.col("year").cast(pl.Int16),
        pl.col("hs6").cast(pl.String),
        pl.col("gpci").cast(pl.Float64),
    )
    _unique_keys(complexity, ["year", "hs6"], name="GPCI")
    green = exports.select(
        pl.col("economy_id").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("hs6").cast(pl.String),
        pl.col("weighted_green_export_usd").cast(pl.Float64),
    )
    invalid = green.filter(
        pl.col("weighted_green_export_usd").is_null()
        | ~pl.col("weighted_green_export_usd").is_finite()
        | (pl.col("weighted_green_export_usd") < 0.0)
    ).height
    if invalid:
        raise ValueError("green exports have invalid weighted values")
    green = green.group_by(["economy_id", "year", "hs6"]).agg(
        pl.col("weighted_green_export_usd").sum()
    )
    annual = green.join(complexity, on=["year", "hs6"], how="left").group_by(
        ["economy_id", "year"]
    ).agg(
        pl.col("weighted_green_export_usd").sum().alias("green_export_usd"),
        pl.len().alias("green_product_rows"),
        pl.col("gpci").is_not_null().sum().alias("gpci_rows"),
        (pl.col("weighted_green_export_usd") * pl.col("gpci"))
        .sum()
        .alias("complexity_numerator"),
    )
    return (
        total.join(annual, on=["economy_id", "year"], how="left")
        .with_columns(
            pl.col("green_export_usd").fill_null(0.0),
            pl.col("green_product_rows").fill_null(0),
            pl.col("gpci_rows").fill_null(0),
        )
        .with_columns(
            pl.when(pl.col("total_export_usd") > 0.0)
            .then(pl.col("green_export_usd") / pl.col("total_export_usd"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("green_export_share"),
            pl.when(
                (pl.col("green_export_usd") > 0.0)
                & (pl.col("gpci_rows") == pl.col("green_product_rows"))
            )
            .then(pl.col("complexity_numerator") / pl.col("green_export_usd"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("green_export_complexity"),
        )
        .select("economy_id", "year", "green_export_complexity", "green_export_share")
        .with_columns(
            pl.col("year").cast(pl.Int16),
            pl.col("green_export_complexity").cast(pl.Float64),
            pl.col("green_export_share").cast(pl.Float64),
        )
        .sort(["year", "economy_id"])
    )


def build_rca_entry_rate(
    rca: pl.DataFrame,
    *,
    treatment_year: int,
    horizon: int,
) -> pl.DataFrame:
    """Calculate a future entry rate against the t-1 non-RCA product set."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    _require_columns(rca, ("economy_id", "hs6", "year", "rca"), name="RCA")
    baseline_year = treatment_year - 1
    future_year = treatment_year + horizon
    prepared = rca.select(
        pl.col("economy_id").cast(pl.String),
        pl.col("hs6").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("rca").cast(pl.Float64),
    )
    _unique_keys(prepared, ["economy_id", "hs6", "year"], name="RCA")
    baseline = prepared.filter(pl.col("year") == baseline_year).filter(
        pl.col("rca").is_not_null() & (pl.col("rca") < 1.0)
    ).select("economy_id", "hs6", pl.col("rca").alias("baseline_rca"))
    future = prepared.filter(pl.col("year") == future_year).select(
        "economy_id", "hs6", pl.col("rca").alias("future_rca")
    )
    return (
        baseline.join(future, on=["economy_id", "hs6"], how="left")
        .group_by("economy_id")
        .agg(
            pl.len().cast(pl.UInt32).alias("eligible_products"),
            pl.col("future_rca").is_not_null().sum().alias("future_observed_products"),
            (pl.col("future_rca") >= 1.0).sum().alias("entered_products"),
        )
        .with_columns(
            pl.lit(treatment_year, dtype=pl.Int16).alias("treatment_year"),
            pl.lit(horizon, dtype=pl.Int16).alias("horizon"),
            pl.when(pl.col("future_observed_products") == pl.col("eligible_products"))
            .then(pl.col("entered_products") / pl.col("eligible_products"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("future_green_rca_entry_rate"),
        )
        .select(
            "economy_id",
            "treatment_year",
            "horizon",
            "eligible_products",
            "entered_products",
            "future_green_rca_entry_rate",
        )
        .sort("economy_id")
    )


def build_value_capture_outcomes(tiva: pl.DataFrame) -> pl.DataFrame:
    """Retain the fixed-weight TiVA raw values in their official source units."""

    _require_columns(
        tiva,
        ("economy_id", "year", "specification_id", "gfvad_raw", "dvashare_raw"),
        name="TiVA measures",
    )
    output = tiva.filter(
        (pl.col("specification_id") == "confirmatory_prod_weight")
        & pl.col("year").is_between(_YEARS[0], _YEARS[-1])
    ).select(
        pl.col("economy_id").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("dvashare_raw").cast(pl.Float64).alias("domestic_value_added_share"),
        pl.col("gfvad_raw").cast(pl.Float64).alias("foreign_value_added_dependence"),
    )
    _unique_keys(output, ["economy_id", "year"], name="confirmatory TiVA measures")
    return output.sort(["year", "economy_id"])


def build_product_entry_inputs(
    all_exports: pl.DataFrame,
    coverage: pl.DataFrame,
    green_products: pl.DataFrame,
) -> pl.DataFrame:
    """Store current RCA primitives only; future entry rows belong to horizon analysis."""

    _require_columns(all_exports, ("economy_id", "hs6", "year", "export_usd"), name="all exports")
    _require_columns(coverage, ("economy_id", "year", "export_coverage_normal"), name="export coverage")
    _require_columns(green_products, ("hs6",), name="green product registry")
    skeleton = coverage.select(
        pl.col("economy_id").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("export_coverage_normal").cast(pl.Boolean).fill_null(False),
    )
    _unique_keys(skeleton, ["economy_id", "year"], name="export coverage")
    products = green_products.select(pl.col("hs6").cast(pl.String)).unique().sort("hs6")
    if products.is_empty():
        raise ValueError("green product registry is empty")
    rca = compute_rca(all_exports).select("economy_id", "hs6", "year", "rca")
    output = (
        skeleton.join(products, how="cross")
        .join(rca, on=["economy_id", "year", "hs6"], how="left")
        .with_columns(
            pl.when(pl.col("export_coverage_normal"))
            .then(pl.col("rca").fill_null(0.0))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("green_product_rca"),
        )
        .with_columns(
            (pl.col("green_product_rca").is_not_null() & (pl.col("green_product_rca") < 1.0))
            .alias("green_product_rca_below_one"),
        )
        .select(
            "economy_id",
            "year",
            "hs6",
            "export_coverage_normal",
            "green_product_rca",
            "green_product_rca_below_one",
        )
        .with_columns(
            pl.col("year").cast(pl.Int16),
            pl.col("green_product_rca").cast(pl.Float64),
            pl.col("green_product_rca_below_one").cast(pl.Boolean),
        )
        .sort(["year", "economy_id", "hs6"])
    )
    _unique_keys(output, ["economy_id", "year", "hs6"], name="product entry inputs")
    return output


def outcome_specs() -> tuple[OutcomeSpec, ...]:
    """Return the complete frozen mapping, including later-horizon outcome records."""

    mapping = load_outcome_gad_map(_OUTCOME_GAD_MAP)
    missing = sorted(set(mapping.outcomes) - set(_OUTCOME_METADATA))
    if missing:
        raise ValueError(f"outcome metadata is missing frozen mappings: {missing}")
    registered = {specification.specification_id for specification in registered_gad_specifications()}
    records: list[OutcomeSpec] = []
    for outcome, variant in sorted(mapping.outcomes.items()):
        if variant not in registered:
            raise ValueError(f"outcome {outcome} maps to unregistered GAD variant: {variant}")
        authority_table, unit, orientation, inputs, flags, materialized = _OUTCOME_METADATA[outcome]
        records.append(
            OutcomeSpec(
                outcome_id=outcome,
                gad_variant=variant,
                horizons=_OUTCOME_HORIZONS[outcome],
                authority_table=authority_table,
                unit=unit,
                orientation=orientation,
                formula_inputs=inputs,
                transformation_flags=flags,
                materialized=materialized,
            )
        )
    return tuple(records)


def choose_gad_variant(outcome: str) -> str:
    """Resolve a registered outcome mapping and reject unknown outcomes by default."""

    variant = load_outcome_gad_map(_OUTCOME_GAD_MAP).variant_for(outcome)
    registered = {specification.specification_id for specification in registered_gad_specifications()}
    if variant not in registered:
        raise ValueError(f"outcome {outcome} maps to unregistered GAD variant: {variant}")
    return variant


def _country_outcomes(
    *,
    wdi: pl.DataFrame,
    irena: pl.DataFrame,
    green_exports: pl.DataFrame,
    totals: pl.DataFrame,
    gpci: pl.DataFrame,
    tiva: pl.DataFrame,
    trade_components: pl.DataFrame,
) -> pl.DataFrame:
    population = _indicator_values(wdi, _WDI_POPULATION, "population")
    environmental = build_environmental_outcomes(wdi, irena, population)
    industrial = build_green_export_outcomes(green_exports, totals, gpci)
    value_capture = build_value_capture_outcomes(tiva)
    _require_columns(trade_components, ("economy_id", "year", "green_imports_usd"), name="trade components")
    imports = trade_components.filter(pl.col("taxonomy_version") == "main_hs96").select(
        pl.col("economy_id").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.when(pl.col("green_imports_usd").is_finite().fill_null(False))
        .then(pl.col("green_imports_usd").cast(pl.Float64).arcsinh())
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("asinh_weighted_green_imports"),
    )
    _unique_keys(imports, ["economy_id", "year"], name="main taxonomy trade components")
    joined = environmental.join(industrial, on=["economy_id", "year"], how="full", coalesce=True)
    joined = joined.join(value_capture, on=["economy_id", "year"], how="full", coalesce=True)
    joined = joined.join(imports, on=["economy_id", "year"], how="full", coalesce=True)
    country_contract = _contract("outcomes_country_year.json")
    materialized = {
        spec.outcome_id
        for spec in outcome_specs()
        if spec.materialized and spec.authority_table == "outcomes_country_year"
    }
    country_columns = [
        column for column in country_contract.columns
        if column in {"economy_id", "year", "renewable_capacity_additions_retirement_or_revision"}
        or column in materialized
    ]
    return (
        joined.select(*country_columns)
        .with_columns(
            pl.col("year").cast(pl.Int16),
            pl.col("renewable_capacity_additions_mw_per_million").cast(pl.Float64),
            pl.col("renewable_capacity_additions_retirement_or_revision").fill_null(False).cast(pl.Boolean),
            pl.col("co2_tonnes_per_million_current_usd").cast(pl.Float64),
            pl.col("energy_intensity_mj_per_ppp_gdp").cast(pl.Float64),
            pl.col("green_export_complexity").cast(pl.Float64),
            pl.col("green_export_share").cast(pl.Float64),
            pl.col("domestic_value_added_share").cast(pl.Float64),
            pl.col("foreign_value_added_dependence").cast(pl.Float64),
            pl.col("asinh_weighted_green_imports").cast(pl.Float64),
        )
        .sort(["year", "economy_id"])
    )


def build_outcome_tables(paths: ProjectPaths, *, build: BuildIdentity) -> OutcomeBuildReport:
    """Build the authoritative annual bases and non-horizon product primitives."""

    wdi_path = paths.normalized / "wdi/wdi_country_year.parquet"
    irena_path = paths.normalized / "irena/irena_country_year.parquet"
    gpci_path = paths.measures / "trade/gpci_product_year.parquet"
    tiva_path = paths.measures / "tiva/tiva_measures.parquet"
    trade_components_path = paths.measures / "trade/trade_components_raw.parquet"
    registry_path = paths.code_root / "02_数据字典/product_registry_hs96_v1.parquet"
    green_export_frames: list[pl.DataFrame] = []
    all_export_frames: list[pl.DataFrame] = []
    totals_frames: list[pl.DataFrame] = []
    country_input_paths: list[Path] = [
        wdi_path,
        irena_path,
        gpci_path,
        tiva_path,
        trade_components_path,
        _OUTCOME_GAD_MAP,
    ]
    product_input_paths: list[Path] = [registry_path]
    for year in _YEARS:
        green_path = (
            paths.normalized / "baci/green_economy_product" / f"year={year}"
            / "taxonomy_version=main_hs96.parquet"
        )
        exports_path = (
            paths.normalized / "baci/exporter_product" / f"year={year}"
            / "taxonomy_version=all_hs96.parquet"
        )
        totals_path = (
            paths.normalized / "baci/economy_year_totals" / f"year={year}"
            / "taxonomy_version=all_hs96.parquet"
        )
        country_input_paths.extend((green_path, totals_path))
        product_input_paths.extend((exports_path, totals_path))
        green_export_frames.append(
            pl.read_parquet(green_path)
            .filter(pl.col("flow_role") == "exporter")
            .select(
                "economy_id",
                "year",
                "hs6",
                pl.col("weighted_green_trade_usd").alias("weighted_green_export_usd"),
            )
        )
        all_export_frames.append(
            pl.read_parquet(exports_path)
            .filter(pl.col("flow_role") == "exporter")
            .select("economy_id", "year", "hs6", pl.col("trade_value_usd").alias("export_usd"))
        )
        totals_frames.append(
            pl.read_parquet(totals_path).select(
                "economy_id",
                "year",
                pl.col("exports_usd").alias("total_export_usd"),
                pl.col("reported_as_exporter").alias("export_coverage_normal"),
            )
        )
    all_exports = pl.concat(all_export_frames)
    totals = pl.concat(totals_frames)
    green_products = (
        pl.read_parquet(registry_path)
        .filter((pl.col("list_name") == "main") & (pl.col("green_weight") > 0.0))
        .select(pl.col("hs96").alias("hs6"))
    )
    country = _country_outcomes(
        wdi=pl.read_parquet(wdi_path),
        irena=pl.read_parquet(irena_path),
        green_exports=pl.concat(green_export_frames),
        totals=totals.select("economy_id", "year", "total_export_usd"),
        gpci=pl.read_parquet(gpci_path).filter(pl.col("taxonomy_version") == "main_hs96"),
        tiva=pl.read_parquet(tiva_path),
        trade_components=pl.read_parquet(trade_components_path),
    )
    product = build_product_entry_inputs(
        all_exports,
        totals.select("economy_id", "year", "export_coverage_normal"),
        green_products,
    )
    country_path = paths.measures / "outcomes/outcomes_country_year.parquet"
    product_path = paths.measures / "outcomes/outcomes_product_year.parquet"
    write_authoritative_table(
        country,
        _contract("outcomes_country_year.json"),
        country_path,
        tuple(InputArtifact.from_path(path) for path in country_input_paths),
        build,
    )
    write_authoritative_table(
        product,
        _contract("outcomes_product_year.json"),
        product_path,
        tuple(InputArtifact.from_path(path) for path in product_input_paths),
        build,
    )
    return OutcomeBuildReport(
        country_rows=country.height,
        product_rows=product.height,
        country_output_path=str(country_path),
        product_output_path=str(product_path),
    )


def audit_outcomes(
    paths: ProjectPaths,
    *,
    audit_path: Path,
    specs: tuple[OutcomeSpec, ...] | None = None,
) -> dict[str, int | str]:
    """Audit independent outcome lineage, mappings, units, signs, and key uniqueness."""

    country_path = paths.measures / "outcomes/outcomes_country_year.parquet"
    product_path = paths.measures / "outcomes/outcomes_product_year.parquet"
    country_manifest = verify_manifest(country_path.with_name(f"{country_path.name}.manifest.json"))
    product_manifest = verify_manifest(product_path.with_name(f"{product_path.name}.manifest.json"))
    records = outcome_specs() if specs is None else specs
    mapping = load_outcome_gad_map(_OUTCOME_GAD_MAP)
    available_variants = {specification.specification_id for specification in registered_gad_specifications()}
    unknown_mappings = sum(variant not in available_variants for variant in mapping.outcomes.values())
    by_outcome = {spec.outcome_id: spec for spec in records}
    missing_specs = sum(outcome not in by_outcome for outcome in mapping.outcomes)
    mismatched_variants = sum(
        by_outcome[outcome].gad_variant != variant
        for outcome, variant in mapping.outcomes.items()
        if outcome in by_outcome
    )
    unregistered_specs = sum(spec.gad_variant not in available_variants for spec in records)
    country_contract = _contract("outcomes_country_year.json")
    product_contract = _contract("outcomes_product_year.json")
    contracts = {
        country_contract.table_id: country_contract,
        product_contract.table_id: product_contract,
    }
    manifests = {
        country_contract.table_id: country_manifest,
        product_contract.table_id: product_manifest,
    }
    materialized_specs = [spec for spec in records if spec.materialized]
    unit_failures = sum(
        spec.authority_table not in contracts
        or spec.outcome_id not in contracts[spec.authority_table].units
        or contracts[spec.authority_table].units[spec.outcome_id] != spec.unit
        or spec.authority_table not in manifests
        or manifests[spec.authority_table].units.get(spec.outcome_id) != spec.unit
        for spec in materialized_specs
    )
    registry = pl.read_csv(paths.code_root / "02_数据字典/indicator_registry_v1.csv")
    energy_registry = registry.filter(
        (pl.col("source_id") == "wdi")
        & (pl.col("source_field") == _WDI_ENERGY_INTENSITY)
    )
    if energy_registry.height != 1:
        raise RuntimeError("frozen energy-intensity registry row is not unique")
    energy_spec = by_outcome.get("energy_intensity_mj_per_ppp_gdp")
    energy_unit_failures = int(
        energy_spec is None
        or energy_spec.unit != energy_registry.get_column("unit").item()
    )
    sign_failures = sum(
        spec.orientation not in _VALID_ORIENTATIONS for spec in records
    )
    interpolation_count = sum(
        "interpolated" in spec.transformation_flags
        or "uninterpolated" not in spec.transformation_flags
        for spec in records
    )
    transformation_flag_failures = sum(
        not _REQUIRED_RAW_FLAGS <= spec.transformation_flags for spec in records
    )
    gad_formula_columns = sum(
        bool(set(spec.formula_inputs) & _GAD_COMPONENT_COLUMNS) for spec in records
    )
    contract_rawness_failures = int(
        "retain_raw_unwinsorized_uninterpolated_annual_outcome_bases"
        not in country_contract.transformations
    )
    gad_value_table_parents = sum(
        "/measures/gad/" in artifact.path.replace("\\", "/")
        for manifest in (country_manifest, product_manifest)
        for artifact in manifest.input_artifacts
    )
    checks = {
        "country_year_duplicate_keys": country_manifest.duplicate_primary_keys,
        "product_year_duplicate_keys": product_manifest.duplicate_primary_keys,
        "outcome_interpolation_count": interpolation_count,
        "unknown_outcome_mappings": unknown_mappings + missing_specs + mismatched_variants + unregistered_specs,
        "unit_registration_failures": unit_failures + energy_unit_failures,
        "sign_registration_failures": sign_failures,
        "transformation_flag_failures": transformation_flag_failures + contract_rawness_failures,
        "gad_component_formula_lineage_columns": gad_formula_columns,
        "gad_value_table_parents": gad_value_table_parents,
    }
    audit = pl.DataFrame(
        {
            "metric": list(checks),
            "value": list(checks.values()),
            "expected": [0] * len(checks),
            "status": ["pass" if value == 0 else "fail" for value in checks.values()],
        }
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.write_csv(audit_path)
    if any(value != 0 for value in checks.values()):
        raise RuntimeError(f"outcome audit failed: {checks}")
    return {**checks, "status": "valid", "audit_path": str(audit_path)}
