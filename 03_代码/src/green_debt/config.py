"""Typed, validated project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


YearRange = tuple[int, int]


def _year_range(value: Any, field: str) -> YearRange:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must contain exactly two years")
    start, end = value
    if not isinstance(start, int) or not isinstance(end, int) or start > end:
        raise ValueError(f"{field} must be an ascending integer year range")
    return start, end


def _integer_tuple(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty integer list")
    if not all(isinstance(item, int) for item in value):
        raise ValueError(f"{field} must contain integers")
    return tuple(value)


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty string list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{field} must contain non-empty strings")
    return tuple(value)


def _strict_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


@dataclass(frozen=True)
class PeriodConfig:
    openalex_history: YearRange
    source_trade: YearRange
    tiva_history: YearRange
    initialization: YearRange
    main: YearRange
    clean_hs07: YearRange
    lite_extension: YearRange


@dataclass(frozen=True)
class StorageConfig:
    absolute_limit_gb: int
    hard_stop_gb: int
    reserve_gb: int
    raw_quota_gb: int | None = None
    intermediate_quota_gb: int | None = None
    temporary_quota_gb: int | None = None
    output_quota_gb: int | None = None


@dataclass(frozen=True)
class NetworkConfig:
    direct_only: bool
    bypass_system_proxy: bool
    rejected_interface_prefixes: tuple[str, ...]
    reject_proxy_environment: bool = True
    require_no_connected_tunnel: bool = True
    allow_logged_user_route_exception: bool = False


@dataclass(frozen=True)
class GADConfig:
    baseline: YearRange
    half_life_years: int
    alternative_half_lives: tuple[int, ...] = ()
    lower_bound: float = 0.0

    @property
    def rho(self) -> float:
        return 2 ** (-1 / self.half_life_years)


@dataclass(frozen=True)
class ThresholdConfig:
    search_percentiles: tuple[int, int]
    minimum_regime_share: float
    selection_outcome: str


@dataclass(frozen=True)
class HorizonConfig:
    environmental: tuple[int, ...]
    environmental_confirmatory: tuple[int, ...]
    industrial: tuple[int, ...]


@dataclass(frozen=True)
class ProjectConfig:
    period: PeriodConfig
    storage: StorageConfig
    network: NetworkConfig
    gad: GADConfig
    threshold: ThresholdConfig
    horizons: HorizonConfig


@dataclass(frozen=True)
class TaxonomyConfig:
    counts: dict[str, int]
    overlap_years: YearRange
    source_hs: str
    main_hs: str
    upstream_bec_uses: tuple[str, ...]


@dataclass(frozen=True)
class OpenAlexConstructionConfig:
    counting_method: str
    include_xpac: bool
    coverage_floor_ratio: float


@dataclass(frozen=True)
class ScalingConfig:
    years: YearRange
    center: str
    scale: str
    fallback: str
    regression_percentiles: tuple[int, int]


@dataclass(frozen=True)
class ConstructionGADConfig:
    start_year: int
    implicit_prior_debt: float
    warmup_observations: int
    half_lives: tuple[int, ...]
    variants: tuple[str, ...]
    scaler_years: YearRange


@dataclass(frozen=True)
class TivaConstructionConfig:
    confirmatory_activities: tuple[str, ...]
    broad_additions: tuple[str, ...]
    equipment_only_removals: tuple[str, ...]
    weight_years: YearRange


@dataclass(frozen=True)
class SampleConfig:
    minimum_population: int
    minimum_valid_main_years: int
    minimum_positive_import_baseline_years: int


@dataclass(frozen=True)
class IVConfig:
    baseline_years: YearRange
    minimum_cell_share: float
    robustness_cell_share: float
    minimum_retained_coverage: float
    cmz_lags: int


@dataclass(frozen=True)
class ConstructionConfig:
    schema_version: int
    taxonomy: TaxonomyConfig
    openalex: OpenAlexConstructionConfig
    scaling: ScalingConfig
    gad: ConstructionGADConfig
    tiva: TivaConstructionConfig
    sample: SampleConfig
    iv: IVConfig


@dataclass(frozen=True)
class OutcomeGADMap:
    schema_version: int
    outcomes: dict[str, str]

    def variant_for(self, outcome: str) -> str:
        try:
            return self.outcomes[outcome]
        except KeyError as exc:
            raise KeyError(f"unregistered outcome: {outcome}") from exc


def _optional_positive_int(mapping: dict[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"storage.{key} must be a positive integer")
    return value


def load_project_config(path: Path) -> ProjectConfig:
    """Load a project YAML file and enforce cross-field design constraints."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "project config")
    period_raw = _mapping(root.get("period"), "period")
    storage_raw = _mapping(root.get("storage"), "storage")
    network_raw = _mapping(root.get("network"), "network")
    gad_raw = _mapping(root.get("gad"), "gad")
    threshold_raw = _mapping(root.get("threshold"), "threshold")
    horizons_raw = _mapping(root.get("horizons"), "horizons")

    storage = StorageConfig(
        absolute_limit_gb=int(storage_raw["absolute_limit_gb"]),
        hard_stop_gb=int(storage_raw["hard_stop_gb"]),
        reserve_gb=int(storage_raw["reserve_gb"]),
        raw_quota_gb=_optional_positive_int(storage_raw, "raw_quota_gb"),
        intermediate_quota_gb=_optional_positive_int(
            storage_raw, "intermediate_quota_gb"
        ),
        temporary_quota_gb=_optional_positive_int(
            storage_raw, "temporary_quota_gb"
        ),
        output_quota_gb=_optional_positive_int(storage_raw, "output_quota_gb"),
    )
    if min(
        storage.absolute_limit_gb,
        storage.hard_stop_gb,
        storage.reserve_gb,
    ) <= 0:
        raise ValueError("storage limits must be positive")
    if storage.hard_stop_gb + storage.reserve_gb > storage.absolute_limit_gb:
        raise ValueError("hard stop plus reserve exceeds absolute limit")

    network = NetworkConfig(
        direct_only=_strict_bool(network_raw["direct_only"], "network.direct_only"),
        bypass_system_proxy=_strict_bool(
            network_raw["bypass_system_proxy"], "network.bypass_system_proxy"
        ),
        rejected_interface_prefixes=tuple(
            str(item).lower()
            for item in network_raw.get("rejected_interface_prefixes", [])
        ),
        reject_proxy_environment=_strict_bool(
            network_raw.get("reject_proxy_environment", True),
            "network.reject_proxy_environment",
        ),
        require_no_connected_tunnel=_strict_bool(
            network_raw.get("require_no_connected_tunnel", True),
            "network.require_no_connected_tunnel",
        ),
        allow_logged_user_route_exception=_strict_bool(
            network_raw.get("allow_logged_user_route_exception", False),
            "network.allow_logged_user_route_exception",
        ),
    )
    if not network.direct_only or not network.bypass_system_proxy:
        raise ValueError("network must enforce direct-only system-proxy bypass")
    if not network.require_no_connected_tunnel:
        raise ValueError("network must require all tunnel services disconnected")

    period = PeriodConfig(
        openalex_history=_year_range(
            period_raw["openalex_history"], "period.openalex_history"
        ),
        source_trade=_year_range(period_raw["source_trade"], "period.source_trade"),
        tiva_history=_year_range(
            period_raw["tiva_history"], "period.tiva_history"
        ),
        initialization=_year_range(
            period_raw["initialization"], "period.initialization"
        ),
        main=_year_range(period_raw["main"], "period.main"),
        clean_hs07=_year_range(period_raw["clean_hs07"], "period.clean_hs07"),
        lite_extension=_year_range(
            period_raw["lite_extension"], "period.lite_extension"
        ),
    )
    if period.openalex_history[1] - period.openalex_history[0] + 1 != 4:
        raise ValueError("period.openalex_history must contain exactly four years")
    if period.openalex_history[1] + 1 != period.source_trade[0]:
        raise ValueError("OpenAlex history must end before the trade source period")
    if period.initialization[1] - period.initialization[0] + 1 != 3:
        raise ValueError("period.initialization must contain exactly three years")
    if period.initialization[1] + 1 != period.main[0]:
        raise ValueError("main period must immediately follow initialization")

    half_life = gad_raw["half_life_years"]
    if not isinstance(half_life, int) or half_life <= 0:
        raise ValueError("gad.half_life_years must be a positive integer")
    gad = GADConfig(
        baseline=_year_range(gad_raw["baseline"], "gad.baseline"),
        half_life_years=half_life,
        alternative_half_lives=tuple(
            int(item) for item in gad_raw.get("alternative_half_lives", [])
        ),
        lower_bound=float(gad_raw.get("lower_bound", 0.0)),
    )
    if not isfinite(gad.rho) or not 0 < gad.rho < 1:
        raise ValueError("gad.rho must lie strictly between zero and one")

    percentiles = _year_range(
        threshold_raw["search_percentiles"],
        "threshold.search_percentiles",
    )
    minimum_share = float(threshold_raw["minimum_regime_share"])
    if not 0 < percentiles[0] < percentiles[1] < 100:
        raise ValueError("threshold percentiles must lie inside 0 and 100")
    if not 0 < minimum_share < 0.5:
        raise ValueError("threshold minimum regime share must lie inside 0 and 0.5")

    config = ProjectConfig(
        period=period,
        storage=storage,
        network=network,
        gad=gad,
        threshold=ThresholdConfig(
            search_percentiles=percentiles,
            minimum_regime_share=minimum_share,
            selection_outcome=str(threshold_raw["selection_outcome"]),
        ),
        horizons=HorizonConfig(
            environmental=_integer_tuple(
                horizons_raw["environmental"], "horizons.environmental"
            ),
            environmental_confirmatory=_integer_tuple(
                horizons_raw["environmental_confirmatory"],
                "horizons.environmental_confirmatory",
            ),
            industrial=_integer_tuple(
                horizons_raw["industrial"], "horizons.industrial"
            ),
        ),
    )
    return config


def load_construction_config(path: Path) -> ConstructionConfig:
    """Load and cross-check the frozen variable-construction contract."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "construction config")
    schema_version = _positive_int(root.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise ValueError("construction schema_version must be 1")

    taxonomy_raw = _mapping(root.get("taxonomy"), "taxonomy")
    counts_raw = _mapping(taxonomy_raw.get("counts"), "taxonomy.counts")
    if set(counts_raw) != {"main", "broad", "apec"}:
        raise ValueError("taxonomy.counts must contain main, broad, and apec")
    counts = {
        name: _positive_int(value, f"taxonomy.counts.{name}")
        for name, value in counts_raw.items()
    }
    if len(set(counts.values())) != len(counts):
        raise ValueError("taxonomy counts must be unique")
    taxonomy = TaxonomyConfig(
        counts=counts,
        overlap_years=_year_range(
            taxonomy_raw["overlap_years"], "taxonomy.overlap_years"
        ),
        source_hs=str(taxonomy_raw["source_hs"]),
        main_hs=str(taxonomy_raw["main_hs"]),
        upstream_bec_uses=_string_tuple(
            taxonomy_raw["upstream_bec_uses"], "taxonomy.upstream_bec_uses"
        ),
    )
    if taxonomy.source_hs == taxonomy.main_hs:
        raise ValueError("source and main HS revisions must differ")

    openalex_raw = _mapping(root.get("openalex"), "openalex")
    coverage_floor = _finite_float(
        openalex_raw["coverage_floor_ratio"], "openalex.coverage_floor_ratio"
    )
    if not 0 <= coverage_floor <= 1:
        raise ValueError("openalex coverage floor must lie in [0,1]")
    openalex = OpenAlexConstructionConfig(
        counting_method=str(openalex_raw["counting_method"]),
        include_xpac=_strict_bool(openalex_raw["include_xpac"], "openalex.include_xpac"),
        coverage_floor_ratio=coverage_floor,
    )

    scaling_raw = _mapping(root.get("scaling"), "scaling")
    regression_percentiles = _year_range(
        scaling_raw["regression_percentiles"], "scaling.regression_percentiles"
    )
    if not 0 < regression_percentiles[0] < regression_percentiles[1] < 100:
        raise ValueError("scaling regression percentiles must lie inside 0 and 100")
    scaling = ScalingConfig(
        years=_year_range(scaling_raw["years"], "scaling.years"),
        center=str(scaling_raw["center"]),
        scale=str(scaling_raw["scale"]),
        fallback=str(scaling_raw["fallback"]),
        regression_percentiles=regression_percentiles,
    )

    gad_raw = _mapping(root.get("gad"), "gad")
    half_lives = _integer_tuple(gad_raw["half_lives"], "gad.half_lives")
    if any(value <= 0 for value in half_lives) or len(set(half_lives)) != len(half_lives):
        raise ValueError("gad.half_lives must be positive and unique")
    variants = _string_tuple(gad_raw["variants"], "gad.variants")
    if len(set(variants)) != len(variants):
        raise ValueError("gad.variants must be unique")
    implicit_prior = _finite_float(
        gad_raw["implicit_prior_debt"], "gad.implicit_prior_debt"
    )
    if implicit_prior < 0:
        raise ValueError("gad.implicit_prior_debt must be nonnegative")
    gad = ConstructionGADConfig(
        start_year=_positive_int(gad_raw["start_year"], "gad.start_year"),
        implicit_prior_debt=implicit_prior,
        warmup_observations=_positive_int(
            gad_raw["warmup_observations"], "gad.warmup_observations"
        ),
        half_lives=half_lives,
        variants=variants,
        scaler_years=scaling.years,
    )

    tiva_raw = _mapping(root.get("tiva"), "tiva")
    confirmatory = _string_tuple(
        tiva_raw["confirmatory_activities"], "tiva.confirmatory_activities"
    )
    removals = _string_tuple(
        tiva_raw["equipment_only_removals"], "tiva.equipment_only_removals"
    )
    if not set(removals) <= set(confirmatory):
        raise ValueError("equipment-only removals must be confirmatory activities")
    tiva = TivaConstructionConfig(
        confirmatory_activities=confirmatory,
        broad_additions=_string_tuple(
            tiva_raw["broad_additions"], "tiva.broad_additions"
        ),
        equipment_only_removals=removals,
        weight_years=_year_range(tiva_raw["weight_years"], "tiva.weight_years"),
    )

    sample_raw = _mapping(root.get("sample"), "sample")
    sample = SampleConfig(
        minimum_population=_positive_int(
            sample_raw["minimum_population"], "sample.minimum_population"
        ),
        minimum_valid_main_years=_positive_int(
            sample_raw["minimum_valid_main_years"],
            "sample.minimum_valid_main_years",
        ),
        minimum_positive_import_baseline_years=_positive_int(
            sample_raw["minimum_positive_import_baseline_years"],
            "sample.minimum_positive_import_baseline_years",
        ),
    )

    iv_raw = _mapping(root.get("iv"), "iv")
    minimum_cell_share = _finite_float(
        iv_raw["minimum_cell_share"], "iv.minimum_cell_share"
    )
    robustness_cell_share = _finite_float(
        iv_raw["robustness_cell_share"], "iv.robustness_cell_share"
    )
    retained_coverage = _finite_float(
        iv_raw["minimum_retained_coverage"], "iv.minimum_retained_coverage"
    )
    if not all(
        0 <= value <= 1
        for value in (minimum_cell_share, robustness_cell_share, retained_coverage)
    ):
        raise ValueError("IV thresholds must lie in [0,1]")
    if minimum_cell_share > robustness_cell_share:
        raise ValueError("IV robustness share cannot be below the main threshold")
    iv = IVConfig(
        baseline_years=_year_range(iv_raw["baseline_years"], "iv.baseline_years"),
        minimum_cell_share=minimum_cell_share,
        robustness_cell_share=robustness_cell_share,
        minimum_retained_coverage=retained_coverage,
        cmz_lags=_positive_int(iv_raw["cmz_lags"], "iv.cmz_lags"),
    )

    config = ConstructionConfig(
        schema_version=schema_version,
        taxonomy=taxonomy,
        openalex=openalex,
        scaling=scaling,
        gad=gad,
        tiva=tiva,
        sample=sample,
        iv=iv,
    )

    project_path = path.with_name("project.yaml")
    if project_path.is_file():
        project = load_project_config(project_path)
        if gad.start_year != project.period.initialization[0]:
            raise ValueError("GAD start must equal the initialization start")
        if gad.warmup_observations != (
            project.period.initialization[1] - project.period.initialization[0] + 1
        ):
            raise ValueError("GAD warm-up count must equal initialization length")
        if not (
            project.period.main[0] <= scaling.years[0]
            <= scaling.years[1] <= project.period.main[1]
        ):
            raise ValueError("scaler years must lie inside the main period")
        if not (
            project.period.source_trade[0] <= iv.baseline_years[0]
            <= iv.baseline_years[1] <= project.period.source_trade[1]
        ):
            raise ValueError("IV baseline must lie inside the trade source period")
        if project.storage.intermediate_quota_gb is None or not (
            project.storage.intermediate_quota_gb
            < project.storage.hard_stop_gb
            < project.storage.absolute_limit_gb
        ):
            raise ValueError("intermediate, hard-stop, and absolute limits are invalid")
    return config


def load_outcome_gad_map(path: Path) -> OutcomeGADMap:
    """Load the fail-closed outcome-to-GAD mapping."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "outcome GAD map")
    schema_version = _positive_int(root.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise ValueError("outcome GAD map schema_version must be 1")
    outcomes_raw = _mapping(root.get("outcomes"), "outcomes")
    if not outcomes_raw:
        raise ValueError("outcomes must not be empty")
    outcomes: dict[str, str] = {}
    for outcome, variant in outcomes_raw.items():
        if not isinstance(outcome, str) or not outcome.strip():
            raise ValueError("outcome names must be non-empty strings")
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError(f"invalid GAD variant for outcome {outcome}")
        outcomes[outcome] = variant
    return OutcomeGADMap(schema_version=schema_version, outcomes=outcomes)
