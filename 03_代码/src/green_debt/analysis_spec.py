"""Frozen, typed specification for the green absorptive-debt analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelCell:
    outcome_id: str
    horizon: int
    gad_version: str
    sample_version: str
    role: str
    analysis_family: str = "confirmatory"


@dataclass(frozen=True)
class OutcomeAnalysisSpec:
    outcome_id: str
    horizons: tuple[int, ...]
    gad_version: str
    role: str
    target_term: str
    expected_sign: str


@dataclass(frozen=True)
class AnalysisColumns:
    outcome: str
    treatment: str
    gad: str
    instrument: str
    interaction_instrument: str
    controls: tuple[str, ...]


@dataclass(frozen=True)
class RobustnessSpec:
    robustness_id: str
    sample_version: str


@dataclass(frozen=True)
class VulnerabilityOutcomeSpec:
    outcome_id: str
    horizons: tuple[int, ...]
    gad_version: str


@dataclass(frozen=True)
class VulnerabilitySpec:
    sample_version: str
    outcomes: tuple[VulnerabilityOutcomeSpec, ...]


@dataclass(frozen=True)
class InferenceSpec:
    exploratory_below_clusters: int
    wild_bootstrap_below_clusters: int
    wild_bootstrap_draws: int
    weak_f_reference: float
    marginal_gad_quantiles: tuple[float, ...]
    ar_grid_points_per_axis: int
    ar_initial_se_span: float
    ar_max_expansions: int


@dataclass(frozen=True)
class OutputSpec:
    root: str
    quota_gb: int


@dataclass(frozen=True)
class ThresholdSpec:
    outcome_id: str
    horizon: int
    gad_version: str
    sample_version: str
    percentiles: tuple[int, ...]
    minimum_regime_share: float
    criterion: str
    quantile_type: int
    tie_break: str


@dataclass(frozen=True)
class AnalysisSpec:
    schema_version: int
    spec_id: str
    seed: int
    period: tuple[int, int]
    sample_version: str
    columns: AnalysisColumns
    fixed_effects: tuple[str, ...]
    cluster: str
    outcomes: tuple[OutcomeAnalysisSpec, ...]
    robustness: tuple[RobustnessSpec, ...]
    vulnerability: VulnerabilitySpec
    threshold: ThresholdSpec
    inference: InferenceSpec
    outputs: OutputSpec

    def confirmatory_cells(self) -> tuple[ModelCell, ...]:
        return tuple(
            ModelCell(
                item.outcome_id,
                horizon,
                item.gad_version,
                self.sample_version,
                item.role,
            )
            for item in self.outcomes
            for horizon in item.horizons
        )

    def threshold_cell(self) -> ModelCell:
        value = self.threshold
        return ModelCell(
            value.outcome_id,
            value.horizon,
            value.gad_version,
            value.sample_version,
            "threshold_selection",
            "threshold_selection",
        )

    def robustness_cells(self) -> tuple[ModelCell, ...]:
        return tuple(
            replace(
                cell,
                sample_version=item.sample_version,
                analysis_family=item.robustness_id,
            )
            for item in self.robustness
            for cell in self.confirmatory_cells()
        )

    def vulnerability_cells(self) -> tuple[ModelCell, ...]:
        return tuple(
            ModelCell(
                item.outcome_id,
                horizon,
                item.gad_version,
                self.vulnerability.sample_version,
                "vulnerability",
                "vulnerability",
            )
            for item in self.vulnerability.outcomes
            for horizon in item.horizons
        )

    def registered_cells(self) -> tuple[ModelCell, ...]:
        return (
            *self.confirmatory_cells(),
            *self.robustness_cells(),
            self.threshold_cell(),
            *self.vulnerability_cells(),
        )

    def require_registered_cell(self, cell: ModelCell) -> None:
        registered = self.registered_cells()
        if cell in registered:
            return
        for expected in registered:
            if (
                expected.outcome_id == cell.outcome_id
                and expected.horizon == cell.horizon
                and expected.sample_version == cell.sample_version
                and expected.role == cell.role
                and expected.analysis_family == cell.analysis_family
                and expected.gad_version != cell.gad_version
            ):
                raise ValueError(
                    f"GAD mapping mismatch for registered analysis cell: {cell}"
                )
        raise ValueError(f"unregistered analysis cell: {cell}")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _integer_tuple(value: Any, field: str) -> tuple[int, ...]:
    result = tuple(
        _integer(item, f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field))
    )
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    result = tuple(
        _string(item, f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field))
    )
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _outcome_spec(value: Any, index: int) -> OutcomeAnalysisSpec:
    item = _mapping(value, f"outcomes[{index}]")
    return OutcomeAnalysisSpec(
        outcome_id=_string(item.get("outcome_id"), f"outcomes[{index}].outcome_id"),
        horizons=_integer_tuple(item.get("horizons"), f"outcomes[{index}].horizons"),
        gad_version=_string(item.get("gad_version"), f"outcomes[{index}].gad_version"),
        role=_string(item.get("role"), f"outcomes[{index}].role"),
        target_term=_string(item.get("target_term"), f"outcomes[{index}].target_term"),
        expected_sign=_string(item.get("expected_sign"), f"outcomes[{index}].expected_sign"),
    )


def _vulnerability_outcome(value: Any, index: int) -> VulnerabilityOutcomeSpec:
    item = _mapping(value, f"vulnerability.outcomes[{index}]")
    return VulnerabilityOutcomeSpec(
        outcome_id=_string(
            item.get("outcome_id"), f"vulnerability.outcomes[{index}].outcome_id"
        ),
        horizons=_integer_tuple(
            item.get("horizons"), f"vulnerability.outcomes[{index}].horizons"
        ),
        gad_version=_string(
            item.get("gad_version"), f"vulnerability.outcomes[{index}].gad_version"
        ),
    )


def _parse_spec(payload: Mapping[str, Any]) -> AnalysisSpec:
    columns = _mapping(payload.get("columns"), "columns")
    robustness_values = _sequence(payload.get("robustness"), "robustness")
    vulnerability = _mapping(payload.get("vulnerability"), "vulnerability")
    threshold = _mapping(payload.get("threshold"), "threshold")
    inference = _mapping(payload.get("inference"), "inference")
    outputs = _mapping(payload.get("outputs"), "outputs")
    period = _integer_tuple(payload.get("period"), "period")
    if len(period) != 2:
        raise ValueError("period must contain exactly two years")
    threshold_bounds = _integer_tuple(
        threshold.get("percentiles"), "threshold.percentiles"
    )
    if len(threshold_bounds) != 2 or threshold_bounds[0] > threshold_bounds[1]:
        raise ValueError("threshold.percentiles must be ascending endpoints")

    return AnalysisSpec(
        schema_version=_integer(payload.get("schema_version"), "schema_version"),
        spec_id=_string(payload.get("spec_id"), "spec_id"),
        seed=_integer(payload.get("seed"), "seed", positive=True),
        period=(period[0], period[1]),
        sample_version=_string(payload.get("sample_version"), "sample_version"),
        columns=AnalysisColumns(
            outcome=_string(columns.get("outcome"), "columns.outcome"),
            treatment=_string(columns.get("treatment"), "columns.treatment"),
            gad=_string(columns.get("gad"), "columns.gad"),
            instrument=_string(columns.get("instrument"), "columns.instrument"),
            interaction_instrument=_string(
                columns.get("interaction_instrument"),
                "columns.interaction_instrument",
            ),
            controls=_string_tuple(columns.get("controls"), "columns.controls"),
        ),
        fixed_effects=_string_tuple(payload.get("fixed_effects"), "fixed_effects"),
        cluster=_string(payload.get("cluster"), "cluster"),
        outcomes=tuple(
            _outcome_spec(value, index)
            for index, value in enumerate(_sequence(payload.get("outcomes"), "outcomes"))
        ),
        robustness=tuple(
            RobustnessSpec(
                robustness_id=_string(
                    _mapping(value, f"robustness[{index}]").get("robustness_id"),
                    f"robustness[{index}].robustness_id",
                ),
                sample_version=_string(
                    _mapping(value, f"robustness[{index}]").get("sample_version"),
                    f"robustness[{index}].sample_version",
                ),
            )
            for index, value in enumerate(robustness_values)
        ),
        vulnerability=VulnerabilitySpec(
            sample_version=_string(
                vulnerability.get("sample_version"), "vulnerability.sample_version"
            ),
            outcomes=tuple(
                _vulnerability_outcome(value, index)
                for index, value in enumerate(
                    _sequence(vulnerability.get("outcomes"), "vulnerability.outcomes")
                )
            ),
        ),
        threshold=ThresholdSpec(
            outcome_id=_string(threshold.get("outcome_id"), "threshold.outcome_id"),
            horizon=_integer(threshold.get("horizon"), "threshold.horizon", positive=True),
            gad_version=_string(threshold.get("gad_version"), "threshold.gad_version"),
            sample_version=_string(
                threshold.get("sample_version"), "threshold.sample_version"
            ),
            percentiles=tuple(range(threshold_bounds[0], threshold_bounds[1] + 1)),
            minimum_regime_share=_number(
                threshold.get("minimum_regime_share"),
                "threshold.minimum_regime_share",
            ),
            criterion=_string(threshold.get("criterion"), "threshold.criterion"),
            quantile_type=_integer(threshold.get("quantile_type"), "threshold.quantile_type"),
            tie_break=_string(threshold.get("tie_break"), "threshold.tie_break"),
        ),
        inference=InferenceSpec(
            exploratory_below_clusters=_integer(
                inference.get("exploratory_below_clusters"),
                "inference.exploratory_below_clusters",
                positive=True,
            ),
            wild_bootstrap_below_clusters=_integer(
                inference.get("wild_bootstrap_below_clusters"),
                "inference.wild_bootstrap_below_clusters",
                positive=True,
            ),
            wild_bootstrap_draws=_integer(
                inference.get("wild_bootstrap_draws"),
                "inference.wild_bootstrap_draws",
                positive=True,
            ),
            weak_f_reference=_number(
                inference.get("weak_f_reference"), "inference.weak_f_reference"
            ),
            marginal_gad_quantiles=tuple(
                _number(value, f"inference.marginal_gad_quantiles[{index}]")
                for index, value in enumerate(
                    _sequence(
                        inference.get("marginal_gad_quantiles"),
                        "inference.marginal_gad_quantiles",
                    )
                )
            ),
            ar_grid_points_per_axis=_integer(
                inference.get("ar_grid_points_per_axis"),
                "inference.ar_grid_points_per_axis",
                positive=True,
            ),
            ar_initial_se_span=_number(
                inference.get("ar_initial_se_span"), "inference.ar_initial_se_span"
            ),
            ar_max_expansions=_integer(
                inference.get("ar_max_expansions"),
                "inference.ar_max_expansions",
                positive=True,
            ),
        ),
        outputs=OutputSpec(
            root=_string(outputs.get("root"), "outputs.root"),
            quota_gb=_integer(outputs.get("quota_gb"), "outputs.quota_gb", positive=True),
        ),
    )


def _validate_frozen_values(spec: AnalysisSpec) -> None:
    expected_columns = AnalysisColumns(
        outcome="delta_outcome",
        treatment="gimc_p01_p99",
        gad="gad_lag_p01_p99",
        instrument="Z_p01_p99",
        interaction_instrument="Z_GAD_p01_p99",
        controls=(
            "renewable_energy_consumption_share_analysis_p01_p99",
            "trade_openness_percent_gdp_analysis_p01_p99",
            "industry_value_added_share_analysis_p01_p99",
            "gdp_per_capita_current_usd_analysis_p01_p99",
        ),
    )
    expected_outcomes = (
        OutcomeAnalysisSpec(
            "co2_tonnes_per_million_current_usd",
            (1, 2, 3),
            "gad_core",
            "h1_primary",
            "gimc_a",
            "negative",
        ),
        OutcomeAnalysisSpec(
            "renewable_capacity_additions_mw_per_million",
            (1, 2, 3),
            "gad_core",
            "h1_deployment",
            "gimc_a",
            "positive",
        ),
        OutcomeAnalysisSpec(
            "energy_intensity_mj_per_ppp_gdp",
            (1, 2, 3),
            "gad_core",
            "h1_secondary",
            "gimc_a",
            "negative",
        ),
        OutcomeAnalysisSpec(
            "future_green_rca_entry_rate",
            (3, 4, 5, 6, 7, 8),
            "gad_no_supp",
            "industrial",
            "gimc_gad_a",
            "negative",
        ),
        OutcomeAnalysisSpec(
            "green_export_complexity",
            (3, 4, 5, 6, 7, 8),
            "gad_no_supp",
            "h2_h3_primary",
            "gimc_gad_a",
            "negative",
        ),
        OutcomeAnalysisSpec(
            "green_export_share",
            (3, 4, 5, 6, 7, 8),
            "gad_no_supp",
            "industrial",
            "gimc_gad_a",
            "negative",
        ),
        OutcomeAnalysisSpec(
            "domestic_value_added_share",
            (3, 4, 5, 6, 7, 8),
            "gad_no_gfvad",
            "h3_primary",
            "gimc_gad_a",
            "negative",
        ),
        OutcomeAnalysisSpec(
            "foreign_value_added_dependence",
            (3, 4, 5, 6, 7, 8),
            "gad_no_gfvad",
            "dependence",
            "gimc_gad_a",
            "positive",
        ),
    )
    expected_vulnerability = VulnerabilitySpec(
        sample_version="core_complete_case_negative_shock",
        outcomes=(
            VulnerabilityOutcomeSpec(
                "asinh_weighted_green_imports", (1, 2, 3), "gad_core"
            ),
            VulnerabilityOutcomeSpec(
                "renewable_capacity_additions_mw_per_million",
                (1, 2, 3),
                "gad_core",
            ),
        ),
    )
    expected_threshold = ThresholdSpec(
        outcome_id="green_industrial_upgrading_index",
        horizon=5,
        gad_version="gad_no_supp",
        sample_version="core_complete_case",
        percentiles=tuple(range(20, 81)),
        minimum_regime_share=0.20,
        criterion="minimum_two_way_fe_ssr",
        quantile_type=7,
        tie_break="lowest_percentile_then_lowest_q",
    )
    expected_inference = InferenceSpec(
        exploratory_below_clusters=20,
        wild_bootstrap_below_clusters=30,
        wild_bootstrap_draws=9999,
        weak_f_reference=10.0,
        marginal_gad_quantiles=(0.25, 0.50, 0.75),
        ar_grid_points_per_axis=121,
        ar_initial_se_span=6.0,
        ar_max_expansions=4,
    )
    frozen_sections = {
        "identity": (spec.schema_version, spec.spec_id, spec.seed),
        "period_and_sample": (spec.period, spec.sample_version),
        "columns": spec.columns,
        "fixed_effects_and_cluster": (spec.fixed_effects, spec.cluster),
        "outcomes": spec.outcomes,
        "robustness": spec.robustness,
        "vulnerability": spec.vulnerability,
        "threshold": spec.threshold,
        "inference": spec.inference,
        "outputs": spec.outputs,
    }
    expected_sections = {
        "identity": (1, "gad_lp_iv_v1", 20260820),
        "period_and_sample": ((2000, 2022), "core_complete_case"),
        "columns": expected_columns,
        "fixed_effects_and_cluster": (
            ("economy_id", "treatment_time"),
            "economy_id",
        ),
        "outcomes": expected_outcomes,
        "robustness": (RobustnessSpec("bounded_controls", "core_bounded_controls"),),
        "vulnerability": expected_vulnerability,
        "threshold": expected_threshold,
        "inference": expected_inference,
        "outputs": OutputSpec("06_结果/analysis", 10),
    }
    drifted = [
        name
        for name, value in frozen_sections.items()
        if value != expected_sections[name]
    ]
    if drifted:
        raise ValueError(
            "frozen analysis specification drift: " + ", ".join(drifted)
        )

    if spec.schema_version != 1 or spec.spec_id != "gad_lp_iv_v1":
        raise ValueError("unexpected analysis schema or spec identity")
    if spec.seed != 20260820 or spec.period != (2000, 2022):
        raise ValueError("unexpected analysis seed or period")
    if spec.sample_version != "core_complete_case":
        raise ValueError("unexpected main sample version")
    if spec.fixed_effects != ("economy_id", "treatment_time"):
        raise ValueError("fixed effects must be economy_id and treatment_time")
    if spec.cluster != "economy_id":
        raise ValueError("cluster must be economy_id")
    if len(spec.confirmatory_cells()) != 39:
        raise ValueError("analysis must register exactly 39 confirmatory cells")
    if len(spec.robustness_cells()) != 39:
        raise ValueError("analysis must register exactly 39 robustness cells")
    if len(spec.vulnerability_cells()) != 6:
        raise ValueError("analysis must register exactly six vulnerability cells")
    if any(cell.horizon == 0 for cell in spec.confirmatory_cells()):
        raise ValueError("h=0 cannot be confirmatory")
    if len(set(spec.confirmatory_cells())) != 39:
        raise ValueError("confirmatory cells must be unique")
    if spec.threshold.percentiles != tuple(range(20, 81)):
        raise ValueError("threshold grid must be integer percentiles 20 through 80")
    if spec.threshold.minimum_regime_share != 0.20:
        raise ValueError("threshold minimum regime share must be 0.20")
    if spec.threshold.criterion != "minimum_two_way_fe_ssr":
        raise ValueError("unexpected threshold criterion")
    if spec.threshold.quantile_type != 7:
        raise ValueError("threshold quantile type must be 7")
    if spec.threshold.tie_break != "lowest_percentile_then_lowest_q":
        raise ValueError("unexpected threshold tie break")
    if any(item.expected_sign not in {"positive", "negative"} for item in spec.outcomes):
        raise ValueError("expected signs must be positive or negative")


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load YAML {path}: {exc}") from exc
    return _mapping(value, str(path))


def _validate_outcome_mapping(spec: AnalysisSpec, config_dir: Path) -> None:
    outcome_map = _load_yaml_mapping(config_dir / "outcome_gad_map.yaml")
    mapped = _mapping(outcome_map.get("outcomes"), "outcome_gad_map.outcomes")
    required = {
        item.outcome_id: item.gad_version for item in spec.outcomes
    } | {
        item.outcome_id: item.gad_version for item in spec.vulnerability.outcomes
    } | {spec.threshold.outcome_id: spec.threshold.gad_version}
    mismatches = {
        outcome_id: (mapped.get(outcome_id), gad_version)
        for outcome_id, gad_version in required.items()
        if mapped.get(outcome_id) != gad_version
    }
    if mismatches:
        raise ValueError(f"analysis GAD mapping mismatch: {mismatches}")


def _validate_project_period(spec: AnalysisSpec, config_dir: Path) -> None:
    project = _load_yaml_mapping(config_dir / "project.yaml")
    period = _mapping(project.get("period"), "project.period")
    if tuple(period.get("main", ())) != spec.period:
        raise ValueError("analysis period does not match project main period")


def _validate_project_threshold(spec: AnalysisSpec, config_dir: Path) -> None:
    project = _load_yaml_mapping(config_dir / "project.yaml")
    threshold = _mapping(project.get("threshold"), "project.threshold")
    expected = {
        "search_percentiles": [
            spec.threshold.percentiles[0], spec.threshold.percentiles[-1]
        ],
        "minimum_regime_share": spec.threshold.minimum_regime_share,
        "selection_outcome": spec.threshold.outcome_id,
    }
    if any(threshold.get(field) != value for field, value in expected.items()):
        raise ValueError("analysis project threshold binding mismatch")


def load_analysis_spec(path: Path) -> AnalysisSpec:
    """Load the frozen analysis YAML and its outcome-to-GAD binding."""

    resolved = path.resolve()
    spec = _parse_spec(_load_yaml_mapping(resolved))
    _validate_frozen_values(spec)
    _validate_outcome_mapping(spec, resolved.parent)
    _validate_project_period(spec, resolved.parent)
    _validate_project_threshold(spec, resolved.parent)
    return spec
