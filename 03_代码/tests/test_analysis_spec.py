from pathlib import Path
import shutil

import pytest
import yaml

from green_debt.analysis_spec import load_analysis_spec


ROOT = Path(__file__).resolve().parents[2]


def _isolated_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in ("analysis.yaml", "project.yaml", "outcome_gad_map.yaml"):
        shutil.copy2(ROOT / "config" / name, config_dir / name)
    return config_dir


def test_confirmatory_grid_has_only_the_frozen_39_cells() -> None:
    spec = load_analysis_spec(ROOT / "config" / "analysis.yaml")

    expected = {
        "co2_tonnes_per_million_current_usd": (
            (1, 2, 3), "gad_core", "h1_primary", "gimc_a", "negative"
        ),
        "renewable_capacity_additions_mw_per_million": (
            (1, 2, 3), "gad_core", "h1_deployment", "gimc_a", "positive"
        ),
        "energy_intensity_mj_per_ppp_gdp": (
            (1, 2, 3), "gad_core", "h1_secondary", "gimc_a", "negative"
        ),
        "future_green_rca_entry_rate": (
            (3, 4, 5, 6, 7, 8),
            "gad_no_supp",
            "industrial",
            "gimc_gad_a",
            "negative",
        ),
        "green_export_complexity": (
            (3, 4, 5, 6, 7, 8),
            "gad_no_supp",
            "h2_h3_primary",
            "gimc_gad_a",
            "negative",
        ),
        "green_export_share": (
            (3, 4, 5, 6, 7, 8),
            "gad_no_supp",
            "industrial",
            "gimc_gad_a",
            "negative",
        ),
        "domestic_value_added_share": (
            (3, 4, 5, 6, 7, 8),
            "gad_no_gfvad",
            "h3_primary",
            "gimc_gad_a",
            "negative",
        ),
        "foreign_value_added_dependence": (
            (3, 4, 5, 6, 7, 8),
            "gad_no_gfvad",
            "dependence",
            "gimc_gad_a",
            "positive",
        ),
    }
    observed = {
        item.outcome_id: (
            item.horizons,
            item.gad_version,
            item.role,
            item.target_term,
            item.expected_sign,
        )
        for item in spec.outcomes
    }

    assert observed == expected
    assert len(spec.confirmatory_cells()) == 39
    assert all(cell.horizon != 0 for cell in spec.confirmatory_cells())


def test_registered_secondary_grids_are_literal() -> None:
    spec = load_analysis_spec(ROOT / "config" / "analysis.yaml")

    assert len(spec.robustness_cells()) == 39
    assert {cell.sample_version for cell in spec.robustness_cells()} == {
        "core_bounded_controls"
    }
    assert {cell.analysis_family for cell in spec.robustness_cells()} == {
        "bounded_controls"
    }
    assert {
        (cell.outcome_id, cell.horizon, cell.gad_version)
        for cell in spec.vulnerability_cells()
    } == {
        ("asinh_weighted_green_imports", 1, "gad_core"),
        ("asinh_weighted_green_imports", 2, "gad_core"),
        ("asinh_weighted_green_imports", 3, "gad_core"),
        ("renewable_capacity_additions_mw_per_million", 1, "gad_core"),
        ("renewable_capacity_additions_mw_per_million", 2, "gad_core"),
        ("renewable_capacity_additions_mw_per_million", 3, "gad_core"),
    }
    assert {cell.sample_version for cell in spec.vulnerability_cells()} == {
        "core_complete_case_negative_shock"
    }


def test_threshold_and_inference_settings_are_frozen() -> None:
    spec = load_analysis_spec(ROOT / "config" / "analysis.yaml")
    cell = spec.threshold_cell()

    assert (cell.outcome_id, cell.horizon, cell.gad_version) == (
        "green_industrial_upgrading_index",
        5,
        "gad_no_supp",
    )
    assert spec.threshold.percentiles == tuple(range(20, 81))
    assert spec.threshold.minimum_regime_share == 0.20
    assert spec.threshold.criterion == "minimum_two_way_fe_ssr"
    assert spec.threshold.quantile_type == 7
    assert spec.threshold.tie_break == "lowest_percentile_then_lowest_q"
    assert spec.seed == 20260820
    assert spec.period == (2000, 2022)
    assert spec.fixed_effects == ("economy_id", "treatment_time")
    assert spec.cluster == "economy_id"
    assert spec.inference.wild_bootstrap_draws == 9999
    assert spec.inference.weak_f_reference == 10.0
    assert spec.inference.marginal_gad_quantiles == (0.25, 0.50, 0.75)
    assert spec.inference.ar_grid_points_per_axis == 121
    assert spec.outputs.quota_gb == 10


def test_analysis_spec_rejects_outcome_gad_mapping_drift(tmp_path: Path) -> None:
    config_dir = _isolated_config(tmp_path)
    mapping_path = config_dir / "outcome_gad_map.yaml"
    payload = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    payload["outcomes"]["green_export_complexity"] = "gad_core"
    mapping_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="GAD mapping mismatch"):
        load_analysis_spec(config_dir / "analysis.yaml")


def test_analysis_spec_rejects_project_period_drift(tmp_path: Path) -> None:
    config_dir = _isolated_config(tmp_path)
    project_path = config_dir / "project.yaml"
    payload = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    payload["period"]["main"] = [2001, 2022]
    project_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="project main period"):
        load_analysis_spec(config_dir / "analysis.yaml")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("search_percentiles", [25, 75]),
        ("minimum_regime_share", 0.25),
        ("selection_outcome", "green_export_complexity"),
    ],
)
def test_analysis_spec_rejects_project_threshold_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    config_dir = _isolated_config(tmp_path)
    project_path = config_dir / "project.yaml"
    payload = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    payload["threshold"][field] = value
    project_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="project threshold"):
        load_analysis_spec(config_dir / "analysis.yaml")


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("columns", "treatment"), "gimc"),
        (("outcomes", 0, "expected_sign"), "positive"),
        (("robustness", 0, "robustness_id"), "unregistered"),
        (("threshold", "horizon"), 4),
        (("inference", "wild_bootstrap_draws"), 999),
        (("outputs", "root"), "06_结果/unregistered"),
    ],
)
def test_analysis_spec_rejects_semantic_drift_inside_analysis_yaml(
    tmp_path: Path, field_path: tuple[object, ...], value: object
) -> None:
    config_dir = _isolated_config(tmp_path)
    analysis_path = config_dir / "analysis.yaml"
    payload = yaml.safe_load(analysis_path.read_text(encoding="utf-8"))
    target = payload
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value
    analysis_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen analysis"):
        load_analysis_spec(analysis_path)
