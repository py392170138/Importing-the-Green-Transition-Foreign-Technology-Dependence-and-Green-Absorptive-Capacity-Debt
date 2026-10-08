from dataclasses import fields
import hashlib
import json

import polars as pl
import pytest

from green_debt.gad import FROZEN_SCALER_HASH
from green_debt.outcomes import OutcomeSpec, fit_giu_scalers, outcome_specs
from green_debt.sample import (
    _audit_parent_derivations,
    build_lp_panel,
    build_vulnerability_panel,
    validate_model_panel,
)


def _annual_fixture() -> pl.DataFrame:
    return pl.DataFrame({
        "economy_id": ["A"] * 10,
        "year": list(range(1999, 2009)),
        "gimc": list(map(float, range(10))),
        "gad_no_supp": list(map(float, range(10))),
        "green_export_complexity": list(map(float, range(10))),
        "control": list(map(float, range(10))),
    })


def test_panel_uses_t_minus_one_state_and_t_plus_h_outcome() -> None:
    out = build_lp_panel(_annual_fixture(), OutcomeSpec("green_export_complexity", "gad_no_supp", (3, 4)))
    row = out.filter((pl.col("treatment_time") == 2000) & (pl.col("horizon") == 3)).row(0, named=True)
    assert row["gad_time"] == 1999
    assert row["control_time"] == 1999
    assert row["baseline_outcome_time"] == 1999
    assert row["outcome_time"] == 2003


def test_outcome_spec_keeps_positional_horizons_and_task14_metadata() -> None:
    assert [field.name for field in fields(OutcomeSpec)][:3] == ["outcome_id", "gad_variant", "horizons"]
    fixture = OutcomeSpec("green_export_complexity", "gad_no_supp", (3, 4))
    assert fixture.horizons == (3, 4)
    registered = next(spec for spec in outcome_specs() if spec.outcome_id == "green_export_complexity")
    assert registered.authority_table == "outcomes_country_year"
    assert registered.orientation == "higher_is_better"
    assert registered.formula_inputs == ("weighted_green_export_usd", "gpci")
    assert "uninterpolated" in registered.transformation_flags


def test_outcome_coverage_is_enforced_for_each_horizon() -> None:
    frame = _annual_fixture().with_columns(
        pl.when(pl.col("year") == 2004)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("green_export_complexity"))
        .alias("green_export_complexity")
    )
    out = build_lp_panel(frame, OutcomeSpec("green_export_complexity", "gad_no_supp", (3, 4)))
    assert out.filter(pl.col("treatment_time") == 2000)["horizon"].to_list() == [3]
    assert out.filter(pl.col("treatment_time") == 2001)["baseline_outcome_time"].item() == 2000


def test_gad_and_outcomes_are_never_interpolated() -> None:
    frame = _annual_fixture().with_columns(
        pl.when(pl.col("year") == 2000).then(None).otherwise(pl.col("gad_no_supp")).alias("gad_no_supp"),
        pl.when(pl.col("year") == 2003).then(None).otherwise(pl.col("green_export_complexity")).alias("green_export_complexity"),
    )
    out = build_lp_panel(frame, OutcomeSpec("green_export_complexity", "gad_no_supp", (3,)))
    assert out.filter(pl.col("treatment_time").is_in([2000, 2001, 2004])).is_empty()


def test_panel_attaches_only_mapped_gad_and_main_iv_key() -> None:
    frame = _annual_fixture().with_columns(
        pl.lit(-0.5).alias("Z"),
        pl.lit(-1.0).alias("Z_GAD"),
        pl.lit(0.25).alias("CMZ"),
        pl.lit("gad_no_supp").alias("iv_gad_version"),
        pl.lit("main_0.0001").alias("iv_share_version"),
        pl.lit(True).alias("confirmatory_iv_eligible"),
    )
    out = build_lp_panel(frame, OutcomeSpec("green_export_complexity", "gad_no_supp", (3,)))
    row = out.filter(pl.col("treatment_time") == 2000).row(0, named=True)
    assert row["gad_version"] == "gad_no_supp"
    assert row["Z"] == pytest.approx(-0.5)
    assert "iv_share_version" not in out.columns

    with pytest.raises(ValueError, match="main_0.0001"):
        build_lp_panel(
            frame.with_columns(pl.lit("all_cells").alias("iv_share_version")),
            OutcomeSpec("green_export_complexity", "gad_no_supp", (3,)),
        )


@pytest.mark.parametrize("column", ["current_import_share", "future_shock"])
def test_panel_rejects_current_share_and_future_shock_columns(column: str) -> None:
    with pytest.raises(ValueError, match=column):
        build_lp_panel(
            _annual_fixture().with_columns(pl.lit(1.0).alias(column)),
            OutcomeSpec("green_export_complexity", "gad_no_supp", (3,)),
        )


def test_future_rca_entry_is_not_differenced_twice() -> None:
    frame = _annual_fixture().with_columns(
        pl.when(pl.col("year") == 1999)
        .then(0.2)
        .when(pl.col("year") == 2003)
        .then(0.75)
        .otherwise(0.0)
        .alias("future_green_rca_entry_rate")
    )
    out = build_lp_panel(frame, OutcomeSpec("future_green_rca_entry_rate", "gad_no_supp", (3,)))
    row = out.filter(pl.col("treatment_time") == 2000).row(0, named=True)
    assert row["delta_outcome"] == pytest.approx(0.75)


def test_giu_scalers_are_horizon_specific_require_all_components_and_are_independent() -> None:
    frame = pl.DataFrame({
        "economy_id": ["A", "B", "C", "A", "B", "C", "D"],
        "treatment_time": [2000, 2001, 2002, 2000, 2001, 2002, 2005],
        "horizon": [3, 3, 3, 4, 4, 4, 3],
        "provisional_core": [True] * 7,
        "future_green_rca_entry_rate": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 999.0],
        "green_export_complexity_change": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 999.0],
        "green_export_share_change": [1.0, 2.0, None, 10.0, 20.0, 30.0, 999.0],
    })
    out, registry = fit_giu_scalers(frame)
    assert registry.by_horizon[3].components["future_green_rca_entry_rate"].center == pytest.approx(1.5)
    assert {component.row_count for component in registry.by_horizon[3].components.values()} == {2}
    assert registry.by_horizon[4].components["future_green_rca_entry_rate"].center == pytest.approx(20.0)
    assert registry.canonical_hash != FROZEN_SCALER_HASH
    assert out.filter((pl.col("economy_id") == "C") & (pl.col("horizon") == 3))["green_industrial_upgrading_index"].item() is None
    assert out.filter((pl.col("economy_id") == "B") & (pl.col("horizon") == 4))["green_industrial_upgrading_index"].item() == pytest.approx(0.0)


def test_vulnerability_panel_keeps_only_negative_shocks_and_unmultiplied_change() -> None:
    panel = pl.DataFrame({
        "outcome_id": ["asinh_weighted_green_imports", "asinh_weighted_green_imports", "renewable_capacity_additions_mw_per_million"],
        "Z": [-2.0, 1.0, -0.5],
        "delta_outcome": [3.0, 4.0, -7.0],
    })
    out = build_vulnerability_panel(panel)
    assert out["delta_outcome"].to_list() == [3.0, -7.0]
    assert out["negative_shock_sample"].to_list() == [True, True]


def _valid_model_row() -> pl.DataFrame:
    return pl.DataFrame({
        "economy_id": ["A"],
        "treatment_time": [2000],
        "horizon": [3],
        "outcome_id": ["green_export_complexity"],
        "gad_version": ["gad_no_supp"],
        "sample_version": ["core_complete_case"],
        "gad_time": [1999],
        "control_time": [1999],
        "baseline_outcome_time": [1999],
        "outcome_time": [2003],
        "gimc": [1.0],
        "gad_lag": [2.0],
        "Z": [-0.5],
        "Z_GAD": [-1.0],
        "CMZ": [0.25],
        "baseline_outcome": [1.0],
        "future_outcome": [4.0],
        "delta_outcome": [3.0],
        "rca_eligible_products": [None],
        "rca_entered_products": [None],
        "outcome_coverage_eligible": [True],
        "confirmatory_iv_eligible": [True],
        "gad_interpolated": [False],
        "outcome_interpolated": [False],
        "negative_shock_sample": [False],
        "core_eligible": [True],
        "lite_eligible": [False],
        "descriptive_only": [False],
        "threshold_selection_only": [False],
        "source_outcome_materialized": [True],
    }).with_columns(
        pl.col("treatment_time", "horizon", "gad_time", "control_time", "baseline_outcome_time", "outcome_time").cast(pl.Int16),
        pl.col("rca_eligible_products", "rca_entered_products").cast(pl.UInt32),
    )


def _valid_model_row_with_controls(*, bounded: bool = False) -> pl.DataFrame:
    expressions: list[pl.Expr] = [
        pl.lit(
            "core_bounded_controls" if bounded else "core_complete_case"
        ).alias("sample_version"),
        pl.lit(False).alias("interpolated_control"),
    ]
    for control in (
        "renewable_energy_consumption_share",
        "trade_openness_percent_gdp",
        "industry_value_added_share",
        "gdp_per_capita_current_usd",
    ):
        expressions.extend(
            [
                pl.lit(1.0).alias(control),
                pl.lit(1.0).alias(f"{control}_analysis"),
                pl.lit(False).alias(f"{control}_interpolated"),
            ]
        )
    return _valid_model_row().with_columns(expressions)


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("gad_time", 2000, "gad_time"),
        ("control_time", 2000, "control_time"),
        ("baseline_outcome_time", 2000, "baseline_outcome_time"),
        ("outcome_time", 2004, "outcome_time"),
    ],
)
def test_model_panel_validator_recomputes_exact_times(column: str, value: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_model_panel(_valid_model_row().with_columns(pl.lit(value, dtype=pl.Int16).alias(column)))


def test_model_panel_validator_rejects_duplicate_frozen_key() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_model_panel(pl.concat([_valid_model_row(), _valid_model_row()]))


def test_model_panel_validator_rejects_core_treatment_before_main_period() -> None:
    altered = _valid_model_row().with_columns(
        pl.lit(1999, dtype=pl.Int16).alias("treatment_time"),
        pl.lit(1998, dtype=pl.Int16).alias("gad_time"),
        pl.lit(1998, dtype=pl.Int16).alias("control_time"),
        pl.lit(1998, dtype=pl.Int16).alias("baseline_outcome_time"),
        pl.lit(2002, dtype=pl.Int16).alias("outcome_time"),
    )
    with pytest.raises(ValueError, match="Core/Lite separation"):
        validate_model_panel(altered)


def test_model_panel_validator_rejects_lite_treatment_after_frozen_extension() -> None:
    altered = _valid_model_row().with_columns(
        pl.lit(2025, dtype=pl.Int16).alias("treatment_time"),
        pl.lit(2024, dtype=pl.Int16).alias("gad_time"),
        pl.lit(2024, dtype=pl.Int16).alias("control_time"),
        pl.lit(2024, dtype=pl.Int16).alias("baseline_outcome_time"),
        pl.lit(2028, dtype=pl.Int16).alias("outcome_time"),
        pl.lit("lite_complete_case").alias("sample_version"),
        pl.lit(False).alias("core_eligible"),
        pl.lit(True).alias("lite_eligible"),
        pl.lit(True).alias("descriptive_only"),
    )
    with pytest.raises(ValueError, match="Core/Lite separation"):
        validate_model_panel(altered)


def test_model_panel_validator_rejects_unknown_sample_version() -> None:
    with pytest.raises(ValueError, match="sample_version"):
        validate_model_panel(
            _valid_model_row().with_columns(pl.lit("core_convenience").alias("sample_version"))
        )


def test_model_panel_validator_rejects_lite_row_with_false_lite_flag() -> None:
    altered = _valid_model_row().with_columns(
        pl.lit(2023, dtype=pl.Int16).alias("treatment_time"),
        pl.lit(2022, dtype=pl.Int16).alias("gad_time"),
        pl.lit(2022, dtype=pl.Int16).alias("control_time"),
        pl.lit(2022, dtype=pl.Int16).alias("baseline_outcome_time"),
        pl.lit(2026, dtype=pl.Int16).alias("outcome_time"),
        pl.lit("lite_complete_case").alias("sample_version"),
        pl.lit(False).alias("lite_eligible"),
        pl.lit(True).alias("descriptive_only"),
    )
    with pytest.raises(ValueError, match="Core/Lite separation"):
        validate_model_panel(altered)


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("core_eligible", False, "Core/Lite separation"),
        ("descriptive_only", True, "discriminator"),
        ("threshold_selection_only", True, "discriminator"),
        ("source_outcome_materialized", False, "discriminator"),
    ],
)
def test_model_panel_validator_derives_discriminator_flags(
    column: str, value: bool, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_model_panel(_valid_model_row().with_columns(pl.lit(value).alias(column)))


def test_model_panel_validator_enforces_derived_outcome_null_fields() -> None:
    derived = _valid_model_row().with_columns(
        pl.lit("future_green_rca_entry_rate").alias("outcome_id"),
        pl.lit("gad_no_supp").alias("gad_version"),
        pl.lit(False).alias("source_outcome_materialized"),
        pl.lit(None, dtype=pl.Float64).alias("baseline_outcome"),
        pl.lit(0.5).alias("future_outcome"),
        pl.lit(0.5).alias("delta_outcome"),
        pl.lit(2, dtype=pl.UInt32).alias("rca_eligible_products"),
        pl.lit(1, dtype=pl.UInt32).alias("rca_entered_products"),
    )
    validate_model_panel(derived)
    with pytest.raises(ValueError, match="derived outcome"):
        validate_model_panel(derived.with_columns(pl.lit(0.0).alias("baseline_outcome")))
    with pytest.raises(ValueError, match="derived outcome"):
        validate_model_panel(derived.with_columns(pl.lit(None, dtype=pl.Float64).alias("future_outcome")))


def test_model_panel_validator_derives_negative_flag_from_sample_key() -> None:
    altered = _valid_model_row().with_columns(
        pl.lit("asinh_weighted_green_imports").alias("outcome_id"),
        pl.lit("gad_core").alias("gad_version"),
        pl.lit("core_complete_case_negative_shock").alias("sample_version"),
        pl.lit(False).alias("negative_shock_sample"),
    )
    with pytest.raises(ValueError, match="discriminator"):
        validate_model_panel(altered)


def test_model_panel_validator_rejects_forged_complete_case_control_copies() -> None:
    altered = _valid_model_row_with_controls().with_columns(
        *[
            expression
            for control in (
                "renewable_energy_consumption_share",
                "trade_openness_percent_gdp",
                "industry_value_added_share",
                "gdp_per_capita_current_usd",
            )
            for expression in (
                pl.lit(999.0).alias(f"{control}_analysis"),
                pl.lit(True).alias(f"{control}_interpolated"),
            )
        ],
        pl.lit(True).alias("interpolated_control"),
    )
    with pytest.raises(ValueError, match="conditional-null"):
        validate_model_panel(altered)


def test_model_panel_validator_rejects_bounded_false_positive_flag() -> None:
    altered = _valid_model_row_with_controls(bounded=True).with_columns(
        pl.lit(True).alias("renewable_energy_consumption_share_interpolated"),
        pl.lit(True).alias("interpolated_control"),
    )
    with pytest.raises(ValueError, match="conditional-null"):
        validate_model_panel(altered)


def test_model_panel_validator_rejects_bounded_false_negative_flag() -> None:
    altered = _valid_model_row_with_controls(bounded=True).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("renewable_energy_consumption_share"),
        pl.lit(2.0).alias("renewable_energy_consumption_share_analysis"),
        pl.lit(False).alias("renewable_energy_consumption_share_interpolated"),
    )
    with pytest.raises(ValueError, match="conditional-null"):
        validate_model_panel(altered)


def test_model_panel_validator_allows_false_flag_when_bounded_control_is_unavailable() -> None:
    altered = _valid_model_row_with_controls(bounded=True).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("renewable_energy_consumption_share"),
        pl.lit(None, dtype=pl.Float64).alias(
            "renewable_energy_consumption_share_analysis"
        ),
        pl.lit(False).alias("renewable_energy_consumption_share_interpolated"),
    )
    validate_model_panel(altered)


def test_model_panel_validator_rejects_row_interpolation_or_mismatch() -> None:
    altered = _valid_model_row_with_controls().with_columns(
        pl.lit(True).alias("interpolated_control")
    )
    with pytest.raises(ValueError, match="conditional-null"):
        validate_model_panel(altered)


def test_model_panel_validator_accepts_valid_complete_and_bounded_control_modes() -> None:
    validate_model_panel(_valid_model_row_with_controls())
    bounded = _valid_model_row_with_controls(bounded=True).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("renewable_energy_consumption_share"),
        pl.lit(2.0).alias("renewable_energy_consumption_share_analysis"),
        pl.lit(True).alias("renewable_energy_consumption_share_interpolated"),
        pl.lit(True).alias("interpolated_control"),
    )
    validate_model_panel(bounded)


def test_model_panel_validator_rejects_complete_case_null_control_flag() -> None:
    altered = _valid_model_row_with_controls().with_columns(
        pl.lit(None, dtype=pl.Boolean).alias(
            "renewable_energy_consumption_share_interpolated"
        )
    )
    with pytest.raises(ValueError, match="control.*flag"):
        validate_model_panel(altered)


def test_model_panel_validator_rejects_bounded_fill_with_null_control_flag() -> None:
    altered = _valid_model_row_with_controls(bounded=True).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("renewable_energy_consumption_share"),
        pl.lit(2.0).alias("renewable_energy_consumption_share_analysis"),
        pl.lit(None, dtype=pl.Boolean).alias(
            "renewable_energy_consumption_share_interpolated"
        ),
    )
    with pytest.raises(ValueError, match="control.*flag"):
        validate_model_panel(altered)


def test_model_panel_validator_rejects_null_row_control_flag() -> None:
    altered = _valid_model_row_with_controls().with_columns(
        pl.lit(None, dtype=pl.Boolean).alias("interpolated_control")
    )
    with pytest.raises(ValueError, match="control.*flag"):
        validate_model_panel(altered)


def test_model_panel_validator_rejects_nonboolean_control_flag() -> None:
    altered = _valid_model_row_with_controls().with_columns(
        pl.lit(1, dtype=pl.Int8).alias(
            "renewable_energy_consumption_share_interpolated"
        )
    )
    with pytest.raises(ValueError, match="Boolean"):
        validate_model_panel(altered)


def test_model_panel_validator_rejects_null_propagating_control_values() -> None:
    altered = _valid_model_row_with_controls(bounded=True).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("renewable_energy_consumption_share"),
        pl.lit(float("nan"), dtype=pl.Float64).alias(
            "renewable_energy_consumption_share_analysis"
        ),
        pl.lit(True).alias("renewable_energy_consumption_share_interpolated"),
        pl.lit(True).alias("interpolated_control"),
    )
    with pytest.raises(ValueError, match="conditional-null"):
        validate_model_panel(altered)


def test_model_panel_validator_rejects_negative_sample_with_nonnegative_shock() -> None:
    altered = _valid_model_row().with_columns(
        pl.lit(True).alias("negative_shock_sample"),
        pl.lit("core_complete_case_negative_shock").alias("sample_version"),
        pl.lit(0.0).alias("Z"),
        pl.lit("asinh_weighted_green_imports").alias("outcome_id"),
        pl.lit("gad_core").alias("gad_version"),
    )
    with pytest.raises(ValueError, match="negative shock"):
        validate_model_panel(altered)


def test_cli_exposes_sample_and_panel_production_commands() -> None:
    from green_debt.cli import build_parser

    parser = build_parser()
    for command in ("freeze-samples", "build-analysis-panels", "audit-analysis-panels"):
        assert parser.parse_args([command]).command == command


def _parent_audit_fixture() -> tuple[pl.DataFrame, ...]:
    product_rows: list[dict[str, object]] = []
    country_rows: list[dict[str, object]] = []
    gad_rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    provisional_rows: list[dict[str, object]] = []
    component_names = (
        "future_green_rca_entry_rate",
        "green_export_complexity_change",
        "green_export_share_change",
    )
    scaler_rows: list[dict[str, object]] = []
    for horizon in range(3, 9):
        for position in range(3):
            economy = f"H{horizon}{position}"
            treatment = 2000 + position
            provisional_rows.append({"economy_id": economy, "provisional_core": True})
            country_rows.extend([
                {"economy_id": economy, "year": treatment - 1, "green_export_complexity": 0.0, "green_export_share": 0.0},
                {"economy_id": economy, "year": treatment + horizon, "green_export_complexity": float(position), "green_export_share": float(position)},
            ])
            for hs6 in ("A", "B"):
                product_rows.append({"economy_id": economy, "year": treatment - 1, "hs6": hs6, "green_product_rca": 0.5})
                entered = position == 2 or (position == 1 and hs6 == "A")
                product_rows.append({"economy_id": economy, "year": treatment + horizon, "hs6": hs6, "green_product_rca": 1.2 if entered else 0.5})
            gad_rows.extend([
                {"economy_id": economy, "year": treatment, "specification_id": "gad_core", "gimc": 10.0 + position, "gad": 9.0, "scaler_hash": FROZEN_SCALER_HASH},
                {"economy_id": economy, "year": treatment - 1, "specification_id": "gad_no_supp", "gimc": 9.0, "gad": 5.0, "scaler_hash": FROZEN_SCALER_HASH},
            ])
            rate = position / 2.0
            giu = (position - 1.0) / 1.4826
            common = {
                "economy_id": economy,
                "treatment_time": treatment,
                "horizon": horizon,
                "gad_version": "gad_no_supp",
                "sample_version": "core_complete_case",
                "gad_time": treatment - 1,
                "gimc": 10.0 + position,
                "gad_lag": 5.0,
                "gad_scaler_hash": FROZEN_SCALER_HASH,
            }
            panel_rows.extend([
                common | {"outcome_id": "future_green_rca_entry_rate", "rca_eligible_products": 2, "rca_entered_products": position, "future_outcome": rate, "delta_outcome": rate},
                common | {"outcome_id": "green_industrial_upgrading_index", "rca_eligible_products": None, "rca_entered_products": None, "future_outcome": giu, "delta_outcome": giu},
            ])
        for component in component_names:
            scaler_rows.append({
                "horizon": horizon,
                "source_column": component,
                "fit_start_year": 2000,
                "fit_end_year": 2004,
                "center": 0.5 if component == "future_green_rca_entry_rate" else 1.0,
                "scale": 0.7413 if component == "future_green_rca_entry_rate" else 1.4826,
                "scale_method": "mad_1_4826",
                "row_count": 3,
                "economy_count": 3,
            })
    payload = {
        "registry_id": "green_industrial_upgrading_outcome_scaler",
        "registry_version": "1.0.0",
        "fit_years": [2000, 2004],
        "horizons": [
            {
                "horizon": horizon,
                "components": {
                    row["source_column"]: {
                        "center": row["center"], "scale": row["scale"],
                        "scale_method": row["scale_method"], "row_count": 3,
                        "economy_count": 3,
                    }
                    for row in scaler_rows if row["horizon"] == horizon
                },
            }
            for horizon in range(3, 9)
        ],
    }
    canonical_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    giu = pl.DataFrame(scaler_rows).with_columns(pl.lit(canonical_hash).alias("canonical_hash"))
    panel = pl.DataFrame(panel_rows).with_columns(pl.lit(canonical_hash).alias("giu_scaler_hash"))
    return panel, giu, pl.DataFrame(gad_rows), pl.DataFrame(country_rows), pl.DataFrame(product_rows), pl.DataFrame(provisional_rows)


@pytest.mark.parametrize(
    ("target", "column", "value", "metric"),
    [
        ("panel_rca", "rca_eligible_products", 3, "rca_parent_failures"),
        ("panel_rca", "rca_entered_products", 2, "rca_parent_failures"),
        ("panel_rca", "future_outcome", 0.25, "rca_parent_failures"),
        ("panel_giu", "future_outcome", 99.0, "giu_panel_parent_failures"),
        ("panel_any", "gimc", 99.0, "gimc_source_failures"),
        ("panel_any", "gad_scaler_hash", "bad", "gad_scaler_source_failures"),
        ("giu", "center", 99.0, "giu_scaler_parent_failures"),
        ("giu", "canonical_hash", "bad", "giu_scaler_parent_failures"),
    ],
)
def test_parent_audit_recomputes_rca_giu_gimc_and_scaler_bindings(
    target: str, column: str, value: object, metric: str
) -> None:
    panel, giu, gad, country, product, provisional = _parent_audit_fixture()
    baseline = _audit_parent_derivations(panel, giu, gad, country, product, provisional)
    assert all(value == 0 for value in baseline.values())
    if target == "giu":
        giu = giu.with_row_index("row").with_columns(
            pl.when(pl.col("row") == 0).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
        ).drop("row")
    else:
        outcome = (
            "future_green_rca_entry_rate" if target == "panel_rca"
            else "green_industrial_upgrading_index" if target == "panel_giu" else None
        )
        selector = pl.col("outcome_id") == outcome if outcome else pl.lit(True)
        indexed = panel.with_row_index("row")
        target_row = indexed.filter(selector).get_column("row").first()
        panel = indexed.with_columns(
            pl.when(pl.col("row") == target_row).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
        ).drop("row")
    assert _audit_parent_derivations(panel, giu, gad, country, product, provisional)[metric] > 0
