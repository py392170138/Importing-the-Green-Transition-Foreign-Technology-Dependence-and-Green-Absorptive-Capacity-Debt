import polars as pl
import pytest

from analysis_fixtures import model_panel_fixture
from green_debt.analysis_spec import ModelCell
from green_debt.diagnostics import exact_model_sample, replication_safe_state_frame


def test_exact_sample_uses_only_frozen_analysis_copies_and_builds_interaction(
    spec,
) -> None:
    panel = model_panel_fixture().with_columns(
        pl.lit(9999.0).alias("gimc"),
        pl.lit(9999.0).alias("gad_lag"),
        pl.lit(9999.0).alias("Z"),
    )
    cell = ModelCell(
        "green_export_complexity",
        3,
        "gad_no_supp",
        "core_complete_case",
        "h2_h3_primary",
    )

    out = exact_model_sample(panel.lazy(), cell, spec)

    assert out.columns == [
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
    ]
    assert out["gimc_gad_a"].to_list() == pytest.approx(
        (out["gimc_a"] * out["gad_a"]).to_list()
    )
    assert out["gimc_a"].max() < 2.0
    assert out.sort("economy_id", "treatment_time").equals(out)


def test_exact_sample_rejects_h0_wrong_gad_and_forged_role(spec) -> None:
    panel = model_panel_fixture().lazy()
    bad_h = ModelCell(
        "co2_tonnes_per_million_current_usd",
        0,
        "gad_core",
        "core_complete_case",
        "h1_primary",
    )
    with pytest.raises(ValueError, match="registered.*cell"):
        exact_model_sample(panel, bad_h, spec)

    bad_gad = ModelCell(
        "green_export_complexity",
        3,
        "gad_core",
        "core_complete_case",
        "h2_h3_primary",
    )
    with pytest.raises(ValueError, match="GAD mapping"):
        exact_model_sample(panel, bad_gad, spec)

    forged_role = ModelCell(
        "green_export_complexity",
        3,
        "gad_no_supp",
        "core_complete_case",
        "h1_primary",
    )
    with pytest.raises(ValueError, match="registered.*cell"):
        exact_model_sample(panel, forged_role, spec)


def test_exact_sample_applies_family_specific_flags(spec) -> None:
    base = model_panel_fixture().head(2)
    threshold = base.with_columns(
        pl.lit("green_industrial_upgrading_index").alias("outcome_id"),
        pl.lit(5).alias("horizon"),
        pl.lit(False).alias("confirmatory_iv_eligible"),
        pl.Series("threshold_selection_only", [True, False]),
    )
    threshold_out = exact_model_sample(
        threshold.lazy(), spec.threshold_cell(), spec
    )
    assert threshold_out.height == 1

    vulnerability_cell = spec.vulnerability_cells()[0]
    vulnerability = base.with_columns(
        pl.lit(vulnerability_cell.outcome_id).alias("outcome_id"),
        pl.lit(vulnerability_cell.horizon).alias("horizon"),
        pl.lit(vulnerability_cell.gad_version).alias("gad_version"),
        pl.lit(vulnerability_cell.sample_version).alias("sample_version"),
        pl.lit(False).alias("confirmatory_iv_eligible"),
        pl.Series("negative_shock_sample", [False, True]),
        pl.Series("Z_p01_p99", [0.1, -0.2]),
        (
            pl.Series("Z_p01_p99", [0.1, -0.2])
            * pl.col("gad_lag_p01_p99")
        ).alias("Z_GAD_p01_p99"),
    )
    vulnerability_out = exact_model_sample(
        vulnerability.lazy(), vulnerability_cell, spec
    )
    assert vulnerability_out.height == 1


def test_vulnerability_sample_rejects_nonnegative_registered_shocks(spec) -> None:
    cell = spec.vulnerability_cells()[0]
    panel = model_panel_fixture().head(2).with_columns(
        pl.lit(cell.outcome_id).alias("outcome_id"),
        pl.lit(cell.horizon).alias("horizon"),
        pl.lit(cell.gad_version).alias("gad_version"),
        pl.lit(cell.sample_version).alias("sample_version"),
        pl.lit(False).alias("confirmatory_iv_eligible"),
        pl.lit(True).alias("negative_shock_sample"),
        pl.lit(0.2).alias("Z_p01_p99"),
        (pl.lit(0.2) * pl.col("gad_lag_p01_p99")).alias("Z_GAD_p01_p99"),
    )

    with pytest.raises(ValueError, match="negative-shock.*Z < 0"):
        exact_model_sample(panel.lazy(), cell, spec)


def test_exact_sample_rejects_nonfinite_frozen_values(spec) -> None:
    panel = model_panel_fixture().with_columns(
        pl.when(pl.col("economy_id") == "E000")
        .then(float("inf"))
        .otherwise(pl.col("Z_p01_p99"))
        .alias("Z_p01_p99")
    )
    cell = ModelCell(
        "green_export_complexity",
        3,
        "gad_no_supp",
        "core_complete_case",
        "h2_h3_primary",
    )

    with pytest.raises(ValueError, match="nonfinite"):
        exact_model_sample(panel.lazy(), cell, spec)


def test_replication_safe_state_frame_deduplicates_only_identical_states() -> None:
    base = model_panel_fixture()
    replicated = pl.concat(
        [
            base,
            base.with_columns(
                pl.lit(4, dtype=base.schema["horizon"]).alias("horizon"),
                pl.lit("green_export_share").alias("outcome_id"),
            ),
        ]
    )

    state = replication_safe_state_frame(replicated.lazy())

    assert state.height == base.height
    assert state.select(
        "economy_id", "treatment_time", "gad_version", "sample_version"
    ).is_duplicated().sum() == 0


def test_replication_safe_state_frame_rejects_disagreement() -> None:
    base = model_panel_fixture()
    conflicting = pl.concat(
        [
            base,
            base.with_columns(
                (pl.col("gimc_p01_p99") + 1.0).alias("gimc_p01_p99")
            ),
        ]
    )

    with pytest.raises(ValueError, match="replicated.*disagree"):
        replication_safe_state_frame(conflicting.lazy())
