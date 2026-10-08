import polars as pl
import pytest

from green_debt.tiva import ActivitySets, build_tiva_measures


def _legacy_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in range(2000, 2005):
        rows += [
            {"economy_id": "AAA", "activity": "C27", "year": year, "measure": "PROD", "obs_value": 75.0},
            {"economy_id": "AAA", "activity": "C28", "year": year, "measure": "PROD", "obs_value": 25.0},
        ]
    rows += [
        {"economy_id": "AAA", "activity": "C27", "year": 2000, "measure": "DFD_FVA", "obs_value": 20.0},
        {"economy_id": "AAA", "activity": "C27", "year": 2000, "measure": "FD_VA", "obs_value": 100.0},
        {"economy_id": "AAA", "activity": "C28", "year": 2000, "measure": "DFD_FVA", "obs_value": 20.0},
        {"economy_id": "AAA", "activity": "C28", "year": 2000, "measure": "FD_VA", "obs_value": 50.0},
        {"economy_id": "AAA", "activity": "C27", "year": 2000, "measure": "EXGR_DVA", "obs_value": 80.0},
        {"economy_id": "AAA", "activity": "C28", "year": 2000, "measure": "EXGR_DVA", "obs_value": 60.0},
    ]
    return rows


def _sets() -> ActivitySets:
    return ActivitySets(
        confirmatory=("C27", "C28"),
        broad=("C27", "C28"),
        equipment_only=("C27", "C28"),
    )


def test_gfvad_uses_fd_va_and_frozen_prod_weights() -> None:
    """Replacing the FD_VA denominator with another level must change this result."""
    out = build_tiva_measures(pl.DataFrame(_legacy_rows()), _sets())
    row = out.confirmatory.filter(pl.col("year") == 2000).row(0, named=True)
    assert row["gfvad_raw"] == pytest.approx(0.25)
    assert row["dvashare_raw"] == pytest.approx(75.0)
    assert row["specification_id"] == "confirmatory_prod_weight"


def test_tiva_normalizes_legacy_only_at_boundary_and_rejects_mixed_schema() -> None:
    """A mixed legacy/current frame could silently select the wrong source values."""
    normalized = pl.DataFrame(_legacy_rows()).rename(
        {"measure": "indicator_id", "obs_value": "value"}
    ).with_columns(
        pl.when(pl.col("indicator_id") == "PROD").then(pl.lit("prod_level"))
        .when(pl.col("indicator_id") == "DFD_FVA").then(pl.lit("dfd_fva_level"))
        .when(pl.col("indicator_id") == "FD_VA").then(pl.lit("fd_va_level"))
        .otherwise(pl.lit("exgr_dva_share"))
        .alias("indicator_id")
    )
    out = build_tiva_measures(normalized, _sets())
    assert out.confirmatory.filter(pl.col("year") == 2000)["gfvad_raw"].item() == pytest.approx(0.25)
    with pytest.raises(ValueError, match="mixed TiVA input schemas"):
        build_tiva_measures(pl.DataFrame(_legacy_rows()).hstack(normalized.select("indicator_id", "value")), _sets())


def test_missing_activity_and_invalid_denominator_stay_missing_without_renormalization() -> None:
    """Dropping an industry or FD_VA<=0 must not inflate the remaining industry's ratio."""
    rows = [row for row in _legacy_rows() if not (row["activity"] == "C28" and row["measure"] == "FD_VA")]
    out = build_tiva_measures(pl.DataFrame(rows), _sets())
    row = out.confirmatory.filter(pl.col("year") == 2000).row(0, named=True)
    assert row["gfvad_raw"] is None
    assert row["gfvad_missing_reason"] == "missing_required_activity_value"


def test_out_of_range_values_are_flagged_and_bounded_spec_excludes_them() -> None:
    """Clipping an abnormal source ratio would hide it instead of producing a robustness value."""
    rows = _legacy_rows()
    for row in rows:
        if row["activity"] == "C27" and row["measure"] == "DFD_FVA":
            row["obs_value"] = 200.0
    out = build_tiva_measures(pl.DataFrame(rows), _sets())
    raw = out.confirmatory.filter(pl.col("year") == 2000).row(0, named=True)
    bounded = out.bounded_confirmatory.filter(pl.col("year") == 2000).row(0, named=True)
    assert raw["gfvad_raw"] == pytest.approx(1.6)
    assert raw["gfvad_out_of_range"] is True
    assert bounded["gfvad_raw"] is None
    assert bounded["gfvad_missing_reason"] == "out_of_range_ratio_in_bounded_robustness"


def test_main_tiva_table_has_only_four_specs_and_bounded_table_is_physical_separate() -> None:
    """Mixing bounded rows into the authority changes its primary specification universe."""
    out = build_tiva_measures(pl.DataFrame(_legacy_rows()), _sets())
    assert set(out.table["specification_id"].unique()) == {
        "confirmatory_prod_weight",
        "broad_prod_weight",
        "equipment_only_prod_weight",
        "equal_industry_weight",
    }
    assert set(out.bounded_confirmatory["specification_id"].unique()) == {
        "bounded_confirmatory_prod_weight"
    }
