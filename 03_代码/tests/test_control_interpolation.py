import polars as pl
import pytest

from green_debt.sample import build_wdi_control_frame, interpolate_single_year_controls


def test_only_one_internal_control_gap_is_interpolated() -> None:
    frame = pl.DataFrame({"economy_id": ["A"] * 4, "year": [2000, 2001, 2002, 2003], "control": [1.0, None, 3.0, None]})
    out = interpolate_single_year_controls(frame, ("control",))
    assert out.filter(pl.col("year") == 2001)["control_analysis"].item() == pytest.approx(2.0)
    assert out.filter(pl.col("year") == 2001)["interpolated_control"].item() is True
    assert out.filter(pl.col("year") == 2003)["control_analysis"].item() is None


def test_two_year_gap_and_nonconsecutive_neighbors_are_not_filled() -> None:
    frame = pl.DataFrame({
        "economy_id": ["A"] * 5,
        "year": [2000, 2001, 2002, 2003, 2005],
        "control": [1.0, None, None, 4.0, 8.0],
    })
    out = interpolate_single_year_controls(frame, ("control",))
    assert out.filter(pl.col("year").is_in([2001, 2002]))["control_analysis"].null_count() == 2
    assert out.filter(pl.col("year") == 2003)["control_analysis"].item() == pytest.approx(4.0)
    assert out["interpolated_control"].sum() == 0


def test_control_interpolation_preserves_the_raw_column_and_per_column_flag() -> None:
    frame = pl.DataFrame({
        "economy_id": ["A"] * 3,
        "year": [2000, 2001, 2002],
        "x": [2.0, None, 6.0],
        "outcome": [10.0, None, 30.0],
    })
    out = interpolate_single_year_controls(frame, ("x",))
    assert out["x"].to_list() == [2.0, None, 6.0]
    assert out.filter(pl.col("year") == 2001)["x_interpolated"].item() is True
    assert out.filter(pl.col("year") == 2001)["outcome"].item() is None


def test_main_controls_use_only_approved_wdi_control_registry_rows() -> None:
    registry = pl.DataFrame({
        "source_id": ["wdi", "wdi", "wdi"],
        "source_field": ["C", "P", "Y"],
        "project_field": ["control", "policy", "outcome"],
        "role": ["control", "robustness_control", "outcome"],
        "status": ["approved_control", "approved_robustness", "approved_main"],
    })
    wdi = pl.DataFrame({
        "economy_id": ["A", "A", "A"],
        "year": [2000, 2000, 2000],
        "indicator_id": ["C", "P", "Y"],
        "value": [1.0, 2.0, 3.0],
    })
    out, columns = build_wdi_control_frame(wdi, registry)
    assert columns == ("control",)
    assert out.columns == ["economy_id", "year", "control"]
    assert out["control"].item() == pytest.approx(1.0)
