import polars as pl
import pytest

from green_debt.sample import build_regression_copies


def test_regression_copy_is_bounded_without_mutating_raw() -> None:
    frame = pl.DataFrame({"economy_id": ["A", "B", "C"], "year": [2000, 2000, 2000], "x": [0.0, 1.0, 100.0]})
    out, registry = build_regression_copies(frame, columns=("x",), lower=1, upper=99)
    assert out["x"].to_list() == [0.0, 1.0, 100.0]
    assert out["x_p01_p99"].max() == registry.bounds["x"].upper
    assert registry.fit_years == (2000, 2022)


def test_bounds_are_fit_once_on_underlying_main_economy_years() -> None:
    frame = pl.DataFrame({
        "economy_id": ["A", "B", "C", "D"],
        "year": [2000, 2001, 2022, 2023],
        "x": [0.0, 1.0, 100.0, 1_000_000.0],
    })
    out, registry = build_regression_copies(frame, columns=("x",), lower=1, upper=99)
    assert registry.fit_row_count == 3
    assert registry.bounds["x"].upper == pytest.approx(98.02)
    assert out["x"].to_list() == [0.0, 1.0, 100.0, 1_000_000.0]
    assert out["x_p01_p99"].to_list()[-1] == pytest.approx(98.02)
    assert len(registry.canonical_hash) == 64
