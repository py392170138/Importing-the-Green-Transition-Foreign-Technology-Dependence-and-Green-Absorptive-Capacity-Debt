from __future__ import annotations

import math

import polars as pl
import pytest

from green_debt.science import audit_gsci_windows, classify_research_coverage, compute_gsci


def test_gsci_requires_complete_window_and_current_population() -> None:
    """Catch partial stocks or a denominator dated before the observation year."""
    works = pl.DataFrame({"economy_id": ["AAA"] * 5, "year": [1992, 1993, 1994, 1995, 1996], "green_works": [1.0, 2.0, 3.0, 4.0, 5.0], "total_works": [100.0] * 5})
    population = pl.DataFrame({"economy_id": ["AAA"] * 5, "year": [1992, 1993, 1994, 1995, 1996], "population": [1_000_000.0, 2_000_000.0, 3_000_000.0, 4_000_000.0, 10_000_000.0]})
    result = compute_gsci(works, population, decay=0.8, window=5)
    assert result.filter(pl.col("year") < 1996)["gsci_raw"].null_count() == 4
    expected = math.log1p((5 + 0.8 * 4 + 0.8**2 * 3 + 0.8**3 * 2 + 0.8**4) / 10)
    assert result.filter(pl.col("year") == 1996)["gsci_raw"].item() == pytest.approx(expected)


def test_abnormal_total_research_zero_is_missing_not_green_zero() -> None:
    works = pl.DataFrame(
        {
            "economy_id": ["AAA", "AAA", "AAA"],
            "year": [2000, 2001, 2002],
            "green_works": [2.0, 0.0, 0.0],
            "total_works": [20.0, 0.0, 20.0],
        }
    )

    result = classify_research_coverage(works, coverage_floor_ratio=0.1)

    abnormal = result.filter(pl.col("year") == 2001).row(0, named=True)
    valid_zero = result.filter(pl.col("year") == 2002).row(0, named=True)
    assert abnormal["green_works"] is None
    assert abnormal["green_works_missing_reason"] == (
        "total_research_coverage_abnormal"
    )
    assert valid_zero["green_works"] == 0.0
    assert valid_zero["green_works_missing_reason"] is None


def test_gsci_does_not_bridge_an_abnormal_missing_year() -> None:
    """Catch a gap being bridged before five new consecutive observations accrue."""
    works = pl.DataFrame({"economy_id": ["AAA"] * 8, "year": list(range(2000, 2008)), "green_works": [1.0, 1.0, None, 1.0, 1.0, 1.0, 1.0, 1.0], "total_works": [10.0, 10.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0]})
    population = pl.DataFrame({"economy_id": ["AAA"] * 8, "year": list(range(2000, 2008)), "population": [1_000_000.0] * 8})
    result = compute_gsci(works, population, decay=0.8, window=5)
    assert result.filter(pl.col("year") == 2006)["gsci_raw"].item() is None
    assert result.filter(pl.col("year") == 2007)["gsci_raw"].item() == pytest.approx(math.log1p(sum(0.8**lag for lag in range(5))))


def test_gsci_marks_missing_window_with_stable_reason() -> None:
    """Catch a missing annual research count becoming an unlabelled partial stock."""
    works = pl.DataFrame({"economy_id": ["AAA"] * 5, "year": [1992, 1993, 1994, 1995, 1996], "green_works": [1.0, 2.0, None, 4.0, 5.0], "total_works": [100.0] * 5})
    population = pl.DataFrame({"economy_id": ["AAA"] * 5, "year": [1992, 1993, 1994, 1995, 1996], "population": [1_000_000.0] * 5})
    row = compute_gsci(works, population).filter(pl.col("year") == 1996).row(0, named=True)
    assert row["gsci_raw"] is None
    assert row["gsci_raw_reason"] == "incomplete_green_works_window"


def test_gsci_window_audit_uses_preoutput_rows_not_filtered_authoritative_output() -> None:
    """Catch a constant-zero partial-window audit after filtering away pre-1996 rows."""
    works = pl.DataFrame({"economy_id": ["AAA"] * 7, "year": list(range(1990, 1997)), "green_works": [1.0] * 7, "total_works": [10.0] * 7})
    population = pl.DataFrame({"economy_id": ["AAA"] * 7, "year": list(range(1990, 1997)), "population": [1_000_000.0] * 7})
    audit = audit_gsci_windows(compute_gsci(works, population))
    assert audit["pre_1996_nonnull_rows"] == 2
    assert audit["invalid_nonnull_windows"] == 0
