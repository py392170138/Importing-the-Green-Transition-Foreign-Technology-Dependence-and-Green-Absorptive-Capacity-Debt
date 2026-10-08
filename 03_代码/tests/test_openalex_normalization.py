from __future__ import annotations

import polars as pl
import pytest

from green_debt.science import normalize_openalex


def test_openalex_union_is_unique_and_complete() -> None:
    old = pl.DataFrame(
        {
            "country_code": ["AA", "AA"],
            "year": [1992, 1993],
            "green_works": [0, 1],
            "total_works": [100, 110],
            "counting_method": ["full_country_participation"] * 2,
            "include_xpac": ["false"] * 2,
        }
    )
    new = pl.DataFrame(
        {
            "country_code": ["AA"],
            "year": [1994],
            "green_works": [2],
            "total_works": [120],
            "counting_method": ["full_country_participation"],
            "include_xpac": ["false"],
        }
    )

    out = normalize_openalex((old, new), expected_period=(1992, 1994))

    assert out.select(pl.struct("country_code", "year").n_unique()).item() == 3
    assert out["year"].to_list() == [1992, 1993, 1994]
    assert out.schema["green_works"] == pl.UInt64
    assert out["include_xpac"].to_list() == [False, False, False]


def test_openalex_rejects_counting_rule_change() -> None:
    bad = pl.DataFrame(
        {
            "country_code": ["AA"],
            "year": [1992],
            "green_works": [1],
            "total_works": [2],
            "counting_method": ["fractional"],
            "include_xpac": ["false"],
        }
    )

    with pytest.raises(ValueError, match="counting method"):
        normalize_openalex((bad,), expected_period=(1992, 1992))


def test_openalex_rejects_invalid_counts_and_xpac() -> None:
    bad_counts = pl.DataFrame(
        {
            "country_code": ["AA"],
            "year": [1992],
            "green_works": [3],
            "total_works": [2],
            "counting_method": ["full_country_participation"],
            "include_xpac": [False],
        }
    )
    with pytest.raises(ValueError, match="green_works exceeds total_works"):
        normalize_openalex((bad_counts,), expected_period=(1992, 1992))

    bad_xpac = bad_counts.with_columns(
        pl.lit(1).alias("green_works"), pl.lit(True).alias("include_xpac")
    )
    with pytest.raises(ValueError, match="include_xpac"):
        normalize_openalex((bad_xpac,), expected_period=(1992, 1992))
