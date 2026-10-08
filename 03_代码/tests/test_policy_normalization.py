from __future__ import annotations

import polars as pl
import pytest

from green_debt.sources.policy import (
    attach_policy_controls,
    normalize_eps,
    normalize_ifcma_snapshot,
)


def _eps_frame(*, component: str, unit: str = "0_TO_6") -> pl.DataFrame:
    years = list(range(1990, 2021))
    return pl.DataFrame(
        {
            "DATAFLOW": ["OECD.ECO.MAD:DSD_EPS@DF_EPS(1.0)"] * len(years),
            "REF_AREA": ["AAA"] * len(years),
            "FREQ": ["A"] * len(years),
            "MEASURE": ["POL_STRINGENCY"] * len(years),
            "CLIM_POL": [component] * len(years),
            "TIME_PERIOD": years,
            "OBS_VALUE": [0.5] * len(years),
            "UNIT_MULT": [0] * len(years),
            "UNIT_MEASURE": [unit] * len(years),
            "DECIMALS": [2] * len(years),
        }
    )


def test_eps_requires_frozen_levels_and_removes_embedded_composite_duplicate() -> None:
    composite = _eps_frame(component="EPS")
    components = pl.concat(
        (composite, _eps_frame(component="TAXCO2"))
    )

    out = normalize_eps(composite, components)

    assert out.height == 62
    assert out["indicator_id"].n_unique() == 2
    assert out["robustness_only"].all()
    with pytest.raises(ValueError, match="dimension or unit levels changed"):
        normalize_eps(
            _eps_frame(component="EPS", unit="percent"), components
        )


def test_ifcma_deduplicates_instrument_subscheme_before_counting() -> None:
    raw = pl.DataFrame(
        {
            "Country ISO": ["AAA", "AAA", "AAA"],
            "Policy Instrument ID": ["P1", "P1", "P2"],
            "Instrument / subscheme": ["Instrument", "Instrument", "Subscheme"],
            "Status": ["In force", "In force", "Ended"],
        }
    )

    out = normalize_ifcma_snapshot(raw, snapshot_year=2026)

    total = out.filter(pl.col("indicator_id") == "climate_policy_instrument_count")
    assert total["value"].item() == 2.0
    assert out["robustness_only"].all()


def test_ifcma_missing_instrument_identity_is_excluded_not_synthesized() -> None:
    raw = pl.DataFrame(
        {
            "Country ISO": ["JPN", "JPN"],
            "Policy Instrument ID": [None, "JPN1"],
            "Instrument / subscheme": ["Instrument", "Instrument"],
            "Status": ["In force", "In force"],
        }
    )

    out = normalize_ifcma_snapshot(raw, snapshot_year=2026)

    total = out.filter(pl.col("indicator_id") == "climate_policy_instrument_count")
    assert total["value"].item() == 1.0


def test_policy_absence_never_changes_main_sample_flag() -> None:
    sample = pl.DataFrame(
        {
            "economy_id": ["AAA", "BBB"],
            "year": [2026, 2026],
            "main_sample_candidate": [True, True],
        }
    )
    policy = normalize_ifcma_snapshot(
        pl.DataFrame(
            {
                "Country ISO": ["AAA"],
                "Policy Instrument ID": ["P1"],
                "Instrument / subscheme": ["Instrument"],
                "Status": ["In force"],
            }
        ),
        snapshot_year=2026,
    )

    out = attach_policy_controls(sample, policy)

    assert out.height == sample.height
    assert out["main_sample_candidate"].to_list() == [True, True]
    assert (
        out.filter(pl.col("economy_id") == "BBB")[
            "policy__climate_policy_instrument_count"
        ].item()
        is None
    )
