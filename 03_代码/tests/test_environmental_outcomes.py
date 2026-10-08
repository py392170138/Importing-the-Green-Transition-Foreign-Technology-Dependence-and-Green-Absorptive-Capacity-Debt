import polars as pl
import pytest

from green_debt.outcomes import build_environmental_outcomes, outcome_specs


def test_co2_intensity_is_tonnes_per_million_current_usd() -> None:
    wdi = pl.DataFrame(
        {
            "economy_id": ["AAA", "AAA"],
            "year": [2000, 2000],
            "indicator_id": ["EN.GHG.CO2.MT.CE.AR5", "NY.GDP.MKTP.CD"],
            "value": [2.0, 1_000_000_000.0],
        }
    )
    irena = pl.DataFrame(
        {
            "economy_id": ["AAA"],
            "year": [2000],
            "renewable_capacity_additions_mw": [10.0],
        }
    )
    population = pl.DataFrame(
        {"economy_id": ["AAA"], "year": [2000], "population": [2_000_000.0]}
    )

    out = build_environmental_outcomes(wdi, irena, population)

    assert out["co2_tonnes_per_million_current_usd"].item() == pytest.approx(2000.0)
    assert out["renewable_capacity_additions_mw_per_million"].item() == pytest.approx(5.0)


def test_negative_capacity_additions_remain_signed_and_flagged() -> None:
    """Clipping retirements would turn a documented loss into a false zero."""
    wdi = pl.DataFrame(
        {
            "economy_id": ["AAA", "AAA"],
            "year": [2000, 2000],
            "indicator_id": ["EN.GHG.CO2.MT.CE.AR5", "NY.GDP.MKTP.CD"],
            "value": [1.0, 1_000_000.0],
        }
    )
    irena = pl.DataFrame(
        {
            "economy_id": ["AAA"],
            "year": [2000],
            "renewable_capacity_additions_mw": [-2.0],
            "retirement_or_revision": [True],
        }
    )
    population = pl.DataFrame(
        {"economy_id": ["AAA"], "year": [2000], "population": [1_000_000.0]}
    )

    out = build_environmental_outcomes(wdi, irena, population)

    assert out["renewable_capacity_additions_mw_per_million"].item() == pytest.approx(-2.0)
    assert out["renewable_capacity_additions_retirement_or_revision"].item() is True


def test_energy_intensity_metadata_uses_the_frozen_2021_ppp_unit() -> None:
    """Registering a generic PPP unit would silently disagree with the source registry."""
    energy = next(spec for spec in outcome_specs() if spec.outcome_id == "energy_intensity_mj_per_ppp_gdp")

    assert energy.unit == "MJ_per_2021_PPP_GDP"
