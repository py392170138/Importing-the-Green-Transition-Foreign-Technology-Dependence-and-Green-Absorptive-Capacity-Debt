import polars as pl
import pytest

from green_debt.outcomes import (
    build_green_export_outcomes,
    build_rca_entry_rate,
    build_value_capture_outcomes,
)


def test_export_complexity_uses_current_gpci_and_green_export_shares() -> None:
    exports = pl.DataFrame(
        {
            "economy_id": ["A", "A"],
            "year": [2000, 2000],
            "hs6": ["p1", "p2"],
            "weighted_green_export_usd": [25.0, 75.0],
        }
    )
    totals = pl.DataFrame(
        {"economy_id": ["A"], "year": [2000], "total_export_usd": [200.0]}
    )
    gpci = pl.DataFrame(
        {"year": [2000, 2000], "hs6": ["p1", "p2"], "gpci": [0.0, 2.0]}
    )

    out = build_green_export_outcomes(exports, totals, gpci)

    assert out["green_export_complexity"].item() == pytest.approx(1.5)
    assert out["green_export_share"].item() == pytest.approx(0.5)


def test_future_rca_entry_denominator_is_baseline_non_rca_products() -> None:
    rca = pl.DataFrame(
        {
            "economy_id": ["A"] * 4,
            "hs6": ["p1", "p2", "p1", "p2"],
            "year": [1999, 1999, 2003, 2003],
            "rca": [0.5, 1.2, 1.1, 1.3],
        }
    )

    out = build_rca_entry_rate(rca, treatment_year=2000, horizon=3)

    assert out["eligible_products"].item() == 1
    assert out["future_green_rca_entry_rate"].item() == pytest.approx(1.0)


def test_domestic_value_added_uses_the_frozen_activity_weighted_share() -> None:
    """Replacing EXGR_DVA with the foreign-value component would break overlap control."""
    tiva = pl.DataFrame(
        {
            "economy_id": ["A"],
            "year": [2000],
            "specification_id": ["confirmatory_prod_weight"],
            "gfvad_raw": [0.2],
            "dvashare_raw": [73.0],
        }
    )

    out = build_value_capture_outcomes(tiva)

    assert out["domestic_value_added_share"].item() == pytest.approx(73.0)
    assert out["foreign_value_added_dependence"].item() == pytest.approx(0.2)


def test_value_capture_excludes_pre_outcome_source_history() -> None:
    """Passing TiVA's 1995 history into the 1996–2024 authority violates its contract."""
    tiva = pl.DataFrame(
        {
            "economy_id": ["A", "A"],
            "year": [1995, 1996],
            "specification_id": ["confirmatory_prod_weight", "confirmatory_prod_weight"],
            "gfvad_raw": [0.2, 0.3],
            "dvashare_raw": [70.0, 71.0],
        }
    )

    out = build_value_capture_outcomes(tiva)

    assert out.get_column("year").to_list() == [1996]
