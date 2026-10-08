import polars as pl
import pytest

from green_debt.instruments import build_partner_shocks, symmetric_growth


def test_destination_only_export_change_does_not_enter_its_shock() -> None:
    trade = pl.DataFrame(
        {
            "year": [2000, 2001, 2000, 2001],
            "exporter": ["J", "J", "J", "J"],
            "importer": ["C", "C", "D", "D"],
            "hs6": ["p"] * 4,
            "weighted_green_trade_usd": [10.0, 20.0, 30.0, 30.0],
        }
    )
    shock = build_partner_shocks(trade)
    c = shock.filter(
        (pl.col("destination_excluded") == "C") & (pl.col("year") == 2001)
    )
    assert c["exporter_growth_excluding_destination"].item() == pytest.approx(0.0)


def test_symmetric_growth_is_bounded() -> None:
    assert symmetric_growth(0.0, 10.0) == pytest.approx(2.0)
    assert symmetric_growth(10.0, 0.0) == pytest.approx(-2.0)
    assert symmetric_growth(0.0, 0.0) is None


def test_sparse_cell_absence_is_observed_zero_when_annual_partition_exists() -> None:
    trade = pl.DataFrame(
        {
            "year": [2000, 2001],
            "exporter": ["J", "K"],
            "importer": ["D", "D"],
            "hs6": ["p", "q"],
            "weighted_green_trade_usd": [10.0, 1.0],
        }
    )
    cells = pl.DataFrame({"importer": ["C"], "exporter": ["J"], "hs6": ["p"]})
    shock = build_partner_shocks(trade, cells=cells, expected_years=(2000, 2001))
    row = shock.filter(pl.col("year") == 2001).row(0, named=True)
    assert row["exports_excluding_destination_lag"] == pytest.approx(10.0)
    assert row["exports_excluding_destination"] == pytest.approx(0.0)
    assert row["exporter_growth_excluding_destination"] == pytest.approx(-2.0)


def test_missing_annual_partition_is_never_silently_filled_with_zero() -> None:
    trade = pl.DataFrame(
        {
            "year": [2000, 2002],
            "exporter": ["J", "J"],
            "importer": ["D", "D"],
            "hs6": ["p", "p"],
            "weighted_green_trade_usd": [10.0, 10.0],
        }
    )
    cells = pl.DataFrame({"importer": ["C"], "exporter": ["J"], "hs6": ["p"]})
    with pytest.raises(ValueError, match="annual partitions.*2001"):
        build_partner_shocks(trade, cells=cells, expected_years=(2000, 2001, 2002))


def test_both_adjacent_zero_levels_produce_explicit_missing_growth() -> None:
    trade = pl.DataFrame(
        {
            "year": [1999, 2000, 2001],
            "exporter": ["J", "K", "K"],
            "importer": ["D", "D", "D"],
            "hs6": ["p", "q", "q"],
            "weighted_green_trade_usd": [1.0, 1.0, 1.0],
        }
    )
    cells = pl.DataFrame({"importer": ["C"], "exporter": ["J"], "hs6": ["p"]})
    row = build_partner_shocks(
        trade, cells=cells, expected_years=(2000, 2001)
    ).filter(pl.col("year") == 2001).row(0, named=True)
    assert row["exporter_growth_excluding_destination"] is None
    assert row["shock_missing_reason"] == "both_adjacent_exporter_levels_zero"


def test_partner_shock_subtracts_leave_destination_out_product_growth() -> None:
    trade = pl.DataFrame(
        {
            "year": [2000, 2001, 2000, 2001],
            "exporter": ["J1", "J1", "J2", "J2"],
            "importer": ["D", "D", "D", "D"],
            "hs6": ["p"] * 4,
            "weighted_green_trade_usd": [10.0, 30.0, 30.0, 30.0],
        }
    )
    cells = pl.DataFrame({"importer": ["C"], "exporter": ["J1"], "hs6": ["p"]})
    row = build_partner_shocks(
        trade, cells=cells, expected_years=(2000, 2001)
    ).filter(pl.col("year") == 2001).row(0, named=True)
    assert row["exporter_growth_excluding_destination"] == pytest.approx(1.0)
    assert row["global_product_growth_excluding_destination"] == pytest.approx(0.4)
    assert row["partner_shock"] == pytest.approx(0.6)


def test_authoritative_all_destination_totals_supply_the_export_level() -> None:
    own_destination_trade = pl.DataFrame(
        {
            "year": [2000, 2001],
            "exporter": ["J", "J"],
            "importer": ["C", "C"],
            "hs6": ["p", "p"],
            "weighted_green_trade_usd": [10.0, 20.0],
        }
    )
    totals = pl.DataFrame(
        {
            "year": [2000, 2001],
            "exporter": ["J", "J"],
            "hs6": ["p", "p"],
            "weighted_green_trade_usd": [40.0, 50.0],
        }
    )
    cells = pl.DataFrame({"importer": ["C"], "exporter": ["J"], "hs6": ["p"]})
    row = build_partner_shocks(
        own_destination_trade,
        cells=cells,
        expected_years=(2000, 2001),
        exporter_product_totals=totals,
    ).filter(pl.col("year") == 2001).row(0, named=True)
    assert row["exports_excluding_destination_lag"] == pytest.approx(30.0)
    assert row["exports_excluding_destination"] == pytest.approx(30.0)
    assert row["exporter_growth_excluding_destination"] == pytest.approx(0.0)
