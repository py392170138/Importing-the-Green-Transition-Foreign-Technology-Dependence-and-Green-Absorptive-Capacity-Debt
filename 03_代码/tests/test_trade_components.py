import sys
from pathlib import Path

import polars as pl
import pytest

import green_debt.trade as trade
from green_debt.artifacts import BuildIdentity
from green_debt.paths import resolve_project_paths
from green_debt.trade import compute_trade_components


def test_import_complexity_uses_previous_year_gpci() -> None:
    """Catch a contemporaneous product-complexity join that leaks current trade."""

    imports = pl.DataFrame(
        {
            "economy_id": ["A"],
            "hs6": ["1"],
            "year": [2001],
            "weighted_green_import_usd": [10.0],
            "trade_coverage_normal": [True],
        }
    )
    gpci = pl.DataFrame(
        {"hs6": ["1", "1"], "year": [2000, 2001], "gpci": [2.0, 99.0]}
    )
    gdp = pl.DataFrame(
        {"economy_id": ["A"], "year": [2001], "gdp_current_usd": [100.0]}
    )
    out = compute_trade_components(imports, gpci, gdp)
    assert out["green_import_complexity_raw"].item() == pytest.approx(2.0)
    assert out["gpci_source_year"].item() == 2000


def test_abnormal_trade_coverage_makes_zero_missing() -> None:
    """Catch treating non-reporting as a valid zero green import observation."""

    imports = pl.DataFrame(
        {
            "economy_id": ["A"],
            "hs6": ["1"],
            "year": [2001],
            "weighted_green_import_usd": [0.0],
            "trade_coverage_normal": [False],
        }
    )
    gpci = pl.DataFrame({"hs6": ["1"], "year": [2000], "gpci": [1.0]})
    gdp = pl.DataFrame(
        {"economy_id": ["A"], "year": [2001], "gdp_current_usd": [100.0]}
    )
    assert (
        compute_trade_components(imports, gpci, gdp)["green_import_intensity_raw"].item()
        is None
    )


def test_missing_green_exports_does_not_become_a_reported_zero() -> None:
    """Catch GNIR treating an absent export input as a valid zero export flow."""

    imports = pl.DataFrame(
        {
            "economy_id": ["A"],
            "hs6": ["1"],
            "year": [2001],
            "weighted_green_import_usd": [10.0],
            "trade_coverage_normal": [True],
        }
    )
    gpci = pl.DataFrame({"hs6": ["1"], "year": [2000], "gpci": [2.0]})
    gdp = pl.DataFrame(
        {"economy_id": ["A"], "year": [2001], "gdp_current_usd": [100.0]}
    )
    out = compute_trade_components(imports, gpci, gdp)
    assert out["gnir_raw"].item() is None
    assert out["gnir_reason"].item() == "missing_green_exports"


def test_import_complexity_ignores_export_only_products_without_gpci() -> None:
    """Catch lagged-GPCI completeness being checked against the export-side union."""

    trade = pl.DataFrame(
        {
            "economy_id": ["A", "A"],
            "hs6": ["1", "2"],
            "year": [2001, 2001],
            "weighted_green_import_usd": [10.0, 0.0],
            "weighted_green_export_usd": [0.0, 7.0],
            "trade_coverage_normal": [True, True],
        }
    )
    gpci = pl.DataFrame({"hs6": ["1"], "year": [2000], "gpci": [2.0]})
    gdp = pl.DataFrame(
        {"economy_id": ["A"], "year": [2001], "gdp_current_usd": [100.0]}
    )
    out = compute_trade_components(trade, gpci, gdp)
    assert out["green_import_complexity_raw"].item() == pytest.approx(2.0)
    assert out["green_import_complexity_reason"].item() is None


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1.0])
def test_invalid_green_trade_value_is_rejected(bad_value: float) -> None:
    """Catch NaN, infinite, or negative trade values reaching raw components."""

    imports = pl.DataFrame(
        {
            "economy_id": ["A"],
            "hs6": ["1"],
            "year": [2001],
            "weighted_green_import_usd": [bad_value],
            "trade_coverage_normal": [True],
        }
    )
    gpci = pl.DataFrame({"hs6": ["1"], "year": [2000], "gpci": [2.0]})
    gdp = pl.DataFrame(
        {"economy_id": ["A"], "year": [2001], "gdp_current_usd": [100.0]}
    )
    with pytest.raises(ValueError, match="invalid green import trade value"):
        compute_trade_components(imports, gpci, gdp)


def test_nonfinite_gdp_is_null_before_the_authoritative_write_boundary() -> None:
    """Catch retaining an infinite raw GDP value after recording its reason code."""

    imports = pl.DataFrame(
        {
            "economy_id": ["A"],
            "hs6": ["1"],
            "year": [2001],
            "weighted_green_import_usd": [10.0],
            "trade_coverage_normal": [True],
        }
    )
    gpci = pl.DataFrame({"hs6": ["1"], "year": [2000], "gpci": [2.0]})
    gdp = pl.DataFrame(
        {"economy_id": ["A"], "year": [2001], "gdp_current_usd": [float("inf")]}
    )
    out = compute_trade_components(imports, gpci, gdp)
    assert out["gdp_current_usd"].item() is None
    assert out["green_import_intensity_reason"].item() == "nonfinite_gdp"


def _write_trade_component_build_fixture(
    tmp_path: Path,
    *,
    green_rows: dict[str, list[object]],
    gpci_rows: dict[str, list[object]],
) -> tuple[object, Path]:
    data_root = tmp_path / "data"
    paths = resolve_project_paths(tmp_path, data_root)
    green_path = (
        paths.normalized
        / "baci/green_economy_product/year=2001/taxonomy_version=main_hs96.parquet"
    )
    totals_path = (
        paths.normalized
        / "baci/economy_year_totals/year=2001/taxonomy_version=all_hs96.parquet"
    )
    gpci_path = paths.measures / "trade/gpci_product_year.parquet"
    wdi_path = paths.normalized / "wdi/wdi_country_year.parquet"
    for path in (green_path, totals_path, gpci_path, wdi_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(green_rows).write_parquet(green_path)
    pl.DataFrame(
        {
            "economy_id": ["A"],
            "year": [2001],
            "reported_as_importer": [True],
            "reported_as_exporter": [True],
        }
    ).write_parquet(totals_path)
    pl.DataFrame(gpci_rows).write_parquet(gpci_path)
    pl.DataFrame(
        {
            "economy_id": ["A"],
            "year": [2001],
            "indicator_id": ["NY.GDP.MKTP.CD"],
            "value": [sys.float_info.max],
        }
    ).write_parquet(wdi_path)
    return paths, data_root


def test_build_rejects_derived_overflow_before_authoritative_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch import-times-GPCI overflow before the build path calls its writer."""

    maximum = sys.float_info.max
    paths, _ = _write_trade_component_build_fixture(
        tmp_path,
        green_rows={
            "economy_id": ["A"],
            "hs6": ["1"],
            "year": [2001],
            "flow_role": ["importer"],
            "weighted_green_trade_usd": [maximum],
        },
        gpci_rows={
            "taxonomy_version": ["main_hs96"],
            "hs6": ["1"],
            "year": [2000],
            "gpci": [maximum],
        },
    )
    writer_calls: list[object] = []
    monkeypatch.setattr(trade, "_YEARS", (2001,))
    monkeypatch.setattr(
        trade,
        "write_authoritative_table",
        lambda *args, **kwargs: writer_calls.append((args, kwargs)),
    )
    with pytest.raises(ValueError, match="nonfinite trade-component values before write"):
        trade.build_trade_components(
            paths,
            taxonomy="main_hs96",
            build=BuildIdentity(command="test", code_commit="test"),
        )
    assert writer_calls == []


def test_build_rejects_grouped_trade_sum_overflow_before_authoritative_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch finite row values becoming infinite during the build's group-by sum."""

    maximum = sys.float_info.max
    paths, _ = _write_trade_component_build_fixture(
        tmp_path,
        green_rows={
            "economy_id": ["A", "A"],
            "hs6": ["1", "2"],
            "year": [2001, 2001],
            "flow_role": ["importer", "importer"],
            "weighted_green_trade_usd": [maximum * 0.75, maximum * 0.75],
        },
        gpci_rows={
            "taxonomy_version": ["main_hs96", "main_hs96"],
            "hs6": ["1", "2"],
            "year": [2000, 2000],
            "gpci": [1.0, 1.0],
        },
    )
    writer_calls: list[object] = []
    monkeypatch.setattr(trade, "_YEARS", (2001,))
    monkeypatch.setattr(
        trade,
        "write_authoritative_table",
        lambda *args, **kwargs: writer_calls.append((args, kwargs)),
    )
    with pytest.raises(ValueError, match="nonfinite trade-component values before write"):
        trade.build_trade_components(
            paths,
            taxonomy="main_hs96",
            build=BuildIdentity(command="test", code_commit="test"),
        )
    assert writer_calls == []
