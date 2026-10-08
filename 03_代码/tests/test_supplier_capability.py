from pathlib import Path

import polars as pl
import pytest

import green_debt.trade as trade
from green_debt.artifacts import BuildIdentity
from green_debt.paths import resolve_project_paths
from green_debt.trade import compute_product_proximity, compute_supplier_raw


def test_proximity_uses_minimum_conditional_probability() -> None:
    """Catch a one-direction co-export probability leaking into product proximity."""
    incidence = pl.DataFrame({"economy_id": ["A", "A", "B"], "hs6": ["p1", "p2", "p1"], "year": [2000, 2000, 2000], "rca_present": [True, True, True]})
    proximity = compute_product_proximity(incidence)
    value = proximity.filter((pl.col("hs6_a") == "p1") & (pl.col("hs6_b") == "p2"))["proximity"].item()
    assert value == pytest.approx(0.5)


def test_proximity_can_limit_temporary_edges_to_upstream_products() -> None:
    """Catch production materializing unrelated product-pair edges before density."""
    incidence = pl.DataFrame({"economy_id": ["A", "A", "A", "B", "B", "B"], "hs6": ["p1", "p2", "p3", "p1", "p2", "p3"], "year": [2000] * 6, "rca_present": [True] * 6})
    proximity = compute_product_proximity(incidence, anchor_products={"p1"})
    assert proximity.height == 4
    assert all("p1" in (row["hs6_a"], row["hs6_b"]) for row in proximity.iter_rows(named=True))


def test_gud_and_grd_exclude_already_exported_opportunity() -> None:
    """Catch treating a current RCA capability as its own relatedness opportunity."""
    incidence = pl.DataFrame({"economy_id": ["A"], "hs6": ["p1"], "year": [2000], "rca_present": [True]})
    upstream = pl.DataFrame({"hs6": ["p1", "p2"], "upstream_weight": [1.0, 0.5]})
    proximity = pl.DataFrame({"year": [2000, 2000], "hs6_a": ["p2", "p2"], "hs6_b": ["p1", "n1"], "proximity": [0.5, 0.5]})
    out = compute_supplier_raw(incidence, upstream, proximity)
    assert out["gud_raw"].item() == pytest.approx(1.0)
    assert out["grd_raw"].item() == pytest.approx(0.5)


def test_supplier_no_export_coverage_is_not_a_valid_zero() -> None:
    """Catch no export rows being silently converted into zero supplier capability."""
    incidence = pl.DataFrame(schema={"economy_id": pl.String, "hs6": pl.String, "year": pl.Int16, "rca_present": pl.Boolean})
    upstream = pl.DataFrame({"hs6": ["p1"], "upstream_weight": [1.0]})
    proximity = pl.DataFrame(schema={"year": pl.Int16, "hs6_a": pl.String, "hs6_b": pl.String, "proximity": pl.Float64})
    out = compute_supplier_raw(incidence, upstream, proximity, economy_years=pl.DataFrame({"economy_id": ["A"], "year": [2000]}))
    row = out.row(0, named=True)
    assert row["gud_raw"] is None
    assert row["grd_raw"] is None
    assert row["gud_raw_reason"] == "no_export_coverage"
    assert row["grd_raw_reason"] == "no_export_coverage"


def test_supplier_keeps_zero_denominator_as_missing() -> None:
    """Catch a product with no eligible proximity being treated as zero density."""
    incidence = pl.DataFrame({"economy_id": ["A"], "hs6": ["p1"], "year": [2000], "rca_present": [True]})
    upstream = pl.DataFrame({"hs6": ["p2"], "upstream_weight": [1.0]})
    proximity = pl.DataFrame(schema={"year": pl.Int16, "hs6_a": pl.String, "hs6_b": pl.String, "proximity": pl.Float64})
    row = compute_supplier_raw(incidence, upstream, proximity).row(0, named=True)
    assert row["grd_raw"] is None
    assert row["grd_raw_reason"] == "zero_proximity_denominator"


def test_sparse_production_density_matches_public_supplier_formula() -> None:
    """Catch a production-only sparse density path drifting from the public formula."""
    incidence = pl.DataFrame({"economy_id": ["A", "B", "B", "B", "B", "C", "C"], "hs6": ["p1", "p1", "p2", "n1", "n2", "p2", "n1"], "year": [2000] * 7, "rca_present": [True] * 7})
    weights = {"p1": 1.0, "p2": 0.5}
    coverage = pl.DataFrame({"economy_id": ["A", "B", "C"], "year": [2000, 2000, 2000], "export_coverage_normal": [True, True, True]})
    expected = compute_supplier_raw(incidence, pl.DataFrame({"hs6": list(weights), "upstream_weight": list(weights.values())}), compute_product_proximity(incidence), economy_years=coverage)
    actual, _ = trade._annual_supplier_raw_sparse(incidence, weights, coverage)
    assert actual.equals(expected)
    assert actual.filter(pl.col("economy_id") == "A")["grd_raw"].item() == pytest.approx(0.25)


def test_supplier_build_keeps_nonexporter_skeleton_key_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch build coverage being derived only from products with export rows."""
    paths = resolve_project_paths(tmp_path, tmp_path / "data")
    registry = paths.code_root / "02_数据字典/product_registry_hs96_v1.parquet"
    exports_path = paths.normalized / "baci/exporter_product/year=1996/taxonomy_version=all_hs96.parquet"
    totals_path = paths.normalized / "baci/economy_year_totals/year=1996/taxonomy_version=all_hs96.parquet"
    for path in (registry, exports_path, totals_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"list_name": ["main"], "hs96": ["p1"], "green_upstream_weight": [1.0]}).write_parquet(registry)
    pl.DataFrame({"economy_id": ["A"], "hs6": ["p1"], "year": [1996], "trade_value_usd": [10.0]}).write_parquet(exports_path)
    pl.DataFrame({"economy_id": ["A", "B"], "year": [1996, 1996], "reported_as_exporter": [True, False]}).write_parquet(totals_path)
    monkeypatch.setattr(trade, "_YEARS", (1996,))
    report = trade.build_supplier_raw(paths, taxonomy="main_hs96", build=BuildIdentity(command="test", code_commit="test"))
    out = pl.read_parquet(report.output_path).filter(pl.col("economy_id") == "B").row(0, named=True)
    assert report.rows == 2
    assert out["gud_raw"] is None and out["grd_raw"] is None
    assert out["gud_raw_reason"] == "no_export_coverage"
    assert out["grd_raw_reason"] == "no_export_coverage"
