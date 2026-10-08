import numpy as np
import polars as pl
import pytest

from green_debt.trade import compute_gpci, compute_gpci_with_audit, compute_rca


def test_rca_matches_balassa_formula() -> None:
    """Catch use of economy or world shares in the wrong Balassa denominator."""

    exports = pl.DataFrame(
        {
            "economy_id": ["A", "A", "B", "B"],
            "hs6": ["1", "2", "1", "2"],
            "year": [2000] * 4,
            "export_usd": [80.0, 20.0, 20.0, 80.0],
        }
    )
    out = compute_rca(exports)
    assert out.filter(
        (pl.col("economy_id") == "A") & (pl.col("hs6") == "1")
    )["rca"].item() == pytest.approx(1.6)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1.0])
def test_rca_rejects_invalid_export_trade_value(bad_value: float) -> None:
    """Catch invalid all-product exports being silently excluded before RCA."""

    exports = pl.DataFrame(
        {
            "economy_id": ["A"],
            "hs6": ["1"],
            "year": [2000],
            "export_usd": [bad_value],
        }
    )
    with pytest.raises(ValueError, match="invalid export trade value"):
        compute_rca(exports)


def test_pci_is_deterministic_oriented_and_standardized() -> None:
    """Catch unstable eigenvector selection, sign inversion, or sample scaling."""

    exports = pl.DataFrame(
        {
            "economy_id": ["A", "A", "B", "B", "B", "C", "C"],
            "hs6": ["1", "2", "1", "2", "3", "2", "3"],
            "year": [2000] * 7,
            "export_usd": [90.0, 10.0, 50.0, 30.0, 20.0, 20.0, 80.0],
        }
    )
    first = compute_gpci(exports)
    second = compute_gpci(exports)
    assert first.equals(second)
    assert first["gpci"].mean() == pytest.approx(0.0, abs=1e-12)
    assert first["gpci"].std(ddof=0) == pytest.approx(1.0)
    assert np.isfinite(first["gpci"].to_numpy()).all()
    assert first["orientation_score"].unique().item() >= 0


def test_pci_records_registry_product_with_no_annual_exports() -> None:
    """Catch silently dropping a registered HS96 product absent from annual exports."""

    exports = pl.DataFrame(
        {
            "economy_id": ["A", "A", "B", "B", "B", "C", "C"],
            "hs6": ["1", "2", "1", "2", "3", "2", "3"],
            "year": [2000] * 7,
            "export_usd": [90.0, 10.0, 50.0, 30.0, 20.0, 20.0, 80.0],
        }
    )
    out, audit = compute_gpci_with_audit(
        exports, product_universe=("1", "2", "3", "999999")
    )
    assert out["zero_ubiquity_products"].unique().item() == 1
    assert audit[0]["zero_ubiquity_product_codes"] == ["999999"]
    assert "999999" not in out["hs6"].to_list()


def test_pci_rejects_repeated_second_eigenvalue() -> None:
    """Catch arbitrary PCI selection when the second eigenspace is degenerate."""

    exports = pl.DataFrame(
        {
            "economy_id": ["A", "A", "A", "B", "B", "B"],
            "hs6": ["1", "2", "3", "1", "2", "3"],
            "year": [2000] * 6,
            "export_usd": [1.0] * 6,
        }
    )
    with pytest.raises(ValueError, match="second eigengap"):
        compute_gpci(exports)


def test_sparse_pci_branch_rejects_repeated_second_eigenvalue() -> None:
    """Catch sparse eigsh accepting a degenerate second eigenspace."""

    products = [f"{index:06d}" for index in range(129)]
    exports = pl.DataFrame(
        {
            "economy_id": ["A"] * 129 + ["B"] * 129,
            "hs6": products * 2,
            "year": [2000] * 258,
            "export_usd": [1.0] * 258,
        }
    )
    with pytest.raises(ValueError, match="second eigengap"):
        compute_gpci(exports)
