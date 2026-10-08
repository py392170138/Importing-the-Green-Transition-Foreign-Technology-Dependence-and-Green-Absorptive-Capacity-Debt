import polars as pl
import pytest

from green_debt.instruments import build_baseline_shares, independent_baseline_audit


def test_small_cells_are_removed_then_remaining_shares_renormalize() -> None:
    baseline = pl.DataFrame(
        {
            "importer": ["C", "C", "C"],
            "exporter": ["J1", "J2", "J3"],
            "hs6": ["p", "p", "p"],
            "mean_weighted_import_usd": [9998.0, 1.0, 1.0],
        }
    )
    out = build_baseline_shares(baseline, minimum_share=0.0001)
    retained = out.retained
    assert retained.height == 1
    assert retained["baseline_share"].sum() == pytest.approx(1.0)
    assert out.coverage["retained_coverage"].item() == pytest.approx(0.9998)


def test_coverage_below_point_95_is_not_confirmatory() -> None:
    baseline = pl.DataFrame(
        {
            "importer": ["C", "C"],
            "exporter": ["J1", "J2"],
            "hs6": ["p1", "p2"],
            "mean_weighted_import_usd": [94.0, 6.0],
        }
    )
    out = build_baseline_shares(baseline, minimum_share=0.10)
    assert out.coverage["retained_coverage"].item() == pytest.approx(0.94)
    assert out.coverage["confirmatory_iv_eligible"].item() is False


def test_annual_baseline_uses_importer_mean_total_as_exact_denominator() -> None:
    baseline = pl.DataFrame(
        {
            "year": [1996, 1997, 1998, 1999] * 2,
            "importer": ["C"] * 8,
            "exporter": ["J1"] * 4 + ["J2"] * 4,
            "hs6": ["p1"] * 4 + ["p2"] * 4,
            "weighted_green_trade_usd": [90.0, 90.0, 90.0, 90.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    out = build_baseline_shares(baseline, minimum_share=0.0)
    shares = {
        row["exporter"]: row["raw_baseline_share"]
        for row in out.raw.iter_rows(named=True)
    }
    assert shares == pytest.approx({"J1": 0.9, "J2": 0.1})
    assert out.coverage["baseline_mean_total_weighted_import_usd"].item() == pytest.approx(100.0)


def test_threshold_is_applied_to_raw_share_before_renormalization() -> None:
    baseline = pl.DataFrame(
        {
            "importer": ["C", "C", "C"],
            "exporter": ["J1", "J2", "J3"],
            "hs6": ["p1", "p2", "p3"],
            "mean_weighted_import_usd": [9998.0, 1.0, 1.0],
        }
    )
    out = build_baseline_shares(baseline, minimum_share=0.0001)
    assert out.raw.filter(pl.col("retained"))["exporter"].to_list() == ["J1"]
    assert out.raw.filter(~pl.col("retained"))["raw_baseline_share"].to_list() == pytest.approx([0.0001, 0.0001])


def test_audit_recomputes_coverage_and_eligibility_bidirectionally() -> None:
    published = pl.DataFrame(
        {
            "share_version": ["main_0.0001", "main_0.0001"],
            "importer": ["C", "C"],
            "exporter": ["J1", "J2"],
            "hs6": ["p1", "p2"],
            "mean_weighted_import_usd": [9999.0, 1.0],
            "baseline_mean_total_weighted_import_usd": [10000.0, 10000.0],
            "raw_baseline_share": [0.9999, 0.0001],
            "retained": [True, False],
            "retained_coverage": [0.9999, 0.9999],
            "baseline_share": [1.0, None],
            "raw_cell_count": [2, 2],
            "retained_cell_count": [1, 1],
            "coverage_eligible": [True, True],
            "confirmatory_specification": [True, True],
            "confirmatory_baseline_eligible": [True, True],
        }
    )
    assert independent_baseline_audit(published) == {
        "coverage_recomputation_failures": 0,
        "coverage_label_failures": 0,
    }

    wrong_coverage = published.with_columns(pl.lit(0.95).alias("retained_coverage"))
    assert independent_baseline_audit(wrong_coverage)["coverage_recomputation_failures"] > 0

    wrong_label = published.with_columns(
        pl.lit(False).alias("coverage_eligible"),
        pl.lit(False).alias("confirmatory_baseline_eligible"),
    )
    assert independent_baseline_audit(wrong_label)["coverage_label_failures"] > 0


def test_audit_rejects_self_consistently_rescaled_published_shares() -> None:
    published = pl.DataFrame(
        {
            "share_version": ["main_0.0001"] * 3,
            "importer": ["C"] * 3,
            "exporter": ["J1", "J2", "J3"],
            "hs6": ["p1", "p2", "p3"],
            "mean_weighted_import_usd": [9998.0, 1.0, 1.0],
            "baseline_mean_total_weighted_import_usd": [10000.0] * 3,
            "raw_baseline_share": [1.9996, 0.0002, 0.0002],
            "retained": [True, True, True],
            "retained_coverage": [2.0, 2.0, 2.0],
            "baseline_share": [0.9998, 0.0001, 0.0001],
            "raw_cell_count": [3, 3, 3],
            "retained_cell_count": [3, 3, 3],
            "coverage_eligible": [True, True, True],
            "confirmatory_specification": [True, True, True],
            "confirmatory_baseline_eligible": [True, True, True],
        }
    )

    failures = independent_baseline_audit(published)

    assert failures["coverage_recomputation_failures"] > 0


def test_audit_derives_raw_share_sum_and_exact_threshold_from_money() -> None:
    boundary = pl.DataFrame(
        {
            "share_version": ["main_0.0001", "main_0.0001"],
            "importer": ["C", "C"],
            "exporter": ["J1", "J2"],
            "hs6": ["p1", "p2"],
            "mean_weighted_import_usd": [9999.0, 1.0],
            "baseline_mean_total_weighted_import_usd": [10000.0, 10000.0],
            "raw_baseline_share": [0.9999, 0.0001],
            "retained": [True, False],
            "retained_coverage": [0.9999, 0.9999],
            "baseline_share": [1.0, None],
            "raw_cell_count": [2, 2],
            "retained_cell_count": [1, 1],
            "coverage_eligible": [True, True],
            "confirmatory_specification": [True, True],
            "confirmatory_baseline_eligible": [True, True],
        }
    )
    assert independent_baseline_audit(boundary)["coverage_recomputation_failures"] == 0

    wrong_boundary = boundary.with_columns(
        pl.Series("raw_baseline_share", [0.9998999999, 0.0001000001]),
        pl.Series("retained", [True, True]),
        pl.lit(1.0).alias("retained_coverage"),
        pl.Series("baseline_share", [0.9998999999, 0.0001000001]),
        pl.lit(2).alias("retained_cell_count"),
    )

    assert (
        independent_baseline_audit(wrong_boundary)["coverage_recomputation_failures"]
        > 0
    )
