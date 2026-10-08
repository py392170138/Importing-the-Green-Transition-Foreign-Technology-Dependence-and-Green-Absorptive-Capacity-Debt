import polars as pl
import pytest

from green_debt.gad import GADSpecification, audit_gad_frame, compute_gad


def _panel(years: list[int], exposure: list[float | None], absorption: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"economy_id": ["AAA"] * len(years), "year": years, "external_exposure": exposure, "absorption": absorption}
    )


def test_gad_starts_in_1997_and_2000_is_first_eligible() -> None:
    panel = _panel([1996, 1997, 1998, 1999, 2000], [None, 2.0, 2.0, 2.0, 2.0], [1.0] * 5)
    out = compute_gad(panel, GADSpecification.core(half_life=5))
    rho = 2 ** (-1 / 5)

    assert out.filter(pl.col("year") == 1997)["gad"].item() == pytest.approx(1.0)
    assert out.filter(pl.col("year") == 1998)["gad"].item() == pytest.approx(rho + 1.0)
    assert out.filter(pl.col("year").is_between(1997, 1999))["warmup_ineligible"].to_list() == [True, True, True]
    assert out.filter(pl.col("year") == 2000)["warmup_ineligible"].item() is False
    assert out.filter(pl.col("year") == 2000)["eligible"].item() is True


def test_missing_component_breaks_debt_and_requires_new_three_row_warmup() -> None:
    panel = _panel(list(range(1996, 2008)), [None, 2.0, 2.0, 2.0, 2.0, 2.0, None, 2.0, 2.0, 2.0, 2.0, 2.0], [1.0] * 12)
    out = compute_gad(panel, GADSpecification.core(half_life=5))

    assert out.filter(pl.col("year") == 2002)["gad"].item() is None
    assert out.filter(pl.col("year").is_between(2003, 2005))["warmup_ineligible"].to_list() == [True, True, True]
    assert out.filter(pl.col("year") == 2003)["gad"].item() == pytest.approx(1.0)
    assert out.filter(pl.col("year") == 2006)["eligible"].item() is True
    assert out.filter(pl.col("year") == 2003)["initialization_run"].item() is False


def test_missing_lagged_absorption_and_calendar_gap_break_runs_without_debt_crossing() -> None:
    panel = _panel([1996, 1997, 1998, 1999, 2000, 2002, 2003, 2004, 2005], [None, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [1.0, 1.0, None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    out = compute_gad(panel, GADSpecification.core(half_life=5))

    assert out.filter(pl.col("year") == 1999)["gap"].item() is None
    assert out.filter(pl.col("year") == 2002)["gad"].item() is None
    assert out.filter(pl.col("year") == 2003)["gad"].item() == pytest.approx(1.0)
    assert out.filter(pl.col("year") == 2003)["run_position"].item() == 1
    assert out.filter(pl.col("year").is_between(2003, 2005))["warmup_ineligible"].to_list() == [True, True, True]


def test_negative_gap_is_floored_and_static_never_uses_prior_debt() -> None:
    panel = _panel([1996, 1997, 1998, 1999], [None, 2.0, 1.0, 0.0], [1.0] * 4)
    recursive = compute_gad(panel, GADSpecification.core(half_life=5))
    static = compute_gad(panel, GADSpecification.static())

    assert recursive.filter(pl.col("year") == 1998)["gad"].item() > 0.0
    assert recursive.filter(pl.col("year") == 1999)["gad"].item() == pytest.approx(0.0)
    assert static.filter(pl.col("year") == 1998)["gad"].item() == pytest.approx(0.0)


def test_authority_audit_rejects_current_absorption_instead_of_strict_lag() -> None:
    panel = _panel([1996, 1997, 1998, 1999], [None, 2.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0])
    out = compute_gad(panel, GADSpecification.core(half_life=5)).with_columns(
        pl.lit("gad_core").alias("specification_id"),
        pl.lit("core").alias("formula_id"),
        pl.lit("half_life").alias("family"),
        pl.lit(5).cast(pl.Int16).alias("half_life"),
        pl.lit("main_hs96").alias("taxonomy_version"),
        pl.lit("confirmatory").alias("sample_version"),
        pl.lit(True).alias("uses_lagged_debt"),
        pl.lit("d44dfe2bd8023e6b9c695b91dc73026a061e5f9b64198b41cf642979de8d612e").alias("scaler_hash"),
        pl.lit(False).alias("descriptive_only"),
    )
    tampered = out.with_columns(
        pl.when(pl.col("year") == 1998).then(pl.col("absorption")).otherwise(pl.col("lagged_absorption")).alias("lagged_absorption")
    )

    with pytest.raises(ValueError, match="strict lagged absorption"):
        audit_gad_frame(tampered)
