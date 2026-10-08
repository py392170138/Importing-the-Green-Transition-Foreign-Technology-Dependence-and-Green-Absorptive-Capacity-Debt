import polars as pl
import pytest

from green_debt.gad import GADSpecification, build_gad_variants, registered_gad_specifications


def test_registered_variants_are_unique_and_cover_exact_decay_set() -> None:
    specs = registered_gad_specifications()
    ids = [spec.specification_id for spec in specs]
    combinations = [(spec.formula_id, spec.half_life, spec.taxonomy_version, spec.sample_version) for spec in specs]

    assert len(ids) == len(set(ids))
    assert len(combinations) == len(set(combinations))
    assert {spec.half_life for spec in specs if spec.family == "half_life"} == {3, 5, 8, 10}
    assert ids.count("gad_core") == 1


def test_registered_decay_and_static_variants_are_exact() -> None:
    by_id = {spec.specification_id: spec for spec in registered_gad_specifications()}

    assert by_id["gad_no_decay"].rho == pytest.approx(1.0)
    assert by_id["gad_static"].uses_lagged_debt is False
    assert by_id["gad_no_gfvad"].formula_id == "no_gfvad"
    assert by_id["gad_no_supp"].formula_id == "no_supp"
    assert by_id["gad_no_gsci"].formula_id == "no_gsci"


def test_core_constructor_rejects_non_positive_half_life() -> None:
    with pytest.raises(ValueError, match="positive"):
        GADSpecification.core(half_life=0)


def test_production_variants_keep_1996_initialization_and_lite_only_extension() -> None:
    years = list(range(1996, 2025))
    source = {"economy_id": ["AAA"] * len(years), "year": years}
    for index, column in enumerate(
        (
            "z0_green_import_intensity_raw", "z0_green_import_complexity_raw", "z0_gfvad_raw",
            "z0_gsci_raw", "z0_gud_raw", "z0_grd_raw", "z0_gnir_raw",
        )
    ):
        source[column] = [float(index + 1)] * len(years)
    output, audit = build_gad_variants(
        pl.DataFrame(source),
        pl.DataFrame({"economy_id": ["AAA"], "sample_version": ["confirmatory"], "provisional_core": [True], "positive_green_import_baseline_eligible": [True]}),
        scaler_hash="d44dfe2bd8023e6b9c695b91dc73026a061e5f9b64198b41cf642979de8d612e",
    )

    assert output.filter((pl.col("year") >= 2023) & (pl.col("specification_id") != "gad_lite")).is_empty()
    assert output.filter(pl.col("specification_id") == "gad_lite").get_column("year").max() == 2024
    assert output.filter((pl.col("specification_id") == "gad_lite") & (pl.col("year") == 2023))["descriptive_only"].item() is True
    assert output.filter((pl.col("specification_id") == "gad_core") & (pl.col("year") == 2000))["confirmatory_eligible"].item() is True
    assert audit["core_eligibility_reason"].item() == "eligible"


def test_cli_registers_build_gad_with_all_registered_variants() -> None:
    from green_debt.cli import build_parser

    args = build_parser().parse_args(["build-gad", "--all-registered-variants"])

    assert args.command == "build-gad"
    assert args.all_registered_variants is True
