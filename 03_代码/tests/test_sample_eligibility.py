import polars as pl
import pytest

from green_debt.sample import freeze_final_sample, validate_final_sample


def test_core_requires_initialization_and_twelve_main_years() -> None:
    coverage = pl.DataFrame({
        "economy_id": ["A", "B"],
        "complete_absorption_1996": [True, True],
        "complete_initialization_1997_1999": [True, False],
        "valid_main_years": [12, 20],
        "positive_import_baseline_years": [2, 4],
        "baseline_share_identifiable": [True, True],
        "economy_rule_eligible": [True, True],
    })
    out = freeze_final_sample(coverage)
    assert out.filter(pl.col("economy_id") == "A")["core_eligible"].item() is True
    assert out.filter(pl.col("economy_id") == "B")["exclusion_reason"].item() == "incomplete_gad_initialization"


def test_structural_sample_does_not_require_an_outcome() -> None:
    coverage = pl.DataFrame({
        "economy_id": ["A"],
        "complete_absorption_1996": [True],
        "complete_initialization_1997_1999": [True],
        "valid_main_years": [12],
        "positive_import_baseline_years": [2],
        "baseline_share_identifiable": [True],
        "economy_rule_eligible": [True],
    })
    out = freeze_final_sample(coverage)
    assert "required_outcome_coverage" not in out.columns
    assert out["core_eligible"].item() is True


def test_lite_eligibility_and_descriptive_years_are_separate_from_core() -> None:
    coverage = pl.DataFrame({
        "economy_id": ["A"],
        "complete_absorption_1996": [False],
        "complete_initialization_1997_1999": [False],
        "valid_main_years": [20],
        "complete_lite_absorption_1996": [True],
        "complete_lite_initialization_1997_1999": [True],
        "valid_lite_years": [20],
        "positive_import_baseline_years": [4],
        "baseline_share_identifiable": [True],
        "economy_rule_eligible": [True],
    })
    out = freeze_final_sample(coverage)
    assert out["core_eligible"].item() is False
    assert out["lite_eligible"].item() is True
    assert out["descriptive_only_2023"].item() is True
    assert out["descriptive_only_2024"].item() is True


def test_first_exclusion_reason_uses_frozen_rule_order() -> None:
    coverage = pl.DataFrame({
        "economy_id": ["A"],
        "complete_absorption_1996": [False],
        "complete_initialization_1997_1999": [False],
        "valid_main_years": [0],
        "positive_import_baseline_years": [0],
        "baseline_share_identifiable": [False],
        "economy_rule_eligible": [False],
    })
    out = freeze_final_sample(coverage)
    assert out["exclusion_reason"].item() == "economy_rule_ineligible"


def test_final_sample_validator_enforces_conditional_exclusion_reason() -> None:
    coverage = pl.DataFrame({
        "economy_id": ["A"],
        "complete_absorption_1996": [True],
        "complete_initialization_1997_1999": [True],
        "valid_main_years": [12],
        "positive_import_baseline_years": [2],
        "baseline_share_identifiable": [True],
        "economy_rule_eligible": [True],
    })
    frozen = freeze_final_sample(coverage)
    validate_final_sample(frozen)
    tampered = frozen.with_columns(pl.lit("fabricated_reason").alias("exclusion_reason"))
    with pytest.raises(ValueError, match="exclusion_reason"):
        validate_final_sample(tampered)
