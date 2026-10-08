import pytest

from green_debt.sample import validate_outcome_gad_pair


@pytest.mark.parametrize(
    ("outcome", "gad_version"),
    [
        ("domestic_value_added_share", "gad_core"),
        ("green_export_complexity", "gad_core"),
        ("green_science_output", "gad_core"),
    ],
)
def test_mechanically_overlapping_pair_is_rejected(outcome: str, gad_version: str) -> None:
    with pytest.raises(ValueError, match="mechanical overlap"):
        validate_outcome_gad_pair(outcome, gad_version)


@pytest.mark.parametrize(
    ("outcome", "gad_version"),
    [
        ("domestic_value_added_share", "gad_no_gfvad"),
        ("green_export_complexity", "gad_no_supp"),
        ("green_science_output", "gad_no_gsci"),
        ("co2_tonnes_per_million_current_usd", "gad_core"),
    ],
)
def test_frozen_nonoverlapping_pair_is_accepted(outcome: str, gad_version: str) -> None:
    validate_outcome_gad_pair(outcome, gad_version)


def test_registered_outcome_with_wrong_leave_one_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="frozen mapping"):
        validate_outcome_gad_pair("co2_tonnes_per_million_current_usd", "gad_no_gfvad")
