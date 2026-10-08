from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from green_debt.artifacts import BuildIdentity, write_authoritative_table
from green_debt.outcomes import (
    _contract,
    audit_outcomes,
    choose_gad_variant,
    outcome_specs,
)
from green_debt.paths import resolve_project_paths


ROOT = Path(__file__).resolve().parents[2]


def test_outcome_mapping_uses_leave_one_variants() -> None:
    assert choose_gad_variant("domestic_value_added_share") == "gad_no_gfvad"
    assert choose_gad_variant("green_export_complexity") == "gad_no_supp"
    assert choose_gad_variant("future_green_rca_entry_rate") == "gad_no_supp"
    assert choose_gad_variant("green_science_output") == "gad_no_gsci"
    assert choose_gad_variant("co2_tonnes_per_million_current_usd") == "gad_core"


def test_unknown_outcome_fails_closed() -> None:
    with pytest.raises(KeyError, match="unregistered outcome"):
        choose_gad_variant("mystery")


def test_cli_exposes_the_outcome_build_and_audit_commands() -> None:
    """Omitting either command prevents the independent authority from being built or checked."""
    from green_debt.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["build-outcomes"]).command == "build-outcomes"
    assert parser.parse_args(["audit-outcomes"]).command == "audit-outcomes"


def _outcome_audit_paths(tmp_path: Path):
    paths = resolve_project_paths(ROOT, tmp_path)
    country_contract = _contract("outcomes_country_year.json")
    product_contract = _contract("outcomes_product_year.json")

    def frame_for(contract):
        data: dict[str, list[object]] = {}
        for column, dtype in contract.columns.items():
            if column == "economy_id":
                data[column] = ["AAA"]
            elif column == "year":
                data[column] = [2000]
            elif column == "hs6":
                data[column] = ["000001"]
            elif dtype == "Boolean":
                data[column] = [False]
            else:
                data[column] = [1.0]
        return pl.DataFrame(data).cast({name: getattr(pl, dtype) for name, dtype in contract.columns.items()})

    build = BuildIdentity(command="test", code_commit="test")
    country_path = paths.measures / "outcomes/outcomes_country_year.parquet"
    product_path = paths.measures / "outcomes/outcomes_product_year.parquet"
    write_authoritative_table(frame_for(country_contract), country_contract, country_path, (), build)
    write_authoritative_table(frame_for(product_contract), product_contract, product_path, (), build)
    return paths


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("orientation", "missing_orientation", "sign_registration_failures"),
        ("transformation_flags", frozenset({"raw", "unwinsorized", "interpolated", "not_shock_multiplied"}), "outcome_interpolation_count"),
        ("formula_inputs", ("gad",), "gad_component_formula_lineage_columns"),
    ],
)
def test_audit_rejects_tampered_outcome_metadata(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    """Literal audit counters must not certify altered orientation, interpolation, or GAD lineage."""
    paths = _outcome_audit_paths(tmp_path)
    original = next(spec for spec in outcome_specs() if spec.outcome_id == "co2_tonnes_per_million_current_usd")
    altered = replace(original, **{field: value})
    specs = tuple(altered if spec.outcome_id == altered.outcome_id else spec for spec in outcome_specs())

    with pytest.raises(RuntimeError, match=match):
        audit_outcomes(paths, audit_path=tmp_path / "outcome-audit.csv", specs=specs)
