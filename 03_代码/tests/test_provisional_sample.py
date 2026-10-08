from __future__ import annotations

import polars as pl

from green_debt.sample import build_provisional_sample


def test_provisional_sample_uses_sources_not_outcomes() -> None:
    coverage = pl.DataFrame(
        {
            "economy_id": ["AAA", "BBB"],
            "population_2000": [2_000_000.0, 2_000_000.0],
            "tiva_member": [True, True],
            "positive_green_import_baseline_years": [2, 1],
            "economy_rule_eligible": [True, True],
        }
    )

    out = build_provisional_sample(coverage)

    assert out.filter(pl.col("economy_id") == "AAA")[
        "provisional_core"
    ].item() is True
    assert out.filter(pl.col("economy_id") == "BBB")[
        "exclusion_reason"
    ].item() == "fewer_than_two_positive_baseline_import_years"
    assert "outcome" not in "|".join(out.columns).lower()


def test_source_availability_is_a_separate_ordered_rule() -> None:
    coverage = pl.DataFrame(
        {
            "economy_id": ["AAA"],
            "population_2000": [2_000_000.0],
            "tiva_member": [True],
            "positive_green_import_baseline_years": [4],
            "economy_rule_eligible": [True],
            "wdi_available": [True],
            "baci_available": [True],
            "openalex_available": [False],
        }
    )

    out = build_provisional_sample(coverage)

    assert out["required_sources_available"].item() is False
    assert out["exclusion_reason"].item() == "required_source_unavailable"
