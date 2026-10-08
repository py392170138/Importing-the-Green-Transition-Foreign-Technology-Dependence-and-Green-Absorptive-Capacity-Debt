from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from green_debt.sources.tiva import append_tiva_history, normalize_tiva_csv
from green_debt.tiva import build_activity_weights


FIXTURE = Path(__file__).parent / "fixtures/tiva/mixed_numeric.csv"


def test_obs_value_is_float_even_when_early_rows_look_integer() -> None:
    out = normalize_tiva_csv(FIXTURE, expected_measure="DFD_FVA")

    assert out.schema["obs_value"] == pl.Float64
    assert out["obs_value"].to_list() == [1.0, 1.25]


def test_wrong_unit_multiplier_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "DATAFLOW,MEASURE,REF_AREA,ACTIVITY,COUNTERPART_AREA,UNIT_MEASURE,FREQ,TIME_PERIOD,OBS_VALUE,UNIT_MULT\n"
        "OECD.STI.PIE:DSD_TIVA_MAINLV@DF_MAINLV(1.1),FD_VA,AAA,C27,W,USD,A,1995,1.0,3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="UNIT_MULT"):
        normalize_tiva_csv(path, expected_measure="FD_VA")


def test_history_append_rejects_an_overlapping_key() -> None:
    history = normalize_tiva_csv(FIXTURE, expected_measure="DFD_FVA")
    current = history.filter(pl.col("year") == 1996)

    with pytest.raises(ValueError, match="overlapping"):
        append_tiva_history(history, current)


def test_prod_weights_are_country_specific_frozen_and_sum_to_one() -> None:
    prod = pl.DataFrame(
        {
            "economy_id": ["AAA"] * 10,
            "year": [year for year in range(2000, 2005) for _ in range(2)],
            "activity": ["C27", "C28"] * 5,
            "indicator_id": ["prod_level"] * 10,
            "value": [1.0, 2.0] * 5,
        }
    )

    out = build_activity_weights(
        prod,
        variants={"confirmatory_prod_weight": ("C27", "C28")},
        equal_variant_activities=("C27", "C28"),
    )

    confirmatory = out.filter(
        pl.col("weight_version") == "confirmatory_prod_weight"
    )
    equal = out.filter(pl.col("weight_version") == "equal_weight")
    assert confirmatory["activity_weight"].sum() == pytest.approx(1.0)
    assert confirmatory.filter(pl.col("activity") == "C28")[
        "activity_weight"
    ].item() == pytest.approx(2 / 3)
    assert equal["activity_weight"].to_list() == [0.5, 0.5]
    assert out["frozen_baseline_start"].unique().item() == 2000
    assert out["frozen_baseline_end"].unique().item() == 2004
