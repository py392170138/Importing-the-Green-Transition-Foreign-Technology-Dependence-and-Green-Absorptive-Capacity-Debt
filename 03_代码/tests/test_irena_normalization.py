import copy
import json
from pathlib import Path

import polars as pl
import pytest

from green_debt.sources.irena import (
    normalize_irena_generation_payloads,
    normalize_irena_payloads,
)


FIXTURES = Path(__file__).parent / "fixtures/irena"


def _payload(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_irena_uses_approved_dimensions_and_sorted_consecutive_additions() -> None:
    out = normalize_irena_payloads(
        capacity_payloads=(_payload("capacity.jsonstat2.json"),),
        generation_payloads=(_payload("generation.jsonstat2.json"),),
    )

    capacity = out.filter(
        pl.col("indicator_id") == "renewable_capacity_mw"
    ).sort("year")
    assert capacity["year"].to_list() == [2000, 2001, 2002]
    assert capacity["value"].to_list() == [10.0, 8.0, 10.0]

    additions = out.filter(
        pl.col("indicator_id") == "renewable_capacity_additions_mw"
    ).sort("year")
    assert additions["value"].to_list() == [None, -2.0, 2.0]
    assert additions["retirement_or_revision"].to_list() == [None, True, False]

    generation = out.filter(
        pl.col("indicator_id") == "renewable_generation_gwh"
    ).sort("year")
    assert generation["value"].to_list() == [100.0, 90.0, 110.0]


def test_generation_without_all_is_missing_not_on_off_sum() -> None:
    payload = _payload("generation.jsonstat2.json")
    grid = payload["dimension"]["Grid connection"]["category"]
    grid["index"] = {"on": 0, "off": 1}
    grid["label"] = {"on": "On-grid", "off": "Off-grid"}
    payload["size"][0] = 2
    payload["value"] = [3.0] * 6

    out = normalize_irena_generation_payloads((payload,)).sort("year")

    assert out["value"].to_list() == [None, None, None]
    assert out["source_status"].unique().to_list() == ["missing_approved_all_cell"]


def test_duplicate_cells_or_changed_target_label_fail() -> None:
    capacity = _payload("capacity.jsonstat2.json")
    with pytest.raises(ValueError, match="duplicate IRENA"):
        normalize_irena_payloads(
            capacity_payloads=(capacity, copy.deepcopy(capacity)),
            generation_payloads=(_payload("generation.jsonstat2.json"),),
        )

    changed = copy.deepcopy(capacity)
    changed["dimension"]["Technology"]["category"]["label"]["total"] = (
        "Renewables total"
    )
    with pytest.raises(ValueError, match="Total renewable energy"):
        normalize_irena_payloads(
            capacity_payloads=(changed,),
            generation_payloads=(_payload("generation.jsonstat2.json"),),
        )
