import json
from pathlib import Path

import polars as pl
import pytest

from green_debt.sources.wdi import normalize_wdi_payload, validate_wdi_metadata


FIXTURE = Path(__file__).parent / "fixtures/wdi/page.json"


def test_wdi_null_stays_missing_and_unit_is_frozen() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    out = normalize_wdi_payload(
        payload,
        unit="Mt CO2e excluding LULUCF",
        expected_indicator="EN.GHG.CO2.MT.CE.AR5",
        expected_period=(2000, 2000),
    )

    assert out.schema["value"] == pl.Float64
    assert out.filter(pl.col("economy_id") == "BBB")["value"].item() is None
    assert out["unit"].unique().to_list() == ["Mt CO2e excluding LULUCF"]
    assert out["source_update"].unique().to_list() == ["2026-07-13"]
    assert out.filter(pl.col("economy_id") == "BBB")["source_status"].item() == "api_null"


def test_wdi_rejects_changed_indicator_or_page_envelope() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload[1][0]["indicator"]["id"] = "EN.ATM.CO2E.KT"

    with pytest.raises(ValueError, match="indicator code"):
        normalize_wdi_payload(
            payload,
            unit="Mt CO2e",
            expected_indicator="EN.GHG.CO2.MT.CE.AR5",
            expected_period=(2000, 2000),
        )

    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload[0]["pages"] = 2
    with pytest.raises(ValueError, match="page envelope"):
        normalize_wdi_payload(
            payload,
            unit="Mt CO2e",
            expected_indicator="EN.GHG.CO2.MT.CE.AR5",
            expected_period=(2000, 2000),
        )


def test_rejected_retired_wdi_code_fails_closed() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for row in payload[1]:
        row["indicator"]["id"] = "EN.ATM.CO2E.KT"

    with pytest.raises(ValueError, match="rejected WDI indicator"):
        normalize_wdi_payload(
            payload,
            unit="kt CO2",
            expected_indicator="EN.ATM.CO2E.KT",
            expected_period=(2000, 2000),
        )


def test_wdi_metadata_envelope_does_not_require_observation_update() -> None:
    metadata = [
        {"page": 1, "pages": 1, "per_page": "50", "total": 1},
        [
            {
                "id": "EN.GHG.CO2.MT.CE.AR5",
                "name": "Carbon dioxide emissions (Mt CO2e)",
                "unit": "",
                "source": {"id": "2", "value": "WDI"},
            }
        ],
    ]

    validate_wdi_metadata(
        metadata, expected_indicator="EN.GHG.CO2.MT.CE.AR5"
    )
