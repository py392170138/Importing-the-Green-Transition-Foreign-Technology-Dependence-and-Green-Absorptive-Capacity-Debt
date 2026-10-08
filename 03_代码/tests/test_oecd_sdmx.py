import json
from pathlib import Path

import httpx
import pytest

from green_debt.network import DirectHttpClient
from green_debt.sources.oecd_sdmx import (
    TivaSupplementSpec,
    acquire_tiva_supplement,
    load_tiva_supplements,
)


ROOT = Path(__file__).resolve().parents[2]
HEADER = (
    "DATAFLOW,MEASURE,REF_AREA,ACTIVITY,COUNTERPART_AREA,UNIT_MEASURE,"
    "FREQ,TIME_PERIOD,OBS_VALUE,UNIT_MULT\n"
)


def spec() -> TivaSupplementSpec:
    return TivaSupplementSpec(
        source_id="oecd_tiva_dfd_fva_history",
        measure="DFD_FVA",
        start_year=1995,
        end_year=1999,
        counterpart="W",
        unit_measure="USD",
        frequency="A",
        destination="DFD_FVA.all_areas_all_activities.world.1995-1999.csv",
        expected_max_bytes=128 * 1024**2,
    )


def test_tiva_spec_is_bounded_to_world_usd_annual() -> None:
    item = spec()

    assert item.key == "DFD_FVA...W.USD.A"
    assert item.query == {
        "startPeriod": "1995",
        "endPeriod": "1999",
        "dimensionAtObservation": "AllDimensions",
    }


def test_only_two_tiva_extensions_are_planned() -> None:
    supplements = load_tiva_supplements(ROOT / "config/tiva_supplements.yaml")

    assert [(item.measure, item.start_year, item.end_year) for item in supplements] == [
        ("DFD_FVA", 1995, 1999),
        ("FD_VA", 1995, 2022),
    ]
    assert {item.source_id for item in supplements} == {
        "oecd_tiva_dfd_fva_history",
        "oecd_tiva_fd_va_full",
    }


def test_supplement_records_user_authorized_route_exception(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.startswith("/sti-public/rest/data/")
        assert request.url.path.endswith("/DFD_FVA...W.USD.A")
        assert dict(request.url.params) == spec().query
        body = (
            HEADER
            + "OECD.STI.PIE:DSD_TIVA_MAINLV@DF_MAINLV(1.1),DFD_FVA,CAN,C27,W,USD,A,1995,1.25,6\n"
            + "OECD.STI.PIE:DSD_TIVA_MAINLV@DF_MAINLV(1.1),DFD_FVA,CAN,C27,W,USD,A,1999,1.50,6\n"
        )
        return httpx.Response(200, text=body, request=request)

    client = DirectHttpClient(transport=httpx.MockTransport(handler))
    log = tmp_path / "download.jsonl"
    try:
        record = acquire_tiva_supplement(
            spec(),
            client=client,
            output_root=tmp_path,
            route_exception_authorization="user_authorized_2026-08-22",
            route_exception_used=True,
            audit_log_path=log,
        )
    finally:
        client.close()

    assert record.proxy_bypass_enforced is True
    assert record.route_exception_authorization == "user_authorized_2026-08-22"
    assert record.route_exception_used is True
    manifest = json.loads(
        (tmp_path / f"{spec().destination}.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["route_exception_authorization"] == "user_authorized_2026-08-22"
    assert json.loads(log.read_text(encoding="utf-8"))["route_exception_used"] is True
    assert not list(tmp_path.rglob("*.partial"))


def test_invalid_tiva_units_fail_before_authoritative_rename(tmp_path: Path) -> None:
    body = (
        HEADER
        + "OECD.STI.PIE:DSD_TIVA_MAINLV@DF_MAINLV(1.1),DFD_FVA,CAN,C27,W,USD,A,1995,1.25,3\n"
    )
    client = DirectHttpClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=body, request=request)
        )
    )
    try:
        with pytest.raises(RuntimeError, match="UNIT_MULT=6"):
            acquire_tiva_supplement(
                spec(),
                client=client,
                output_root=tmp_path,
                route_exception_authorization=None,
            )
    finally:
        client.close()

    assert not (tmp_path / spec().destination).exists()
    assert not list(tmp_path.rglob("*.partial"))
