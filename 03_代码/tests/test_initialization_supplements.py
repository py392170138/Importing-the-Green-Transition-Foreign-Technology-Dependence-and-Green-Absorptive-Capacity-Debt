import json
from pathlib import Path

import httpx
import pytest

from green_debt.acquire import AcquisitionRunner, DownloadSpec, manifest_path_for
from green_debt.cli import main
from green_debt.network import DirectHttpClient, DirectRouteGuard
from green_debt.night_direct import DataHostPolicy, NightDirectGate
from green_debt.sources.openalex import (
    acquire_country_year_aggregates,
    initialization_supplement_spec,
    load_included_topic_ids,
    validate_initialization_supplement_range,
)
from green_debt.storage import DiskBudgetGuard, GIB


ROOT = Path(__file__).resolve().parents[2]


def test_openalex_extension_is_exactly_1992_1995() -> None:
    registry = ROOT / "02_数据字典/openalex_green_topics_v1.csv"
    frozen = initialization_supplement_spec(registry)

    assert validate_initialization_supplement_range(1992, 1995) == (1992, 1995)
    assert len(load_included_topic_ids(registry)) == 59
    assert frozen.start_year == 1992
    assert frozen.end_year == 1995
    assert frozen.topic_count == 59
    assert frozen.counting_method == "full_country_participation"
    assert frozen.include_xpac is False
    assert frozen.destination.name == "openalex_country_year_counts_1992_1995.csv"
    with pytest.raises(ValueError, match="1992-1995"):
        validate_initialization_supplement_range(1991, 1995)
    with pytest.raises(ValueError, match="1992-1995"):
        validate_initialization_supplement_range(1992, 1996)


def test_openalex_page_manifests_record_route_authorization(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        is_green = "topics.id" in request.url.params["filter"]
        return httpx.Response(
            200,
            json={
                "meta": {"count": 1, "next_cursor": None},
                "group_by": [
                    {
                        "key": "US",
                        "key_display_name": "United States",
                        "count": 1 if is_green else 2,
                    }
                ],
            },
            request=request,
        )

    client = DirectHttpClient(transport=httpx.MockTransport(handler))
    try:
        acquire_country_year_aggregates(
            client=client,
            base_url="https://api.openalex.org/works",
            topic_ids=("T1001",),
            start_year=1992,
            end_year=1992,
            api_key=None,
            output_root=tmp_path,
            response_budget_bytes=100_000,
            route_exception_authorization="user_authorized_2026-08-22",
            route_exception_used=False,
        )
    finally:
        client.close()

    page_manifest = next(tmp_path.glob("green/1992/*.manifest.json"))
    payload = json.loads(page_manifest.read_text(encoding="utf-8"))
    assert payload["route_exception_authorization"] == "user_authorized_2026-08-22"
    assert payload["route_exception_used"] is False


def test_authorized_route_exception_is_explicit_and_logged() -> None:
    gate = NightDirectGate(
        policy=DataHostPolicy(("api.openalex.org",)),
        route_guard=DirectRouteGuard(
            system_proxy_reader=lambda: {"HTTPEnable": "1"},
            route_reader=lambda host, ip: "utun7",
            resolver=lambda host: "203.0.113.1",
            rejected_interface_prefixes=("utun",),
        ),
        disk_guard=DiskBudgetGuard(
            project_root=ROOT,
            hard_stop_bytes=120 * GIB,
            reserve_bytes=30 * GIB,
            usage_reader=lambda path: 5 * GIB,
            free_reader=lambda path: 100 * GIB,
        ),
        tunnel_reader=lambda: ("Shadowrocket",),
        allow_logged_user_route_exception=True,
    )

    evidence = gate.check(
        "https://api.openalex.org/works",
        projected_additional_bytes=1,
        route_exception_authorization="user_authorized_2026-08-22",
    )

    assert evidence.connected_tunnel_count == 1
    assert evidence.route_interface == "utun7"
    assert evidence.route_exception_used is True
    assert evidence.route_exception_authorization == "user_authorized_2026-08-22"


def test_direct_route_retains_optional_authorization_without_using_it() -> None:
    gate = NightDirectGate(
        policy=DataHostPolicy(("api.openalex.org",)),
        route_guard=DirectRouteGuard(
            system_proxy_reader=lambda: {},
            route_reader=lambda host, ip: "en0",
            resolver=lambda host: "203.0.113.1",
            rejected_interface_prefixes=("utun",),
        ),
        disk_guard=DiskBudgetGuard(
            project_root=ROOT,
            hard_stop_bytes=120 * GIB,
            reserve_bytes=30 * GIB,
            usage_reader=lambda path: 5 * GIB,
            free_reader=lambda path: 100 * GIB,
        ),
        tunnel_reader=lambda: (),
        allow_logged_user_route_exception=True,
    )

    evidence = gate.check(
        "https://api.openalex.org/works",
        projected_additional_bytes=1,
        route_exception_authorization="user_authorized_2026-08-22",
    )

    assert evidence.route_exception_used is False
    assert evidence.route_exception_authorization == "user_authorized_2026-08-22"


def test_generic_acquisition_manifest_and_log_retain_route_fields(tmp_path: Path) -> None:
    payload = b"ok"
    runner = AcquisitionRunner.for_test(
        tmp_path,
        httpx.MockTransport(
            lambda request: httpx.Response(200, content=payload, request=request)
        ),
    )
    spec = DownloadSpec(
        source_id="route_fixture",
        source_version="1",
        url="https://fixture.example/data",
        allowed_hosts=("fixture.example",),
        destination=tmp_path / "raw/data.bin",
        expected_max_bytes=10,
        expected_sha256=None,
        projected_working_bytes=10,
    )

    record = runner.acquire(
        spec,
        route_exception_authorization="user_authorized_2026-08-22",
        route_exception_used=False,
    )

    assert record.route_exception_authorization == "user_authorized_2026-08-22"
    assert record.route_exception_used is False
    manifest = json.loads(manifest_path_for(spec.destination).read_text(encoding="utf-8"))
    log = json.loads((tmp_path / "下载日志.jsonl").read_text(encoding="utf-8"))
    assert manifest["route_exception_authorization"] == log[
        "route_exception_authorization"
    ]


def test_initialization_dry_run_is_bounded_and_downloads_no_bodies(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "acquire-initialization-supplements",
            "--data-root",
            str(tmp_path),
            "--dry-run",
            "--route-exception-authorization",
            "user_authorized_2026-08-22",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["downloads_response_bodies"] is False
    assert payload["source_ids"] == [
        "oecd_tiva_dfd_fva_history",
        "oecd_tiva_fd_va_full",
        "openalex_initialization_aggregates",
    ]
    assert payload["openalex_range"] == [1992, 1995]
    assert payload["projected_additional_bytes"] < GIB
    assert not (tmp_path / "04_原始数据").exists()
