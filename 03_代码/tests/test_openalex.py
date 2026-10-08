from __future__ import annotations

import csv
import json
from pathlib import Path

import httpx
import pytest

from green_debt.cli import main
from green_debt.network import DirectHttpClient
from green_debt.sources.openalex import (
    acquire_country_year_aggregates,
    audit_topic_registry,
    build_topic_candidates,
    build_works_query,
    fetch_country_groups,
    reject_snapshot_mode,
    safe_request_hash,
    write_topic_registry,
)


FIXTURES = Path(__file__).parent / "fixtures" / "openalex"


def test_openalex_query_is_topic_year_and_cursor_bounded() -> None:
    query = build_works_query(
        topic_ids=("T1001", "T1002"),
        year=2000,
        api_key="secret",
        cursor="*",
    )

    assert query["filter"] == (
        "topics.id:T1001|T1002,"
        "from_publication_date:2000-01-01,to_publication_date:2000-12-31"
    )
    assert query["group_by"] == "authorships.institutions.country_code"
    assert query["cursor"] == "*"
    assert query["per_page"] == "200"
    assert query["include_xpac"] == "false"
    assert query["api_key"] == "secret"


def test_request_hash_never_depends_on_or_contains_api_key() -> None:
    first = build_works_query(
        topic_ids=("T1001",), year=2000, api_key="first-secret", cursor="*"
    )
    second = build_works_query(
        topic_ids=("T1001",), year=2000, api_key="second-secret", cursor="*"
    )

    first_hash, first_safe_query = safe_request_hash(first)
    second_hash, second_safe_query = safe_request_hash(second)

    assert first_hash == second_hash
    assert "api_key" not in first_safe_query
    assert "first-secret" not in json.dumps(first_safe_query)


def test_snapshot_mode_is_always_rejected() -> None:
    with pytest.raises(RuntimeError, match="OpenAlex full snapshot is forbidden"):
        reject_snapshot_mode("full_snapshot")


def test_topic_registry_resolves_ids_from_recorded_response(tmp_path: Path) -> None:
    candidates = build_topic_candidates(
        response_files=(FIXTURES / "topic_search_fixture.json",),
        decisions_path=FIXTURES / "topic_decisions_fixture.yaml",
    )
    registry = tmp_path / "topics.csv"

    write_topic_registry(candidates, registry)
    report = audit_topic_registry(registry)

    with registry.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    wind = next(row for row in rows if row["topic_name"] == "Wind Energy Technology")
    generic = next(row for row in rows if row["topic_name"] == "Generic Management")
    assert wind["topic_id"] == "T1001"
    assert wind["include_main"] == "true"
    assert generic["include_main"] == "false"
    assert report.included_topics == 1
    assert report.duplicate_topic_ids == 0
    assert report.unreviewed_candidates == 0


def test_country_group_fetch_follows_cursor_and_never_persists_key(
    tmp_path: Path,
) -> None:
    seen_cursors: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params["cursor"]
        seen_cursors.append(cursor)
        if cursor == "*":
            payload = {
                "meta": {"count": 100, "next_cursor": "next-page"},
                "group_by": [
                    {"key": "CN", "key_display_name": "China", "count": 60},
                    {"key": "US", "key_display_name": "United States", "count": 40},
                ],
            }
        else:
            payload = {
                "meta": {"count": 100, "next_cursor": None},
                "group_by": [
                    {"key": "DE", "key_display_name": "Germany", "count": 10}
                ],
            }
        return httpx.Response(200, json=payload, request=request)

    client = DirectHttpClient(transport=httpx.MockTransport(handler))
    try:
        result = fetch_country_groups(
            client=client,
            base_url="https://api.openalex.org/works",
            topic_ids=("T1001",),
            year=2000,
            api_key="never-write-this-key",
            destination=tmp_path / "green" / "2000",
            response_budget_bytes=100_000,
        )
    finally:
        client.close()

    assert seen_cursors == ["*", "next-page"]
    assert result.country_counts == {"CN": 60, "DE": 10, "US": 40}
    assert result.pages == 2
    assert not list(tmp_path.rglob("*.partial"))
    persisted = "".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert "never-write-this-key" not in persisted


def test_country_group_fetch_rejects_duplicate_country_across_pages(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = {
            "meta": {"count": 2, "next_cursor": "again" if calls == 1 else None},
            "group_by": [{"key": "US", "key_display_name": "US", "count": 1}],
        }
        return httpx.Response(200, json=payload, request=request)

    client = DirectHttpClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(RuntimeError, match="duplicate country group US"):
            fetch_country_groups(
                client=client,
                base_url="https://api.openalex.org/works",
                topic_ids=("T1001",),
                year=2000,
                api_key=None,
                destination=tmp_path / "green" / "2000",
                response_budget_bytes=100_000,
            )
    finally:
        client.close()


def test_country_entity_url_group_key_is_normalized_to_iso2(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "meta": {"count": 3, "next_cursor": None},
                "group_by": [
                    {
                        "key": "https://openalex.org/countries/AE",
                        "key_display_name": "United Arab Emirates",
                        "count": 3,
                    }
                ],
            },
            request=request,
        )

    client = DirectHttpClient(transport=httpx.MockTransport(handler))
    try:
        result = fetch_country_groups(
            client=client,
            base_url="https://api.openalex.org/works",
            topic_ids=("T1001",),
            year=2000,
            api_key=None,
            destination=tmp_path / "green" / "2000",
            response_budget_bytes=100_000,
        )
    finally:
        client.close()

    assert result.country_counts == {"AE": 3}


def test_second_country_group_fetch_uses_verified_cache(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "meta": {"count": 1, "next_cursor": None},
                "group_by": [
                    {"key": "US", "key_display_name": "US", "count": 1}
                ],
            },
            request=request,
        )

    client = DirectHttpClient(transport=httpx.MockTransport(handler))
    try:
        first = fetch_country_groups(
            client=client,
            base_url="https://api.openalex.org/works",
            topic_ids=("T1001",),
            year=2000,
            api_key=None,
            destination=tmp_path / "green" / "2000",
            response_budget_bytes=100_000,
        )
        second = fetch_country_groups(
            client=client,
            base_url="https://api.openalex.org/works",
            topic_ids=("T1001",),
            year=2000,
            api_key=None,
            destination=tmp_path / "green" / "2000",
            response_budget_bytes=100_000,
        )
    finally:
        client.close()

    assert first == second
    assert calls == 1


def test_year_range_writes_unique_complete_country_year_panel(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        is_green = "topics.id" in request.url.params["filter"]
        year = int(request.url.params["filter"].split("from_publication_date:")[1][:4])
        groups = [
            {"key": "US", "key_display_name": "US", "count": year - 1990}
        ]
        if not is_green:
            groups.append(
                {"key": "DE", "key_display_name": "DE", "count": year - 1980}
            )
        return httpx.Response(
            200,
            json={
                "meta": {"count": 1000, "next_cursor": None},
                "group_by": groups,
            },
            request=request,
        )

    client = DirectHttpClient(transport=httpx.MockTransport(handler))
    try:
        report = acquire_country_year_aggregates(
            client=client,
            base_url="https://api.openalex.org/works",
            topic_ids=("T1001",),
            start_year=2000,
            end_year=2001,
            api_key=None,
            output_root=tmp_path / "aggregates",
            response_budget_bytes=1_000_000,
        )
    finally:
        client.close()

    with report.panel_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert calls == 4
    assert report.country_year_rows == 4
    assert len({(row["country_code"], row["year"]) for row in rows}) == 4
    de_rows = [row for row in rows if row["country_code"] == "DE"]
    assert {row["green_works"] for row in de_rows} == {"0"}
    assert {row["counting_method"] for row in rows} == {
        "full_country_participation"
    }


def test_year_range_rejects_green_country_absent_from_total(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        is_green = "topics.id" in request.url.params["filter"]
        groups = (
            [{"key": "US", "key_display_name": "US", "count": 1}]
            if is_green
            else [{"key": "DE", "key_display_name": "DE", "count": 2}]
        )
        return httpx.Response(
            200,
            json={"meta": {"count": 2, "next_cursor": None}, "group_by": groups},
            request=request,
        )

    client = DirectHttpClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(RuntimeError, match="green countries absent from total"):
            acquire_country_year_aggregates(
                client=client,
                base_url="https://api.openalex.org/works",
                topic_ids=("T1001",),
                start_year=2000,
                end_year=2000,
                api_key=None,
                output_root=tmp_path / "aggregates",
                response_budget_bytes=1_000_000,
            )
    finally:
        client.close()


def test_openalex_topic_audit_cli_reports_ready_registry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidates = build_topic_candidates(
        response_files=(FIXTURES / "topic_search_fixture.json",),
        decisions_path=FIXTURES / "topic_decisions_fixture.yaml",
    )
    registry = tmp_path / "topics.csv"
    write_topic_registry(candidates, registry)

    exit_code = main(["openalex-topic-audit", "--registry", str(registry)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "ready"
    assert payload["included_topics"] == 1


def test_openalex_acquisition_dry_run_writes_no_response_body(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidates = build_topic_candidates(
        response_files=(FIXTURES / "topic_search_fixture.json",),
        decisions_path=FIXTURES / "topic_decisions_fixture.yaml",
    )
    registry = tmp_path / "topics.csv"
    write_topic_registry(candidates, registry)
    output_root = tmp_path / "raw"

    exit_code = main(
        [
            "acquire-openalex-aggregates",
            "--registry",
            str(registry),
            "--output-root",
            str(output_root),
            "--start",
            "2000",
            "--end",
            "2001",
            "--response-budget-gb",
            "1",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["downloads_response_bodies"] is False
    assert payload["topic_count"] == 1
    assert not output_root.exists()


def test_science_audit_cli_rejects_duplicate_country_year(
    tmp_path: Path,
) -> None:
    panel = tmp_path / "counts.csv"
    panel.write_text(
        "country_code,year,green_works,total_works\n"
        "US,2000,1,2\n"
        "US,2000,1,2\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="duplicate country-year keys"):
        main(["science-audit", "--counts-csv", str(panel)])
