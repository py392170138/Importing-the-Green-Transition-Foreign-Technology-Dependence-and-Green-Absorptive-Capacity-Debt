"""Bounded OpenAlex topic auditing and country-year aggregate acquisition."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterable, Mapping

import httpx
import yaml

from green_debt.network import DirectHttpClient


TOPIC_ID_PATTERN = re.compile(r"^T[0-9]+$")
COUNTRY_CODE_PATTERN = re.compile(r"^[A-Z]{2}$")
COUNTRY_ENTITY_PREFIX = "https://openalex.org/countries/"
REGISTRY_FIELDS = (
    "topic_id",
    "topic_name",
    "description",
    "domain",
    "field",
    "subfield",
    "works_count",
    "seed_families",
    "seed_queries",
    "best_match_rank",
    "include_main",
    "canonical_seed_family",
    "inclusion_rule",
    "review_status",
    "source_response_files",
    "source_response_sha256",
)


@dataclass(frozen=True)
class TopicAuditReport:
    candidate_topics: int
    included_topics: int
    duplicate_topic_ids: int
    unreviewed_candidates: int
    invalid_included_rows: int


@dataclass(frozen=True)
class CountryGroupFetchResult:
    year: int
    country_counts: dict[str, int]
    unknown_country_count: int
    total_matching_works: int
    pages: int
    bytes: int
    request_hashes: tuple[str, ...]


@dataclass(frozen=True)
class AggregateRangeReport:
    start_year: int
    end_year: int
    country_year_rows: int
    pages: int
    bytes: int
    panel_path: Path


@dataclass(frozen=True)
class CountryYearPanelAudit:
    rows: int
    economies: int
    start_year: int
    end_year: int
    duplicate_country_year_keys: int
    negative_counts: int
    green_exceeds_total: int


@dataclass(frozen=True)
class OpenAlexInitializationSpec:
    start_year: int
    end_year: int
    topic_count: int
    counting_method: str
    include_xpac: bool
    destination: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_topic_ids(topic_ids: Iterable[str]) -> tuple[str, ...]:
    values = tuple(topic_ids)
    if len(values) > 100:
        raise ValueError("OpenAlex allows at most 100 OR topic IDs")
    if len(set(values)) != len(values):
        raise ValueError("topic IDs must be unique")
    if any(TOPIC_ID_PATTERN.fullmatch(value) is None for value in values):
        raise ValueError("invalid OpenAlex topic ID")
    return values


def build_works_query(
    *,
    topic_ids: Iterable[str],
    year: int,
    api_key: str | None,
    cursor: str,
) -> dict[str, str]:
    """Build one year- and cursor-bounded country aggregation query."""

    if year < 1800 or year > 2100:
        raise ValueError("year is outside the supported range")
    topics = _validate_topic_ids(topic_ids)
    filters = []
    if topics:
        filters.append(f"topics.id:{'|'.join(topics)}")
    filters.extend(
        (
            f"from_publication_date:{year}-01-01",
            f"to_publication_date:{year}-12-31",
        )
    )
    query = {
        "filter": ",".join(filters),
        "group_by": "authorships.institutions.country_code",
        "per_page": "200",
        "cursor": cursor,
        "include_xpac": "false",
    }
    if api_key:
        query["api_key"] = api_key
    return query


def safe_request_hash(query: Mapping[str, str]) -> tuple[str, dict[str, str]]:
    """Hash a canonical request after removing credentials."""

    safe_query = {
        str(key): str(value)
        for key, value in query.items()
        if str(key).lower() not in {"api_key", "mailto"}
    }
    canonical = json.dumps(
        safe_query,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), safe_query


def reject_snapshot_mode(mode: str) -> None:
    if mode.strip().lower() != "aggregate_api":
        raise RuntimeError("OpenAlex full snapshot is forbidden")


def _response_seed(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" not in stem:
        return "supplemental_audit", stem.replace("_", " ")
    family, query = stem.split("__", 1)
    return family, query.replace("_", " ")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _display_name(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("display_name") or "").strip()
    return ""


def build_topic_candidates(
    *,
    response_files: Iterable[Path],
    decisions_path: Path,
) -> list[dict[str, object]]:
    """Resolve frozen topic IDs only from recorded API responses."""

    decisions_raw = yaml.safe_load(decisions_path.read_text(encoding="utf-8"))
    if not isinstance(decisions_raw, dict) or decisions_raw.get("schema_version") != 1:
        raise ValueError("topic decisions schema_version must be 1")
    default_rule = str(decisions_raw.get("default_exclusion_rule") or "").strip()
    included_raw = decisions_raw.get("included_topics")
    if not default_rule or not isinstance(included_raw, dict):
        raise ValueError("topic decisions require default rule and included_topics")

    candidates: dict[str, dict[str, object]] = {}
    seen_names: set[str] = set()
    for path in response_files:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes)
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError(f"OpenAlex topic response has no results list: {path}")
        seed_family, seed_query = _response_seed(path)
        source_evidence = f"{path.name}:{_sha256_bytes(raw_bytes)}"
        for rank, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                raise ValueError(f"invalid topic result in {path}")
            topic_id = str(result.get("id") or "").rsplit("/", 1)[-1]
            topic_name = str(result.get("display_name") or "").strip()
            if TOPIC_ID_PATTERN.fullmatch(topic_id) is None or not topic_name:
                raise ValueError(f"invalid topic identity in {path}")
            current = candidates.get(topic_id)
            if current is None:
                current = {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "description": str(result.get("description") or "").strip(),
                    "domain": _display_name(result.get("domain")),
                    "field": _display_name(result.get("field")),
                    "subfield": _display_name(result.get("subfield")),
                    "works_count": int(result.get("works_count") or 0),
                    "seed_families": set(),
                    "seed_queries": set(),
                    "best_match_rank": rank,
                    "source_response_files": set(),
                    "source_response_sha256": set(),
                }
                candidates[topic_id] = current
            elif current["topic_name"] != topic_name:
                raise ValueError(f"topic ID {topic_id} changed display name")
            current["seed_families"].add(seed_family)  # type: ignore[union-attr]
            current["seed_queries"].add(seed_query)  # type: ignore[union-attr]
            current["best_match_rank"] = min(
                int(current["best_match_rank"]), rank
            )
            current["source_response_files"].add(path.name)  # type: ignore[union-attr]
            current["source_response_sha256"].add(source_evidence)  # type: ignore[union-attr]

    rows: list[dict[str, object]] = []
    for topic_id, current in candidates.items():
        topic_name = str(current["topic_name"])
        decision = included_raw.get(topic_name)
        include_main = decision is not None
        if include_main:
            if not isinstance(decision, dict):
                raise ValueError(f"invalid inclusion decision for {topic_name}")
            canonical_family = str(
                decision.get("canonical_seed_family") or ""
            ).strip()
            inclusion_rule = str(decision.get("inclusion_rule") or "").strip()
            if not canonical_family or not inclusion_rule:
                raise ValueError(f"incomplete inclusion decision for {topic_name}")
            seen_names.add(topic_name)
        else:
            canonical_family = ""
            inclusion_rule = default_rule
        rows.append(
            {
                **current,
                "seed_families": "|".join(sorted(current["seed_families"])),
                "seed_queries": "|".join(sorted(current["seed_queries"])),
                "include_main": include_main,
                "canonical_seed_family": canonical_family,
                "inclusion_rule": inclusion_rule,
                "review_status": "reviewed",
                "source_response_files": "|".join(
                    sorted(current["source_response_files"])
                ),
                "source_response_sha256": "|".join(
                    sorted(current["source_response_sha256"])
                ),
            }
        )
    missing = sorted(set(str(name) for name in included_raw) - seen_names)
    if missing:
        raise ValueError(f"included topic names absent from responses: {missing}")
    return sorted(rows, key=lambda row: str(row["topic_id"]))


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    if partial.exists():
        partial.unlink()
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_topic_registry(rows: Iterable[Mapping[str, object]], path: Path) -> None:
    buffer: list[str] = []
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=REGISTRY_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        normalized = dict(row)
        normalized["include_main"] = (
            "true" if bool(normalized.get("include_main")) else "false"
        )
        writer.writerow({field: normalized.get(field, "") for field in REGISTRY_FIELDS})
    buffer.append(stream.getvalue())
    _write_atomic(path, "".join(buffer).encode("utf-8"))


def audit_topic_registry(path: Path) -> TopicAuditReport:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    topic_ids = [row.get("topic_id", "") for row in rows]
    duplicates = len(topic_ids) - len(set(topic_ids))
    unreviewed = sum(row.get("review_status") != "reviewed" for row in rows)
    included = [row for row in rows if row.get("include_main") == "true"]
    invalid_included = sum(
        not row.get("topic_id")
        or not row.get("canonical_seed_family")
        or not row.get("inclusion_rule")
        for row in included
    )
    return TopicAuditReport(
        candidate_topics=len(rows),
        included_topics=len(included),
        duplicate_topic_ids=duplicates,
        unreviewed_candidates=unreviewed,
        invalid_included_rows=invalid_included,
    )


def load_included_topic_ids(path: Path) -> tuple[str, ...]:
    report = audit_topic_registry(path)
    if (
        report.included_topics == 0
        or report.duplicate_topic_ids
        or report.unreviewed_candidates
        or report.invalid_included_rows
    ):
        raise RuntimeError(f"OpenAlex topic registry audit failed: {report}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        values = tuple(
            row["topic_id"] for row in rows if row.get("include_main") == "true"
        )
    return _validate_topic_ids(values)


def validate_initialization_supplement_range(
    start_year: int,
    end_year: int,
) -> tuple[int, int]:
    """Accept only the frozen four-year pre-trade OpenAlex extension."""

    if (start_year, end_year) != (1992, 1995):
        raise ValueError("OpenAlex initialization supplement must be 1992-1995")
    return start_year, end_year


def initialization_supplement_spec(registry: Path) -> OpenAlexInitializationSpec:
    """Build the immutable initialization query contract from the topic registry."""

    topics = load_included_topic_ids(registry)
    if len(topics) != 59:
        raise RuntimeError("OpenAlex initialization supplement requires 59 topics")
    return OpenAlexInitializationSpec(
        start_year=1992,
        end_year=1995,
        topic_count=len(topics),
        counting_method="full_country_participation",
        include_xpac=False,
        destination=Path("openalex_country_year_counts_1992_1995.csv"),
    )


def _safe_headers(headers: httpx.Headers) -> dict[str, str]:
    allowed = {
        "content-type",
        "content-length",
        "date",
        "etag",
        "last-modified",
        "x-ratelimit-cost-usd",
        "x-ratelimit-credits-used",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
    }
    return {key.lower(): value for key, value in headers.items() if key.lower() in allowed}


def _country_code_from_group_key(key: str) -> str:
    candidate = (
        key.removeprefix(COUNTRY_ENTITY_PREFIX)
        if key.startswith(COUNTRY_ENTITY_PREFIX)
        else key
    )
    if COUNTRY_CODE_PATTERN.fullmatch(candidate) is None:
        raise RuntimeError(f"invalid OpenAlex country group {key}")
    return candidate


def _load_cached_country_groups(
    destination: Path,
    *,
    topic_ids: tuple[str, ...],
    year: int,
) -> CountryGroupFetchResult | None:
    summary_path = destination / "result.json"
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_bytes())
    if payload.get("year") != year or tuple(payload.get("topic_ids", [])) != topic_ids:
        raise RuntimeError("cached OpenAlex result does not match requested query")
    for page_file in payload.get("page_files", []):
        raw_path = destination / str(page_file)
        manifest_path = raw_path.with_name(
            raw_path.name.removesuffix(".json") + ".manifest.json"
        )
        if not raw_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("cached OpenAlex page or manifest is missing")
        manifest = json.loads(manifest_path.read_bytes())
        raw = raw_path.read_bytes()
        if len(raw) != manifest.get("bytes") or _sha256_bytes(raw) != manifest.get(
            "sha256"
        ):
            raise RuntimeError("cached OpenAlex page failed checksum verification")
    return CountryGroupFetchResult(
        year=year,
        country_counts={
            str(key): int(value)
            for key, value in payload.get("country_counts", {}).items()
        },
        unknown_country_count=int(payload.get("unknown_country_count") or 0),
        total_matching_works=int(payload.get("total_matching_works") or 0),
        pages=int(payload.get("pages") or 0),
        bytes=int(payload.get("bytes") or 0),
        request_hashes=tuple(payload.get("request_hashes", [])),
    )


def fetch_country_groups(
    *,
    client: DirectHttpClient,
    base_url: str,
    topic_ids: Iterable[str],
    year: int,
    api_key: str | None,
    destination: Path,
    response_budget_bytes: int,
    route_exception_authorization: str | None = None,
    route_exception_used: bool = False,
) -> CountryGroupFetchResult:
    """Fetch every cursor page for one country-year aggregation."""

    if response_budget_bytes <= 0:
        raise ValueError("response budget must be positive")
    topics = _validate_topic_ids(topic_ids)
    cached = _load_cached_country_groups(
        destination,
        topic_ids=topics,
        year=year,
    )
    if cached is not None:
        if cached.bytes > response_budget_bytes:
            raise RuntimeError("cached OpenAlex response exceeds response budget")
        return cached
    cursor = "*"
    seen_cursors: set[str] = set()
    country_counts: dict[str, int] = {}
    unknown_count = 0
    total_bytes = 0
    total_matching: int | None = None
    request_hashes: list[str] = []
    page_files: list[str] = []
    page = 0
    while cursor:
        if cursor in seen_cursors:
            raise RuntimeError("OpenAlex repeated a pagination cursor")
        seen_cursors.add(cursor)
        page += 1
        if page > 100:
            raise RuntimeError("OpenAlex country grouping exceeded 100 pages")
        query = build_works_query(
            topic_ids=topics,
            year=year,
            api_key=api_key,
            cursor=cursor,
        )
        request_hash, safe_query = safe_request_hash(query)
        request_url = httpx.URL(base_url, params=query)
        with client.stream("GET", str(request_url)) as response:
            if response.is_redirect:
                raise RuntimeError("OpenAlex redirect refused")
            if response.status_code != 200:
                raise RuntimeError(f"OpenAlex HTTP status {response.status_code}")
            declared = response.headers.get("Content-Length")
            if declared is not None and total_bytes + int(declared) > response_budget_bytes:
                raise RuntimeError("OpenAlex response budget exceeded")
            raw = b"".join(response.iter_bytes())
            total_bytes += len(raw)
            if total_bytes > response_budget_bytes:
                raise RuntimeError("OpenAlex response budget exceeded")
            safe_headers = _safe_headers(response.headers)
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("group_by"), list):
            raise RuntimeError("OpenAlex group response schema changed")
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise RuntimeError("OpenAlex response metadata is missing")
        matching = int(meta.get("count") or 0)
        if total_matching is None:
            total_matching = matching
        elif total_matching != matching:
            raise RuntimeError("OpenAlex total count changed during pagination")

        raw_path = destination / f"page_{page:03d}_{request_hash[:16]}.json"
        manifest_path = destination / f"page_{page:03d}_{request_hash[:16]}.manifest.json"
        _write_atomic(raw_path, raw)
        _write_atomic(
            manifest_path,
            (
                json.dumps(
                    {
                        "bytes": len(raw),
                        "downloaded_at_utc": _utc_now(),
                        "request_hash": request_hash,
                        "safe_query": safe_query,
                        "sha256": _sha256_bytes(raw),
                        "response_headers": safe_headers,
                        "route_exception_authorization": (
                            route_exception_authorization
                        ),
                        "route_exception_used": route_exception_used,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )
        request_hashes.append(request_hash)
        page_files.append(raw_path.name)

        for group in payload["group_by"]:
            if not isinstance(group, dict):
                raise RuntimeError("OpenAlex country group schema changed")
            key = str(group.get("key") or "unknown")
            count = int(group.get("count") or 0)
            if count < 0:
                raise RuntimeError("OpenAlex returned a negative group count")
            if key.lower() == "unknown":
                unknown_count += count
                continue
            country_code = _country_code_from_group_key(key)
            if country_code in country_counts:
                raise RuntimeError(f"duplicate country group {country_code}")
            country_counts[country_code] = count
        next_cursor = meta.get("next_cursor")
        cursor = str(next_cursor) if next_cursor else ""

    result = CountryGroupFetchResult(
        year=year,
        country_counts=dict(sorted(country_counts.items())),
        unknown_country_count=unknown_count,
        total_matching_works=total_matching or 0,
        pages=page,
        bytes=total_bytes,
        request_hashes=tuple(request_hashes),
    )
    _write_atomic(
        destination / "result.json",
        (
            json.dumps(
                {
                    "bytes": result.bytes,
                    "country_counts": result.country_counts,
                    "pages": result.pages,
                    "page_files": page_files,
                    "request_hashes": list(result.request_hashes),
                    "schema_version": 1,
                    "topic_ids": list(topics),
                    "total_matching_works": result.total_matching_works,
                    "unknown_country_count": result.unknown_country_count,
                    "year": result.year,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return result


def _write_panel_csv(rows: list[dict[str, object]], path: Path) -> None:
    from io import StringIO

    fields = (
        "country_code",
        "year",
        "green_works",
        "total_works",
        "green_total_matching_works",
        "total_all_matching_works",
        "counting_method",
        "include_xpac",
    )
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_atomic(path, stream.getvalue().encode("utf-8"))


def acquire_country_year_aggregates(
    *,
    client: DirectHttpClient,
    base_url: str,
    topic_ids: Iterable[str],
    start_year: int,
    end_year: int,
    api_key: str | None,
    output_root: Path,
    response_budget_bytes: int,
    route_exception_authorization: str | None = None,
    route_exception_used: bool = False,
    source_id: str = "openalex_works_api",
) -> AggregateRangeReport:
    """Acquire bounded green and total works counts for every requested year."""

    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year")
    topics = _validate_topic_ids(topic_ids)
    if not topics:
        raise ValueError("at least one frozen green topic is required")
    if response_budget_bytes <= 0:
        raise ValueError("response budget must be positive")
    rows: list[dict[str, object]] = []
    used_bytes = 0
    pages = 0
    for year in range(start_year, end_year + 1):
        green = fetch_country_groups(
            client=client,
            base_url=base_url,
            topic_ids=topics,
            year=year,
            api_key=api_key,
            destination=output_root / "green" / str(year),
            response_budget_bytes=response_budget_bytes - used_bytes,
            route_exception_authorization=route_exception_authorization,
            route_exception_used=route_exception_used,
        )
        used_bytes += green.bytes
        pages += green.pages
        total = fetch_country_groups(
            client=client,
            base_url=base_url,
            topic_ids=(),
            year=year,
            api_key=api_key,
            destination=output_root / "total" / str(year),
            response_budget_bytes=response_budget_bytes - used_bytes,
            route_exception_authorization=route_exception_authorization,
            route_exception_used=route_exception_used,
        )
        used_bytes += total.bytes
        pages += total.pages
        if used_bytes > response_budget_bytes:
            raise RuntimeError("OpenAlex response budget exceeded")
        missing_total = sorted(set(green.country_counts) - set(total.country_counts))
        if missing_total:
            raise RuntimeError(
                f"green countries absent from total counts: {missing_total}"
            )
        for country_code, total_works in sorted(total.country_counts.items()):
            green_works = green.country_counts.get(country_code, 0)
            if green_works > total_works:
                raise RuntimeError(
                    f"green works exceed total works for {country_code}-{year}"
                )
            rows.append(
                {
                    "country_code": country_code,
                    "year": year,
                    "green_works": green_works,
                    "total_works": total_works,
                    "green_total_matching_works": green.total_matching_works,
                    "total_all_matching_works": total.total_matching_works,
                    "counting_method": "full_country_participation",
                    "include_xpac": "false",
                }
            )
    keys = [(str(row["country_code"]), int(row["year"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate OpenAlex country-year keys")
    panel_path = output_root / (
        f"openalex_country_year_counts_{start_year}_{end_year}.csv"
    )
    _write_panel_csv(rows, panel_path)
    panel_bytes = panel_path.read_bytes()
    _write_atomic(
        panel_path.with_name(f"{panel_path.name}.manifest.json"),
        (
            json.dumps(
                {
                    "bytes": len(panel_bytes),
                    "source_id": source_id,
                    "counting_method": "full_country_participation",
                    "end_year": end_year,
                    "include_xpac": False,
                    "route_exception_authorization": (
                        route_exception_authorization
                    ),
                    "route_exception_used": route_exception_used,
                    "sha256": _sha256_bytes(panel_bytes),
                    "start_year": start_year,
                    "topic_ids": list(topics),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )
    _write_atomic(
        output_root / "acquisition_summary.json",
        (
            json.dumps(
                {
                    "bytes": used_bytes,
                    "counting_method": "full_country_participation",
                    "country_year_rows": len(rows),
                    "end_year": end_year,
                    "include_xpac": False,
                    "pages": pages,
                    "panel_file": panel_path.name,
                    "schema_version": 1,
                    "source_id": source_id,
                    "start_year": start_year,
                    "topic_ids": list(topics),
                    "route_exception_authorization": (
                        route_exception_authorization
                    ),
                    "route_exception_used": route_exception_used,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return AggregateRangeReport(
        start_year=start_year,
        end_year=end_year,
        country_year_rows=len(rows),
        pages=pages,
        bytes=used_bytes,
        panel_path=panel_path,
    )


def audit_country_year_panel(path: Path) -> CountryYearPanelAudit:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"country_code", "year", "green_works", "total_works"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("OpenAlex country-year panel schema is incomplete")
    keys: list[tuple[str, int]] = []
    countries: set[str] = set()
    years: list[int] = []
    negative = 0
    green_exceeds_total = 0
    for row in rows:
        country = row["country_code"]
        year = int(row["year"])
        green = int(row["green_works"])
        total = int(row["total_works"])
        if COUNTRY_CODE_PATTERN.fullmatch(country) is None:
            raise RuntimeError(f"invalid country code in OpenAlex panel: {country}")
        keys.append((country, year))
        countries.add(country)
        years.append(year)
        negative += int(green < 0 or total < 0)
        green_exceeds_total += int(green > total)
    duplicate_keys = len(keys) - len(set(keys))
    if duplicate_keys:
        raise RuntimeError("duplicate country-year keys in OpenAlex panel")
    if negative:
        raise RuntimeError("negative counts in OpenAlex panel")
    if green_exceeds_total:
        raise RuntimeError("green works exceed total works in OpenAlex panel")
    return CountryYearPanelAudit(
        rows=len(rows),
        economies=len(countries),
        start_year=min(years),
        end_year=max(years),
        duplicate_country_year_keys=duplicate_keys,
        negative_counts=negative,
        green_exceeds_total=green_exceeds_total,
    )
