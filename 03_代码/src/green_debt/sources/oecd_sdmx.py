"""Frozen, bounded OECD TiVA SDMX-CSV initialization supplements."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from math import isfinite
import os
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

import httpx
import yaml

from green_debt.acquire import append_acquisition_log
from green_debt.network import DirectHttpClient
from green_debt.storage import sha256_file


DATAFLOW_ID = "OECD.STI.PIE:DSD_TIVA_MAINLV@DF_MAINLV(1.1)"
BASE_URL = (
    "https://sdmx.oecd.org/sti-public/rest/data/"
    "OECD.STI.PIE,DSD_TIVA_MAINLV@DF_MAINLV,1.1"
)
MAX_RESPONSE_BYTES = 512 * 1024**2
CSV_HEADER = (
    "DATAFLOW",
    "MEASURE",
    "REF_AREA",
    "ACTIVITY",
    "COUNTERPART_AREA",
    "UNIT_MEASURE",
    "FREQ",
    "TIME_PERIOD",
    "OBS_VALUE",
    "UNIT_MULT",
)
AUTHORIZATION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _safe_headers(headers: httpx.Headers) -> dict[str, str]:
    allowed = {
        "content-type",
        "content-length",
        "date",
        "etag",
        "last-modified",
    }
    return {
        key.lower(): value
        for key, value in headers.items()
        if key.lower() in allowed
    }


@dataclass(frozen=True)
class TivaSupplementSpec:
    source_id: str
    measure: str
    start_year: int
    end_year: int
    counterpart: str
    unit_measure: str
    frequency: str
    destination: str
    expected_max_bytes: int

    @property
    def key(self) -> str:
        return (
            f"{self.measure}...{self.counterpart}."
            f"{self.unit_measure}.{self.frequency}"
        )

    @property
    def query(self) -> dict[str, str]:
        return {
            "startPeriod": str(self.start_year),
            "endPeriod": str(self.end_year),
            "dimensionAtObservation": "AllDimensions",
        }


@dataclass(frozen=True)
class TivaSupplementRecord:
    source_id: str
    measure: str
    url: str
    destination: str
    status: str
    rows: int
    start_year: int
    end_year: int
    bytes: int
    sha256: str
    response_headers: dict[str, str]
    proxy_bypass_enforced: bool
    automatic_redirects: bool
    route_exception_authorization: str | None
    route_exception_used: bool
    completed_at_utc: str


@dataclass(frozen=True)
class TivaSupplementAudit:
    source_id: str
    measure: str
    rows: int
    start_year: int
    end_year: int
    duplicate_keys: int


_FROZEN_SPECS = (
    ("oecd_tiva_dfd_fva_history", "DFD_FVA", 1995, 1999),
    ("oecd_tiva_fd_va_full", "FD_VA", 1995, 2022),
)


def _validate_spec(spec: TivaSupplementSpec) -> None:
    signature = (spec.source_id, spec.measure, spec.start_year, spec.end_year)
    if signature not in _FROZEN_SPECS:
        raise ValueError(f"unapproved TiVA supplement: {signature}")
    if (spec.counterpart, spec.unit_measure, spec.frequency) != ("W", "USD", "A"):
        raise ValueError("TiVA supplement must be world/USD/annual")
    if Path(spec.destination).name != spec.destination:
        raise ValueError("TiVA supplement destination must be a file name")
    if not 0 < spec.expected_max_bytes <= MAX_RESPONSE_BYTES:
        raise ValueError("TiVA response cap must lie within 512 MiB")


def load_tiva_supplements(path: Path) -> tuple[TivaSupplementSpec, ...]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("TiVA supplement config schema_version must be 1")
    if raw.get("dataflow") != "OECD.STI.PIE,DSD_TIVA_MAINLV@DF_MAINLV,1.1":
        raise ValueError("unexpected TiVA dataflow")
    if raw.get("base_url") != BASE_URL:
        raise ValueError("unexpected TiVA base URL")
    entries = raw.get("supplements")
    if not isinstance(entries, list):
        raise ValueError("TiVA supplements must be a list")
    specs: list[TivaSupplementSpec] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("TiVA supplement entry must be a mapping")
        spec = TivaSupplementSpec(
            source_id=str(entry.get("source_id") or ""),
            measure=str(entry.get("measure") or ""),
            start_year=int(entry.get("start_year")),
            end_year=int(entry.get("end_year")),
            counterpart=str(entry.get("counterpart") or ""),
            unit_measure=str(entry.get("unit_measure") or ""),
            frequency=str(entry.get("frequency") or ""),
            destination=str(entry.get("destination") or ""),
            expected_max_bytes=int(entry.get("expected_max_bytes")),
        )
        _validate_spec(spec)
        specs.append(spec)
    signatures = tuple(
        (item.source_id, item.measure, item.start_year, item.end_year)
        for item in specs
    )
    if signatures != _FROZEN_SPECS:
        raise ValueError("TiVA supplement set or order is not frozen")
    return tuple(specs)


def _validate_csv(path: Path, spec: TivaSupplementSpec) -> TivaSupplementAudit:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_HEADER:
            raise RuntimeError("TiVA CSV header changed")
        rows = list(reader)
    if not rows:
        raise RuntimeError("TiVA supplement returned no observations")
    keys: list[tuple[str, ...]] = []
    years: list[int] = []
    for row in rows:
        if row["DATAFLOW"] != DATAFLOW_ID:
            raise RuntimeError("TiVA dataflow version changed")
        if row["MEASURE"] != spec.measure:
            raise RuntimeError("TiVA response contains an unexpected measure")
        if row["COUNTERPART_AREA"] != spec.counterpart:
            raise RuntimeError("TiVA response counterpart is not world")
        if row["UNIT_MEASURE"] != spec.unit_measure or row["FREQ"] != spec.frequency:
            raise RuntimeError("TiVA response unit or frequency changed")
        if row["UNIT_MULT"] != "6":
            raise RuntimeError("TiVA response must use UNIT_MULT=6")
        year = int(row["TIME_PERIOD"])
        if not spec.start_year <= year <= spec.end_year:
            raise RuntimeError("TiVA response year is outside the frozen request")
        value = float(row["OBS_VALUE"])
        if not isfinite(value):
            raise RuntimeError("TiVA OBS_VALUE must be finite Float64")
        years.append(year)
        keys.append(
            (
                row["MEASURE"],
                row["REF_AREA"],
                row["ACTIVITY"],
                row["COUNTERPART_AREA"],
                row["UNIT_MEASURE"],
                row["FREQ"],
                row["TIME_PERIOD"],
            )
        )
    duplicates = len(keys) - len(set(keys))
    if duplicates:
        raise RuntimeError("duplicate TiVA supplement keys")
    if min(years) != spec.start_year or max(years) != spec.end_year:
        raise RuntimeError("TiVA supplement does not span the frozen period")
    return TivaSupplementAudit(
        source_id=spec.source_id,
        measure=spec.measure,
        rows=len(rows),
        start_year=min(years),
        end_year=max(years),
        duplicate_keys=duplicates,
    )


def audit_tiva_supplement(
    path: Path,
    spec: TivaSupplementSpec,
) -> TivaSupplementAudit:
    _validate_spec(spec)
    return _validate_csv(path, spec)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _record_from_payload(payload: dict[str, object]) -> TivaSupplementRecord:
    return TivaSupplementRecord(
        source_id=str(payload["source_id"]),
        measure=str(payload["measure"]),
        url=str(payload["url"]),
        destination=str(payload["destination"]),
        status=str(payload["status"]),
        rows=int(payload["rows"]),
        start_year=int(payload["start_year"]),
        end_year=int(payload["end_year"]),
        bytes=int(payload["bytes"]),
        sha256=str(payload["sha256"]),
        response_headers={
            str(key): str(value)
            for key, value in dict(payload["response_headers"]).items()
        },
        proxy_bypass_enforced=bool(payload["proxy_bypass_enforced"]),
        automatic_redirects=bool(payload["automatic_redirects"]),
        route_exception_authorization=(
            str(payload["route_exception_authorization"])
            if payload.get("route_exception_authorization") is not None
            else None
        ),
        route_exception_used=bool(payload["route_exception_used"]),
        completed_at_utc=str(payload["completed_at_utc"]),
    )


def acquire_tiva_supplement(
    spec: TivaSupplementSpec,
    client: DirectHttpClient,
    output_root: Path,
    route_exception_authorization: str | None,
    *,
    route_exception_used: bool = False,
    audit_log_path: Path | None = None,
) -> TivaSupplementRecord:
    """Download and validate one of exactly two approved TiVA CSV responses."""

    _validate_spec(spec)
    authorization = (
        route_exception_authorization.strip()
        if route_exception_authorization is not None
        else None
    )
    if authorization == "" or (
        authorization is not None
        and AUTHORIZATION_PATTERN.fullmatch(authorization) is None
    ):
        raise ValueError("route exception authorization is invalid")
    if route_exception_used and authorization is None:
        raise ValueError("a used route exception requires authorization")
    root = output_root.resolve()
    destination = root / spec.destination
    manifest_path = destination.with_name(f"{destination.name}.manifest.json")
    headers_path = destination.with_name(f"{destination.name}.headers.json")
    if destination.exists():
        if not manifest_path.is_file():
            raise RuntimeError("existing TiVA supplement lacks a manifest")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if destination.stat().st_size != payload.get("bytes"):
            raise RuntimeError("existing TiVA supplement byte count changed")
        if sha256_file(destination) != payload.get("sha256"):
            raise RuntimeError("existing TiVA supplement hash changed")
        _validate_csv(destination, spec)
        record = _record_from_payload(payload)
        if audit_log_path is not None:
            append_acquisition_log(audit_log_path, record)
        return record

    request_url = str(httpx.URL(f"{BASE_URL}/{spec.key}", params=spec.query))
    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with client.stream(
            "GET",
            request_url,
            headers={"Accept": "text/csv;version=2.0"},
        ) as response:
            if response.is_redirect:
                raise RuntimeError("OECD SDMX redirect refused")
            if response.status_code != 200:
                raise RuntimeError(f"OECD SDMX HTTP status {response.status_code}")
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > spec.expected_max_bytes:
                raise RuntimeError("OECD SDMX content length exceeds frozen cap")
            written = 0
            with partial.open("xb") as handle:
                for chunk in response.iter_bytes():
                    written += len(chunk)
                    if written > spec.expected_max_bytes or written > MAX_RESPONSE_BYTES:
                        raise RuntimeError("OECD SDMX response exceeds frozen cap")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            response_headers = _safe_headers(response.headers)
        audit = _validate_csv(partial, spec)
        digest = sha256_file(partial)
        record = TivaSupplementRecord(
            source_id=spec.source_id,
            measure=spec.measure,
            url=_safe_url(request_url),
            destination=str(destination.relative_to(root)),
            status="downloaded",
            rows=audit.rows,
            start_year=audit.start_year,
            end_year=audit.end_year,
            bytes=written,
            sha256=digest,
            response_headers=response_headers,
            proxy_bypass_enforced=(
                client.trust_env is False and client.explicit_proxy is None
            ),
            automatic_redirects=client.follow_redirects,
            route_exception_authorization=authorization,
            route_exception_used=route_exception_used,
            completed_at_utc=_utc_now(),
        )
        os.replace(partial, destination)
        _write_atomic(
            headers_path,
            (json.dumps(response_headers, sort_keys=True, indent=2) + "\n").encode(
                "utf-8"
            ),
        )
        _write_atomic(
            manifest_path,
            (
                json.dumps(
                    asdict(record), ensure_ascii=False, sort_keys=True, indent=2
                )
                + "\n"
            ).encode("utf-8"),
        )
        if audit_log_path is not None:
            append_acquisition_log(audit_log_path, record)
        return record
    except Exception:
        partial.unlink(missing_ok=True)
        if destination.exists() and not manifest_path.exists():
            destination.unlink()
        headers_path.unlink(missing_ok=True)
        raise
