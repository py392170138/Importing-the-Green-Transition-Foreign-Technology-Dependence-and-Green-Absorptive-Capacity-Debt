"""Strict World Development Indicators normalization with frozen units."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any

import polars as pl

from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    verify_manifest,
    write_authoritative_table,
)


REJECTED_INDICATORS = frozenset({"EN.ATM.CO2E.KT"})
OUTPUT_SCHEMA = {
    "economy_id": pl.String,
    "year": pl.Int16,
    "indicator_id": pl.String,
    "value": pl.Float64,
    "unit": pl.String,
    "source_update": pl.String,
    "source_status": pl.String,
}


@dataclass(frozen=True)
class WdiBuildReport:
    source_files: int
    indicators: tuple[str, ...]
    raw_rows: int
    retained_rows: int
    excluded_rows: int
    null_values: int
    zero_values: int
    duplicate_keys: int
    rejected_response_bodies_loaded: int
    output_path: str
    output_bytes: int


@dataclass(frozen=True)
class WdiAuditReport:
    indicators: tuple[str, ...]
    rows: int
    null_values: int
    zero_values: int
    duplicate_keys: int
    rejected_indicators: int
    unit_mismatches: int
    years: tuple[int, ...]
    status: str


def _source_code(observation: dict[str, Any]) -> str:
    iso3 = str(observation.get("countryiso3code") or "").strip()
    if iso3:
        return iso3
    country = observation.get("country")
    if not isinstance(country, dict):
        raise ValueError("WDI observation lacks a country object")
    wb2 = str(country.get("id") or "").strip()
    if not wb2:
        raise ValueError("WDI observation lacks both ISO3 and World Bank country code")
    return f"WB2:{wb2}"


def _validate_page_envelope(
    payload: Any, *, require_source_update: bool = True
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if (
        not isinstance(payload, list)
        or len(payload) != 2
        or not isinstance(payload[0], dict)
        or not isinstance(payload[1], list)
        or not all(isinstance(row, dict) for row in payload[1])
    ):
        raise ValueError("invalid WDI page envelope")
    envelope = payload[0]
    rows = payload[1]
    try:
        page = int(envelope["page"])
        pages = int(envelope["pages"])
        total = int(envelope["total"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid WDI page envelope") from exc
    if page != 1 or pages != 1 or total != len(rows):
        raise ValueError(
            "invalid WDI page envelope: expected one complete page with matching total"
        )
    if str(envelope.get("sourceid") or "2") != "2":
        raise ValueError("WDI page envelope sourceid changed")
    if require_source_update and not str(envelope.get("lastupdated") or "").strip():
        raise ValueError("WDI page envelope lacks lastupdated")
    return envelope, rows


def _mapping_dict(economies: pl.DataFrame | None) -> dict[str, str | None] | None:
    if economies is None:
        return None
    if not {"source_code", "economy_id"} <= set(economies.columns):
        raise ValueError("WDI economy mapping requires source_code and economy_id")
    mapping: dict[str, str | None] = {}
    for row in economies.select("source_code", "economy_id").iter_rows(named=True):
        source_code = str(row["source_code"] or "").strip()
        if not source_code or source_code in mapping:
            raise ValueError(f"duplicate or empty WDI source code: {source_code!r}")
        value = row["economy_id"]
        mapping[source_code] = str(value).strip() if value is not None else None
    return mapping


def normalize_wdi_payload(
    payload: Any,
    *,
    unit: str,
    expected_indicator: str,
    expected_period: tuple[int, int],
    economies: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Validate one WDI response and retain API nulls as typed nulls."""

    if expected_indicator in REJECTED_INDICATORS:
        raise ValueError(f"rejected WDI indicator cannot be normalized: {expected_indicator}")
    if not unit:
        raise ValueError("WDI canonical unit must be non-empty")
    if expected_period[0] > expected_period[1]:
        raise ValueError("WDI expected period must be ascending")
    envelope, observations = _validate_page_envelope(payload)
    mapping = _mapping_dict(economies)
    output: list[dict[str, Any]] = []
    observed_years: set[int] = set()
    for row_number, observation in enumerate(observations, start=1):
        indicator = observation.get("indicator")
        indicator_id = (
            str(indicator.get("id") or "").strip()
            if isinstance(indicator, dict)
            else ""
        )
        if indicator_id != expected_indicator:
            raise ValueError(
                f"WDI indicator code changed at row {row_number}: {indicator_id!r}"
            )
        try:
            year = int(observation["date"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid WDI year at row {row_number}") from exc
        if not expected_period[0] <= year <= expected_period[1]:
            raise ValueError(f"WDI year outside expected period: {year}")
        observed_years.add(year)
        value = observation.get("value")
        if isinstance(value, bool) or (
            value is not None
            and (not isinstance(value, (int, float)) or not isfinite(float(value)))
        ):
            raise ValueError(f"invalid WDI numeric value at row {row_number}")
        source_code = _source_code(observation)
        if mapping is None:
            economy_id = source_code
        else:
            if source_code not in mapping:
                raise ValueError(f"unmapped WDI source code: {source_code}")
            economy_id = mapping[source_code]
            if economy_id is None:
                continue
        raw_status = str(observation.get("obs_status") or "").strip()
        source_status = (
            f"raw:{raw_status}"
            if raw_status
            else ("api_null" if value is None else "reported")
        )
        output.append(
            {
                "economy_id": economy_id,
                "year": year,
                "indicator_id": indicator_id,
                "value": float(value) if value is not None else None,
                "unit": unit,
                "source_update": str(envelope["lastupdated"]),
                "source_status": source_status,
            }
        )
    expected_years = set(range(expected_period[0], expected_period[1] + 1))
    if observed_years != expected_years:
        raise ValueError(
            f"WDI source period differs: expected {sorted(expected_years)}, "
            f"got {sorted(observed_years)}"
        )
    frame = pl.DataFrame(output, schema=OUTPUT_SCHEMA).sort(
        "economy_id", "year", "indicator_id"
    )
    duplicates = frame.group_by("economy_id", "year", "indicator_id").len().filter(
        pl.col("len") > 1
    ).height
    if duplicates:
        raise ValueError(f"duplicate WDI economy-year-indicator keys: {duplicates}")
    return frame


def validate_wdi_metadata(payload: Any, *, expected_indicator: str) -> None:
    """Freeze the WDI metadata envelope and its currently blank unit field."""

    _, rows = _validate_page_envelope(payload, require_source_update=False)
    if len(rows) != 1:
        raise ValueError("WDI metadata must contain exactly one indicator")
    row = rows[0]
    if str(row.get("id") or "") != expected_indicator:
        raise ValueError("WDI metadata indicator code changed")
    if row.get("unit") != "":
        raise ValueError("WDI metadata unit field changed from the frozen blank value")
    if not str(row.get("name") or "").strip():
        raise ValueError("WDI metadata indicator name is empty")
    source = row.get("source")
    if not isinstance(source, dict) or str(source.get("id") or "") != "2":
        raise ValueError("WDI metadata source changed")


def _approved_registry(registry: pl.DataFrame) -> pl.DataFrame:
    required = {
        "source_id",
        "source_field",
        "unit",
        "start_year",
        "end_year",
        "status",
    }
    if not required <= set(registry.columns):
        raise ValueError(f"indicator registry lacks columns: {sorted(required)}")
    approved = registry.filter(
        (pl.col("source_id") == "wdi")
        & ~pl.col("status").str.starts_with("rejected")
    )
    if approved.is_empty():
        raise ValueError("indicator registry has no approved WDI rows")
    if approved.group_by("source_field").len().filter(pl.col("len") > 1).height:
        raise ValueError("indicator registry has duplicate WDI source fields")
    return approved.sort("source_field")


def normalize_wdi(
    paths: tuple[Path, ...],
    registry: pl.DataFrame,
    economies: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Normalize the complete approved WDI file set into one typed long table."""

    approved = _approved_registry(registry)
    specifications = {
        row["source_field"]: row for row in approved.iter_rows(named=True)
    }
    seen: set[str] = set()
    frames: list[pl.DataFrame] = []
    for path in sorted(paths):
        if path.name.endswith(".error.json"):
            raise ValueError(f"rejected WDI response body was supplied: {path.name}")
        matches = [code for code in specifications if path.name.startswith(f"{code}.")]
        if len(matches) != 1:
            raise ValueError(f"WDI data file is not uniquely registered: {path.name}")
        indicator = matches[0]
        if indicator in seen:
            raise ValueError(f"duplicate WDI data file for {indicator}")
        seen.add(indicator)
        spec = specifications[indicator]
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames.append(
            normalize_wdi_payload(
                payload,
                unit=str(spec["unit"]),
                expected_indicator=indicator,
                expected_period=(int(spec["start_year"]), int(spec["end_year"])),
                economies=economies,
            )
        )
    missing = sorted(set(specifications) - seen)
    if missing:
        raise ValueError(f"approved WDI data files are missing: {missing}")
    frame = pl.concat(frames).sort("economy_id", "year", "indicator_id")
    duplicates = frame.group_by("economy_id", "year", "indicator_id").len().filter(
        pl.col("len") > 1
    ).height
    if duplicates:
        raise ValueError(f"duplicate WDI economy-year-indicator keys: {duplicates}")
    return frame


def _load_contract(path: Path) -> TableContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    period = tuple(int(value) for value in payload["period"])
    return TableContract(
        table_id=str(payload["table_id"]),
        schema_version=str(payload["schema_version"]),
        primary_key=tuple(payload["primary_key"]),
        columns={str(key): str(value) for key, value in payload["columns"].items()},
        units={str(key): str(value) for key, value in payload["units"].items()},
        period=(period[0], period[1]),
        zero_semantics={
            str(key): str(value)
            for key, value in payload.get("zero_semantics", {}).items()
        },
        transformations=tuple(payload.get("transformations", [])),
    )


def build_wdi_table(
    *,
    data_paths: tuple[Path, ...],
    metadata_paths: tuple[Path, ...],
    registry: pl.DataFrame,
    economies: pl.DataFrame,
    destination: Path,
    contract_path: Path,
    inputs: tuple[InputArtifact, ...],
    build: BuildIdentity,
) -> WdiBuildReport:
    approved = _approved_registry(registry)
    specifications = {
        row["source_field"]: row for row in approved.iter_rows(named=True)
    }
    metadata_seen: set[str] = set()
    for path in metadata_paths:
        matches = [code for code in specifications if path.name.startswith(f"{code}.")]
        if len(matches) != 1:
            raise ValueError(f"WDI metadata file is not uniquely registered: {path.name}")
        indicator = matches[0]
        validate_wdi_metadata(
            json.loads(path.read_text(encoding="utf-8")),
            expected_indicator=indicator,
        )
        metadata_seen.add(indicator)
    if metadata_seen != set(specifications):
        raise ValueError("approved WDI metadata file set is incomplete")

    raw_rows = 0
    for path in data_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _, rows = _validate_page_envelope(payload)
        raw_rows += len(rows)
    frame = normalize_wdi(data_paths, registry, economies)
    manifest = write_authoritative_table(
        frame,
        _load_contract(contract_path),
        destination,
        inputs,
        build,
    )
    duplicate_keys = frame.group_by("economy_id", "year", "indicator_id").len().filter(
        pl.col("len") > 1
    ).height
    return WdiBuildReport(
        source_files=len(data_paths),
        indicators=tuple(sorted(specifications)),
        raw_rows=raw_rows,
        retained_rows=frame.height,
        excluded_rows=raw_rows - frame.height,
        null_values=frame.get_column("value").null_count(),
        zero_values=frame.filter(pl.col("value") == 0.0).height,
        duplicate_keys=duplicate_keys,
        rejected_response_bodies_loaded=0,
        output_path=str(destination.resolve()),
        output_bytes=manifest.bytes,
    )


def audit_wdi_table(
    *,
    manifest_path: Path,
    registry: pl.DataFrame,
) -> WdiAuditReport:
    manifest = verify_manifest(manifest_path)
    frame = pl.read_parquet(manifest.destination)
    approved = _approved_registry(registry)
    expected_units = {
        row["source_field"]: row["unit"] for row in approved.iter_rows(named=True)
    }
    unit_mismatches = frame.filter(
        pl.struct("indicator_id", "unit").map_elements(
            lambda row: expected_units.get(row["indicator_id"]) != row["unit"],
            return_dtype=pl.Boolean,
        )
    ).height
    duplicate_keys = frame.group_by("economy_id", "year", "indicator_id").len().filter(
        pl.col("len") > 1
    ).height
    rejected = frame.filter(pl.col("indicator_id").is_in(REJECTED_INDICATORS)).height
    years = tuple(sorted(frame.get_column("year").unique().to_list()))
    if tuple(sorted(expected_units)) != tuple(
        sorted(frame.get_column("indicator_id").unique().to_list())
    ):
        raise RuntimeError("WDI authoritative indicator set differs from registry")
    if years != tuple(range(1996, 2025)):
        raise RuntimeError(f"WDI authoritative period differs: {years}")
    if duplicate_keys or rejected or unit_mismatches:
        raise RuntimeError(
            "WDI audit failed: "
            f"duplicates={duplicate_keys}, rejected={rejected}, units={unit_mismatches}"
        )
    return WdiAuditReport(
        indicators=tuple(sorted(expected_units)),
        rows=frame.height,
        null_values=frame.get_column("value").null_count(),
        zero_values=frame.filter(pl.col("value") == 0.0).height,
        duplicate_keys=duplicate_keys,
        rejected_indicators=rejected,
        unit_mismatches=unit_mismatches,
        years=years,
        status="valid",
    )
