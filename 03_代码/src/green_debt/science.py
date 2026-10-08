"""Green-science coverage checks and absorptive-capacity construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import polars as pl

from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    verify_manifest,
    write_authoritative_table,
)


REQUIRED_WORK_COLUMNS = {"economy_id", "year", "green_works", "total_works"}
_POPULATION_INDICATOR = "SP.POP.TOTL"
_SCIENCE_CONTRACT_ROOT = Path(__file__).resolve().parents[2] / "contracts"
OPENALEX_REQUIRED_COLUMNS = {
    "country_code",
    "year",
    "green_works",
    "total_works",
    "counting_method",
    "include_xpac",
}
OPENALEX_OPTIONAL_COUNT_COLUMNS = (
    "green_total_matching_works",
    "total_all_matching_works",
)
OPENALEX_NORMALIZED_SCHEMA = {
    "country_code": pl.String,
    "year": pl.Int16,
    "green_works": pl.UInt64,
    "total_works": pl.UInt64,
    "green_total_matching_works": pl.UInt64,
    "total_all_matching_works": pl.UInt64,
    "counting_method": pl.String,
    "include_xpac": pl.Boolean,
}
OPENALEX_OUTPUT_SCHEMA = {
    "economy_id": pl.String,
    "year": pl.Int16,
    "green_works": pl.UInt64,
    "total_works": pl.UInt64,
    "green_total_matching_works": pl.UInt64,
    "total_all_matching_works": pl.UInt64,
    "counting_method": pl.String,
    "include_xpac": pl.Boolean,
    "source_status": pl.String,
}


@dataclass(frozen=True)
class OpenAlexBuildReport:
    source_files: int
    raw_rows: int
    retained_rows: int
    excluded_rows: int
    economies: int
    zero_green_counts: int
    duplicate_keys: int
    years: tuple[int, ...]
    output_path: str
    output_bytes: int


@dataclass(frozen=True)
class OpenAlexAuditReport:
    rows: int
    economies: int
    zero_green_counts: int
    duplicate_keys: int
    invalid_count_rows: int
    years: tuple[int, ...]
    status: str


@dataclass(frozen=True)
class GsciBuildReport:
    rows: int
    first_computable_year: int | None
    partial_window_rows: int
    nonfinite_values: int
    output_path: str
    audit_path: str


def _require_columns(frame: pl.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")


def _canonical_economy_key(frame: pl.DataFrame) -> pl.DataFrame:
    """Accept the retired test-only economy key while emitting economy_id."""

    if "economy_id" in frame.columns:
        return frame
    if "economy" in frame.columns:
        return frame.rename({"economy": "economy_id"})
    return frame


def _load_science_contract(name: str) -> TableContract:
    payload = json.loads((_SCIENCE_CONTRACT_ROOT / name).read_text(encoding="utf-8"))
    period = tuple(int(value) for value in payload["period"])
    return TableContract(
        table_id=str(payload["table_id"]),
        schema_version=str(payload["schema_version"]),
        primary_key=tuple(str(value) for value in payload["primary_key"]),
        columns={str(key): str(value) for key, value in payload["columns"].items()},
        units={str(key): str(value) for key, value in payload["units"].items()},
        period=(period[0], period[1]),
        zero_semantics={str(key): str(value) for key, value in payload.get("zero_semantics", {}).items()},
        transformations=tuple(str(value) for value in payload.get("transformations", [])),
    )


def _nonnegative_integer(value: object, *, field: str, row_number: int) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"OpenAlex {field} is invalid at row {row_number}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"OpenAlex {field} is invalid at row {row_number}"
        ) from exc
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"OpenAlex {field} is invalid at row {row_number}")
    return int(numeric)


def _false_value(value: object, *, row_number: int) -> bool:
    if value is False or value == 0:
        return False
    if isinstance(value, str) and value.strip().lower() in {"false", "0"}:
        return False
    raise ValueError(f"OpenAlex include_xpac must be false at row {row_number}")


def normalize_openalex(
    frames: tuple[pl.DataFrame, ...],
    *,
    expected_period: tuple[int, int] = (1992, 2024),
) -> pl.DataFrame:
    """Union bounded country-year counts under one unchanged counting rule."""

    if not frames:
        raise ValueError("OpenAlex normalization requires at least one frame")
    if expected_period[0] > expected_period[1]:
        raise ValueError("OpenAlex expected period must be ascending")
    rows: list[dict[str, Any]] = []
    for frame_number, frame in enumerate(frames, start=1):
        _require_columns(frame, OPENALEX_REQUIRED_COLUMNS)
        for row_number, row in enumerate(frame.iter_rows(named=True), start=1):
            country_code = str(row.get("country_code") or "").strip().upper()
            if not country_code:
                raise ValueError(
                    f"OpenAlex country_code is empty in frame {frame_number}, "
                    f"row {row_number}"
                )
            try:
                year = int(row["year"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"OpenAlex year is invalid at row {row_number}"
                ) from exc
            if not expected_period[0] <= year <= expected_period[1]:
                raise ValueError(f"OpenAlex year outside expected period: {year}")
            counting_method = str(row.get("counting_method") or "").strip()
            if counting_method != "full_country_participation":
                raise ValueError(
                    "OpenAlex counting method must be full_country_participation"
                )
            green = _nonnegative_integer(
                row.get("green_works"), field="green_works", row_number=row_number
            )
            total = _nonnegative_integer(
                row.get("total_works"), field="total_works", row_number=row_number
            )
            if green > total:
                raise ValueError(
                    f"OpenAlex green_works exceeds total_works at row {row_number}"
                )
            normalized: dict[str, Any] = {
                "country_code": country_code,
                "year": year,
                "green_works": green,
                "total_works": total,
                "counting_method": counting_method,
                "include_xpac": _false_value(
                    row.get("include_xpac"), row_number=row_number
                ),
            }
            for field in OPENALEX_OPTIONAL_COUNT_COLUMNS:
                value = row.get(field)
                normalized[field] = (
                    None
                    if value is None
                    else _nonnegative_integer(
                        value, field=field, row_number=row_number
                    )
                )
            global_green = normalized["green_total_matching_works"]
            global_total = normalized["total_all_matching_works"]
            if (global_green is None) != (global_total is None):
                raise ValueError("OpenAlex global count columns must be jointly present")
            if (
                global_green is not None
                and global_total is not None
                and global_green > global_total
            ):
                raise ValueError("OpenAlex global green count exceeds global total")
            rows.append(normalized)
    output = pl.DataFrame(rows, schema=OPENALEX_NORMALIZED_SCHEMA).sort(
        "country_code", "year"
    )
    duplicates = output.group_by("country_code", "year").len().filter(
        pl.col("len") > 1
    )
    if duplicates.height:
        raise ValueError(
            f"duplicate OpenAlex country-year keys: {duplicates.height}"
        )
    expected_years = tuple(range(expected_period[0], expected_period[1] + 1))
    observed_years = tuple(sorted(output.get_column("year").unique().to_list()))
    if observed_years != expected_years:
        raise ValueError(
            f"OpenAlex source period differs: expected {expected_years}, "
            f"got {observed_years}"
        )
    global_counts = output.filter(
        pl.col("green_total_matching_works").is_not_null()
    ).group_by("year").agg(
        pl.col("green_total_matching_works").n_unique().alias("green_versions"),
        pl.col("total_all_matching_works").n_unique().alias("total_versions"),
    )
    if global_counts.filter(
        (pl.col("green_versions") != 1) | (pl.col("total_versions") != 1)
    ).height:
        raise ValueError("OpenAlex global counts differ within a year")
    return output


def _map_openalex_economies(
    frame: pl.DataFrame, economies: pl.DataFrame
) -> pl.DataFrame:
    _require_columns(economies, {"source_code", "economy_id"})
    mapping = economies.select(
        pl.col("source_code").cast(pl.String).str.strip_chars().alias("country_code"),
        pl.col("economy_id").cast(pl.String).str.strip_chars(),
    ).with_columns(pl.lit(True).alias("_declared"))
    if mapping.filter(
        pl.col("country_code").is_null() | (pl.col("country_code") == "")
    ).height:
        raise ValueError("OpenAlex economy mapping has an empty source code")
    if mapping.group_by("country_code").len().filter(pl.col("len") > 1).height:
        raise ValueError("OpenAlex economy mapping has duplicate source codes")
    joined = frame.join(mapping, on="country_code", how="left")
    missing = joined.filter(pl.col("_declared").is_null())
    if missing.height:
        codes = missing.get_column("country_code").unique().sort().to_list()
        raise ValueError(f"unmapped OpenAlex source codes: {codes}")
    return (
        joined.filter(pl.col("economy_id").is_not_null())
        .with_columns(pl.lit("reported", dtype=pl.String).alias("source_status"))
        .select(*OPENALEX_OUTPUT_SCHEMA)
        .sort("economy_id", "year")
    )


def _load_openalex_contract(path: Path) -> TableContract:
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


def build_openalex_table(
    *,
    data_paths: tuple[Path, ...],
    economies: pl.DataFrame,
    destination: Path,
    contract_path: Path,
    inputs: tuple[InputArtifact, ...],
    build: BuildIdentity,
) -> OpenAlexBuildReport:
    frames = tuple(
        pl.read_csv(
            path,
            schema_overrides={
                "country_code": pl.String,
                "year": pl.Int16,
                "green_works": pl.UInt64,
                "total_works": pl.UInt64,
                "green_total_matching_works": pl.UInt64,
                "total_all_matching_works": pl.UInt64,
            },
        )
        for path in data_paths
    )
    normalized = normalize_openalex(frames, expected_period=(1992, 2024))
    if any(
        normalized.get_column(name).null_count()
        for name in OPENALEX_OPTIONAL_COUNT_COLUMNS
    ):
        raise ValueError("production OpenAlex files lack global coverage counts")
    output = _map_openalex_economies(normalized, economies)
    manifest = write_authoritative_table(
        output,
        _load_openalex_contract(contract_path),
        destination,
        inputs,
        build,
    )
    return OpenAlexBuildReport(
        source_files=len(data_paths),
        raw_rows=normalized.height,
        retained_rows=output.height,
        excluded_rows=normalized.height - output.height,
        economies=output.get_column("economy_id").n_unique(),
        zero_green_counts=output.filter(pl.col("green_works") == 0).height,
        duplicate_keys=0,
        years=tuple(sorted(output.get_column("year").unique().to_list())),
        output_path=str(destination.resolve()),
        output_bytes=manifest.bytes,
    )


def audit_openalex_table(*, manifest_path: Path) -> OpenAlexAuditReport:
    manifest = verify_manifest(manifest_path)
    frame = pl.read_parquet(manifest.destination)
    years = tuple(sorted(frame.get_column("year").unique().to_list()))
    duplicates = frame.group_by("economy_id", "year").len().filter(
        pl.col("len") > 1
    ).height
    invalid = frame.filter(
        (pl.col("green_works") > pl.col("total_works"))
        | (pl.col("green_total_matching_works") > pl.col("total_all_matching_works"))
        | (pl.col("counting_method") != "full_country_participation")
        | pl.col("include_xpac")
    ).height
    if years != tuple(range(1992, 2025)) or duplicates or invalid:
        raise RuntimeError(
            "OpenAlex audit failed: "
            f"years={years}, duplicates={duplicates}, invalid={invalid}"
        )
    return OpenAlexAuditReport(
        rows=frame.height,
        economies=frame.get_column("economy_id").n_unique(),
        zero_green_counts=frame.filter(pl.col("green_works") == 0).height,
        duplicate_keys=duplicates,
        invalid_count_rows=invalid,
        years=years,
        status="valid",
    )


def classify_research_coverage(
    works: pl.DataFrame,
    *,
    coverage_floor_ratio: float = 0.1,
) -> pl.DataFrame:
    """Keep a zero only when total research coverage is within its normal band."""

    works = _canonical_economy_key(works)
    _require_columns(works, REQUIRED_WORK_COLUMNS)
    if not 0 < coverage_floor_ratio <= 1:
        raise ValueError("coverage_floor_ratio must be in (0, 1]")
    rows = works.sort(["economy_id", "year"]).to_dicts()
    positive_by_economy: dict[str, list[float]] = {}
    for row in rows:
        total = row["total_works"]
        if total is not None and float(total) > 0:
            positive_by_economy.setdefault(str(row["economy_id"]), []).append(
                float(total)
            )
    baselines = {
        economy: median(values) for economy, values in positive_by_economy.items()
    }
    normalized: list[dict[str, object]] = []
    for row in rows:
        current = dict(row)
        economy = str(current["economy_id"])
        total_raw = current["total_works"]
        baseline = baselines.get(economy)
        abnormal = (
            total_raw is None
            or not math.isfinite(float(total_raw))
            or float(total_raw) <= 0
            or baseline is None
            or float(total_raw) < baseline * coverage_floor_ratio
        )
        if abnormal:
            current["green_works"] = None
            current["green_works_missing_reason"] = (
                "total_research_coverage_abnormal"
            )
        elif current["green_works"] is None:
            current["green_works_missing_reason"] = "green_works_unavailable"
        elif not math.isfinite(float(current["green_works"])) or float(current["green_works"]) < 0:
            current["green_works"] = None
            current["green_works_missing_reason"] = "nonfinite_or_negative_green_works"
        else:
            current["green_works"] = float(current["green_works"])
            current["green_works_missing_reason"] = None
        normalized.append(current)
    return pl.DataFrame(normalized).sort(["economy_id", "year"])


def compute_gsci(
    green_works: pl.DataFrame,
    population: pl.DataFrame,
    *,
    decay: float = 0.8,
    window: int = 5,
) -> pl.DataFrame:
    """Compute a log, population-scaled, finite decayed green-science stock."""

    green_works = _canonical_economy_key(green_works)
    population = _canonical_economy_key(population)
    _require_columns(green_works, REQUIRED_WORK_COLUMNS)
    _require_columns(population, {"economy_id", "year", "population"})
    if not 0 < decay <= 1:
        raise ValueError("decay must be in (0, 1]")
    if window <= 0:
        raise ValueError("window must be positive")
    covered = classify_research_coverage(green_works)
    joined = covered.join(
        population.select("economy_id", "year", "population"),
        on=["economy_id", "year"],
        how="left",
        validate="1:1",
    ).sort(["economy_id", "year"])
    rows = joined.to_dicts()
    output: list[dict[str, object]] = []
    by_economy: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_economy.setdefault(str(row["economy_id"]), []).append(row)
    for economy_rows in by_economy.values():
        observed = {int(row["year"]): row for row in economy_rows}
        for row in economy_rows:
            current = dict(row)
            year = int(current["year"])
            pop = current["population"]
            if pop is None:
                current["gsci_raw"] = None
                current["gsci_raw_reason"] = "missing_population"
            elif not math.isfinite(float(pop)):
                current["population"] = None
                current["gsci_raw"] = None
                current["gsci_raw_reason"] = "nonfinite_population"
            elif float(pop) <= 0.0:
                current["gsci_raw"] = None
                current["gsci_raw_reason"] = "nonpositive_population"
            else:
                window_rows = [observed.get(year - lag) for lag in range(window)]
                if any(item is None or item["green_works"] is None for item in window_rows):
                    current["gsci_raw"] = None
                    current["gsci_raw_reason"] = "incomplete_green_works_window"
                else:
                    stock = sum(
                        decay**lag * float(window_rows[lag]["green_works"])
                        for lag in range(window)
                    ) / (float(pop) / 1_000_000.0)
                    if not math.isfinite(stock) or stock < 0.0:
                        current["gsci_raw"] = None
                        current["gsci_raw_reason"] = "nonfinite_gsci_stock"
                    else:
                        value = math.log1p(stock)
                        current["gsci_raw"] = value if math.isfinite(value) else None
                        current["gsci_raw_reason"] = (
                            None if math.isfinite(value) else "nonfinite_gsci_stock"
                        )
            output.append(current)
    return pl.DataFrame(output).sort(["economy_id", "year"])


def audit_gsci_windows(constructed: pl.DataFrame, *, window: int = 5) -> dict[str, int]:
    """Audit complete-window evidence before restricting the authoritative period."""

    _require_columns(constructed, {"economy_id", "year", "green_works", "gsci_raw"})
    if window <= 0:
        raise ValueError("window must be positive")
    invalid_nonnull = 0
    pre_1996_nonnull = 0
    for economy_rows in constructed.sort(["economy_id", "year"]).partition_by("economy_id"):
        observed = {int(row["year"]): row for row in economy_rows.iter_rows(named=True)}
        for row in economy_rows.iter_rows(named=True):
            if row["gsci_raw"] is None:
                continue
            year = int(row["year"])
            if year < 1996:
                pre_1996_nonnull += 1
            if any(
                observed.get(year - lag) is None
                or observed[year - lag]["green_works"] is None
                for lag in range(window)
            ):
                invalid_nonnull += 1
    return {
        "pre_1996_nonnull_rows": pre_1996_nonnull,
        "invalid_nonnull_windows": invalid_nonnull,
    }


def build_gsci(paths: Any, *, build: BuildIdentity) -> GsciBuildReport:
    """Construct 1996-2024 GSCI using the 1992-1995 OpenAlex supplement."""

    openalex_path = paths.normalized / "openalex/openalex_country_year.parquet"
    wdi_path = paths.normalized / "wdi/wdi_country_year.parquet"
    works = pl.read_parquet(openalex_path).select(
        "economy_id", "year", "green_works", "total_works"
    )
    population = (
        pl.read_parquet(wdi_path)
        .filter(pl.col("indicator_id") == _POPULATION_INDICATOR)
        .select("economy_id", "year", pl.col("value").alias("population"))
    )
    all_constructed = compute_gsci(works, population)
    window_audit = audit_gsci_windows(all_constructed)
    if window_audit["invalid_nonnull_windows"]:
        raise RuntimeError("GSCI audit found nonnull values without complete windows")
    constructed = all_constructed.filter(
        pl.col("year").is_between(1996, 2024, closed="both")
    ).select("economy_id", "year", "gsci_raw", "gsci_raw_reason").with_columns(
        pl.col("economy_id").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("gsci_raw").cast(pl.Float64),
        pl.col("gsci_raw_reason").cast(pl.String),
    ).sort(["year", "economy_id"])
    nonfinite = int(constructed.select(
        (pl.col("gsci_raw").is_not_null() & ~pl.col("gsci_raw").is_finite()).sum()
    ).item())
    if nonfinite:
        raise ValueError("nonfinite GSCI before authoritative write")
    destination = paths.measures / "science/gsci_raw.parquet"
    write_authoritative_table(
        constructed,
        _load_science_contract("gsci_raw.json"),
        destination,
        (InputArtifact.from_path(openalex_path), InputArtifact.from_path(wdi_path)),
        build,
    )
    computable = constructed.filter(pl.col("gsci_raw").is_not_null())
    first_year = int(computable.get_column("year").min()) if computable.height else None
    audit_path = paths.measures / "science/gsci_raw_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "first_computable_year": first_year,
        "partial_window_rows": window_audit["pre_1996_nonnull_rows"],
        **window_audit,
        "nonfinite_values": nonfinite,
        "rows": constructed.height,
        "years": sorted(constructed.get_column("year").unique().to_list()),
        "1996_absorption_input_coverage": int(constructed.filter((pl.col("year") == 1996) & pl.col("gsci_raw").is_not_null()).height),
        "missing_reasons": constructed.filter(pl.col("gsci_raw_reason").is_not_null()).group_by("gsci_raw_reason").len().sort("gsci_raw_reason").to_dicts(),
        "duplicate_keys": int(constructed.group_by(["economy_id", "year"]).len().filter(pl.col("len") > 1).height),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return GsciBuildReport(
        rows=constructed.height,
        first_computable_year=first_year,
        partial_window_rows=window_audit["pre_1996_nonnull_rows"],
        nonfinite_values=nonfinite,
        output_path=str(destination),
        audit_path=str(audit_path),
    )
