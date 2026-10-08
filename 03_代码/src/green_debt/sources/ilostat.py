"""ILOSTAT RDS extraction and source-consistent occupation skill shares."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import polars as pl

from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    verify_manifest,
    write_authoritative_table,
)


TECHNICAL_ISCO08_CODES = (
    "OC2_ISCO08_31",
    "OC2_ISCO08_35",
    "OC2_ISCO08_71",
    "OC2_ISCO08_72",
    "OC2_ISCO08_74",
    "OC2_ISCO08_81",
    "OC2_ISCO08_82",
)
ISCO08_TOTAL = "OC2_ISCO08_TOTAL"
RAW_COLUMNS = (
    "ref_area",
    "source",
    "indicator",
    "sex",
    "classif1",
    "time",
    "obs_value",
    "obs_status",
    "note_classif",
    "note_indicator",
    "note_source",
    "best_source",
)
RAW_SCHEMA_OVERRIDES = {
    "ref_area": pl.String,
    "source": pl.String,
    "indicator": pl.String,
    "sex": pl.String,
    "classif1": pl.String,
    "time": pl.Int16,
    "obs_value": pl.Float64,
    "obs_status": pl.String,
    "note_classif": pl.String,
    "note_indicator": pl.String,
    "note_source": pl.String,
    "best_source": pl.Int8,
}
OUTPUT_SCHEMA = {
    "economy_id": pl.String,
    "year": pl.Int16,
    "series_id": pl.String,
    "skill_share": pl.Float64,
    "numerator_thousands": pl.Float64,
    "denominator_thousands": pl.Float64,
    "unit": pl.String,
    "source_ref": pl.String,
    "source_status": pl.String,
    "robustness_only": pl.Boolean,
    "missing_reason": pl.String,
}


@dataclass(frozen=True)
class IlostatBuildReport:
    source_files: int
    source_rows: int
    series: tuple[str, ...]
    retained_rows: int
    null_skill_shares: int
    same_source_denominator_missing: int
    duplicate_keys: int
    temporary_csv_files_remaining: int
    output_path: str
    output_bytes: int


@dataclass(frozen=True)
class IlostatAuditReport:
    rows: int
    series: tuple[str, ...]
    null_skill_shares: int
    out_of_range_shares: int
    primary_rows: int
    robustness_rows: int
    duplicate_keys: int
    status: str


def _require_columns(frame: pl.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"ILOSTAT frame lacks columns: {missing}")


def _validated_raw(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "ref_area",
        "source",
        "indicator",
        "sex",
        "classif1",
        "time",
        "obs_value",
        "best_source",
    }
    _require_columns(frame, required)
    additions: list[pl.Expr] = []
    if "obs_status" not in frame.columns:
        additions.append(pl.lit(None, dtype=pl.String).alias("obs_status"))
    if additions:
        frame = frame.with_columns(additions)
    output = frame.select(
        pl.col("ref_area").cast(pl.String, strict=True).str.strip_chars(),
        pl.col("source").cast(pl.String, strict=True).str.strip_chars(),
        pl.col("indicator").cast(pl.String, strict=True).str.strip_chars(),
        pl.col("sex").cast(pl.String, strict=True).str.strip_chars(),
        pl.col("classif1").cast(pl.String, strict=True).str.strip_chars(),
        pl.col("time").cast(pl.Int16, strict=True),
        pl.col("obs_value").cast(pl.Float64, strict=True),
        pl.col("obs_status").cast(pl.String, strict=True).str.strip_chars(),
        pl.col("best_source").cast(pl.Int8, strict=True),
    )
    if output.filter(
        pl.col("ref_area").is_null()
        | (pl.col("ref_area") == "")
        | pl.col("source").is_null()
        | (pl.col("source") == "")
        | pl.col("indicator").is_null()
        | (pl.col("indicator") == "")
        | pl.col("classif1").is_null()
        | (pl.col("classif1") == "")
        | pl.col("time").is_null()
        | pl.col("best_source").is_null()
    ).height:
        raise ValueError("ILOSTAT source has a null or empty key field")
    if output.filter(
        pl.col("obs_value").is_not_null()
        & (~pl.col("obs_value").is_finite() | (pl.col("obs_value") < 0.0))
    ).height:
        raise ValueError("ILOSTAT employment values must be finite and nonnegative")
    return output


def _combined_status(statuses: list[object]) -> str:
    values = sorted(
        {
            str(value).strip()
            for value in statuses
            if value is not None and str(value).strip()
        }
    )
    return "raw:" + "|".join(values) if values else "reported"


def normalize_ilostat_skill(
    frame: pl.DataFrame,
    *,
    indicator: str,
    series_id: str,
    robustness_only: bool,
    expected_period: tuple[int, int] | None = None,
) -> pl.DataFrame:
    """Construct a seven-category share without crossing source denominators."""

    if not indicator or not series_id:
        raise ValueError("ILOSTAT indicator and series_id are required")
    raw = _validated_raw(frame).filter(
        (pl.col("indicator") == indicator)
        & (pl.col("sex") == "SEX_T")
        & (pl.col("best_source") == 1)
        & pl.col("classif1").is_in((*TECHNICAL_ISCO08_CODES, ISCO08_TOTAL))
    )
    if expected_period is not None:
        if expected_period[0] > expected_period[1]:
            raise ValueError("ILOSTAT expected period must be ascending")
        raw = raw.filter(pl.col("time").is_between(*expected_period))
    duplicates = raw.group_by(
        "ref_area", "source", "indicator", "sex", "classif1", "time"
    ).len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError(f"duplicate ILOSTAT selected cells: {duplicates.height}")

    totals: dict[tuple[str, str, int], dict[str, Any]] = {}
    numerator_groups: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in raw.iter_rows(named=True):
        key = (str(row["ref_area"]), str(row["source"]), int(row["time"]))
        category = str(row["classif1"])
        if category == ISCO08_TOTAL:
            totals[key] = row
        else:
            numerator_groups.setdefault(key, {})[category] = row

    output: list[dict[str, Any]] = []
    expected_categories = set(TECHNICAL_ISCO08_CODES)
    for key in sorted(numerator_groups):
        ref_area, source, year = key
        categories = numerator_groups[key]
        missing_categories = expected_categories - set(categories)
        values = [categories[code]["obs_value"] for code in TECHNICAL_ISCO08_CODES if code in categories]
        total_row = totals.get(key)
        denominator = total_row["obs_value"] if total_row is not None else None
        numerator = (
            float(sum(float(value) for value in values))
            if not missing_categories and all(value is not None for value in values)
            else None
        )
        reason: str | None = None
        share: float | None = None
        if missing_categories:
            reason = "approved_categories_incomplete"
        elif any(value is None for value in values):
            reason = "numerator_source_null"
        elif total_row is None:
            reason = "same_source_denominator_missing"
        elif denominator is None:
            reason = "denominator_source_null"
        elif float(denominator) <= 0.0:
            reason = "denominator_nonpositive"
        else:
            share = float(numerator) / float(denominator)
            if share > 1.0 + 1e-12:
                raise ValueError(
                    f"ILOSTAT technical numerator exceeds total employment: {key}"
                )
        statuses = [row["obs_status"] for row in categories.values()]
        if total_row is not None:
            statuses.append(total_row["obs_status"])
        output.append(
            {
                "economy_id": ref_area,
                "year": year,
                "series_id": series_id,
                "skill_share": share,
                "numerator_thousands": numerator,
                "denominator_thousands": (
                    float(denominator) if denominator is not None else None
                ),
                "unit": "share_0_to_1",
                "source_ref": source,
                "source_status": _combined_status(statuses),
                "robustness_only": robustness_only,
                "missing_reason": reason,
            }
        )
    return pl.DataFrame(output, schema=OUTPUT_SCHEMA).sort(
        "economy_id", "year", "series_id"
    )


def normalize_ilostat_stem_share(
    stem: pl.DataFrame,
    employment: pl.DataFrame,
    *,
    expected_period: tuple[int, int] = (2000, 2024),
) -> pl.DataFrame:
    """Divide STEM employment by same-survey ICLS-13 total employment."""

    stem_raw = _validated_raw(stem).filter(
        (pl.col("indicator") == "EMP_STEM_SEX_OC2_NB")
        & (pl.col("sex") == "SEX_T")
        & (pl.col("best_source") == 1)
        & (pl.col("classif1") == ISCO08_TOTAL)
        & pl.col("time").is_between(*expected_period)
    )
    denominator_raw = _validated_raw(employment).filter(
        (pl.col("indicator") == "EMP_TEMP_SEX_OC2_NB")
        & (pl.col("sex") == "SEX_T")
        & (pl.col("best_source") == 1)
        & (pl.col("classif1") == ISCO08_TOTAL)
        & pl.col("time").is_between(*expected_period)
    )
    key = ["ref_area", "source", "time"]
    for label, selected in (("STEM", stem_raw), ("employment", denominator_raw)):
        duplicates = selected.group_by(key).len().filter(pl.col("len") > 1).height
        if duplicates:
            raise ValueError(f"duplicate ILOSTAT {label} denominator keys: {duplicates}")
    denominators = {
        (str(row["ref_area"]), str(row["source"]), int(row["time"])): row
        for row in denominator_raw.iter_rows(named=True)
    }
    output: list[dict[str, Any]] = []
    for row in stem_raw.sort(key).iter_rows(named=True):
        row_key = (str(row["ref_area"]), str(row["source"]), int(row["time"]))
        denominator_row = denominators.get(row_key)
        numerator = row["obs_value"]
        denominator = (
            denominator_row["obs_value"] if denominator_row is not None else None
        )
        if numerator is None:
            share = None
            reason = "numerator_source_null"
        elif denominator_row is None:
            share = None
            reason = "same_source_denominator_missing"
        elif denominator is None:
            share = None
            reason = "denominator_source_null"
        elif float(denominator) <= 0.0:
            share = None
            reason = "denominator_nonpositive"
        else:
            share = float(numerator) / float(denominator)
            reason = None
            if share > 1.0 + 1e-12:
                raise ValueError(f"ILOSTAT STEM employment exceeds total: {row_key}")
        statuses = [row["obs_status"]]
        if denominator_row is not None:
            statuses.append(denominator_row["obs_status"])
        output.append(
            {
                "economy_id": row_key[0],
                "year": row_key[2],
                "series_id": "stem_occupation_share_robustness",
                "skill_share": share,
                "numerator_thousands": (
                    float(numerator) if numerator is not None else None
                ),
                "denominator_thousands": (
                    float(denominator) if denominator is not None else None
                ),
                "unit": "share_0_to_1",
                "source_ref": row_key[1],
                "source_status": _combined_status(statuses),
                "robustness_only": True,
                "missing_reason": reason,
            }
        )
    return pl.DataFrame(output, schema=OUTPUT_SCHEMA).sort(
        "economy_id", "year", "series_id"
    )


def _map_economies(frame: pl.DataFrame, economies: pl.DataFrame) -> pl.DataFrame:
    _require_columns(economies, {"source_code", "economy_id"})
    mapping = economies.select(
        pl.col("source_code").cast(pl.String).str.strip_chars().alias("_source_code"),
        pl.col("economy_id").cast(pl.String).str.strip_chars().alias("_economy_id"),
    ).with_columns(pl.lit(True).alias("_declared"))
    if mapping.group_by("_source_code").len().filter(pl.col("len") > 1).height:
        raise ValueError("ILOSTAT economy mapping has duplicate source codes")
    joined = frame.rename({"economy_id": "_source_code"}).join(
        mapping, on="_source_code", how="left"
    )
    missing = joined.filter(pl.col("_declared").is_null())
    if missing.height:
        codes = missing.get_column("_source_code").unique().sort().to_list()
        raise ValueError(f"unmapped ILOSTAT source codes: {codes}")
    output = (
        joined.filter(pl.col("_economy_id").is_not_null())
        .drop("_source_code", "_declared")
        .rename({"_economy_id": "economy_id"})
        .select(*OUTPUT_SCHEMA)
        .sort("economy_id", "year", "series_id")
    )
    duplicates = output.group_by("economy_id", "year", "series_id").len().filter(
        pl.col("len") > 1
    ).height
    if duplicates:
        raise ValueError(f"duplicate normalized ILOSTAT keys: {duplicates}")
    return output


def extract_ilostat_rds(
    *, r_script: Path, source: Path, destination: Path
) -> None:
    """Extract exactly one RDS data frame into one bounded temporary CSV."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"ILOSTAT temporary CSV already exists: {destination}")
    completed = subprocess.run(
        ["Rscript", "--vanilla", str(r_script), str(source), str(destination)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ILOSTAT RDS extraction failed for {source.name}: "
            f"{completed.stderr.strip()}"
        )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"ILOSTAT extractor wrote no CSV: {source.name}")


def _read_extracted_csv(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(
        path,
        schema_overrides=RAW_SCHEMA_OVERRIDES,
        null_values="",
        infer_schema_length=10_000,
    )
    if tuple(frame.columns) != RAW_COLUMNS:
        raise ValueError(
            f"ILOSTAT extracted schema changed: expected {RAW_COLUMNS}, "
            f"got {tuple(frame.columns)}"
        )
    _validated_raw(frame)
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


def build_ilostat_table(
    *,
    primary_path: Path,
    icls19_path: Path,
    stem_path: Path,
    r_script: Path,
    temporary_directory: Path,
    economies: pl.DataFrame,
    destination: Path,
    contract_path: Path,
    inputs: tuple[InputArtifact, ...],
    build: BuildIdentity,
) -> IlostatBuildReport:
    sources = (primary_path, icls19_path, stem_path)
    temporary_paths = tuple(
        temporary_directory / f"{path.stem}.{os.getpid()}.csv" for path in sources
    )
    frames: list[pl.DataFrame] = []
    for source, temporary in zip(sources, temporary_paths, strict=True):
        extract_ilostat_rds(
            r_script=r_script, source=source, destination=temporary
        )
        frames.append(_read_extracted_csv(temporary))

    primary, icls19, stem = frames
    output = _map_economies(
        pl.concat(
            (
                normalize_ilostat_skill(
                    primary,
                    indicator="EMP_TEMP_SEX_OC2_NB",
                    series_id="technical_skill_share_primary",
                    robustness_only=False,
                    expected_period=(1996, 2024),
                ),
                normalize_ilostat_skill(
                    icls19,
                    indicator="EMP_5EMP_SEX_OC2_NB",
                    series_id="technical_skill_share_icls19_robustness",
                    robustness_only=True,
                    expected_period=(2000, 2024),
                ),
                normalize_ilostat_stem_share(
                    stem, primary, expected_period=(2000, 2024)
                ),
            )
        ),
        economies,
    )
    manifest = write_authoritative_table(
        output, _load_contract(contract_path), destination, inputs, build
    )
    for temporary in temporary_paths:
        temporary.unlink()
    remaining = sum(path.exists() for path in temporary_paths)
    return IlostatBuildReport(
        source_files=len(sources),
        source_rows=sum(frame.height for frame in frames),
        series=tuple(sorted(output.get_column("series_id").unique().to_list())),
        retained_rows=output.height,
        null_skill_shares=output.get_column("skill_share").null_count(),
        same_source_denominator_missing=output.filter(
            pl.col("missing_reason") == "same_source_denominator_missing"
        ).height,
        duplicate_keys=0,
        temporary_csv_files_remaining=remaining,
        output_path=str(destination.resolve()),
        output_bytes=manifest.bytes,
    )


def audit_ilostat_table(*, manifest_path: Path) -> IlostatAuditReport:
    manifest = verify_manifest(manifest_path)
    frame = pl.read_parquet(manifest.destination)
    expected_series = (
        "stem_occupation_share_robustness",
        "technical_skill_share_icls19_robustness",
        "technical_skill_share_primary",
    )
    series = tuple(sorted(frame.get_column("series_id").unique().to_list()))
    duplicates = frame.group_by("economy_id", "year", "series_id").len().filter(
        pl.col("len") > 1
    ).height
    out_of_range = frame.filter(
        pl.col("skill_share").is_not_null()
        & ((pl.col("skill_share") < 0.0) | (pl.col("skill_share") > 1.0))
    ).height
    flag_mismatches = frame.filter(
        (
            (pl.col("series_id") == "technical_skill_share_primary")
            & pl.col("robustness_only")
        )
        | (
            (pl.col("series_id") != "technical_skill_share_primary")
            & ~pl.col("robustness_only")
        )
    ).height
    if series != expected_series or duplicates or out_of_range or flag_mismatches:
        raise RuntimeError(
            "ILOSTAT audit failed: "
            f"series={series}, duplicates={duplicates}, ranges={out_of_range}, "
            f"flags={flag_mismatches}"
        )
    return IlostatAuditReport(
        rows=frame.height,
        series=series,
        null_skill_shares=frame.get_column("skill_share").null_count(),
        out_of_range_shares=out_of_range,
        primary_rows=frame.filter(~pl.col("robustness_only")).height,
        robustness_rows=frame.filter(pl.col("robustness_only")).height,
        duplicate_keys=duplicates,
        status="valid",
    )
