"""Bounded BACI ZIP streaming into authoritative annual trade layers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from decimal import Decimal
import json
from pathlib import Path
import re
import shutil
from typing import Any
from zipfile import ZipFile, ZipInfo

import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv

from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    quarantine_rows,
    verify_manifest,
    write_authoritative_table,
)
from green_debt.storage import directory_usage_bytes


SOURCE_COLUMNS = ("t", "i", "j", "k", "v", "q")
MAX_CSV_BLOCK_SIZE_BYTES = 64 * 1024**2
CSV_READER_USE_THREADS = True
# The frozen source survey found up to 16/17 fractional digits in v/q.  Scale
# 18 retains them exactly.  The integer gates plus <=20m annual rows guarantee
# that v*1000 and every annual sum remain below Decimal128's 20 integer digits.
MAX_ANNUAL_SOURCE_ROWS = 20_000_000
DECIMAL_DTYPE = pl.Decimal(precision=38, scale=18)
_TRADE_DECIMAL = re.compile(r"^[+-]?[0-9]{1,9}(?:[.][0-9]{1,18})?$")
_QUANTITY_DECIMAL = re.compile(r"^[+-]?[0-9]{1,11}(?:[.][0-9]{1,18})?$")
_HS6 = re.compile(r"^[0-9]{6}$")
_SOURCE_CODE = re.compile(r"^[0-9]+$")
_SAFE_PARTITION_VALUE = re.compile(r"^[a-z0-9_]+$")
_CONTRACT_ROOT = Path(__file__).resolve().parents[3] / "contracts"


@dataclass(frozen=True)
class TradeBuildReport:
    revision: str
    years: tuple[int, ...]
    source_rows: int
    valid_rows: int
    quarantined_rows: int
    partitions_written: int
    output_rows: int
    output_bytes: int
    expanded_csv_files_written: int
    max_csv_block_size_bytes: int
    scratch_peak_bytes: int
    scratch_remaining_bytes: int


@dataclass(frozen=True)
class TradeAuditReport:
    hs96_years: tuple[int, ...]
    hs07_years: tuple[int, ...]
    manifests_verified: int
    duplicate_primary_keys: int
    weighted_value_violations: int
    expanded_csv_files: int
    source_hash_mismatches: int
    scratch_bytes: int
    status: str


def _load_contract(name: str) -> TableContract:
    payload = json.loads((_CONTRACT_ROOT / name).read_text(encoding="utf-8"))
    period_value = payload.get("period")
    period = (
        (int(period_value[0]), int(period_value[1]))
        if period_value is not None
        else None
    )
    return TableContract(
        table_id=str(payload["table_id"]),
        schema_version=str(payload["schema_version"]),
        primary_key=tuple(str(value) for value in payload["primary_key"]),
        columns={str(key): str(value) for key, value in payload["columns"].items()},
        units={str(key): str(value) for key, value in payload["units"].items()},
        period=period,
        zero_semantics={
            str(key): str(value)
            for key, value in payload.get("zero_semantics", {}).items()
        },
        transformations=tuple(str(value) for value in payload.get("transformations", [])),
    )


def _annual_members(archive: ZipFile, revision: str) -> dict[int, ZipInfo]:
    pattern = re.compile(
        rf"(?:^|/)BACI_{re.escape(revision)}_Y([0-9]{{4}})_V[^/]*[.]csv$"
    )
    candidates: dict[int, list[ZipInfo]] = {}
    for info in archive.infolist():
        match = pattern.search(info.filename)
        if match is not None:
            candidates.setdefault(int(match.group(1)), []).append(info)
    if not candidates:
        raise RuntimeError(f"no BACI annual members found for {revision}")
    ambiguous = {
        year: tuple(info.filename for info in infos)
        for year, infos in candidates.items()
        if len(infos) != 1
    }
    if ambiguous:
        raise RuntimeError(f"ambiguous BACI member: {ambiguous}")
    return {year: infos[0] for year, infos in sorted(candidates.items())}


def _validate_member_headers(
    archive: ZipFile, members: dict[int, ZipInfo]
) -> None:
    for year, info in members.items():
        with archive.open(info) as handle:
            header = handle.readline().decode("utf-8-sig").rstrip("\r\n")
        if tuple(header.split(",")) != SOURCE_COLUMNS:
            raise RuntimeError(
                f"BACI {year} must have exact columns {SOURCE_COLUMNS}; got {header!r}"
            )


def _validate_inputs(
    revision: str,
    taxonomy: pl.DataFrame,
    economies: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if revision not in {"HS96", "HS07"}:
        raise ValueError("BACI revision must be HS96 or HS07")
    required_taxonomy = {"hs6", "green_weight", "taxonomy_version"}
    if not required_taxonomy <= set(taxonomy.columns):
        raise ValueError(f"taxonomy requires columns {sorted(required_taxonomy)}")
    required_economies = {"source_code", "economy_id"}
    if not required_economies <= set(economies.columns):
        raise ValueError(f"economies require columns {sorted(required_economies)}")

    weights = taxonomy.select(required_taxonomy).with_columns(
        pl.col("hs6").cast(pl.String),
        pl.col("green_weight").cast(pl.Float64),
        pl.col("taxonomy_version").cast(pl.String),
    )
    if weights.filter(
        pl.col("hs6").is_null()
        | ~pl.col("hs6").str.contains(r"^[0-9]{6}$")
        | pl.col("taxonomy_version").is_null()
        | (pl.col("taxonomy_version") == "")
        | pl.col("green_weight").is_null()
        | ~pl.col("green_weight").is_finite()
        | (pl.col("green_weight") < 0.0)
        | (pl.col("green_weight") > 1.0)
    ).height:
        raise ValueError("taxonomy contains invalid HS6 codes, versions, or weights")
    if weights.group_by("taxonomy_version", "hs6").len().filter(
        pl.col("len") > 1
    ).height:
        raise ValueError("taxonomy contains duplicate taxonomy-version/HS6 keys")
    versions = set(weights.get_column("taxonomy_version").unique().to_list())
    if any(_SAFE_PARTITION_VALUE.fullmatch(value) is None for value in versions):
        raise ValueError("taxonomy_version must be a safe lowercase partition value")
    if revision == "HS96" and any(not value.endswith("_hs96") for value in versions):
        raise ValueError("HS96 build received a non-HS96 taxonomy version")
    if revision == "HS07" and versions != {"hs07_native"}:
        raise ValueError("HS07 build must use taxonomy_version=hs07_native")
    weights = weights.filter(pl.col("green_weight") > 0.0).sort(
        ["taxonomy_version", "hs6"]
    ).with_columns(
        pl.col("green_weight")
        .cast(pl.String)
        .cast(DECIMAL_DTYPE, strict=True)
    )

    mapping_columns = ["source_code", "economy_id"]
    if "exclusion_reason" in economies.columns:
        mapping_columns.append("exclusion_reason")
    mapping = economies.select(mapping_columns).with_columns(
        pl.col("source_code").cast(pl.String).str.strip_chars(),
        pl.col("economy_id").cast(pl.String).str.strip_chars(),
    )
    if "exclusion_reason" not in mapping.columns:
        mapping = mapping.with_columns(
            pl.lit(None, dtype=pl.String).alias("exclusion_reason")
        )
    else:
        mapping = mapping.with_columns(
            pl.col("exclusion_reason").cast(pl.String).str.strip_chars()
        )
    if mapping.filter(
        pl.col("source_code").is_null()
        | ~pl.col("source_code").str.contains(r"^[0-9]+$")
        | (
            (pl.col("economy_id").is_null() | (pl.col("economy_id") == ""))
            & (
                pl.col("exclusion_reason").is_null()
                | (pl.col("exclusion_reason") == "")
            )
        )
    ).height:
        raise ValueError(
            "economy mapping contains invalid source codes or undeclared exclusions"
        )
    if mapping.group_by("source_code").len().filter(pl.col("len") > 1).height:
        raise ValueError("economy mapping contains duplicate BACI source codes")
    return weights, mapping.sort("source_code")


def _all_taxonomy_version(revision: str) -> str:
    return "all_hs96" if revision == "HS96" else "all_hs07_native"


def _read_batches(archive: ZipFile, info: ZipInfo) -> Iterable[pa.RecordBatch]:
    convert_options = pacsv.ConvertOptions(
        column_types={name: pa.string() for name in SOURCE_COLUMNS},
        strings_can_be_null=True,
        null_values=["", "NA", "NaN", "nan"],
    )
    read_options = pacsv.ReadOptions(
        block_size=MAX_CSV_BLOCK_SIZE_BYTES,
        use_threads=CSV_READER_USE_THREADS,
    )
    parse_options = pacsv.ParseOptions(delimiter=",")
    with archive.open(info) as handle:
        reader = pacsv.open_csv(
            handle,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )
        yield from reader


def _normalise_batch(
    batch: pa.RecordBatch,
    *,
    expected_year: int,
    member: str,
    source_row_offset: int,
    exporter_map: pl.DataFrame,
    importer_map: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    raw = pl.from_arrow(batch).select(SOURCE_COLUMNS).with_columns(
        *(pl.col(name).str.strip_chars().alias(name) for name in SOURCE_COLUMNS),
        pl.Series(
            "source_row_number",
            range(source_row_offset + 2, source_row_offset + batch.num_rows + 2),
            dtype=pl.UInt64,
        ),
        pl.lit(member, dtype=pl.String).alias("source_member"),
    )
    parsed = (
        raw.with_columns(
            pl.col("t").cast(pl.Int16, strict=False).alias("_year"),
            pl.col("v").str.contains(_TRADE_DECIMAL.pattern).fill_null(False).alias(
                "_trade_decimal_syntax"
            ),
            pl.col("q")
            .str.contains(_QUANTITY_DECIMAL.pattern)
            .fill_null(False)
            .alias("_quantity_decimal_syntax"),
            pl.col("v").cast(DECIMAL_DTYPE, strict=False).alias("_trade_kusd"),
            pl.col("q").cast(DECIMAL_DTYPE, strict=False).alias("_quantity_tonnes"),
        )
        .join(exporter_map, left_on="i", right_on="exporter_code", how="left")
        .join(importer_map, left_on="j", right_on="importer_code", how="left")
        .with_columns(
            pl.when(pl.col("_year").is_null() | (pl.col("_year") != expected_year))
            .then(pl.lit("invalid_year"))
            .when(
                pl.col("i").is_null()
                | ~pl.col("i").str.contains(r"^[0-9]+$").fill_null(False)
            )
            .then(pl.lit("invalid_exporter_code"))
            .when(
                pl.col("j").is_null()
                | ~pl.col("j").str.contains(r"^[0-9]+$").fill_null(False)
            )
            .then(pl.lit("invalid_importer_code"))
            .when(
                pl.col("k").is_null()
                | ~pl.col("k").str.contains(r"^[0-9]{6}$").fill_null(False)
            )
            .then(pl.lit("invalid_hs6"))
            .when(
                ~pl.col("_trade_decimal_syntax")
                | pl.col("_trade_kusd").is_null()
            )
            .then(pl.lit("invalid_trade_value"))
            .when(
                pl.col("_trade_kusd")
                < pl.lit(Decimal("0"), dtype=DECIMAL_DTYPE)
            )
            .then(pl.lit("negative_trade_value"))
            .when(
                pl.col("q").is_not_null()
                & (
                    ~pl.col("_quantity_decimal_syntax")
                    | pl.col("_quantity_tonnes").is_null()
                )
            )
            .then(pl.lit("invalid_quantity"))
            .when(
                pl.col("_quantity_tonnes")
                < pl.lit(Decimal("0"), dtype=DECIMAL_DTYPE)
            )
            .then(pl.lit("negative_quantity"))
            .when(
                pl.col("exporter_id").is_null()
                & pl.col("exporter_declared").fill_null(False)
            )
            .then(pl.lit("excluded_exporter"))
            .when(pl.col("exporter_id").is_null())
            .then(pl.lit("unmapped_exporter"))
            .when(
                pl.col("importer_id").is_null()
                & pl.col("importer_declared").fill_null(False)
            )
            .then(pl.lit("excluded_importer"))
            .when(pl.col("importer_id").is_null())
            .then(pl.lit("unmapped_importer"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("quarantine_reason")
        )
    )
    quarantined = parsed.filter(pl.col("quarantine_reason").is_not_null()).select(
        "source_member",
        "source_row_number",
        *SOURCE_COLUMNS,
        "exporter_exclusion_reason",
        "importer_exclusion_reason",
        "quarantine_reason",
    )
    valid = (
        parsed.filter(pl.col("quarantine_reason").is_null())
        .select(
            pl.col("_year").alias("year"),
            "exporter_id",
            "importer_id",
            pl.col("k").alias("hs6"),
            (
                pl.col("_trade_kusd")
                * pl.lit(Decimal("1000"), dtype=DECIMAL_DTYPE)
            ).alias("trade_value_usd"),
            pl.col("_quantity_tonnes").alias("quantity_tonnes"),
        )
        .with_columns(
            pl.col("year").cast(pl.Int16),
            pl.col("trade_value_usd").cast(DECIMAL_DTYPE),
            pl.col("quantity_tonnes").cast(DECIMAL_DTYPE),
        )
    )
    return valid, quarantined


def _quantity_aggregations() -> tuple[pl.Expr, pl.Expr]:
    return (
        pl.col("quantity_tonnes").sum().alias("quantity_tonnes"),
        pl.col("quantity_tonnes")
        .is_not_null()
        .sum()
        .cast(pl.UInt64)
        .alias("quantity_observation_count"),
    )


def _write_scratch(frame: pl.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")
    return path.stat().st_size


def _batch_aggregates(
    valid: pl.DataFrame,
    *,
    taxonomy: pl.DataFrame,
    all_taxonomy_version: str,
    scratch: Path,
    batch_index: int,
) -> tuple[list[Path], list[Path], list[Path], int]:
    exporter_path = scratch / "exporter" / f"batch-{batch_index:05d}.parquet"
    importer_path = scratch / "importer" / f"batch-{batch_index:05d}.parquet"
    green_path = scratch / "green" / f"batch-{batch_index:05d}.parquet"
    scratch_bytes = 0

    exporter = (
        valid.group_by("year", "exporter_id", "hs6")
        .agg(
            pl.col("trade_value_usd").sum(),
            *_quantity_aggregations(),
            pl.len().cast(pl.UInt64).alias("source_row_count"),
        )
        .rename({"exporter_id": "economy_id"})
        .with_columns(
            pl.lit(all_taxonomy_version, dtype=pl.String).alias("taxonomy_version"),
            pl.lit("exporter", dtype=pl.String).alias("flow_role"),
        )
        .select(
            "taxonomy_version",
            "year",
            "flow_role",
            "economy_id",
            "hs6",
            "trade_value_usd",
            "quantity_tonnes",
            "quantity_observation_count",
            "source_row_count",
        )
    )
    importer = (
        valid.group_by("year", "importer_id", "hs6")
        .agg(
            pl.col("trade_value_usd").sum(),
            *_quantity_aggregations(),
            pl.len().cast(pl.UInt64).alias("source_row_count"),
        )
        .rename({"importer_id": "economy_id"})
        .with_columns(
            pl.lit(all_taxonomy_version, dtype=pl.String).alias("taxonomy_version"),
            pl.lit("importer", dtype=pl.String).alias("flow_role"),
        )
        .select(exporter.columns)
    )
    scratch_bytes += _write_scratch(exporter, exporter_path)
    scratch_bytes += _write_scratch(importer, importer_path)

    green = (
        valid.join(taxonomy, on="hs6", how="inner")
        .group_by(
            "taxonomy_version",
            "year",
            "exporter_id",
            "importer_id",
            "hs6",
            "green_weight",
        )
        .agg(
            pl.col("trade_value_usd").sum(),
            *_quantity_aggregations(),
            pl.len().cast(pl.UInt64).alias("source_row_count"),
        )
    )
    green_paths: list[Path] = []
    if not green.is_empty():
        scratch_bytes += _write_scratch(green, green_path)
        green_paths.append(green_path)
    return [exporter_path], [importer_path], green_paths, scratch_bytes


def _scan_paths(paths: list[Path]) -> pl.LazyFrame:
    if not paths:
        raise ValueError("cannot scan an empty scratch path list")
    return pl.scan_parquet([str(path) for path in paths])


def _empty_product() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "taxonomy_version": pl.String,
            "year": pl.Int16,
            "flow_role": pl.String,
            "economy_id": pl.String,
            "hs6": pl.String,
            "trade_value_usd": pl.Float64,
            "quantity_tonnes": pl.Float64,
            "quantity_observation_count": pl.UInt64,
            "weighted_green_trade_usd": pl.Float64,
            "source_row_count": pl.UInt64,
        }
    )


def _finalise_product(paths: list[Path]) -> pl.DataFrame:
    if not paths:
        return _empty_product()
    keys = ("taxonomy_version", "year", "flow_role", "economy_id", "hs6")
    return (
        _scan_paths(paths)
        .group_by(*keys)
        .agg(
            pl.col("trade_value_usd").sum(),
            pl.col("quantity_tonnes").sum(),
            pl.col("quantity_observation_count").sum(),
            pl.col("source_row_count").sum(),
        )
        .with_columns(
            pl.when(pl.col("quantity_observation_count") > 0)
            .then(pl.col("quantity_tonnes"))
            .otherwise(pl.lit(None, dtype=DECIMAL_DTYPE))
            .alias("quantity_tonnes"),
            pl.lit(None, dtype=pl.Float64).alias("weighted_green_trade_usd"),
        )
        .select(
            *keys,
            "trade_value_usd",
            "quantity_tonnes",
            "quantity_observation_count",
            "weighted_green_trade_usd",
            "source_row_count",
        )
        .sort(keys)
        .collect(engine="streaming")
    )


def _empty_green_bilateral() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "taxonomy_version": pl.String,
            "year": pl.Int16,
            "exporter_id": pl.String,
            "importer_id": pl.String,
            "hs6": pl.String,
            "trade_value_usd": pl.Float64,
            "quantity_tonnes": pl.Float64,
            "quantity_observation_count": pl.UInt64,
            "green_weight": pl.Float64,
            "weighted_green_trade_usd": pl.Float64,
            "source_row_count": pl.UInt64,
        }
    )


def _finalise_green(paths: list[Path]) -> pl.DataFrame:
    if not paths:
        return _empty_green_bilateral()
    keys = (
        "taxonomy_version",
        "year",
        "exporter_id",
        "importer_id",
        "hs6",
    )
    return (
        _scan_paths(paths)
        .group_by(*keys)
        .agg(
            pl.col("trade_value_usd").sum(),
            pl.col("quantity_tonnes").sum(),
            pl.col("quantity_observation_count").sum(),
            pl.col("green_weight").first(),
            pl.col("green_weight").n_unique().alias("_weight_count"),
            pl.col("source_row_count").sum(),
        )
        .with_columns(
            pl.when(pl.col("quantity_observation_count") > 0)
            .then(pl.col("quantity_tonnes"))
            .otherwise(pl.lit(None, dtype=DECIMAL_DTYPE))
            .alias("quantity_tonnes"),
            (pl.col("trade_value_usd") * pl.col("green_weight")).alias(
                "weighted_green_trade_usd"
            ),
        )
        .collect(engine="streaming")
        .pipe(
            lambda frame: (
                frame
                if frame.filter(pl.col("_weight_count") != 1).is_empty()
                else _raise_inconsistent_weight()
            )
        )
        .drop("_weight_count")
        .select(_empty_green_bilateral().columns)
        .sort(keys)
    )


def _raise_inconsistent_weight() -> pl.DataFrame:
    raise RuntimeError("green weight changed within an annual taxonomy partition")


def _green_economy_product(bilateral: pl.DataFrame) -> pl.DataFrame:
    schema = {
        "taxonomy_version": pl.String,
        "year": pl.Int16,
        "flow_role": pl.String,
        "economy_id": pl.String,
        "hs6": pl.String,
        "trade_value_usd": pl.Float64,
        "quantity_tonnes": pl.Float64,
        "quantity_observation_count": pl.UInt64,
        "weighted_green_trade_usd": pl.Float64,
        "source_row_count": pl.UInt64,
    }
    if bilateral.is_empty():
        return pl.DataFrame(schema=schema)

    def aggregate(role: str, column: str) -> pl.DataFrame:
        return (
            bilateral.group_by("taxonomy_version", "year", column, "hs6")
            .agg(
                pl.col("trade_value_usd").sum(),
                pl.col("quantity_tonnes").sum(),
                pl.col("quantity_observation_count").sum(),
                pl.col("weighted_green_trade_usd").sum(),
                pl.col("source_row_count").sum(),
            )
            .rename({column: "economy_id"})
            .with_columns(
                pl.when(pl.col("quantity_observation_count") > 0)
                .then(pl.col("quantity_tonnes"))
                .otherwise(pl.lit(None, dtype=DECIMAL_DTYPE))
                .alias("quantity_tonnes"),
                pl.lit(role, dtype=pl.String).alias("flow_role"),
            )
            .select(*schema)
        )

    return pl.concat(
        [aggregate("exporter", "exporter_id"), aggregate("importer", "importer_id")]
    ).sort("taxonomy_version", "year", "flow_role", "economy_id", "hs6")


def _economy_year_totals(
    exporter: pl.DataFrame,
    importer: pl.DataFrame,
    economies: pl.DataFrame,
    *,
    all_taxonomy_version: str,
    year: int,
) -> pl.DataFrame:
    export_totals = (
        exporter.group_by("economy_id")
        .agg(
            pl.col("trade_value_usd").sum().alias("exports_usd"),
            pl.col("quantity_tonnes").sum().alias("export_quantity_tonnes"),
            pl.col("quantity_observation_count")
            .sum()
            .alias("export_quantity_observation_count"),
            pl.col("source_row_count").sum().alias("export_source_rows"),
        )
    )
    import_totals = (
        importer.group_by("economy_id")
        .agg(
            pl.col("trade_value_usd").sum().alias("imports_usd"),
            pl.col("quantity_tonnes").sum().alias("import_quantity_tonnes"),
            pl.col("quantity_observation_count")
            .sum()
            .alias("import_quantity_observation_count"),
            pl.col("source_row_count").sum().alias("import_source_rows"),
        )
    )
    return (
        economies.select("economy_id")
        .unique()
        .join(export_totals, on="economy_id", how="left")
        .join(import_totals, on="economy_id", how="left")
        .with_columns(
            pl.lit(all_taxonomy_version, dtype=pl.String).alias("taxonomy_version"),
            pl.lit(year, dtype=pl.Int16).alias("year"),
            pl.col("exports_usd").fill_null(Decimal("0")),
            pl.col("imports_usd").fill_null(Decimal("0")),
            pl.col("export_quantity_observation_count").fill_null(0).cast(pl.UInt64),
            pl.col("import_quantity_observation_count").fill_null(0).cast(pl.UInt64),
            pl.col("export_source_rows").fill_null(0).cast(pl.UInt64),
            pl.col("import_source_rows").fill_null(0).cast(pl.UInt64),
        )
        .with_columns(
            pl.when(pl.col("export_quantity_observation_count") > 0)
            .then(pl.col("export_quantity_tonnes"))
            .otherwise(pl.lit(None, dtype=DECIMAL_DTYPE))
            .alias("export_quantity_tonnes"),
            pl.when(pl.col("import_quantity_observation_count") > 0)
            .then(pl.col("import_quantity_tonnes"))
            .otherwise(pl.lit(None, dtype=DECIMAL_DTYPE))
            .alias("import_quantity_tonnes"),
            (pl.col("export_source_rows") > 0).alias("reported_as_exporter"),
            (pl.col("import_source_rows") > 0).alias("reported_as_importer"),
        )
        .select(
            "taxonomy_version",
            "year",
            "economy_id",
            "exports_usd",
            "imports_usd",
            "export_quantity_tonnes",
            "import_quantity_tonnes",
            "export_quantity_observation_count",
            "import_quantity_observation_count",
            "export_source_rows",
            "import_source_rows",
            "reported_as_exporter",
            "reported_as_importer",
        )
        .sort("economy_id")
    )


def _empty_quarantine() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "source_member": pl.String,
            "source_row_number": pl.UInt64,
            "t": pl.String,
            "i": pl.String,
            "j": pl.String,
            "k": pl.String,
            "v": pl.String,
            "q": pl.String,
            "exporter_exclusion_reason": pl.String,
            "importer_exclusion_reason": pl.String,
            "quarantine_reason": pl.String,
        }
    )


def _contract_for_partition(
    base: TableContract, taxonomy_version: str, year: int
) -> TableContract:
    return replace(
        base,
        table_id=f"{base.table_id}__{taxonomy_version}__{year}",
        period=(year, year),
    )


def _publish(
    frame: pl.DataFrame,
    *,
    base_contract: TableContract,
    output_root: Path,
    dataset: str,
    taxonomy_version: str,
    year: int,
    inputs: tuple[InputArtifact, ...],
    build: BuildIdentity,
) -> tuple[int, int]:
    decimal_columns = tuple(
        name for name, dtype in frame.schema.items() if dtype.is_decimal()
    )
    if decimal_columns:
        frame = frame.with_columns(
            *(pl.col(name).cast(pl.Float64) for name in decimal_columns)
        )
    destination = (
        output_root
        / dataset
        / f"year={year}"
        / f"taxonomy_version={taxonomy_version}.parquet"
    )
    manifest = write_authoritative_table(
        frame,
        _contract_for_partition(base_contract, taxonomy_version, year),
        destination,
        inputs,
        build,
    )
    return manifest.rows, manifest.bytes


def stream_baci_aggregates(
    zip_path: Path,
    hs_revision: str,
    taxonomy: pl.DataFrame,
    economies: pl.DataFrame,
    output_root: Path,
    *,
    scratch_root: Path | None = None,
    input_artifacts: tuple[InputArtifact, ...] | None = None,
    build_identity: BuildIdentity | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    years: tuple[int, ...] | None = None,
) -> TradeBuildReport:
    """Stream one BACI archive and publish only bounded annual aggregates."""

    archive_path = zip_path.resolve()
    revision = hs_revision.upper()
    output = output_root.resolve()
    scratch_base = (
        scratch_root.resolve()
        if scratch_root is not None
        else (output / "_scratch").resolve()
    )

    with ZipFile(archive_path) as archive:
        members = _annual_members(archive, revision)
        _validate_member_headers(archive, members)
    if years is not None:
        requested_years = tuple(sorted(set(years)))
        unavailable = tuple(year for year in requested_years if year not in members)
        if unavailable:
            raise ValueError(f"requested BACI years are unavailable: {unavailable}")
        members = {year: members[year] for year in requested_years}
    weights, mapping = _validate_inputs(revision, taxonomy, economies)
    taxonomy_versions = tuple(
        sorted(weights.get_column("taxonomy_version").unique().to_list())
    )
    if not taxonomy_versions:
        raise ValueError("taxonomy contains no positive green weights")

    inputs = input_artifacts or (InputArtifact.from_path(archive_path),)
    build = build_identity or BuildIdentity(
        command=f"stream_baci_aggregates --revision {revision}",
        code_commit="test_or_library_call",
    )
    bilateral_contract = _load_contract("baci_green_bilateral.json")
    product_contract = _load_contract("baci_economy_product.json")
    totals_contract = _load_contract("baci_economy_year_totals.json")
    exporter_map = mapping.rename(
        {
            "source_code": "exporter_code",
            "economy_id": "exporter_id",
            "exclusion_reason": "exporter_exclusion_reason",
        }
    ).with_columns(pl.lit(True).alias("exporter_declared"))
    importer_map = mapping.rename(
        {
            "source_code": "importer_code",
            "economy_id": "importer_id",
            "exclusion_reason": "importer_exclusion_reason",
        }
    ).with_columns(pl.lit(True).alias("importer_declared"))
    all_version = _all_taxonomy_version(revision)

    source_rows = 0
    valid_rows = 0
    quarantined_rows = 0
    partitions_written = 0
    output_rows = 0
    output_bytes = 0
    scratch_peak = 0

    with ZipFile(archive_path) as archive:
        for year, info in members.items():
            if progress is not None:
                progress(
                    {
                        "revision": revision,
                        "year": year,
                        "status": "streaming",
                        "source_member_bytes": info.file_size,
                    }
                )
            year_scratch = scratch_base / revision.lower() / f"year={year}"
            if year_scratch.exists():
                if not year_scratch.is_relative_to(scratch_base):
                    raise RuntimeError("BACI scratch path escaped the declared scratch root")
                shutil.rmtree(year_scratch)
            exporter_paths: list[Path] = []
            importer_paths: list[Path] = []
            green_paths: list[Path] = []
            quarantine_paths: list[Path] = []
            row_offset = 0
            year_valid_rows = 0
            year_quarantined_rows = 0
            scratch_bytes = 0
            for batch_index, batch in enumerate(_read_batches(archive, info)):
                batch_source_rows = batch.num_rows
                valid, quarantined = _normalise_batch(
                    batch,
                    expected_year=year,
                    member=info.filename,
                    source_row_offset=row_offset,
                    exporter_map=exporter_map,
                    importer_map=importer_map,
                )
                source_rows += batch_source_rows
                valid_rows += valid.height
                quarantined_rows += quarantined.height
                year_valid_rows += valid.height
                year_quarantined_rows += quarantined.height
                row_offset += batch_source_rows
                if row_offset > MAX_ANNUAL_SOURCE_ROWS:
                    raise RuntimeError(
                        "BACI annual member exceeds the exact Decimal accumulator row gate"
                    )
                if not valid.is_empty():
                    exp, imp, green, bytes_written = _batch_aggregates(
                        valid,
                        taxonomy=weights,
                        all_taxonomy_version=all_version,
                        scratch=year_scratch,
                        batch_index=batch_index,
                    )
                    exporter_paths.extend(exp)
                    importer_paths.extend(imp)
                    green_paths.extend(green)
                    scratch_bytes += bytes_written
                if not quarantined.is_empty():
                    path = year_scratch / "quarantine" / f"batch-{batch_index:05d}.parquet"
                    scratch_bytes += _write_scratch(quarantined, path)
                    quarantine_paths.append(path)
                scratch_peak = max(scratch_peak, scratch_bytes)

            if row_offset == 0:
                raise RuntimeError(f"BACI annual member is empty: {info.filename}")
            exporter = _finalise_product(exporter_paths)
            importer = _finalise_product(importer_paths)
            green = _finalise_green(green_paths)
            green_economy = _green_economy_product(green)
            totals = _economy_year_totals(
                exporter,
                importer,
                mapping.filter(pl.col("economy_id").is_not_null()),
                all_taxonomy_version=all_version,
                year=year,
            )
            if quarantine_paths:
                quarantined = (
                    _scan_paths(quarantine_paths)
                    .sort("source_row_number")
                    .collect(engine="streaming")
                    .select(_empty_quarantine().columns)
                )
            else:
                quarantined = _empty_quarantine()
            quarantine_rows(
                quarantined,
                "quarantine_reason",
                output
                / "quarantine"
                / f"revision={revision.lower()}"
                / f"year={year}"
                / "quarantine.parquet",
            )

            for dataset, frame in (
                ("exporter_product", exporter),
                ("importer_product", importer),
            ):
                rows, byte_count = _publish(
                    frame,
                    base_contract=product_contract,
                    output_root=output,
                    dataset=dataset,
                    taxonomy_version=all_version,
                    year=year,
                    inputs=inputs,
                    build=build,
                )
                output_rows += rows
                output_bytes += byte_count
                partitions_written += 1
            rows, byte_count = _publish(
                totals,
                base_contract=totals_contract,
                output_root=output,
                dataset="economy_year_totals",
                taxonomy_version=all_version,
                year=year,
                inputs=inputs,
                build=build,
            )
            output_rows += rows
            output_bytes += byte_count
            partitions_written += 1

            for taxonomy_version in taxonomy_versions:
                bilateral_partition = green.filter(
                    pl.col("taxonomy_version") == taxonomy_version
                )
                rows, byte_count = _publish(
                    bilateral_partition,
                    base_contract=bilateral_contract,
                    output_root=output,
                    dataset="green_bilateral",
                    taxonomy_version=taxonomy_version,
                    year=year,
                    inputs=inputs,
                    build=build,
                )
                output_rows += rows
                output_bytes += byte_count
                partitions_written += 1
                economy_partition = green_economy.filter(
                    pl.col("taxonomy_version") == taxonomy_version
                )
                rows, byte_count = _publish(
                    economy_partition,
                    base_contract=product_contract,
                    output_root=output,
                    dataset="green_economy_product",
                    taxonomy_version=taxonomy_version,
                    year=year,
                    inputs=inputs,
                    build=build,
                )
                output_rows += rows
                output_bytes += byte_count
                partitions_written += 1

            shutil.rmtree(year_scratch)
            if progress is not None:
                progress(
                    {
                        "revision": revision,
                        "year": year,
                        "source_rows": row_offset,
                        "valid_rows": year_valid_rows,
                        "quarantined_rows": year_quarantined_rows,
                        "status": "complete",
                    }
                )

    return TradeBuildReport(
        revision=revision,
        years=tuple(members),
        source_rows=source_rows,
        valid_rows=valid_rows,
        quarantined_rows=quarantined_rows,
        partitions_written=partitions_written,
        output_rows=output_rows,
        output_bytes=output_bytes,
        expanded_csv_files_written=0,
        max_csv_block_size_bytes=MAX_CSV_BLOCK_SIZE_BYTES,
        scratch_peak_bytes=scratch_peak,
        scratch_remaining_bytes=directory_usage_bytes(scratch_base),
    )


def _snapshot_hashes(snapshot: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    pattern = re.compile(r"^([0-9a-f]{64})  [.]?/(.+)$")
    for line_number, line in enumerate(
        snapshot.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line or line.startswith("#"):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid raw hash snapshot line {line_number}")
        digest, relative = match.groups()
        result[relative] = digest
    return result


def audit_trade_normalization(
    *,
    data_root: Path,
    raw_snapshot: Path | None = None,
) -> TradeAuditReport:
    """Verify BACI periods, manifests, keys, weights, hashes, and size bounds."""

    data = data_root.resolve()
    root = data / "05_中间数据/normalized/baci"
    manifest_paths = sorted(root.rglob("*.parquet.manifest.json"))
    if not manifest_paths:
        raise RuntimeError("no BACI normalization manifests found")
    manifests = [verify_manifest(path) for path in manifest_paths]
    duplicate_primary_keys = sum(item.duplicate_primary_keys for item in manifests)

    green_paths = sorted((root / "green_bilateral").rglob("*.parquet"))
    if not green_paths:
        raise RuntimeError("no green BACI bilateral partitions found")
    green = pl.scan_parquet([str(path) for path in green_paths])
    weighted_violations = (
        green.filter(
            (pl.col("weighted_green_trade_usd") < -1e-9)
            | (
                pl.col("weighted_green_trade_usd")
                > pl.col("trade_value_usd") + 1e-6
            )
        )
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    versions = (
        green.select("taxonomy_version", "year")
        .unique()
        .collect(engine="streaming")
        .sort("taxonomy_version", "year")
    )
    hs96_years = tuple(
        versions.filter(pl.col("taxonomy_version") == "main_hs96")
        .get_column("year")
        .to_list()
    )
    hs07_years = tuple(
        versions.filter(pl.col("taxonomy_version") == "hs07_native")
        .get_column("year")
        .to_list()
    )
    if hs96_years != tuple(range(1996, 2025)):
        raise RuntimeError(f"HS96 normalized period differs: {hs96_years}")
    if hs07_years != tuple(range(2007, 2025)):
        raise RuntimeError(f"HS07 normalized period differs: {hs07_years}")

    expanded_csv_files = len(list(root.rglob("*.csv")))
    snapshot = raw_snapshot or data / "04_原始数据/SHA256SUMS_20260823.txt"
    expected_hashes = _snapshot_hashes(snapshot)
    archive_relatives = {
        "baci/202601/BACI_HS96_V202601.zip",
        "baci/202601/BACI_HS07_V202601.zip",
    }
    expected_archive_hashes = {
        expected_hashes[relative] for relative in archive_relatives
    }
    observed_archive_hashes = {
        artifact.sha256
        for manifest in manifests
        for artifact in manifest.input_artifacts
        if Path(artifact.path).name in {
            "BACI_HS96_V202601.zip",
            "BACI_HS07_V202601.zip",
        }
    }
    source_hash_mismatches = len(
        expected_archive_hashes.symmetric_difference(observed_archive_hashes)
    )
    scratch_bytes = directory_usage_bytes(data / "05_中间数据/_tmp/baci")
    if duplicate_primary_keys:
        raise RuntimeError(f"BACI normalized duplicate keys: {duplicate_primary_keys}")
    if weighted_violations:
        raise RuntimeError(f"BACI weighted value violations: {weighted_violations}")
    if expanded_csv_files:
        raise RuntimeError(f"expanded BACI CSV files found: {expanded_csv_files}")
    if source_hash_mismatches:
        raise RuntimeError(f"BACI source hash mismatches: {source_hash_mismatches}")
    if scratch_bytes >= 25 * 1024**3:
        raise RuntimeError("BACI scratch reaches 25 GB")
    return TradeAuditReport(
        hs96_years=hs96_years,
        hs07_years=hs07_years,
        manifests_verified=len(manifests),
        duplicate_primary_keys=duplicate_primary_keys,
        weighted_value_violations=weighted_violations,
        expanded_csv_files=expanded_csv_files,
        source_hash_mismatches=source_hash_mismatches,
        scratch_bytes=scratch_bytes,
        status="valid",
    )
