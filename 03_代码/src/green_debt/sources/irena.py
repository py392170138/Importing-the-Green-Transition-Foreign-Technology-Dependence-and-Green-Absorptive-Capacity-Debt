"""IRENA JSON-stat2 decoding with exact dimension and zero semantics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from math import isfinite, prod
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    verify_manifest,
    write_authoritative_table,
)


CAPACITY_DIMS = frozenset({"Country/area", "Technology", "Grid connection", "Year"})
GENERATION_DIMS = frozenset(
    {"Country/area", "Technology", "Data Type", "Grid connection", "Year"}
)
SHARE_DIMS = frozenset({"Region/country/area", "Indicator", "Year"})
OUTPUT_SCHEMA = {
    "economy_id": pl.String,
    "year": pl.Int16,
    "indicator_id": pl.String,
    "value": pl.Float64,
    "unit": pl.String,
    "source_update": pl.String,
    "source_status": pl.String,
    "retirement_or_revision": pl.Boolean,
}
SOURCE_SCHEMA = {
    "source_code": pl.String,
    "year": pl.Int16,
    "value": pl.Float64,
    "source_update": pl.String,
    "source_status": pl.String,
}


@dataclass(frozen=True)
class IrenaBuildReport:
    source_files: int
    source_cells: int
    indicators: tuple[str, ...]
    retained_rows: int
    null_values: int
    negative_additions: int
    duplicate_keys: int
    output_path: str
    output_bytes: int


@dataclass(frozen=True)
class IrenaAuditReport:
    indicators: tuple[str, ...]
    rows: int
    null_values: int
    negative_additions: int
    retirement_flag_mismatches: int
    duplicate_keys: int
    unit_mismatches: int
    indicator_periods: dict[str, tuple[int, ...]]
    status: str


@dataclass(frozen=True)
class _Dataset:
    dimensions: tuple[str, ...]
    categories: dict[str, tuple[tuple[str, str], ...]]
    sizes: tuple[int, ...]
    values: list[Any] | dict[str, Any]
    statuses: list[Any] | dict[str, Any]
    source_update: str
    total_cells: int


def _ordered_categories(dimension: str, payload: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    category = payload.get("category")
    if not isinstance(category, dict):
        raise ValueError(f"IRENA dimension {dimension} lacks category metadata")
    index = category.get("index")
    labels = category.get("label")
    if not isinstance(labels, dict):
        raise ValueError(f"IRENA dimension {dimension} lacks category labels")
    if isinstance(index, dict):
        try:
            positions = {str(code): int(position) for code, position in index.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"IRENA dimension {dimension} has invalid category index") from exc
        if sorted(positions.values()) != list(range(len(positions))):
            raise ValueError(f"IRENA dimension {dimension} category index is not contiguous")
        codes = [code for code, _ in sorted(positions.items(), key=lambda item: item[1])]
    elif isinstance(index, list) and all(isinstance(code, str) for code in index):
        codes = list(index)
    else:
        raise ValueError(f"IRENA dimension {dimension} has invalid category index")
    if set(codes) != set(labels):
        raise ValueError(f"IRENA dimension {dimension} labels differ from category index")
    return tuple((code, str(labels[code])) for code in codes)


def _dataset(payload: Any, *, expected_dimensions: frozenset[str]) -> _Dataset:
    if not isinstance(payload, dict) or payload.get("class") != "dataset":
        raise ValueError("IRENA JSON-stat2 root must be a dataset")
    dimensions_value = payload.get("id")
    sizes_value = payload.get("size")
    dimension_payload = payload.get("dimension")
    if (
        not isinstance(dimensions_value, list)
        or not all(isinstance(value, str) for value in dimensions_value)
        or len(set(dimensions_value)) != len(dimensions_value)
        or set(dimensions_value) != expected_dimensions
        or not isinstance(sizes_value, list)
        or len(sizes_value) != len(dimensions_value)
        or not isinstance(dimension_payload, dict)
    ):
        raise ValueError(
            f"IRENA dimensions changed: expected {sorted(expected_dimensions)}, "
            f"got {dimensions_value!r}"
        )
    dimensions = tuple(dimensions_value)
    categories: dict[str, tuple[tuple[str, str], ...]] = {}
    sizes: list[int] = []
    for dimension, raw_size in zip(dimensions, sizes_value, strict=True):
        item = dimension_payload.get(dimension)
        if not isinstance(item, dict) or item.get("label") != dimension:
            raise ValueError(f"IRENA dimension label changed: {dimension}")
        ordered = _ordered_categories(dimension, item)
        try:
            size = int(raw_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"IRENA dimension size is invalid: {dimension}") from exc
        if size != len(ordered):
            raise ValueError(f"IRENA dimension size differs from categories: {dimension}")
        categories[dimension] = ordered
        sizes.append(size)
    total_cells = prod(sizes)
    values = payload.get("value")
    if isinstance(values, list):
        if len(values) != total_cells:
            raise ValueError("IRENA dense value vector length differs from dimensions")
    elif isinstance(values, dict):
        if any(not str(key).isdigit() or not 0 <= int(key) < total_cells for key in values):
            raise ValueError("IRENA sparse value index is outside dimensions")
    else:
        raise ValueError("IRENA value must be a dense list or sparse object")
    statuses = payload.get("status", {})
    if isinstance(statuses, list):
        if len(statuses) != total_cells:
            raise ValueError("IRENA dense status vector length differs from dimensions")
    elif isinstance(statuses, dict):
        if any(not str(key).isdigit() or not 0 <= int(key) < total_cells for key in statuses):
            raise ValueError("IRENA sparse status index is outside dimensions")
    else:
        raise ValueError("IRENA status must be a dense list or sparse object")
    source_update = str(payload.get("updated") or "").strip()
    if not source_update:
        raise ValueError("IRENA dataset lacks updated metadata")
    return _Dataset(
        dimensions=dimensions,
        categories=categories,
        sizes=tuple(sizes),
        values=values,
        statuses=statuses,
        source_update=source_update,
        total_cells=total_cells,
    )


def _indexed_value(container: list[Any] | dict[str, Any], index: int) -> Any:
    return container[index] if isinstance(container, list) else container.get(str(index))


def _numeric_value(value: Any, index: int) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"IRENA cell {index} is not numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"IRENA cell {index} is nonfinite")
    return result


def _cells(dataset: _Dataset) -> Iterator[tuple[dict[str, tuple[str, str]], float | None, str]]:
    ordered = [dataset.categories[dimension] for dimension in dataset.dimensions]
    for flat_index, coordinate in enumerate(product(*ordered)):
        mapping = dict(zip(dataset.dimensions, coordinate, strict=True))
        value = _numeric_value(_indexed_value(dataset.values, flat_index), flat_index)
        status_value = _indexed_value(dataset.statuses, flat_index)
        status = str(status_value).strip() if status_value is not None else ""
        yield mapping, value, status


def _target_code(dataset: _Dataset, dimension: str, label: str) -> str:
    matches = [code for code, value in dataset.categories[dimension] if value == label]
    if len(matches) != 1:
        raise ValueError(
            f"IRENA dimension {dimension} must contain exactly one {label!r} category"
        )
    return matches[0]


def _status_for_value(value: float | None, raw_status: str) -> str:
    if value is None:
        return "source_null"
    return f"raw:{raw_status}" if raw_status else "reported"


def _combined_status(values: tuple[float | None, ...], statuses: tuple[str, ...]) -> str:
    if any(value is None for value in values):
        return "source_null"
    raw = sorted({status for status in statuses if status})
    return "raw:" + "|".join(raw) if raw else "reported"


def _source_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=SOURCE_SCHEMA).sort("source_code", "year")


def normalize_irena_capacity_payloads(payloads: tuple[dict[str, Any], ...]) -> pl.DataFrame:
    """Select total renewable capacity and sum OnGrid plus OffGrid only."""

    cells: dict[tuple[str, int, str], tuple[float | None, str, str]] = {}
    for payload in payloads:
        dataset = _dataset(payload, expected_dimensions=CAPACITY_DIMS)
        technology = _target_code(dataset, "Technology", "Total renewable energy")
        on_grid = _target_code(dataset, "Grid connection", "OnGrid")
        off_grid = _target_code(dataset, "Grid connection", "OffGrid")
        for coordinate, value, status in _cells(dataset):
            if coordinate["Technology"][0] != technology:
                continue
            grid_code = coordinate["Grid connection"][0]
            if grid_code not in {on_grid, off_grid}:
                continue
            source_code = coordinate["Country/area"][0]
            try:
                year = int(coordinate["Year"][1])
            except ValueError as exc:
                raise ValueError("IRENA capacity year label is not numeric") from exc
            grid = "on" if grid_code == on_grid else "off"
            key = (source_code, year, grid)
            if key in cells:
                raise ValueError(f"duplicate IRENA capacity cell: {key}")
            cells[key] = (value, status, dataset.source_update)
    groups = sorted({(source, year) for source, year, _ in cells})
    rows: list[dict[str, Any]] = []
    for source_code, year in groups:
        on_key = (source_code, year, "on")
        off_key = (source_code, year, "off")
        if on_key not in cells or off_key not in cells:
            raise ValueError(f"IRENA capacity lacks OnGrid/OffGrid cell: {source_code}/{year}")
        on_value, on_status, on_update = cells[on_key]
        off_value, off_status, off_update = cells[off_key]
        if on_update != off_update:
            raise ValueError("IRENA capacity update differs within country-year")
        value = (
            on_value + off_value
            if on_value is not None and off_value is not None
            else None
        )
        rows.append(
            {
                "source_code": source_code,
                "year": year,
                "value": value,
                "source_update": on_update,
                "source_status": _combined_status(
                    (on_value, off_value), (on_status, off_status)
                ),
            }
        )
    return _source_frame(rows)


def normalize_irena_generation_payloads(payloads: tuple[dict[str, Any], ...]) -> pl.DataFrame:
    """Select the approved total-renewable, All-grid generation cell only."""

    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for payload in payloads:
        dataset = _dataset(payload, expected_dimensions=GENERATION_DIMS)
        technology = _target_code(dataset, "Technology", "Total renewable")
        data_type = _target_code(
            dataset, "Data Type", "Electricity Generation (GWh)"
        )
        all_matches = [
            code
            for code, label in dataset.categories["Grid connection"]
            if label == "All"
        ]
        if len(all_matches) > 1:
            raise ValueError("IRENA generation has duplicate All grid categories")
        if not all_matches:
            for country_code, _ in dataset.categories["Country/area"]:
                for _, year_label in dataset.categories["Year"]:
                    try:
                        year = int(year_label)
                    except ValueError as exc:
                        raise ValueError("IRENA generation year label is not numeric") from exc
                    key = (country_code, year)
                    if key in rows_by_key:
                        raise ValueError(f"duplicate IRENA generation cell: {key}")
                    rows_by_key[key] = {
                        "source_code": country_code,
                        "year": year,
                        "value": None,
                        "source_update": dataset.source_update,
                        "source_status": "missing_approved_all_cell",
                    }
            continue
        all_grid = all_matches[0]
        for coordinate, value, status in _cells(dataset):
            if (
                coordinate["Technology"][0] != technology
                or coordinate["Data Type"][0] != data_type
                or coordinate["Grid connection"][0] != all_grid
            ):
                continue
            source_code = coordinate["Country/area"][0]
            try:
                year = int(coordinate["Year"][1])
            except ValueError as exc:
                raise ValueError("IRENA generation year label is not numeric") from exc
            key = (source_code, year)
            if key in rows_by_key:
                raise ValueError(f"duplicate IRENA generation cell: {key}")
            rows_by_key[key] = {
                "source_code": source_code,
                "year": year,
                "value": value,
                "source_update": dataset.source_update,
                "source_status": _status_for_value(value, status),
            }
    return _source_frame(list(rows_by_key.values()))


def normalize_irena_share_payloads(payloads: tuple[dict[str, Any], ...]) -> pl.DataFrame:
    targets = {
        "RE share of electricity capacity (%)": "renewable_capacity_share",
        "RE share of electricity generation (%)": "renewable_generation_share",
    }
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, int, str]] = set()
    for payload in payloads:
        dataset = _dataset(payload, expected_dimensions=SHARE_DIMS)
        target_codes = {
            _target_code(dataset, "Indicator", label): indicator
            for label, indicator in targets.items()
        }
        for coordinate, value, status in _cells(dataset):
            indicator_code = coordinate["Indicator"][0]
            if indicator_code not in target_codes:
                continue
            source_code = coordinate["Region/country/area"][0]
            try:
                year = int(coordinate["Year"][1])
            except ValueError as exc:
                raise ValueError("IRENA share year label is not numeric") from exc
            indicator = target_codes[indicator_code]
            key = (source_code, year, indicator)
            if key in keys:
                raise ValueError(f"duplicate IRENA share cell: {key}")
            keys.add(key)
            rows.append(
                {
                    "source_code": source_code,
                    "year": year,
                    "indicator_id": indicator,
                    "value": value,
                    "unit": "percent",
                    "source_update": dataset.source_update,
                    "source_status": _status_for_value(value, status),
                    "retirement_or_revision": None,
                }
            )
    return pl.DataFrame(
        rows,
        schema={"source_code": pl.String, **{k: v for k, v in OUTPUT_SCHEMA.items() if k != "economy_id"}},
    ).sort("source_code", "year", "indicator_id")


def _economy_mapping(economies: pl.DataFrame | None) -> pl.DataFrame | None:
    if economies is None:
        return None
    if not {"source_code", "economy_id"} <= set(economies.columns):
        raise ValueError("IRENA economy mapping requires source_code and economy_id")
    mapping = economies.select("source_code", "economy_id").with_columns(
        pl.col("source_code").cast(pl.String).str.strip_chars(),
        pl.col("economy_id").cast(pl.String).str.strip_chars(),
        pl.lit(True).alias("_declared"),
    )
    if mapping.filter(pl.col("source_code").is_null() | (pl.col("source_code") == "")).height:
        raise ValueError("IRENA economy mapping has an empty source code")
    if mapping.group_by("source_code").len().filter(pl.col("len") > 1).height:
        raise ValueError("IRENA economy mapping has duplicate source codes")
    return mapping


def _map_frame(frame: pl.DataFrame, mapping: pl.DataFrame | None) -> pl.DataFrame:
    if mapping is None:
        return frame.rename({"source_code": "economy_id"})
    joined = frame.join(mapping, on="source_code", how="left")
    missing = joined.filter(pl.col("_declared").is_null())
    if missing.height:
        codes = missing.get_column("source_code").unique().sort().to_list()
        raise ValueError(f"unmapped IRENA source codes: {codes}")
    return joined.filter(pl.col("economy_id").is_not_null()).drop(
        "source_code", "_declared"
    )


def _long_source(
    frame: pl.DataFrame,
    *,
    indicator_id: str,
    unit: str,
) -> pl.DataFrame:
    return frame.with_columns(
        pl.lit(indicator_id, dtype=pl.String).alias("indicator_id"),
        pl.lit(unit, dtype=pl.String).alias("unit"),
        pl.lit(None, dtype=pl.Boolean).alias("retirement_or_revision"),
    ).select(*OUTPUT_SCHEMA)


def _capacity_additions(capacity: pl.DataFrame) -> pl.DataFrame:
    ordered = capacity.sort("economy_id", "year").with_columns(
        pl.col("year").shift(1).over("economy_id").alias("_prior_year"),
        pl.col("value").shift(1).over("economy_id").alias("_prior_value"),
    )
    consecutive = (
        (pl.col("year") - pl.col("_prior_year") == 1)
        & pl.col("value").is_not_null()
        & pl.col("_prior_value").is_not_null()
    )
    return (
        ordered.with_columns(
            pl.when(consecutive)
            .then(pl.col("value") - pl.col("_prior_value"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("value"),
            pl.when(consecutive)
            .then((pl.col("value") - pl.col("_prior_value")) < 0.0)
            .otherwise(pl.lit(None, dtype=pl.Boolean))
            .alias("retirement_or_revision"),
            pl.when(pl.col("_prior_year").is_null() | (pl.col("year") - pl.col("_prior_year") != 1))
            .then(pl.lit("not_computable_no_consecutive_prior"))
            .when(pl.col("value").is_null() | pl.col("_prior_value").is_null())
            .then(pl.lit("source_null"))
            .otherwise(pl.lit("derived_consecutive_capacity"))
            .alias("source_status"),
            pl.lit("renewable_capacity_additions_mw", dtype=pl.String).alias(
                "indicator_id"
            ),
        )
        .drop("_prior_year", "_prior_value")
        .select(*OUTPUT_SCHEMA)
    )


def normalize_irena_payloads(
    *,
    capacity_payloads: tuple[dict[str, Any], ...],
    generation_payloads: tuple[dict[str, Any], ...],
    share_payloads: tuple[dict[str, Any], ...] = (),
    economies: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Normalize approved IRENA cells and derive consecutive-year capacity additions."""

    mapping = _economy_mapping(economies)
    capacity_source = normalize_irena_capacity_payloads(capacity_payloads)
    generation_source = normalize_irena_generation_payloads(generation_payloads)
    capacity = _long_source(
        _map_frame(capacity_source, mapping),
        indicator_id="renewable_capacity_mw",
        unit="MW",
    )
    generation = _long_source(
        _map_frame(generation_source, mapping),
        indicator_id="renewable_generation_gwh",
        unit="GWh",
    )
    frames = [capacity, _capacity_additions(capacity), generation]
    if share_payloads:
        shares = normalize_irena_share_payloads(share_payloads)
        frames.append(_map_frame(shares, mapping).select(*OUTPUT_SCHEMA))
    output = pl.concat(frames).sort("economy_id", "year", "indicator_id")
    duplicates = output.group_by("economy_id", "year", "indicator_id").len().filter(
        pl.col("len") > 1
    ).height
    if duplicates:
        raise ValueError(f"duplicate IRENA economy-year-indicator keys: {duplicates}")
    return output


def _validate_indicator_registry(registry: pl.DataFrame) -> None:
    required = {
        "source_id",
        "source_field",
        "project_field",
        "unit",
        "start_year",
        "end_year",
        "status",
    }
    if not required <= set(registry.columns):
        raise ValueError(f"indicator registry lacks columns: {sorted(required)}")
    rows = registry.filter(
        (pl.col("source_id") == "irena")
        & pl.col("status").str.starts_with("approved")
    )
    observed = {
        str(row["source_field"]): (
            str(row["project_field"]),
            str(row["unit"]),
            int(row["start_year"]),
            int(row["end_year"]),
        )
        for row in rows.iter_rows(named=True)
    }
    expected = {
        "Country_ELECCAP_2026_H1": ("renewable_capacity_mw", "MW", 2000, 2025),
        "Country_ELECGEN_2025_H2": (
            "renewable_generation_gwh",
            "GWh",
            2000,
            2023,
        ),
        "RE-SHARE_2026_H1:capacity": (
            "renewable_capacity_share",
            "percent",
            2000,
            2025,
        ),
        "RE-SHARE_2026_H1:generation": (
            "renewable_generation_share",
            "percent",
            2000,
            2025,
        ),
    }
    if observed != expected:
        raise ValueError(f"IRENA indicator registry differs: {observed}")


def normalize_irena(
    paths: tuple[Path, ...],
    registry: pl.DataFrame,
    economies: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Normalize the registered IRENA data paths into the long L1 table."""

    _validate_indicator_registry(registry)
    capacity_paths = tuple(
        sorted(path for path in paths if path.name.startswith("Country_ELECCAP_"))
    )
    generation_paths = tuple(
        sorted(path for path in paths if path.name.startswith("Country_ELECGEN_"))
    )
    share_paths = tuple(
        sorted(path for path in paths if path.name.startswith("RE-SHARE_"))
    )
    if (
        len(capacity_paths) != 4
        or len(generation_paths) != 4
        or len(share_paths) != 1
        or len(paths) != 9
    ):
        raise ValueError(
            "IRENA approved file set must be 4 capacity, 4 generation, and 1 share"
        )
    return normalize_irena_payloads(
        capacity_payloads=tuple(
            json.loads(path.read_text(encoding="utf-8")) for path in capacity_paths
        ),
        generation_payloads=tuple(
            json.loads(path.read_text(encoding="utf-8")) for path in generation_paths
        ),
        share_payloads=tuple(
            json.loads(path.read_text(encoding="utf-8")) for path in share_paths
        ),
        economies=economies,
    )


def validate_irena_metadata(payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Require JSON-stat category codes and labels to agree with frozen metadata."""

    if payload.get("label") != metadata.get("title"):
        raise ValueError("IRENA dataset title differs from metadata")
    variables = metadata.get("variables")
    if not isinstance(variables, list) or not all(isinstance(item, dict) for item in variables):
        raise ValueError("IRENA metadata variables are invalid")
    by_code = {str(item.get("code")): item for item in variables}
    dimensions = payload.get("id")
    if not isinstance(dimensions, list) or set(dimensions) != set(by_code):
        raise ValueError("IRENA metadata dimension set changed")
    for dimension in dimensions:
        item = by_code[dimension]
        if item.get("text") != dimension:
            raise ValueError(f"IRENA metadata label changed: {dimension}")
        values = item.get("values")
        texts = item.get("valueTexts")
        if (
            not isinstance(values, list)
            or not isinstance(texts, list)
            or len(values) != len(texts)
        ):
            raise ValueError(f"IRENA metadata categories are invalid: {dimension}")
        expected = dict(zip((str(value) for value in values), (str(text) for text in texts), strict=True))
        raw_dimension = payload.get("dimension", {}).get(dimension)
        if not isinstance(raw_dimension, dict):
            raise ValueError(f"IRENA payload dimension is missing: {dimension}")
        categories = _ordered_categories(dimension, raw_dimension)
        observed = dict(categories)
        if any(code not in expected or expected[code] != label for code, label in categories):
            raise ValueError(f"IRENA category label changed: {dimension}")
        if dimension != "Year" and set(observed) != set(expected):
            raise ValueError(f"IRENA non-year category set changed: {dimension}")


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


def build_irena_table(
    *,
    capacity_paths: tuple[Path, ...],
    generation_paths: tuple[Path, ...],
    share_paths: tuple[Path, ...],
    capacity_metadata_path: Path,
    generation_metadata_path: Path,
    share_metadata_path: Path,
    registry: pl.DataFrame,
    economies: pl.DataFrame,
    destination: Path,
    contract_path: Path,
    inputs: tuple[InputArtifact, ...],
    build: BuildIdentity,
) -> IrenaBuildReport:
    _validate_indicator_registry(registry)
    capacity_payloads = tuple(
        json.loads(path.read_text(encoding="utf-8")) for path in capacity_paths
    )
    generation_payloads = tuple(
        json.loads(path.read_text(encoding="utf-8")) for path in generation_paths
    )
    share_payloads = tuple(
        json.loads(path.read_text(encoding="utf-8")) for path in share_paths
    )
    capacity_metadata = json.loads(capacity_metadata_path.read_text(encoding="utf-8"))
    generation_metadata = json.loads(generation_metadata_path.read_text(encoding="utf-8"))
    share_metadata = json.loads(share_metadata_path.read_text(encoding="utf-8"))
    for payload in capacity_payloads:
        validate_irena_metadata(payload, capacity_metadata)
    for payload in generation_payloads:
        validate_irena_metadata(payload, generation_metadata)
    for payload in share_payloads:
        validate_irena_metadata(payload, share_metadata)
    frame = normalize_irena_payloads(
        capacity_payloads=capacity_payloads,
        generation_payloads=generation_payloads,
        share_payloads=share_payloads,
        economies=economies,
    )
    manifest = write_authoritative_table(
        frame,
        _load_contract(contract_path),
        destination,
        inputs,
        build,
    )
    source_cells = sum(
        prod(int(value) for value in payload["size"])
        for payload in (*capacity_payloads, *generation_payloads, *share_payloads)
    )
    additions = frame.filter(
        pl.col("indicator_id") == "renewable_capacity_additions_mw"
    )
    return IrenaBuildReport(
        source_files=len(capacity_paths) + len(generation_paths) + len(share_paths),
        source_cells=source_cells,
        indicators=tuple(sorted(frame.get_column("indicator_id").unique().to_list())),
        retained_rows=frame.height,
        null_values=frame.get_column("value").null_count(),
        negative_additions=additions.filter(pl.col("value") < 0.0).height,
        duplicate_keys=0,
        output_path=str(destination.resolve()),
        output_bytes=manifest.bytes,
    )


def audit_irena_table(
    *, manifest_path: Path, registry: pl.DataFrame | None = None
) -> IrenaAuditReport:
    if registry is not None:
        _validate_indicator_registry(registry)
    manifest = verify_manifest(manifest_path)
    frame = pl.read_parquet(manifest.destination)
    expected_units = {
        "renewable_capacity_mw": "MW",
        "renewable_capacity_additions_mw": "MW",
        "renewable_generation_gwh": "GWh",
        "renewable_capacity_share": "percent",
        "renewable_generation_share": "percent",
    }
    expected_periods = {
        "renewable_capacity_mw": tuple(range(2000, 2026)),
        "renewable_capacity_additions_mw": tuple(range(2000, 2026)),
        "renewable_generation_gwh": tuple(range(2000, 2024)),
        "renewable_capacity_share": tuple(range(2000, 2026)),
        "renewable_generation_share": tuple(range(2000, 2026)),
    }
    indicators = tuple(sorted(frame.get_column("indicator_id").unique().to_list()))
    if indicators != tuple(sorted(expected_units)):
        raise RuntimeError(f"IRENA authoritative indicator set differs: {indicators}")
    periods = {
        indicator: tuple(
            sorted(
                frame.filter(pl.col("indicator_id") == indicator)
                .get_column("year")
                .unique()
                .to_list()
            )
        )
        for indicator in indicators
    }
    if periods != expected_periods:
        raise RuntimeError(f"IRENA indicator periods differ: {periods}")
    unit_mismatches = frame.filter(
        pl.struct("indicator_id", "unit").map_elements(
            lambda row: expected_units.get(row["indicator_id"]) != row["unit"],
            return_dtype=pl.Boolean,
        )
    ).height
    duplicates = frame.group_by("economy_id", "year", "indicator_id").len().filter(
        pl.col("len") > 1
    ).height
    additions = frame.filter(
        pl.col("indicator_id") == "renewable_capacity_additions_mw"
    )
    flag_mismatches = additions.filter(
        (pl.col("value").is_null() & pl.col("retirement_or_revision").is_not_null())
        | (
            pl.col("value").is_not_null()
            & (pl.col("retirement_or_revision") != (pl.col("value") < 0.0))
        )
    ).height + frame.filter(
        (pl.col("indicator_id") != "renewable_capacity_additions_mw")
        & pl.col("retirement_or_revision").is_not_null()
    ).height
    if duplicates or unit_mismatches or flag_mismatches:
        raise RuntimeError(
            "IRENA audit failed: "
            f"duplicates={duplicates}, units={unit_mismatches}, flags={flag_mismatches}"
        )
    return IrenaAuditReport(
        indicators=indicators,
        rows=frame.height,
        null_values=frame.get_column("value").null_count(),
        negative_additions=additions.filter(pl.col("value") < 0.0).height,
        retirement_flag_mismatches=flag_mismatches,
        duplicate_keys=duplicates,
        unit_mismatches=unit_mismatches,
        indicator_periods=periods,
        status="valid",
    )
