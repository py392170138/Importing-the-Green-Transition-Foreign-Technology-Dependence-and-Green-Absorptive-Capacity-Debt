"""Atomic, contract-checked Parquet artifacts and lineage manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

import polars as pl

from green_debt.storage import (
    enforce_construction_capacity,
    measure_layer_usage,
    sha256_file,
)


_REASON_CODE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_FLOAT_DTYPES = {"Float32", "Float64"}
_NUMERIC_PREFIXES = ("Int", "UInt", "Float", "Decimal")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class InputArtifact:
    path: str
    bytes: int
    sha256: str
    parent_manifest_sha256: str | None = None

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        manifest_path: Path | None = None,
    ) -> InputArtifact:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        sidecar = manifest_path
        if sidecar is None:
            candidate = resolved.with_name(f"{resolved.name}.manifest.json")
            sidecar = candidate if candidate.is_file() else None
        parent_hash = None
        if sidecar is not None:
            if not sidecar.is_file():
                raise FileNotFoundError(sidecar)
            parent_hash = sha256_file(sidecar)
        return cls(
            path=str(resolved),
            bytes=resolved.stat().st_size,
            sha256=sha256_file(resolved),
            parent_manifest_sha256=parent_hash,
        )


@dataclass(frozen=True)
class BuildIdentity:
    command: str
    code_commit: str
    created_at_utc: str = field(default_factory=_utc_now)


@dataclass(frozen=True)
class TableContract:
    table_id: str
    schema_version: str
    primary_key: tuple[str, ...]
    columns: dict[str, str]
    units: dict[str, str]
    period: tuple[int, int] | None = None
    zero_semantics: dict[str, str] = field(default_factory=dict)
    null_semantics: dict[str, str] = field(default_factory=dict)
    transformations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.table_id or not self.schema_version:
            raise ValueError("table_id and schema_version are required")
        if not self.primary_key or not set(self.primary_key) <= set(self.columns):
            raise ValueError("primary key must be non-empty and declared in columns")
        if not set(self.units) <= set(self.columns):
            raise ValueError("unit columns must be declared in columns")
        if not set(self.zero_semantics) <= set(self.columns):
            raise ValueError("zero-semantics columns must be declared in columns")
        if not set(self.null_semantics) <= set(self.columns):
            raise ValueError("null-semantics columns must be declared in columns")
        if self.period is not None and self.period[0] > self.period[1]:
            raise ValueError("contract period must be ascending")


@dataclass(frozen=True)
class TableManifest:
    table_id: str
    schema_version: str
    destination: str
    schema_path: str
    output_sha256: str
    schema_sha256: str
    rows: int
    bytes: int
    duplicate_primary_keys: int
    primary_key: tuple[str, ...]
    column_order: tuple[str, ...]
    columns: dict[str, str]
    units: dict[str, str]
    null_counts: dict[str, int]
    null_semantics: dict[str, str]
    zero_counts: dict[str, int]
    zero_semantics: dict[str, str]
    transformations: tuple[str, ...]
    period: tuple[int, int] | None
    coverage: dict[str, dict[str, int | float | str | None]]
    input_artifacts: tuple[InputArtifact, ...]
    input_hashes: tuple[str, ...]
    parent_manifest_hashes: tuple[str, ...]
    command: str
    code_commit: str
    created_at_utc: str


def _dtype_names(frame: pl.DataFrame) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in frame.schema.items()}


def _duplicate_primary_keys(frame: pl.DataFrame, key: tuple[str, ...]) -> int:
    return frame.group_by(list(key)).len().filter(pl.col("len") > 1).height


def _validate_frame(frame: pl.DataFrame, contract: TableContract) -> int:
    expected_names = tuple(contract.columns)
    if tuple(frame.columns) != expected_names:
        raise ValueError(
            f"table must have exact columns {expected_names}; got {tuple(frame.columns)}"
        )
    actual_dtypes = _dtype_names(frame)
    if actual_dtypes != contract.columns:
        raise ValueError(
            f"table dtypes do not match contract: expected {contract.columns}, "
            f"got {actual_dtypes}"
        )
    for name, dtype in actual_dtypes.items():
        if dtype not in _FLOAT_DTYPES:
            continue
        nonfinite = frame.select(
            (pl.col(name).is_not_null() & ~pl.col(name).is_finite()).sum()
        ).item()
        if nonfinite:
            raise ValueError(f"nonfinite values in {name}: {nonfinite}")
    null_key_counts = {
        name: frame.get_column(name).null_count()
        for name in contract.primary_key
        if frame.get_column(name).null_count()
    }
    if null_key_counts:
        raise ValueError(f"null primary key values: {null_key_counts}")
    if contract.null_semantics:
        undeclared = sorted(
            name
            for name in frame.columns
            if frame.get_column(name).null_count() and name not in contract.null_semantics
        )
        if undeclared:
            raise ValueError(f"undeclared null semantics for columns: {undeclared}")
    duplicate_count = _duplicate_primary_keys(frame, contract.primary_key)
    if duplicate_count:
        raise ValueError(
            f"duplicate primary key groups in {contract.table_id}: {duplicate_count}"
        )
    if contract.period is not None:
        if "year" not in frame.columns:
            raise ValueError("a declared period requires a year column")
        years = frame.get_column("year").drop_nulls()
        if not years.is_empty():
            minimum = int(years.min())
            maximum = int(years.max())
            if minimum < contract.period[0] or maximum > contract.period[1]:
                raise ValueError(
                    f"year outside declared period {contract.period}: {minimum}-{maximum}"
                )
    return duplicate_count


def _null_counts(frame: pl.DataFrame) -> dict[str, int]:
    return {name: int(frame.get_column(name).null_count()) for name in frame.columns}


def _zero_counts(frame: pl.DataFrame, contract: TableContract) -> dict[str, int]:
    counts: dict[str, int] = {}
    dtypes = _dtype_names(frame)
    for name in contract.zero_semantics:
        if not dtypes[name].startswith(_NUMERIC_PREFIXES):
            raise ValueError(f"declared zero count requires a numeric column: {name}")
        counts[name] = int(frame.select((pl.col(name) == 0).sum()).item())
    return counts


def _coverage(frame: pl.DataFrame) -> dict[str, dict[str, int | float | str | None]]:
    coverage: dict[str, dict[str, int | float | str | None]] = {}
    candidates = ("economy_id", "product_id", "hs6", "activity", "year")
    for name in candidates:
        if name not in frame.columns:
            continue
        series = frame.get_column(name).drop_nulls()
        summary: dict[str, int | float | str | None] = {
            "unique": int(series.n_unique()),
            "min": None,
            "max": None,
        }
        if not series.is_empty():
            minimum = series.min()
            maximum = series.max()
            if hasattr(minimum, "item"):
                minimum = minimum.item()
            if hasattr(maximum, "item"):
                maximum = maximum.item()
            summary["min"] = minimum
            summary["max"] = maximum
        coverage[name] = summary
    return coverage


def _json_type(dtype: str, *, nullable: bool) -> dict[str, Any]:
    if dtype == "String" or dtype.startswith(("Categorical", "Enum", "Date", "Datetime")):
        value: dict[str, Any] = {"type": "string"}
    elif dtype == "Boolean":
        value = {"type": "boolean"}
    elif dtype.startswith(("Int", "UInt")):
        value = {"type": "integer"}
    elif dtype.startswith(("Float", "Decimal")):
        value = {"type": "number"}
    else:
        value = {"description": f"Polars dtype {dtype}"}
    return {"anyOf": [value, {"type": "null"}]} if nullable else value


def _schema_payload(contract: TableContract) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for name, dtype in contract.columns.items():
        definition = _json_type(
            dtype,
            nullable=not contract.null_semantics or name in contract.null_semantics,
        )
        definition["x-polars-dtype"] = dtype
        if name in contract.units:
            definition["x-unit"] = contract.units[name]
        if name in contract.zero_semantics:
            definition["x-zero-semantics"] = contract.zero_semantics[name]
        if name in contract.null_semantics:
            definition["x-null-semantics"] = contract.null_semantics[name]
        properties[name] = definition
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": contract.table_id,
        "type": "object",
        "additionalProperties": False,
        "required": list(contract.columns),
        "properties": properties,
        "x-schema-version": contract.schema_version,
        "x-primary-key": list(contract.primary_key),
        "x-period": list(contract.period) if contract.period is not None else None,
        "x-transformations": list(contract.transformations),
    }


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _write_pending(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    if partial.exists():
        partial.unlink()
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return partial


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commit_files(pending_to_target: tuple[tuple[Path, Path], ...]) -> None:
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for _, target in pending_to_target:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = target.with_name(f"{target.name}.rollback")
                if backup.exists():
                    raise RuntimeError(f"stale rollback file blocks commit: {backup}")
                os.replace(target, backup)
                backups.append((backup, target))
        for pending, target in pending_to_target:
            os.replace(pending, target)
            installed.append(target)
        for directory in {target.parent for _, target in pending_to_target}:
            _fsync_directory(directory)
    except Exception:
        for target in reversed(installed):
            if target.exists():
                target.unlink()
        for backup, target in reversed(backups):
            if backup.exists():
                os.replace(backup, target)
        raise
    else:
        for backup, _ in backups:
            backup.unlink(missing_ok=True)


def _runtime_paths(destination: Path, table_id: str) -> tuple[Path, Path] | None:
    resolved = destination.resolve()
    parts = resolved.parts
    try:
        marker = len(parts) - 1 - parts[::-1].index("05_中间数据")
    except ValueError:
        return None
    data_root = Path(*parts[:marker])
    intermediate = data_root / "05_中间数据"
    return (
        intermediate / "schemas" / f"{table_id}.schema.json",
        intermediate / "manifests" / f"{table_id}.manifest.json",
    )


def _enforce_destination_capacity(destination: Path) -> None:
    runtime = _runtime_paths(destination, "capacity_probe")
    if runtime is None:
        return
    data_root = runtime[0].parents[2]
    usage = measure_layer_usage(data_root)
    enforce_construction_capacity(usage)


def _manifest_payload(manifest: TableManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    payload["primary_key"] = list(manifest.primary_key)
    payload["column_order"] = list(manifest.column_order)
    payload["transformations"] = list(manifest.transformations)
    payload["period"] = list(manifest.period) if manifest.period is not None else None
    payload["input_hashes"] = list(manifest.input_hashes)
    payload["parent_manifest_hashes"] = list(manifest.parent_manifest_hashes)
    return payload


def _manifest_from_payload(payload: Mapping[str, Any]) -> TableManifest:
    artifacts = tuple(
        InputArtifact(
            path=str(item["path"]),
            bytes=int(item["bytes"]),
            sha256=str(item["sha256"]),
            parent_manifest_sha256=(
                str(item["parent_manifest_sha256"])
                if item.get("parent_manifest_sha256") is not None
                else None
            ),
        )
        for item in payload["input_artifacts"]
    )
    period_value = payload.get("period")
    period = (
        (int(period_value[0]), int(period_value[1]))
        if period_value is not None
        else None
    )
    column_order = tuple(str(item) for item in payload["column_order"])
    raw_columns = payload["columns"]
    if set(column_order) != set(raw_columns):
        raise ValueError("manifest column_order does not match columns")
    return TableManifest(
        table_id=str(payload["table_id"]),
        schema_version=str(payload["schema_version"]),
        destination=str(payload["destination"]),
        schema_path=str(payload["schema_path"]),
        output_sha256=str(payload["output_sha256"]),
        schema_sha256=str(payload["schema_sha256"]),
        rows=int(payload["rows"]),
        bytes=int(payload["bytes"]),
        duplicate_primary_keys=int(payload["duplicate_primary_keys"]),
        primary_key=tuple(str(item) for item in payload["primary_key"]),
        column_order=column_order,
        columns={name: str(raw_columns[name]) for name in column_order},
        units={str(key): str(value) for key, value in payload["units"].items()},
        null_counts={
            str(key): int(value) for key, value in payload["null_counts"].items()
        },
        null_semantics={
            str(key): str(value)
            for key, value in payload.get("null_semantics", {}).items()
        },
        zero_counts={
            str(key): int(value) for key, value in payload["zero_counts"].items()
        },
        zero_semantics={
            str(key): str(value)
            for key, value in payload["zero_semantics"].items()
        },
        transformations=tuple(str(item) for item in payload["transformations"]),
        period=period,
        coverage={str(key): dict(value) for key, value in payload["coverage"].items()},
        input_artifacts=artifacts,
        input_hashes=tuple(str(item) for item in payload["input_hashes"]),
        parent_manifest_hashes=tuple(
            str(item) for item in payload["parent_manifest_hashes"]
        ),
        command=str(payload["command"]),
        code_commit=str(payload["code_commit"]),
        created_at_utc=str(payload["created_at_utc"]),
    )


def write_authoritative_table(
    frame: pl.DataFrame,
    contract: TableContract,
    destination: Path,
    inputs: tuple[InputArtifact, ...],
    build: BuildIdentity,
) -> TableManifest:
    """Validate, round-trip, and atomically publish Parquet plus sidecars."""

    _validate_frame(frame, contract)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    parquet_partial = destination.with_name(f"{destination.name}.partial")
    schema_path = destination.with_name(f"{destination.name}.schema.json")
    manifest_path = destination.with_name(f"{destination.name}.manifest.json")
    pending: list[Path] = [parquet_partial]
    try:
        parquet_partial.unlink(missing_ok=True)
        frame.write_parquet(parquet_partial)
        _fsync_file(parquet_partial)
        round_trip = pl.read_parquet(parquet_partial)
        _validate_frame(round_trip, contract)
        if not round_trip.equals(frame):
            raise ValueError("Parquet round trip changed table values or row order")
        _enforce_destination_capacity(parquet_partial)

        schema_bytes = _canonical_json(_schema_payload(contract))
        schema_partial = _write_pending(schema_path, schema_bytes)
        pending.append(schema_partial)
        schema_sha256 = sha256_file(schema_partial)
        output_sha256 = sha256_file(parquet_partial)
        input_hashes = tuple(sorted(item.sha256 for item in inputs))
        parent_hashes = tuple(
            sorted(
                item.parent_manifest_sha256
                for item in inputs
                if item.parent_manifest_sha256 is not None
            )
        )
        manifest = TableManifest(
            table_id=contract.table_id,
            schema_version=contract.schema_version,
            destination=str(destination),
            schema_path=str(schema_path),
            output_sha256=output_sha256,
            schema_sha256=schema_sha256,
            rows=frame.height,
            bytes=parquet_partial.stat().st_size,
            duplicate_primary_keys=0,
            primary_key=contract.primary_key,
            column_order=tuple(contract.columns),
            columns=dict(contract.columns),
            units=dict(contract.units),
            null_counts=_null_counts(frame),
            null_semantics=dict(contract.null_semantics),
            zero_counts=_zero_counts(frame, contract),
            zero_semantics=dict(contract.zero_semantics),
            transformations=contract.transformations,
            period=contract.period,
            coverage=_coverage(frame),
            input_artifacts=inputs,
            input_hashes=input_hashes,
            parent_manifest_hashes=parent_hashes,
            command=build.command,
            code_commit=build.code_commit,
            created_at_utc=build.created_at_utc,
        )
        manifest_bytes = _canonical_json(_manifest_payload(manifest))
        manifest_partial = _write_pending(manifest_path, manifest_bytes)
        pending.append(manifest_partial)

        pairs: list[tuple[Path, Path]] = [
            (parquet_partial, destination),
            (schema_partial, schema_path),
            (manifest_partial, manifest_path),
        ]
        runtime = _runtime_paths(destination, contract.table_id)
        if runtime is not None:
            central_schema, central_manifest = runtime
            if central_schema != schema_path:
                central_schema_partial = _write_pending(central_schema, schema_bytes)
                pending.append(central_schema_partial)
                pairs.append((central_schema_partial, central_schema))
            if central_manifest != manifest_path:
                central_manifest_partial = _write_pending(
                    central_manifest, manifest_bytes
                )
                pending.append(central_manifest_partial)
                pairs.append((central_manifest_partial, central_manifest))
        _commit_files(tuple(pairs))
        return verify_manifest(manifest_path)
    except Exception:
        for path in pending:
            path.unlink(missing_ok=True)
        raise


def verify_manifest(manifest_path: Path) -> TableManifest:
    """Verify sidecar structure, schema hash, output hash, and table contract."""

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be an object")
    try:
        manifest = _manifest_from_payload(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid manifest fields: {exc}") from exc
    output = Path(manifest.destination)
    schema = Path(manifest.schema_path)
    if not output.is_file():
        raise ValueError(f"manifest output is missing: {output}")
    if sha256_file(output) != manifest.output_sha256:
        raise ValueError("output hash mismatch")
    if output.stat().st_size != manifest.bytes:
        raise ValueError("output byte count mismatch")
    if not schema.is_file() or sha256_file(schema) != manifest.schema_sha256:
        raise ValueError("schema hash mismatch")
    frame = pl.read_parquet(output)
    contract = TableContract(
        table_id=manifest.table_id,
        schema_version=manifest.schema_version,
        primary_key=manifest.primary_key,
        columns=manifest.columns,
        units=manifest.units,
        period=manifest.period,
        zero_semantics=manifest.zero_semantics,
        null_semantics=manifest.null_semantics,
        transformations=manifest.transformations,
    )
    duplicate_count = _validate_frame(frame, contract)
    if duplicate_count != manifest.duplicate_primary_keys:
        raise ValueError("duplicate-key audit mismatch")
    if frame.height != manifest.rows:
        raise ValueError("output row count mismatch")
    if _null_counts(frame) != manifest.null_counts:
        raise ValueError("null-count audit mismatch")
    if _zero_counts(frame, contract) != manifest.zero_counts:
        raise ValueError("zero-count audit mismatch")
    return manifest


def quarantine_rows(
    frame: pl.DataFrame,
    reason_column: str,
    destination: Path,
) -> int:
    """Atomically write invalid rows after enforcing stable reason codes."""

    if reason_column not in frame.columns:
        raise ValueError(f"missing quarantine reason column: {reason_column}")
    reasons = frame.get_column(reason_column)
    if reasons.dtype != pl.String or reasons.null_count():
        raise ValueError("quarantine reason must be a non-null String")
    if any(_REASON_CODE.fullmatch(value) is None for value in reasons.to_list()):
        raise ValueError("quarantine rows require a stable reason code")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    try:
        partial.unlink(missing_ok=True)
        frame.write_parquet(partial)
        _fsync_file(partial)
        if not pl.read_parquet(partial).equals(frame):
            raise ValueError("quarantine Parquet round trip changed rows")
        _enforce_destination_capacity(partial)
        _commit_files(((partial, destination),))
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return frame.height


def manifest_is_current(
    manifest: Mapping[str, object],
    current_input_hashes: tuple[str, ...],
) -> bool:
    """Return whether a manifest was built from exactly the current inputs."""

    stored = manifest.get("input_hashes")
    if not isinstance(stored, list) or not all(isinstance(item, str) for item in stored):
        return False
    return sorted(stored) == sorted(current_input_hashes)
