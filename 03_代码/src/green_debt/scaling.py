"""Immutable robust scaling for the pre-outcome GAD component frame."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import polars as pl


FROZEN_COMPONENT_COLUMNS = (
    "green_import_intensity_raw",
    "green_import_complexity_raw",
    "gfvad_raw",
    "gsci_raw",
    "gud_raw",
    "grd_raw",
    "gnir_raw",
)

FROZEN_SAMPLE_SEMANTIC_FIELDS = (
    "table_id",
    "rows",
    "primary_key",
    "output_sha256",
    "schema_sha256",
    "period",
    "duplicate_primary_keys",
    "column_order",
    "columns",
    "null_semantics",
    "zero_semantics",
    "units",
    "transformations",
    "coverage",
)


@dataclass(frozen=True)
class ScalerRow:
    source_column: str
    center: float
    scale: float
    scale_method: str
    row_count: int
    economy_count: int


@dataclass(frozen=True)
class ScalerRegistry:
    rows: tuple[ScalerRow, ...]
    years: tuple[int, int]
    provisional_sample_manifest_hash: str
    canonical_hash: str

    def canonical_payload(self) -> dict[str, Any]:
        """Return the exact, hashable registry payload (the hash is excluded)."""

        return {
            "registry_version": "1.0.0",
            "columns": [row.source_column for row in self.rows],
            "years": list(self.years),
            "provisional_sample_manifest_hash": self.provisional_sample_manifest_hash,
            "rows": [asdict(row) for row in self.rows],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.canonical_payload()
        payload["canonical_hash"] = self.canonical_hash
        return payload


@dataclass(frozen=True)
class ScalerAnchor:
    """Git-frozen independent commitment to one registry file and lineage."""

    canonical_hash: str
    registry_file_sha256: str
    provisional_sample_manifest_hash: str
    implementation_commit: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_version": "1.0.0",
            "canonical_hash": self.canonical_hash,
            "registry_file_sha256": self.registry_file_sha256,
            "provisional_sample_manifest_hash": self.provisional_sample_manifest_hash,
            "implementation_commit": self.implementation_commit,
        }


def canonical_registry_hash(payload: dict[str, Any]) -> str:
    """Hash a precise UTF-8 JSON payload, stable across process invocations."""

    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def registry_from_dict(payload: dict[str, Any]) -> ScalerRegistry:
    try:
        rows = tuple(
            ScalerRow(
                source_column=str(row["source_column"]),
                center=float(row["center"]),
                scale=float(row["scale"]),
                scale_method=str(row["scale_method"]),
                row_count=int(row["row_count"]),
                economy_count=int(row["economy_count"]),
            )
            for row in payload["rows"]
        )
        years = tuple(int(year) for year in payload["years"])
        if len(years) != 2:
            raise ValueError
        registry = ScalerRegistry(
            rows=rows,
            years=(years[0], years[1]),
            provisional_sample_manifest_hash=str(payload["provisional_sample_manifest_hash"]),
            canonical_hash=str(payload["canonical_hash"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid scaler registry: {exc}") from exc
    _validate_registry(registry)
    return registry


def scaler_anchor_from_dict(payload: dict[str, Any]) -> ScalerAnchor:
    try:
        anchor = ScalerAnchor(
            canonical_hash=str(payload["canonical_hash"]),
            registry_file_sha256=str(payload["registry_file_sha256"]),
            provisional_sample_manifest_hash=str(payload["provisional_sample_manifest_hash"]),
            implementation_commit=str(payload["implementation_commit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid scaler anchor: {exc}") from exc
    for value in (
        anchor.canonical_hash,
        anchor.registry_file_sha256,
        anchor.provisional_sample_manifest_hash,
    ):
        _require_valid_manifest_hash(value)
    if len(anchor.implementation_commit) != 40 or any(
        character not in "0123456789abcdef" for character in anchor.implementation_commit
    ):
        raise ValueError("scaler anchor implementation commit must be a lowercase git SHA")
    return anchor


def create_scaler_anchor(
    registry: ScalerRegistry,
    *,
    registry_file_sha256: str,
    implementation_commit: str,
) -> ScalerAnchor:
    _validate_registry(registry)
    return scaler_anchor_from_dict(
        {
            "canonical_hash": registry.canonical_hash,
            "registry_file_sha256": registry_file_sha256,
            "provisional_sample_manifest_hash": registry.provisional_sample_manifest_hash,
            "implementation_commit": implementation_commit,
        }
    )


def verify_scaler_anchor(
    registry: ScalerRegistry,
    anchor: ScalerAnchor,
    *,
    registry_file_sha256: str,
    implementation_commit: str,
) -> None:
    _validate_registry(registry)
    anchor = scaler_anchor_from_dict(anchor.to_dict())
    if registry.canonical_hash != anchor.canonical_hash:
        raise ValueError("external anchor canonical hash mismatch")
    if registry_file_sha256 != anchor.registry_file_sha256:
        raise ValueError("external anchor registry file hash mismatch")
    if registry.provisional_sample_manifest_hash != anchor.provisional_sample_manifest_hash:
        raise ValueError("external anchor provisional sample manifest hash mismatch")
    if implementation_commit != anchor.implementation_commit:
        raise ValueError("external anchor implementation commit mismatch")


def _assert_unique_source(frame: pl.DataFrame, name: str, required: tuple[str, ...]) -> pl.DataFrame:
    missing = sorted(set(("economy_id", "year", *required)) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} source lacks columns: {missing}")
    selected = frame.select("economy_id", "year", *required)
    duplicates = selected.group_by("economy_id", "year").len().filter(pl.col("len") > 1).height
    if duplicates:
        raise ValueError(f"{name} source has duplicate economy-year keys: {duplicates}")
    return selected


def build_scaler_authority(
    provisional_sample: pl.DataFrame,
    trade: pl.DataFrame,
    tiva: pl.DataFrame,
    gsci: pl.DataFrame,
    supplier: pl.DataFrame,
    *,
    years: tuple[int, int] = (1996, 2024),
) -> pl.DataFrame:
    """Build the full core-economy calendar before left-joining raw components.

    This table is authoritative.  A component's unavailable period remains a
    null on its economy-year; it cannot delete initialization or Lite rows.
    """

    if years[0] > years[1]:
        raise ValueError("scaler authority years must be ascending")
    if not {"economy_id", "provisional_core"} <= set(provisional_sample.columns):
        raise ValueError("provisional sample lacks economy_id/provisional_core")
    core = provisional_sample.filter(pl.col("provisional_core")).select("economy_id")
    if core.group_by("economy_id").len().filter(pl.col("len") > 1).height:
        raise ValueError("provisional sample has duplicate core economy ids")
    calendar = pl.DataFrame({"year": list(range(years[0], years[1] + 1))}, schema={"year": pl.Int16})
    skeleton = core.join(calendar, how="cross").sort("economy_id", "year")
    sources = (
        _assert_unique_source(trade, "trade", ("green_import_intensity_raw", "green_import_complexity_raw", "gnir_raw")),
        _assert_unique_source(tiva, "TiVA", ("gfvad_raw",)),
        _assert_unique_source(gsci, "GSCI", ("gsci_raw",)),
        _assert_unique_source(supplier, "supplier", ("gud_raw", "grd_raw")),
    )
    output = skeleton
    for source in sources:
        output = output.join(source, on=("economy_id", "year"), how="left")
    return output.select("economy_id", "year", *FROZEN_COMPONENT_COLUMNS).sort("economy_id", "year")


def _require_valid_manifest_hash(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("provisional sample manifest hash must be a lowercase SHA-256")


def _validate_columns(columns: tuple[str, ...]) -> None:
    if len(columns) != len(set(columns)) or not columns:
        raise ValueError("scaler component columns must be non-empty and unique")
    if set(columns) == set(FROZEN_COMPONENT_COLUMNS) and columns != FROZEN_COMPONENT_COLUMNS:
        raise ValueError("scaler requires exact frozen component order")


def _validate_registry(registry: ScalerRegistry) -> None:
    _validate_columns(tuple(row.source_column for row in registry.rows))
    _require_valid_manifest_hash(registry.provisional_sample_manifest_hash)
    if registry.years != (2000, 2004) and tuple(row.source_column for row in registry.rows) == FROZEN_COMPONENT_COLUMNS:
        raise ValueError("frozen GAD scaler years must be exactly 2000-2004")
    if registry.years[0] > registry.years[1]:
        raise ValueError("scaler years must be ascending")
    for row in registry.rows:
        if not math.isfinite(row.center) or not math.isfinite(row.scale) or row.scale <= 0.0:
            raise ValueError(f"invalid finite scaler row: {row.source_column}")
        if row.scale_method not in {"mad_1_4826", "iqr_div_1_349"}:
            raise ValueError(f"invalid scaler method: {row.scale_method}")
        if row.row_count <= 0 or row.economy_count <= 0:
            raise ValueError(f"invalid scaler population: {row.source_column}")
    if canonical_registry_hash(registry.canonical_payload()) != registry.canonical_hash:
        raise ValueError("scaler canonical hash mismatch")


def fit_scaler(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    years: tuple[int, int],
    *,
    provisional_sample_manifest_hash: str,
) -> ScalerRegistry:
    """Fit one pooled robust scaler, never silently skipping unmatched values."""

    columns = tuple(columns)
    _validate_columns(columns)
    _require_valid_manifest_hash(provisional_sample_manifest_hash)
    if years[0] > years[1]:
        raise ValueError("scaler years must be ascending")
    required = {"economy_id", "year", *columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"scaler frame lacks columns: {missing}")
    matched = frame.filter(pl.col("year").is_between(*years))
    if matched.is_empty():
        raise ValueError("scaler has no baseline rows")
    nonfinite_or_null = matched.select(
        [
            (pl.col(column).is_null() | ~pl.col(column).is_finite()).any().alias(column)
            for column in columns
        ]
    ).row(0, named=True)
    bad = [column for column in columns if nonfinite_or_null[column]]
    if bad:
        raise ValueError(f"scaler requires complete matched rows: {bad}")
    duplicate_keys = matched.group_by("economy_id", "year").len().filter(pl.col("len") > 1).height
    if duplicate_keys:
        raise ValueError(f"scaler matched frame has duplicate economy-year keys: {duplicate_keys}")
    rows: list[ScalerRow] = []
    for column in columns:
        values = matched.get_column(column)
        center = float(values.median())
        deviations = (values - center).abs()
        mad = float(deviations.median())
        if mad > 0.0 and math.isfinite(mad):
            scale = 1.4826 * mad
            method = "mad_1_4826"
        else:
            q25 = float(values.quantile(0.25, interpolation="linear"))
            q75 = float(values.quantile(0.75, interpolation="linear"))
            iqr = q75 - q25
            if not math.isfinite(iqr) or iqr <= 0.0:
                raise ValueError(f"non-identifiable component {column}")
            scale = iqr / 1.349
            method = "iqr_div_1_349"
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"non-identifiable component {column}")
        rows.append(
            ScalerRow(
                source_column=column,
                center=center,
                scale=scale,
                scale_method=method,
                row_count=matched.height,
                economy_count=matched.get_column("economy_id").n_unique(),
            )
        )
    partial = ScalerRegistry(
        rows=tuple(rows),
        years=(int(years[0]), int(years[1])),
        provisional_sample_manifest_hash=provisional_sample_manifest_hash,
        canonical_hash="",
    )
    return ScalerRegistry(
        rows=partial.rows,
        years=partial.years,
        provisional_sample_manifest_hash=partial.provisional_sample_manifest_hash,
        canonical_hash=canonical_registry_hash(partial.canonical_payload()),
    )


def verify_scaler(
    registry: ScalerRegistry,
    *,
    expected_manifest_hash: str,
    expected_columns: tuple[str, ...] = FROZEN_COMPONENT_COLUMNS,
    expected_years: tuple[int, int] = (2000, 2004),
) -> None:
    """Fail closed if registry identity, composition, or canonical bytes drift."""

    _validate_registry(registry)
    if registry.provisional_sample_manifest_hash != expected_manifest_hash:
        raise ValueError("scaler provisional sample manifest hash mismatch")
    if tuple(row.source_column for row in registry.rows) != tuple(expected_columns):
        raise ValueError("scaler component columns mismatch")
    if registry.years != tuple(expected_years):
        raise ValueError("scaler years mismatch")


def _sample_semantic_projection(manifest: Mapping[str, object]) -> dict[str, object]:
    missing = sorted(set(FROZEN_SAMPLE_SEMANTIC_FIELDS) - set(manifest))
    # Older reviewed manifests omitted an empty null_semantics mapping.  Treat
    # absence and an explicit empty mapping identically, but no other field.
    if missing == ["null_semantics"]:
        manifest = {**manifest, "null_semantics": {}}
        missing = []
    if missing:
        raise ValueError(f"sample manifest lacks frozen semantic fields: {missing}")
    return {name: manifest[name] for name in FROZEN_SAMPLE_SEMANTIC_FIELDS}


def verify_frozen_sample_semantics(
    registry: ScalerRegistry,
    current_manifest: Mapping[str, object],
    approved_anchor: Mapping[str, object],
) -> None:
    """Allow metadata-only manifest drift while preserving the approved sample.

    The scaler continues to bind the exact approved manifest SHA.  A separate
    executable-input anchor records that manifest's stable table semantics;
    current manifests must match every field, including content/schema hashes,
    primary key, period, and null/zero rules, before the registry may be applied.
    """

    _validate_registry(registry)
    if approved_anchor.get("schema_version") != "1.0.0":
        raise ValueError("invalid frozen-scaler sample semantic anchor")
    approved_hash = approved_anchor.get("approved_manifest_sha256")
    if approved_hash != registry.provisional_sample_manifest_hash:
        raise ValueError("semantic anchor manifest identity mismatch")
    semantics = approved_anchor.get("manifest_semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError("invalid frozen-scaler sample semantic anchor")
    approved = _sample_semantic_projection(semantics)
    current = _sample_semantic_projection(current_manifest)
    if current != approved:
        changed = sorted(name for name in approved if current[name] != approved[name])
        raise ValueError(
            "approved frozen-scaler sample semantics mismatch: " + ",".join(changed)
        )


def apply_scaler(frame: pl.DataFrame, registry: ScalerRegistry) -> pl.DataFrame:
    """Append immutable ``z0_`` columns after independently verifying the hash."""

    _validate_registry(registry)
    missing = [row.source_column for row in registry.rows if row.source_column not in frame.columns]
    if missing:
        raise ValueError(f"scaler input lacks columns: {missing}")
    return frame.with_columns(
        [
            ((pl.col(row.source_column) - row.center) / row.scale).alias(f"z0_{row.source_column}")
            for row in registry.rows
        ]
    )


def verify_scaled_authority(frame: pl.DataFrame, registry: ScalerRegistry) -> None:
    """Check every published z0 value against immutable raw values and registry."""

    _validate_registry(registry)
    for row in registry.rows:
        raw = row.source_column
        scaled = f"z0_{raw}"
        if raw not in frame.columns or scaled not in frame.columns:
            raise ValueError(f"scaled authority lacks columns: {raw}, {scaled}")
        for value in frame.select(raw, scaled).iter_rows():
            raw_value, scaled_value = value
            if raw_value is None:
                if scaled_value is not None:
                    raise ValueError(f"raw null requires z0 null: {raw}")
                continue
            if not _finite_number(raw_value) or not _finite_number(scaled_value):
                raise ValueError(f"scaled authority contains nonfinite values: {raw}")
            expected = (float(raw_value) - row.center) / row.scale
            if not math.isclose(float(scaled_value), expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"scaled z0 mismatch: {raw}")


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))
