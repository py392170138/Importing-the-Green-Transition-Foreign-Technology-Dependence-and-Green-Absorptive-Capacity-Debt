"""OECD EPS and IFCMA robustness-only policy controls."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import polars as pl

from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    verify_manifest,
    write_authoritative_table,
)


EPS_COLUMNS = {
    "DATAFLOW",
    "REF_AREA",
    "FREQ",
    "MEASURE",
    "CLIM_POL",
    "TIME_PERIOD",
    "OBS_VALUE",
    "UNIT_MULT",
    "UNIT_MEASURE",
    "DECIMALS",
}
IFCMA_KEY = ("Country ISO", "Policy Instrument ID", "Instrument / subscheme")
IFCMA_STATUSES = {
    "In force": "in_force",
    "Ended": "ended",
    "Planned": "planned",
    "Non-existent": "non_existent",
    "N/A": "unknown",
}
OUTPUT_SCHEMA = {
    "economy_id": pl.String,
    "year": pl.Int16,
    "indicator_id": pl.String,
    "value": pl.Float64,
    "unit": pl.String,
    "source_status": pl.String,
    "robustness_only": pl.Boolean,
}


@dataclass(frozen=True)
class PolicyBuildReport:
    eps_source_rows: int
    eps_retained_rows: int
    ifcma_source_rows: int
    ifcma_embedded_headers_removed: int
    ifcma_invalid_identity_rows_excluded: int
    ifcma_duplicate_rows_removed: int
    ifcma_nonexistent_records: int
    retained_rows: int
    duplicate_keys: int
    policy_dependent_sample_exclusions: int
    output_path: str
    output_bytes: int


@dataclass(frozen=True)
class PolicyAuditReport:
    rows: int
    indicators: int
    economies: int
    duplicate_keys: int
    non_robustness_rows: int
    invalid_values: int
    policy_dependent_sample_exclusions: int
    status: str


def _require_columns(frame: pl.DataFrame, columns: set[str], *, source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} frame lacks columns: {missing}")


def _validated_eps(frame: pl.DataFrame, *, label: str) -> pl.DataFrame:
    _require_columns(frame, EPS_COLUMNS, source=f"EPS {label}")
    output = frame.select(
        pl.col("DATAFLOW").cast(pl.String, strict=True).str.strip_chars(),
        pl.col("REF_AREA").cast(pl.String, strict=True).str.strip_chars(),
        pl.col("FREQ").cast(pl.String, strict=True).str.strip_chars(),
        pl.col("MEASURE").cast(pl.String, strict=True).str.strip_chars(),
        pl.col("CLIM_POL").cast(pl.String, strict=True).str.strip_chars(),
        pl.col("TIME_PERIOD").cast(pl.Int16, strict=True),
        pl.col("OBS_VALUE").cast(pl.Float64, strict=True),
        pl.col("UNIT_MULT").cast(pl.Int8, strict=True),
        pl.col("UNIT_MEASURE").cast(pl.String, strict=True).str.strip_chars(),
        pl.col("DECIMALS").cast(pl.Int8, strict=True),
    )
    invalid_levels = output.filter(
        (pl.col("FREQ") != "A")
        | (pl.col("MEASURE") != "POL_STRINGENCY")
        | (pl.col("UNIT_MULT") != 0)
        | (pl.col("UNIT_MEASURE") != "0_TO_6")
        | pl.col("DATAFLOW").is_null()
        | (pl.col("DATAFLOW") == "")
    ).height
    if invalid_levels:
        raise ValueError(f"EPS {label} dimension or unit levels changed")
    invalid_values = output.filter(
        pl.col("OBS_VALUE").is_null()
        | ~pl.col("OBS_VALUE").is_finite()
        | (pl.col("OBS_VALUE") < 0.0)
        | (pl.col("OBS_VALUE") > 6.0)
    ).height
    if invalid_values:
        raise ValueError(f"EPS {label} has invalid 0-to-6 values: {invalid_values}")
    duplicates = output.group_by("REF_AREA", "TIME_PERIOD", "CLIM_POL").len().filter(
        pl.col("len") > 1
    ).height
    if duplicates:
        raise ValueError(f"EPS {label} has duplicate country-year-policy keys")
    return output


def normalize_eps(
    composite: pl.DataFrame, components: pl.DataFrame
) -> pl.DataFrame:
    """Normalize the composite EPS and its non-duplicated component panel."""

    main = _validated_eps(composite, label="composite")
    detail = _validated_eps(components, label="components")
    if main.get_column("CLIM_POL").unique().to_list() != ["EPS"]:
        raise ValueError("EPS composite file must contain only CLIM_POL=EPS")
    embedded = detail.filter(pl.col("CLIM_POL") == "EPS").select(
        "REF_AREA", "TIME_PERIOD", pl.col("OBS_VALUE").alias("component_value")
    )
    comparison = main.select(
        "REF_AREA", "TIME_PERIOD", pl.col("OBS_VALUE").alias("composite_value")
    ).join(embedded, on=["REF_AREA", "TIME_PERIOD"], how="full", coalesce=True)
    if comparison.filter(
        pl.col("composite_value").is_null()
        | pl.col("component_value").is_null()
        | ((pl.col("composite_value") - pl.col("component_value")).abs() > 1e-12)
    ).height:
        raise ValueError("EPS component file's embedded composite differs")
    years = tuple(sorted(main.get_column("TIME_PERIOD").unique().to_list()))
    if years != tuple(range(1990, 2021)):
        raise ValueError(f"EPS composite period changed: {years}")

    main_output = main.select(
        pl.col("REF_AREA").alias("economy_id"),
        pl.col("TIME_PERIOD").alias("year"),
        pl.lit("environmental_policy_stringency_index").alias("indicator_id"),
        pl.col("OBS_VALUE").alias("value"),
        pl.lit("index_0_to_6").alias("unit"),
        pl.lit("reported").alias("source_status"),
        pl.lit(True).alias("robustness_only"),
    )
    component_output = detail.filter(pl.col("CLIM_POL") != "EPS").select(
        pl.col("REF_AREA").alias("economy_id"),
        pl.col("TIME_PERIOD").alias("year"),
        (
            pl.lit("environmental_policy_component:")
            + pl.col("CLIM_POL").str.to_lowercase()
        ).alias("indicator_id"),
        pl.col("OBS_VALUE").alias("value"),
        pl.lit("index_0_to_6").alias("unit"),
        pl.lit("reported").alias("source_status"),
        pl.lit(True).alias("robustness_only"),
    )
    return pl.concat((main_output, component_output)).cast(OUTPUT_SCHEMA).sort(
        "economy_id", "year", "indicator_id"
    )


def _clean_ifcma(frame: pl.DataFrame) -> tuple[pl.DataFrame, int, int, int]:
    _require_columns(frame, {*IFCMA_KEY, "Status"}, source="IFCMA")
    selected = frame.select(
        *(pl.col(name).cast(pl.String, strict=True).str.strip_chars() for name in IFCMA_KEY),
        pl.col("Status").cast(pl.String, strict=True).str.strip_chars(),
    )
    embedded_header = (
        (pl.col("Country ISO") == "Country ISO")
        & (pl.col("Policy Instrument ID") == "Policy Instrument ID")
        & (pl.col("Instrument / subscheme") == "Instrument / subscheme")
    )
    headers = selected.filter(embedded_header).height
    selected = selected.filter(~embedded_header)
    invalid_identity = pl.any_horizontal(
        *(
            pl.col(name).is_null() | (pl.col(name) == "")
            for name in (*IFCMA_KEY, "Status")
        )
    )
    invalid_rows = selected.filter(invalid_identity).height
    selected = selected.filter(~invalid_identity)
    unknown_statuses = sorted(set(selected.get_column("Status").unique()) - set(IFCMA_STATUSES))
    if unknown_statuses:
        raise ValueError(f"IFCMA status categories changed: {unknown_statuses}")
    conflicts = selected.group_by(*IFCMA_KEY).agg(
        pl.col("Status").n_unique().alias("status_versions")
    ).filter(pl.col("status_versions") > 1)
    if conflicts.height:
        raise ValueError(f"IFCMA duplicate keys have conflicting status: {conflicts.height}")
    deduplicated = selected.unique(subset=list(IFCMA_KEY), keep="first", maintain_order=True)
    return (
        deduplicated,
        headers,
        invalid_rows,
        selected.height - deduplicated.height,
    )


def normalize_ifcma_snapshot(
    frame: pl.DataFrame, *, snapshot_year: int = 2026
) -> pl.DataFrame:
    """Count deduplicated instrument/subscheme records at the frozen snapshot."""

    if not 1990 <= snapshot_year <= 2100:
        raise ValueError("IFCMA snapshot year is outside the supported range")
    deduplicated, _, _, _ = _clean_ifcma(frame)
    rows: list[dict[str, object]] = []
    for country in sorted(deduplicated.get_column("Country ISO").unique()):
        country_rows = deduplicated.filter(pl.col("Country ISO") == country)
        enacted = country_rows.filter(pl.col("Status") != "Non-existent").height
        rows.append(
            {
                "economy_id": country,
                "year": snapshot_year,
                "indicator_id": "climate_policy_instrument_count",
                "value": float(enacted),
                "unit": "count",
                "source_status": "snapshot_enacted_records",
                "robustness_only": True,
            }
        )
        for status, slug in IFCMA_STATUSES.items():
            rows.append(
                {
                    "economy_id": country,
                    "year": snapshot_year,
                    "indicator_id": f"climate_policy_status:{slug}_count",
                    "value": float(country_rows.filter(pl.col("Status") == status).height),
                    "unit": "count",
                    "source_status": f"snapshot_status:{status}",
                    "robustness_only": True,
                }
            )
    return pl.DataFrame(rows, schema=OUTPUT_SCHEMA).sort(
        "economy_id", "year", "indicator_id"
    )


def _map_economies(
    frame: pl.DataFrame, economies: pl.DataFrame, *, source: str
) -> pl.DataFrame:
    _require_columns(economies, {"source_code", "economy_id"}, source=source)
    mapping = economies.select(
        pl.col("source_code").cast(pl.String).str.strip_chars().alias("_source_code"),
        pl.col("economy_id").cast(pl.String).str.strip_chars().alias("_economy_id"),
    ).with_columns(pl.lit(True).alias("_declared"))
    if mapping.group_by("_source_code").len().filter(pl.col("len") > 1).height:
        raise ValueError(f"{source} economy mapping has duplicate source codes")
    joined = frame.rename({"economy_id": "_source_code"}).join(
        mapping, on="_source_code", how="left"
    )
    missing = joined.filter(pl.col("_declared").is_null())
    if missing.height:
        codes = missing.get_column("_source_code").unique().sort().to_list()
        raise ValueError(f"unmapped {source} source codes: {codes}")
    return (
        joined.filter(pl.col("_economy_id").is_not_null())
        .drop("_source_code", "_declared")
        .rename({"_economy_id": "economy_id"})
        .select(*OUTPUT_SCHEMA)
    )


def attach_policy_controls(
    sample: pl.DataFrame, policy: pl.DataFrame
) -> pl.DataFrame:
    """Left-attach policy values without altering eligibility or row support."""

    _require_columns(sample, {"economy_id", "year"}, source="sample")
    _require_columns(
        policy,
        {"economy_id", "year", "indicator_id", "value", "robustness_only"},
        source="policy",
    )
    if policy.filter(~pl.col("robustness_only")).height:
        raise ValueError("policy controls must all be robustness_only")
    if policy.group_by("economy_id", "year", "indicator_id").len().filter(
        pl.col("len") > 1
    ).height:
        raise ValueError("policy controls have duplicate economy-year indicators")
    protected = {
        column: sample.get_column(column).to_list()
        for column in sample.columns
        if column.endswith("sample_candidate") or column.endswith("eligible")
    }
    wide = policy.select("economy_id", "year", "indicator_id", "value").pivot(
        on="indicator_id",
        index=["economy_id", "year"],
        values="value",
    )
    wide = wide.rename(
        {
            column: f"policy__{column}"
            for column in wide.columns
            if column not in {"economy_id", "year"}
        }
    )
    output = sample.join(wide, on=["economy_id", "year"], how="left", validate="m:1")
    if output.height != sample.height:
        raise RuntimeError("policy attachment changed sample row count")
    for column, values in protected.items():
        if output.get_column(column).to_list() != values:
            raise RuntimeError(f"policy attachment changed {column}")
    return output


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


def build_policy_table(
    *,
    eps_composite_path: Path,
    eps_components_path: Path,
    ifcma_path: Path,
    eps_economies: pl.DataFrame,
    ifcma_economies: pl.DataFrame,
    destination: Path,
    contract_path: Path,
    inputs: tuple[InputArtifact, ...],
    build: BuildIdentity,
) -> PolicyBuildReport:
    eps_schema = {
        "REF_AREA": pl.String,
        "TIME_PERIOD": pl.Int16,
        "OBS_VALUE": pl.Float64,
        "UNIT_MULT": pl.Int8,
        "DECIMALS": pl.Int8,
    }
    composite = pl.read_csv(
        eps_composite_path,
        schema_overrides=eps_schema,
        infer_schema_length=10_000,
    )
    components = pl.read_csv(
        eps_components_path,
        schema_overrides=eps_schema,
        infer_schema_length=10_000,
    )
    ifcma = pl.read_csv(
        ifcma_path,
        columns=[*IFCMA_KEY, "Status"],
        schema_overrides={name: pl.String for name in (*IFCMA_KEY, "Status")},
        encoding="utf8-lossy",
        infer_schema_length=10_000,
    )
    deduplicated, headers, invalid_rows, duplicate_rows = _clean_ifcma(ifcma)
    eps_output = _map_economies(
        normalize_eps(composite, components), eps_economies, source="OECD EPS"
    )
    ifcma_output = _map_economies(
        normalize_ifcma_snapshot(ifcma, snapshot_year=2026),
        ifcma_economies,
        source="OECD IFCMA",
    )
    output = pl.concat((eps_output, ifcma_output)).sort(
        "economy_id", "year", "indicator_id"
    )
    duplicates = output.group_by("economy_id", "year", "indicator_id").len().filter(
        pl.col("len") > 1
    ).height
    if duplicates:
        raise ValueError(f"duplicate policy economy-year indicators: {duplicates}")
    manifest = write_authoritative_table(
        output, _load_contract(contract_path), destination, inputs, build
    )
    return PolicyBuildReport(
        eps_source_rows=composite.height + components.height,
        eps_retained_rows=eps_output.height,
        ifcma_source_rows=ifcma.height,
        ifcma_embedded_headers_removed=headers,
        ifcma_invalid_identity_rows_excluded=invalid_rows,
        ifcma_duplicate_rows_removed=duplicate_rows,
        ifcma_nonexistent_records=deduplicated.filter(
            pl.col("Status") == "Non-existent"
        ).height,
        retained_rows=output.height,
        duplicate_keys=duplicates,
        policy_dependent_sample_exclusions=0,
        output_path=str(destination.resolve()),
        output_bytes=manifest.bytes,
    )


def audit_policy_table(*, manifest_path: Path) -> PolicyAuditReport:
    manifest = verify_manifest(manifest_path)
    frame = pl.read_parquet(manifest.destination)
    duplicates = frame.group_by("economy_id", "year", "indicator_id").len().filter(
        pl.col("len") > 1
    ).height
    non_robustness = frame.filter(~pl.col("robustness_only")).height
    invalid = frame.filter(
        pl.col("value").is_null()
        | ~pl.col("value").is_finite()
        | (pl.col("value") < 0.0)
        | (
            (pl.col("unit") == "index_0_to_6")
            & (pl.col("value") > 6.0)
        )
        | (
            (pl.col("unit") == "count")
            & (pl.col("value") != pl.col("value").floor())
        )
    ).height
    if duplicates or non_robustness or invalid:
        raise RuntimeError(
            "policy audit failed: "
            f"duplicates={duplicates}, non_robustness={non_robustness}, "
            f"invalid={invalid}"
        )
    return PolicyAuditReport(
        rows=frame.height,
        indicators=frame.get_column("indicator_id").n_unique(),
        economies=frame.get_column("economy_id").n_unique(),
        duplicate_keys=duplicates,
        non_robustness_rows=non_robustness,
        invalid_values=invalid,
        policy_dependent_sample_exclusions=0,
        status="valid",
    )
