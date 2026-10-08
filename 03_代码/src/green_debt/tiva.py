"""TiVA baseline activity weights used by GVC-dependent measures."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import polars as pl


WEIGHT_OUTPUT_SCHEMA = {
    "economy_id": pl.String,
    "activity": pl.String,
    "weight_version": pl.String,
    "activity_weight": pl.Float64,
    "baseline_mean_prod_usd_millions": pl.Float64,
    "baseline_years_observed": pl.UInt8,
    "frozen_baseline_start": pl.Int16,
    "frozen_baseline_end": pl.Int16,
    "unit": pl.String,
    "missing_reason": pl.String,
}

OFFICIAL_LEAF_ACTIVITIES = frozenset(
    (
        "A01", "A02", "A03", "B05", "B06", "B07", "B08", "B09",
        "C10T12", "C13T15", "C16", "C17_18", "C19", "C20", "C21",
        "C22", "C23", "C241_2431", "C242_2432", "C25", "C26", "C27",
        "C28", "C29", "C301", "C302T309", "C31T33", "D", "E", "F",
        "G", "H49", "H50", "H51", "H52", "H53", "I", "J58T60", "J61",
        "J62_63", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    )
)

_LEGACY_INDICATORS = {
    "PROD": "prod_level",
    "DFD_FVA": "dfd_fva_level",
    "FD_VA": "fd_va_level",
    "EXGR_DVA": "exgr_dva_share",
}
_MEASURE_SCHEMA = {
    "economy_id": pl.String,
    "year": pl.Int16,
    "specification_id": pl.String,
    "gfvad_raw": pl.Float64,
    "dvashare_raw": pl.Float64,
    "gfvad_missing_reason": pl.String,
    "dvashare_missing_reason": pl.String,
    "gfvad_out_of_range": pl.Boolean,
    "dvashare_out_of_range": pl.Boolean,
}


@dataclass(frozen=True)
class ActivitySets:
    """Frozen TiVA subsets.  All members must be official 50-leaf activities."""

    confirmatory: tuple[str, ...]
    broad: tuple[str, ...]
    equipment_only: tuple[str, ...]


@dataclass(frozen=True)
class TivaMeasures:
    """Authoritative raw and bounded-ratio TiVA measures, plus frozen weights."""

    confirmatory: pl.DataFrame
    broad: pl.DataFrame
    equipment_only: pl.DataFrame
    equal_industry_weight: pl.DataFrame
    bounded_confirmatory: pl.DataFrame
    table: pl.DataFrame
    weights: pl.DataFrame


def _normalized_measure_input(frame: pl.DataFrame) -> pl.DataFrame:
    """Accept one documented input dialect and immediately normalize it.

    Legacy ``measure/obs_value`` exists only for pure-function fixtures.  No
    caller can accidentally combine it with the current production
    ``indicator_id/value`` dialect, and returned data never exposes legacy names.
    """

    current = {"indicator_id", "value"}
    legacy = {"measure", "obs_value"}
    has_current = current <= set(frame.columns)
    has_legacy = legacy <= set(frame.columns)
    if has_current and has_legacy:
        raise ValueError("mixed TiVA input schemas are not allowed")
    if not has_current and not has_legacy:
        raise ValueError("TiVA input requires indicator_id/value or measure/obs_value")
    _require_columns(frame, {"economy_id", "activity", "year"})
    if has_legacy:
        unknown = set(frame.get_column("measure").drop_nulls().unique()) - set(_LEGACY_INDICATORS)
        if unknown:
            raise ValueError(f"unknown legacy TiVA measures: {sorted(unknown)}")
        output = frame.select(
            pl.col("economy_id").cast(pl.String),
            pl.col("activity").cast(pl.String),
            pl.col("year").cast(pl.Int16),
            pl.col("measure").replace_strict(_LEGACY_INDICATORS).alias("indicator_id"),
            pl.col("obs_value").cast(pl.Float64).alias("value"),
        )
    else:
        output = frame.select(
            pl.col("economy_id").cast(pl.String),
            pl.col("activity").cast(pl.String),
            pl.col("year").cast(pl.Int16),
            pl.col("indicator_id").cast(pl.String),
            pl.col("value").cast(pl.Float64),
        )
    if output.filter(
        pl.col("economy_id").is_null() | pl.col("activity").is_null() | pl.col("year").is_null()
    ).height:
        raise ValueError("TiVA keys must be non-null")
    unknown_activities = set(output.get_column("activity").unique()) - OFFICIAL_LEAF_ACTIVITIES
    if unknown_activities:
        raise ValueError(f"non-leaf or unregistered TiVA activities: {sorted(unknown_activities)}")
    duplicates = output.group_by("economy_id", "activity", "year", "indicator_id").len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError(f"duplicate TiVA activity indicator keys: {duplicates.height}")
    return output


def _validate_activity_sets(sets: ActivitySets) -> dict[str, tuple[str, ...]]:
    variants = {
        "confirmatory_prod_weight": sets.confirmatory,
        "broad_prod_weight": sets.broad,
        "equipment_only_prod_weight": sets.equipment_only,
    }
    for name, activities in variants.items():
        if not activities or len(activities) != len(set(activities)):
            raise ValueError(f"invalid TiVA activity set: {name}")
        unregistered = set(activities) - OFFICIAL_LEAF_ACTIVITIES
        if unregistered:
            raise ValueError(f"TiVA set contains non-leaf activities: {sorted(unregistered)}")
    return variants


def _finite(value: object) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def _measure_rows(
    normalized: pl.DataFrame,
    weights: pl.DataFrame,
    *,
    specification_id: str,
    weight_version: str | None = None,
    bounded: bool,
) -> list[dict[str, Any]]:
    version = weight_version or (
        "equal_weight" if specification_id == "equal_industry_weight" else specification_id
    )
    selected = weights.filter(pl.col("weight_version") == version)
    activities_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in selected.iter_rows(named=True):
        activities_by_group.setdefault(str(row["economy_id"]), []).append(row)
    value_map = {
        (str(row["economy_id"]), int(row["year"]), str(row["activity"]), str(row["indicator_id"])): row["value"]
        for row in normalized.iter_rows(named=True)
    }
    years_by_economy: dict[str, set[int]] = {}
    for row in normalized.iter_rows(named=True):
        years_by_economy.setdefault(str(row["economy_id"]), set()).add(int(row["year"]))
    rows: list[dict[str, Any]] = []
    for economy_id, activities in sorted(activities_by_group.items()):
        for year in sorted(years_by_economy.get(economy_id, set())):
            invalid_weight = any(item["activity_weight"] is None for item in activities)
            gfvad_reason = next((str(item["missing_reason"]) for item in activities if item["missing_reason"] is not None), None)
            dva_reason = gfvad_reason
            gfvad_value: float | None = None
            dva_value: float | None = None
            gfvad_out = False
            dva_out = False
            if invalid_weight:
                gfvad_reason = gfvad_reason or "invalid_frozen_weight"
                dva_reason = dva_reason or "invalid_frozen_weight"
            else:
                gfvad_parts: list[float] = []
                dva_parts: list[float] = []
                for item in activities:
                    activity = str(item["activity"])
                    weight = float(item["activity_weight"])
                    dfd = value_map.get((economy_id, year, activity, "dfd_fva_level"))
                    fd = value_map.get((economy_id, year, activity, "fd_va_level"))
                    dva = value_map.get((economy_id, year, activity, "exgr_dva_share"))
                    if not _finite(dfd) or not _finite(fd):
                        gfvad_reason = "missing_required_activity_value"
                    elif float(fd) <= 0.0:
                        gfvad_reason = "invalid_fd_va_denominator"
                    elif gfvad_reason is None:
                        ratio = float(dfd) / float(fd)
                        gfvad_parts.append(weight * ratio)
                        gfvad_out = gfvad_out or ratio < 0.0 or ratio > 1.0
                    if not _finite(dva):
                        dva_reason = "missing_required_activity_value"
                    elif dva_reason is None:
                        share = float(dva)
                        dva_parts.append(weight * share)
                        dva_out = dva_out or share < 0.0 or share > 100.0
                if gfvad_reason is None:
                    gfvad_value = sum(gfvad_parts)
                    if bounded and gfvad_out:
                        gfvad_value = None
                        gfvad_reason = "out_of_range_ratio_in_bounded_robustness"
                if dva_reason is None:
                    dva_value = sum(dva_parts)
                    if bounded and dva_out:
                        dva_value = None
                        dva_reason = "out_of_range_ratio_in_bounded_robustness"
            rows.append(
                {
                    "economy_id": economy_id,
                    "year": year,
                    "specification_id": specification_id,
                    "gfvad_raw": gfvad_value,
                    "dvashare_raw": dva_value,
                    "gfvad_missing_reason": gfvad_reason,
                    "dvashare_missing_reason": dva_reason,
                    "gfvad_out_of_range": gfvad_out,
                    "dvashare_out_of_range": dva_out,
                }
            )
    return rows


def build_tiva_measures(frame: pl.DataFrame, activity_sets: ActivitySets) -> TivaMeasures:
    """Calculate un-clipped TiVA ratios with strict frozen-weight completeness.

    A required industry's missing value invalidates that entire economy-year
    measure.  It is deliberately never removed from the denominator.
    """

    normalized = _normalized_measure_input(frame)
    variants = _validate_activity_sets(activity_sets)
    weights = build_activity_weights(
        normalized,
        variants=variants,
        equal_variant_activities=activity_sets.confirmatory,
    )
    grouped: dict[str, pl.DataFrame] = {}
    for specification in (*variants, "equal_industry_weight"):
        grouped[specification] = pl.DataFrame(
            _measure_rows(normalized, weights, specification_id=specification, bounded=False),
            schema=_MEASURE_SCHEMA,
        ).sort("economy_id", "year", "specification_id")
    bounded = pl.DataFrame(
        _measure_rows(
            normalized,
            weights,
            specification_id="bounded_confirmatory_prod_weight",
            weight_version="confirmatory_prod_weight",
            bounded=True,
        ),
        schema=_MEASURE_SCHEMA,
    ).sort("economy_id", "year", "specification_id")
    table = pl.concat(tuple(grouped.values())).sort("economy_id", "year", "specification_id")
    return TivaMeasures(
        confirmatory=grouped["confirmatory_prod_weight"],
        broad=grouped["broad_prod_weight"],
        equipment_only=grouped["equipment_only_prod_weight"],
        equal_industry_weight=grouped["equal_industry_weight"],
        bounded_confirmatory=bounded,
        table=table,
        weights=weights,
    )


def _require_columns(frame: pl.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"TiVA PROD frame lacks columns: {missing}")


def build_activity_weights(
    prod: pl.DataFrame,
    *,
    variants: Mapping[str, tuple[str, ...]],
    equal_variant_activities: tuple[str, ...],
    baseline_period: tuple[int, int] = (2000, 2004),
) -> pl.DataFrame:
    """Freeze country-specific PROD weights and a same-set equal-weight version."""

    _require_columns(
        prod, {"economy_id", "year", "activity", "indicator_id", "value"}
    )
    if baseline_period[0] > baseline_period[1]:
        raise ValueError("TiVA weight baseline period must be ascending")
    expected_years = baseline_period[1] - baseline_period[0] + 1
    if not variants or not equal_variant_activities:
        raise ValueError("TiVA weight variants must be non-empty")
    normalized_variants: dict[str, tuple[str, ...]] = {}
    for name, activities in variants.items():
        ordered = tuple(dict.fromkeys(activities))
        if not name or not ordered or len(ordered) != len(activities):
            raise ValueError(f"invalid TiVA weight variant: {name!r}")
        normalized_variants[name] = ordered
    equal_activities = tuple(dict.fromkeys(equal_variant_activities))
    if len(equal_activities) != len(equal_variant_activities):
        raise ValueError("equal-weight TiVA activities contain duplicates")

    baseline = (
        prod.filter(
            (pl.col("indicator_id") == "prod_level")
            & pl.col("year").is_between(*baseline_period)
        )
        .select(
            pl.col("economy_id").cast(pl.String),
            pl.col("year").cast(pl.Int16),
            pl.col("activity").cast(pl.String),
            pl.col("value").cast(pl.Float64),
        )
    )
    if baseline.filter(
        pl.col("value").is_not_null()
        & (~pl.col("value").is_finite() | (pl.col("value") < 0.0))
    ).height:
        raise ValueError("TiVA baseline PROD values must be finite and nonnegative")
    duplicates = baseline.group_by("economy_id", "year", "activity").len().filter(
        pl.col("len") > 1
    ).height
    if duplicates:
        raise ValueError(f"duplicate TiVA baseline PROD keys: {duplicates}")
    summary = baseline.group_by("economy_id", "activity").agg(
        pl.col("value").mean().alias("baseline_mean"),
        pl.col("value").count().cast(pl.UInt8).alias("years_observed"),
    )
    by_economy: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summary.iter_rows(named=True):
        by_economy.setdefault(str(row["economy_id"]), {})[
            str(row["activity"])
        ] = row

    rows: list[dict[str, Any]] = []
    all_variants = {
        **normalized_variants,
        "equal_weight": equal_activities,
    }
    for economy_id in sorted(by_economy):
        values = by_economy[economy_id]
        for version, activities in all_variants.items():
            selected = [values.get(activity) for activity in activities]
            complete = all(
                item is not None
                and item["baseline_mean"] is not None
                and int(item["years_observed"]) == expected_years
                for item in selected
            )
            denominator = (
                sum(float(item["baseline_mean"]) for item in selected if item is not None)
                if complete
                else None
            )
            if not complete:
                reason = "baseline_activity_years_incomplete"
            elif denominator is None or denominator <= 0.0:
                reason = "baseline_prod_denominator_nonpositive"
            else:
                reason = None
            for activity, item in zip(activities, selected, strict=True):
                mean_value = (
                    float(item["baseline_mean"])
                    if item is not None and item["baseline_mean"] is not None
                    else None
                )
                years_observed = (
                    int(item["years_observed"]) if item is not None else 0
                )
                if reason is not None:
                    weight = None
                elif version == "equal_weight":
                    weight = 1.0 / len(activities)
                else:
                    weight = float(mean_value) / float(denominator)
                rows.append(
                    {
                        "economy_id": economy_id,
                        "activity": activity,
                        "weight_version": version,
                        "activity_weight": weight,
                        "baseline_mean_prod_usd_millions": mean_value,
                        "baseline_years_observed": years_observed,
                        "frozen_baseline_start": baseline_period[0],
                        "frozen_baseline_end": baseline_period[1],
                        "unit": "share_0_to_1",
                        "missing_reason": reason,
                    }
                )
    output = pl.DataFrame(rows, schema=WEIGHT_OUTPUT_SCHEMA).sort(
        "economy_id", "weight_version", "activity"
    )
    valid_sums = (
        output.filter(pl.col("activity_weight").is_not_null())
        .group_by("economy_id", "weight_version")
        .agg(pl.col("activity_weight").sum().alias("weight_sum"))
    )
    if valid_sums.filter((pl.col("weight_sum") - 1.0).abs() > 1e-12).height:
        raise ValueError("valid TiVA activity weights do not sum to one")
    partial_groups = output.group_by("economy_id", "weight_version").agg(
        pl.col("activity_weight").is_null().sum().alias("null_weights"),
        pl.len().alias("rows"),
    ).filter(
        (pl.col("null_weights") > 0) & (pl.col("null_weights") < pl.col("rows"))
    )
    if partial_groups.height:
        raise ValueError("TiVA weight group is partially missing")
    return output
