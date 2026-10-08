"""Deterministic source-to-canonical economy mappings and sample flags."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import re
import subprocess
import unicodedata
from zipfile import ZipFile

from jsonschema import validate as validate_json
import polars as pl
import pycountry

from green_debt.artifacts import TableContract
from green_debt.storage import sha256_file


MAPPING_VERSION = "economy_crosswalk_v1"
REEXPORT_HUBS = frozenset({"HKG", "SGP", "NLD"})
EXCLUSION_REASONS = frozenset(
    {
        "aggregate_or_row",
        "asia_nes",
        "taiwan_confirmatory_exclusion",
        "population_below_1m",
        "unresolved_source_code",
    }
)
AGGREGATE_CODES = frozenset({"ROW", "W", "WORLD", "WLD", "999"})
IRENA_AGGREGATE_CODES = frozenset(
    {"GLO", "RAF", "RAS", "RCC", "REA", "RER", "RME", "RNA", "ROC", "RSA", "OCA"}
)
AGGREGATE_SOURCE_IDS = frozenset(
    {"wdi", "oecd_tiva", "oecd_eps", "oecd_ifcma", "irena", "ilostat"}
)
AGGREGATE_LABEL_TOKENS = (
    "world",
    "income",
    "region",
    "european union",
    "euro area",
    "small states",
    "oecd members",
    "ida",
    "ibrd",
    "demographic dividend",
    "arab world",
    "asia and the pacific",
    "africa eastern and southern",
    "africa western and central",
)
SOURCE_COLUMNS = (
    "source_id",
    "source_code",
    "source_label",
    "source_native_iso2",
    "source_native_iso3",
    "source_dictionary_sha256",
)
OUTPUT_COLUMNS = (
    "source_id",
    "source_code",
    "source_label",
    "source_native_iso2",
    "source_native_iso3",
    "economy_id",
    "economy_name",
    "mapping_method",
    "mapping_version",
    "review_status",
    "confirmatory_eligible",
    "exclusion_reason",
    "reexport_hub",
    "micro_economy",
    "sample_version",
    "population",
    "population_year",
    "source_dictionary_sha256",
    "mapping_notes",
)


@dataclass(frozen=True)
class EconomyBuildReport:
    source_rows: int
    canonical_economies: int
    duplicate_source_keys: int
    unresolved_codes: int
    exclusions_without_reason: int
    retained_without_economy_id: int
    reexport_hub_rows: int
    micro_robustness_rows: int
    taiwan_robustness_rows: int
    crosswalk_path: str
    audit_path: str


def _normalize_label(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _country_label_map() -> dict[str, str]:
    candidates: dict[str, set[str]] = {}
    for country in pycountry.countries:
        for attribute in ("name", "official_name", "common_name"):
            value = getattr(country, attribute, None)
            if value:
                candidates.setdefault(_normalize_label(value), set()).add(country.alpha_3)
    result = {
        label: next(iter(codes)) for label, codes in candidates.items() if len(codes) == 1
    }
    result.update(
        {
            "bolivia plurinational state of": "BOL",
            "china hong kong sar": "HKG",
            "hong kong sar china": "HKG",
            "china macao sar": "MAC",
            "iran islamic republic of": "IRN",
            "korea democratic people s republic of": "PRK",
            "korea republic of": "KOR",
            "lao people s democratic republic": "LAO",
            "micronesia federated states of": "FSM",
            "moldova republic of": "MDA",
            "state of palestine": "PSE",
            "tanzania united republic of": "TZA",
            "venezuela bolivarian republic of": "VEN",
            "viet nam": "VNM",
        }
    )
    return result


COUNTRY_LABELS = _country_label_map()


def _lookup_alpha3(code: object) -> tuple[str, str] | None:
    text = str(code or "").strip().upper()
    if not text:
        return None
    country = None
    method = ""
    if len(text) == 3 and text.isalpha():
        country = pycountry.countries.get(alpha_3=text)
        method = "iso3_code"
    elif len(text) == 2 and text.isalpha():
        country = pycountry.countries.get(alpha_2=text)
        method = "iso2_code"
    elif text.isdigit():
        country = pycountry.countries.get(numeric=text.zfill(3))
        method = "numeric_iso_code"
    if country is None:
        return None
    return country.alpha_3, method


def _economy_name(economy_id: str | None, fallback: str) -> str | None:
    if economy_id is None:
        return None
    country = pycountry.countries.get(alpha_3=economy_id)
    return country.name if country is not None else (fallback or economy_id)


def _override_rows(overrides: pl.DataFrame) -> tuple[dict[tuple[str, str], dict], dict[str, dict]]:
    if overrides.is_empty() or not {"source_id", "source_code"} <= set(overrides.columns):
        return {}, {}
    specific: dict[tuple[str, str], dict] = {}
    canonical: dict[str, dict] = {}
    for row in overrides.iter_rows(named=True):
        source_id = str(row.get("source_id") or "").strip()
        source_code = str(row.get("source_code") or "").strip()
        if not source_id or not source_code:
            raise ValueError("economy override requires source_id and source_code")
        if source_id == "canonical":
            canonical[source_code] = row
        else:
            key = (source_id, source_code)
            if key in specific:
                raise ValueError(f"duplicate economy override: {key}")
            specific[key] = row
    return specific, canonical


def _bool_value(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid Boolean override value: {value!r}")


def _apply_override(result: dict[str, object], override: dict) -> None:
    economy_value = override.get("economy_id")
    economy_id = str(economy_value).strip() if economy_value is not None else ""
    result["economy_id"] = economy_id or None
    result["economy_name"] = _economy_name(
        result["economy_id"], str(override.get("source_label") or result["source_label"])
    )
    result["confirmatory_eligible"] = _bool_value(
        override.get("confirmatory_eligible"), default=False
    )
    reason_value = override.get("exclusion_reason")
    reason = str(reason_value).strip() if reason_value is not None else ""
    result["exclusion_reason"] = reason or None
    result["reexport_hub"] = _bool_value(override.get("reexport_hub"), default=False)
    result["review_status"] = str(override.get("review_status") or "reviewed")
    result["mapping_method"] = "frozen_override"
    result["mapping_notes"] = (
        str(override.get("notes")).strip() if override.get("notes") is not None else None
    )


def _is_aggregate(source_id: str, source_code: str, source_label: str) -> bool:
    code = source_code.upper()
    label = _normalize_label(source_label)
    if code in AGGREGATE_CODES or label in {"world", "rest of world"}:
        return True
    if source_id == "ilostat" and code.startswith("X"):
        return True
    if source_id == "irena" and code in IRENA_AGGREGATE_CODES:
        return True
    if source_id == "wdi" and _lookup_alpha3(code) is None:
        return True
    if source_id in AGGREGATE_SOURCE_IDS:
        padded_label = f" {label} "
        if any(f" {token} " in padded_label for token in AGGREGATE_LABEL_TOKENS):
            return True
        if source_id in {"oecd_tiva", "oecd_eps", "oecd_ifcma"} and _lookup_alpha3(code) is None:
            return True
    return False


def _base_mapping(row: dict[str, object]) -> dict[str, object]:
    source_id = str(row["source_id"]).strip()
    source_code = str(row["source_code"]).strip()
    source_label = str(row["source_label"]).strip()
    iso2 = str(row.get("source_native_iso2") or "").strip()
    iso3 = str(row.get("source_native_iso3") or "").strip()
    result: dict[str, object] = {
        "source_id": source_id,
        "source_code": source_code,
        "source_label": source_label,
        "source_native_iso2": iso2 or None,
        "source_native_iso3": iso3 or None,
        "economy_id": None,
        "economy_name": None,
        "mapping_method": "unresolved",
        "mapping_version": MAPPING_VERSION,
        "review_status": "rule_reviewed",
        "confirmatory_eligible": False,
        "exclusion_reason": "unresolved_source_code",
        "reexport_hub": False,
        "micro_economy": False,
        "sample_version": "excluded",
        "population": (
            float(row["population"]) if row.get("population") is not None else None
        ),
        "population_year": (
            int(row["population_year"])
            if row.get("population_year") is not None
            else None
        ),
        "source_dictionary_sha256": str(
            row.get("source_dictionary_sha256") or ""
        ),
        "mapping_notes": None,
    }
    normalized_label = _normalize_label(source_label)
    if source_id == "baci" and (
        source_code == "490" or normalized_label in {"asia nes", "other asia nes"}
    ):
        result.update(
            mapping_method="asia_nes_rule",
            exclusion_reason="asia_nes",
            review_status="reviewed",
        )
        return result
    if _is_aggregate(source_id, source_code, source_label):
        result.update(
            mapping_method="aggregate_rule",
            exclusion_reason="aggregate_or_row",
            review_status="reviewed",
        )
        return result

    mapping: tuple[str, str] | None = None
    for value, method_prefix in (
        (source_code, "source"),
        (iso3, "native"),
        (iso2, "native"),
    ):
        candidate = _lookup_alpha3(value)
        if candidate is not None:
            economy_id, method = candidate
            mapping = (
                economy_id,
                method if method_prefix == "source" else f"source_native_{method}",
            )
            break
    if mapping is None:
        economy_id = COUNTRY_LABELS.get(normalized_label)
        if economy_id is not None:
            mapping = (economy_id, "normalized_label")
    if mapping is not None:
        economy_id, method = mapping
        result.update(
            economy_id=economy_id,
            economy_name=_economy_name(economy_id, source_label),
            mapping_method=method,
            confirmatory_eligible=economy_id != "TWN",
            exclusion_reason=(
                "taiwan_confirmatory_exclusion" if economy_id == "TWN" else None
            ),
            reexport_hub=economy_id in REEXPORT_HUBS,
            sample_version=(
                "taiwan_robustness" if economy_id == "TWN" else "confirmatory"
            ),
        )
    return result


def build_economy_crosswalk(
    source: pl.DataFrame,
    *,
    overrides: pl.DataFrame,
) -> pl.DataFrame:
    """Map exact codes, then exact normalized labels, then frozen overrides."""

    required = {"source_id", "source_code", "source_label"}
    if not required <= set(source.columns):
        raise ValueError("economy source rows lack source_id/source_code/source_label")
    frame = source
    for name in SOURCE_COLUMNS:
        if name not in frame.columns:
            frame = frame.with_columns(pl.lit("").cast(pl.String).alias(name))
    for name in ("population", "population_year"):
        if name not in frame.columns:
            dtype = pl.Float64 if name == "population" else pl.Int16
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(name))
    frame = frame.select([*SOURCE_COLUMNS, "population", "population_year"]).with_columns(
        pl.col("source_id").cast(pl.String),
        pl.col("source_code").cast(pl.String),
        pl.col("source_label").cast(pl.String),
        pl.col("source_native_iso2").cast(pl.String),
        pl.col("source_native_iso3").cast(pl.String),
        pl.col("source_dictionary_sha256").cast(pl.String),
        pl.col("population").cast(pl.Float64),
        pl.col("population_year").cast(pl.Int16),
    )
    if frame.filter(pl.col("source_id").str.strip_chars() == "").height:
        raise ValueError("source_id cannot be empty")
    if frame.filter(pl.col("source_code").str.strip_chars() == "").height:
        raise ValueError("source_code cannot be empty")
    if frame.group_by("source_id", "source_code").len().filter(pl.col("len") > 1).height:
        raise ValueError("duplicate source_id/source_code economy rows")

    specific_overrides, canonical_overrides = _override_rows(overrides)
    rows: list[dict[str, object]] = []
    for source_row in frame.iter_rows(named=True):
        result = _base_mapping(source_row)
        specific = specific_overrides.get(
            (str(result["source_id"]), str(result["source_code"]))
        )
        if specific is not None:
            _apply_override(result, specific)
        economy_id = result.get("economy_id")
        if economy_id is not None and str(economy_id) in canonical_overrides:
            canonical = canonical_overrides[str(economy_id)]
            result["reexport_hub"] = _bool_value(
                canonical.get("reexport_hub"), default=bool(result["reexport_hub"])
            )
            result["review_status"] = str(
                canonical.get("review_status") or result["review_status"]
            )
        reason = result.get("exclusion_reason")
        if reason is not None and str(reason) not in EXCLUSION_REASONS:
            raise ValueError(f"unregistered economy exclusion reason: {reason}")
        if result["confirmatory_eligible"]:
            result["sample_version"] = "confirmatory"
        elif result["economy_id"] == "TWN":
            result["sample_version"] = "taiwan_robustness"
        else:
            result["sample_version"] = "excluded"
        rows.append(result)

    schema = {
        "source_id": pl.String,
        "source_code": pl.String,
        "source_label": pl.String,
        "source_native_iso2": pl.String,
        "source_native_iso3": pl.String,
        "economy_id": pl.String,
        "economy_name": pl.String,
        "mapping_method": pl.String,
        "mapping_version": pl.String,
        "review_status": pl.String,
        "confirmatory_eligible": pl.Boolean,
        "exclusion_reason": pl.String,
        "reexport_hub": pl.Boolean,
        "micro_economy": pl.Boolean,
        "sample_version": pl.String,
        "population": pl.Float64,
        "population_year": pl.Int16,
        "source_dictionary_sha256": pl.String,
        "mapping_notes": pl.String,
    }
    return pl.DataFrame(rows, schema=schema).select(OUTPUT_COLUMNS).sort(
        ["source_id", "source_code"]
    )


def apply_sample_flags(
    frame: pl.DataFrame,
    *,
    minimum_population: int,
) -> pl.DataFrame:
    """Apply the inclusive population floor without erasing prior exclusions."""

    if minimum_population <= 0:
        raise ValueError("minimum_population must be positive")
    if not {"economy_id", "population"} <= set(frame.columns):
        raise ValueError("sample flags require economy_id and population")
    out = frame
    additions: list[pl.Expr] = []
    if "confirmatory_eligible" not in out.columns:
        additions.append(pl.lit(True).alias("confirmatory_eligible"))
    if "exclusion_reason" not in out.columns:
        additions.append(pl.lit(None).cast(pl.String).alias("exclusion_reason"))
    if "sample_version" not in out.columns:
        additions.append(pl.lit("confirmatory").alias("sample_version"))
    if "micro_economy" not in out.columns:
        additions.append(pl.lit(False).alias("micro_economy"))
    if additions:
        out = out.with_columns(additions)
    out = out.with_columns(
        (
            pl.col("population").is_not_null()
            & (pl.col("population").cast(pl.Float64) < float(minimum_population))
        ).alias("_new_micro")
    )
    has_prior_reason = pl.col("exclusion_reason").is_not_null() & (
        pl.col("exclusion_reason").str.len_chars() > 0
    )
    out = out.with_columns(
        (pl.col("confirmatory_eligible") & ~pl.col("_new_micro")).alias(
            "confirmatory_eligible"
        ),
        pl.when(has_prior_reason)
        .then(pl.col("exclusion_reason"))
        .when(pl.col("_new_micro"))
        .then(pl.lit("population_below_1m"))
        .otherwise(pl.lit(None).cast(pl.String))
        .alias("exclusion_reason"),
        pl.col("_new_micro").alias("micro_economy"),
        pl.when(pl.col("sample_version") != "confirmatory")
        .then(pl.col("sample_version"))
        .when(pl.col("_new_micro"))
        .then(pl.lit("micro_robustness"))
        .otherwise(pl.lit("confirmatory"))
        .alias("sample_version"),
    ).drop("_new_micro")
    return out


def _combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _row(
    source_id: str,
    source_code: object,
    source_label: object,
    source_hash: str,
    *,
    iso2: object = "",
    iso3: object = "",
) -> dict[str, str]:
    return {
        "source_id": source_id,
        "source_code": str(source_code).strip(),
        "source_label": str(source_label).strip(),
        "source_native_iso2": str(iso2 or "").strip(),
        "source_native_iso3": str(iso3 or "").strip(),
        "source_dictionary_sha256": source_hash,
    }


def _baci_rows(raw: Path) -> list[dict[str, str]]:
    path = raw / "baci/202601/BACI_HS96_V202601.zip"
    source_hash = sha256_file(path)
    with ZipFile(path) as archive:
        text = archive.read("country_codes_V202601.csv").decode(
            "utf-8-sig", errors="replace"
        )
    return [
        _row(
            "baci",
            item["country_code"],
            item["country_name"],
            source_hash,
            iso2=item["country_iso2"],
            iso3=item["country_iso3"],
        )
        for item in csv.DictReader(StringIO(text))
    ]


def _wdi_payload(raw: Path) -> tuple[Path, list[dict]]:
    path = raw / "wdi/20260821/SP.POP.TOTL.1996-2024.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 2 or not isinstance(payload[1], list):
        raise ValueError("invalid WDI population payload")
    return path, payload[1]


def _wdi_rows(raw: Path) -> list[dict[str, str]]:
    path, observations = _wdi_payload(raw)
    source_hash = sha256_file(path)
    unique: dict[str, dict[str, str]] = {}
    for observation in observations:
        iso3 = str(observation.get("countryiso3code") or "").strip()
        wb2 = str(observation["country"]["id"]).strip()
        source_code = iso3 or f"WB2:{wb2}"
        unique[source_code] = _row(
            "wdi",
            source_code,
            observation["country"]["value"],
            source_hash,
            iso2=wb2 if len(wb2) == 2 else "",
            iso3=iso3,
        )
    return list(unique.values())


def _openalex_rows(raw: Path) -> list[dict[str, str]]:
    paths = [
        raw / "openalex/20260823/works_aggregate/openalex_country_year_counts_1992_1995.csv",
        raw / "openalex/20260822/works_aggregate/openalex_country_year_counts_1996_2024.csv",
    ]
    source_hash = _combined_hash(paths)
    codes: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            codes.update(row["country_code"] for row in csv.DictReader(handle))
    return [
        _row("openalex", code, code, source_hash, iso2=code) for code in sorted(codes)
    ]


def _irena_rows(raw: Path) -> list[dict[str, str]]:
    data = raw / "irena/20260821/data"
    sources = (
        (
            data / "Country_ELECCAP_2026_H1.2000-2007.jsonstat2.json",
            "Country/area",
        ),
        (
            data / "Country_ELECGEN_2025_H2.2000-2006.jsonstat2.json",
            "Country/area",
        ),
        (
            data / "RE-SHARE_2026_H1.2000-2025.jsonstat2.json",
            "Region/country/area",
        ),
    )
    labels: dict[str, str] = {}
    paths: list[Path] = []
    for path, dimension in sources:
        payload = json.loads(path.read_text(encoding="utf-8"))
        category = payload["dimension"][dimension]["category"]
        category_labels = category.get("label", {})
        for code in category["index"]:
            labels.setdefault(code, category_labels.get(code, code))
        paths.append(path)
    source_hash = _combined_hash(paths)
    return [
        _row("irena", code, label, source_hash, iso3=code)
        for code, label in sorted(labels.items())
    ]


def _ilostat_rows(raw: Path) -> list[dict[str, str]]:
    path = raw / "ilostat/20260821/dictionaries/ref_area_en.rds"
    expression = (
        'x<-readRDS(commandArgs(trailingOnly=TRUE)[1]);'
        'write.table(x,file="",sep="\\t",row.names=FALSE,col.names=TRUE,'
        'quote=FALSE,na="")'
    )
    completed = subprocess.run(
        ["Rscript", "--vanilla", "-e", expression, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Rscript could not read ILOSTAT dictionary: {completed.stderr}")
    rows = csv.DictReader(StringIO(completed.stdout), delimiter="\t")
    source_hash = sha256_file(path)
    return [
        _row(
            "ilostat",
            item["ref_area"],
            item["ref_area.label"],
            source_hash,
            iso3=item["ref_area"] if len(item["ref_area"]) == 3 else "",
        )
        for item in rows
    ]


def _tiva_area_labels(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data", payload)
    for codelist in data.get("codelists", []):
        if codelist.get("id") != "CL_AREA":
            continue
        return {
            str(code["id"]): str(code.get("name") or code["id"])
            for code in codelist.get("codes", [])
        }
    return {}


def _tiva_rows(raw: Path) -> list[dict[str, str]]:
    constraint = raw / "oecd_tiva/20260821/metadata/DSD_TIVA_MAINLV_DF_MAINLV.availableconstraint.json"
    structure = raw / "oecd_tiva/20260821/metadata/DSD_TIVA_MAINLV_DF_MAINLV.structure.json"
    payload = json.loads(constraint.read_text(encoding="utf-8"))
    values: list[str] | None = None
    for region in payload["data"]["contentConstraints"][0]["cubeRegions"]:
        for key_value in region["keyValues"]:
            if key_value.get("id") == "REF_AREA":
                values = [str(value) for value in key_value["values"]]
                break
    if values is None:
        raise ValueError("TiVA constraint lacks REF_AREA values")
    labels = _tiva_area_labels(structure)
    source_hash = _combined_hash([constraint, structure])
    return [
        _row("oecd_tiva", code, labels.get(code, code), source_hash, iso3=code)
        for code in values
    ]


def _eps_rows(raw: Path) -> list[dict[str, str]]:
    data_path = raw / "oecd_eps/20260821/data/EPS_composite_all_countries_1990-2020.csv"
    structure = raw / "oecd_eps/20260821/metadata/DSD_EPS_DF_EPS.structure.json"
    with data_path.open(encoding="utf-8", newline="") as handle:
        codes = sorted({row["REF_AREA"] for row in csv.DictReader(handle)})
    source_hash = _combined_hash([data_path, structure])
    return [_row("oecd_eps", code, code, source_hash, iso3=code) for code in codes]


def _ifcma_rows(raw: Path) -> list[dict[str, str]]:
    path = (
        raw
        / "oecd_ifcma/202604/IFCMA_ClimatePolicyDatabase_Data_April_2026.csv"
    )
    labels: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            code = str(row.get("Country ISO") or "").strip()
            if not code or code == "Country ISO":
                continue
            label = str(row.get("Country") or code).strip()
            labels.setdefault(code, label or code)
    source_hash = sha256_file(path)
    return [
        _row("oecd_ifcma", code, label, source_hash, iso3=code)
        for code, label in sorted(labels.items())
    ]


def collect_production_source_rows(data_root: Path) -> pl.DataFrame:
    raw = data_root / "04_原始数据"
    rows = [
        *_baci_rows(raw),
        *_wdi_rows(raw),
        *_openalex_rows(raw),
        *_irena_rows(raw),
        *_ilostat_rows(raw),
        *_tiva_rows(raw),
        *_eps_rows(raw),
        *_ifcma_rows(raw),
    ]
    frame = pl.DataFrame(
        rows,
        schema={name: pl.String for name in SOURCE_COLUMNS},
    ).unique(subset=["source_id", "source_code"], keep="first").sort(
        ["source_id", "source_code"]
    )
    if frame.filter(pl.col("source_code") == "").height:
        raise ValueError("production source dictionary contains an empty source code")
    return frame


def _population_reference(raw: Path, wdi_crosswalk: pl.DataFrame) -> pl.DataFrame:
    _, observations = _wdi_payload(raw)
    candidates: list[dict[str, object]] = []
    for observation in observations:
        value = observation.get("value")
        year = int(observation["date"])
        if value is None or year > 2022:
            continue
        iso3 = str(observation.get("countryiso3code") or "").strip()
        wb2 = str(observation["country"]["id"]).strip()
        candidates.append(
            {
                "source_code": iso3 or f"WB2:{wb2}",
                "population": float(value),
                "population_year": year,
            }
        )
    population = pl.DataFrame(
        candidates,
        schema={
            "source_code": pl.String,
            "population": pl.Float64,
            "population_year": pl.Int16,
        },
    ).sort("population_year", descending=True).unique("source_code", keep="first")
    return (
        wdi_crosswalk.select("source_code", "economy_id")
        .filter(pl.col("economy_id").is_not_null())
        .join(population, on="source_code", how="inner")
        .sort("population_year", descending=True)
        .unique("economy_id", keep="first")
        .select("economy_id", "population", "population_year")
    )


def _load_contract(path: Path) -> TableContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = json.loads(
        (path.parent / "base_table_contract.schema.json").read_text(encoding="utf-8")
    )
    validate_json(instance=payload, schema=schema)
    return TableContract(
        table_id=str(payload["table_id"]),
        schema_version=str(payload["schema_version"]),
        primary_key=tuple(payload["primary_key"]),
        columns={str(key): str(value) for key, value in payload["columns"].items()},
        units={str(key): str(value) for key, value in payload["units"].items()},
        period=None,
        zero_semantics={
            str(key): str(value) for key, value in payload.get("zero_semantics", {}).items()
        },
        transformations=tuple(payload.get("transformations", [])),
    )


def _dtype_map() -> dict[str, pl.DataType]:
    return {
        "String": pl.String,
        "Boolean": pl.Boolean,
        "Float64": pl.Float64,
        "Int16": pl.Int16,
    }


def _validate_crosswalk(frame: pl.DataFrame, contract: TableContract) -> None:
    if tuple(frame.columns) != tuple(contract.columns):
        raise ValueError("economy crosswalk columns differ from contract")
    actual = {name: str(dtype) for name, dtype in frame.schema.items()}
    if actual != contract.columns:
        raise ValueError(f"economy crosswalk dtypes differ: {actual}")
    duplicates = frame.group_by(list(contract.primary_key)).len().filter(
        pl.col("len") > 1
    ).height
    if duplicates:
        raise ValueError(f"duplicate economy source keys: {duplicates}")
    if any(frame.get_column(key).null_count() for key in contract.primary_key):
        raise ValueError("null economy source key")


def _atomic_write_crosswalk(
    frame: pl.DataFrame,
    contract: TableContract,
    destination: Path,
) -> None:
    casted = frame.select(
        [
            pl.col(name).cast(_dtype_map()[dtype]).alias(name)
            for name, dtype in contract.columns.items()
        ]
    )
    _validate_crosswalk(casted, contract)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        casted.write_csv(partial)
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        reread = pl.read_csv(
            partial,
            schema_overrides={
                name: _dtype_map()[dtype] for name, dtype in contract.columns.items()
            },
            null_values="",
        )
        _validate_crosswalk(reread, contract)
        if not reread.equals(casted):
            raise ValueError("economy crosswalk CSV round trip changed values")
        os.replace(partial, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _atomic_write_audit(rows: list[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    frame = pl.DataFrame(
        rows,
        schema={
            "metric": pl.String,
            "source_id": pl.String,
            "source_code": pl.String,
            "source_label": pl.String,
            "value": pl.String,
            "expected": pl.String,
            "status": pl.String,
        },
    )
    try:
        frame.write_csv(partial)
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _audit_rows(crosswalk: pl.DataFrame) -> tuple[list[dict[str, object]], dict[str, int]]:
    duplicates = crosswalk.group_by("source_id", "source_code").len().filter(
        pl.col("len") > 1
    ).height
    unresolved = crosswalk.filter(pl.col("exclusion_reason") == "unresolved_source_code")
    exclusions_without_reason = crosswalk.filter(
        ~pl.col("confirmatory_eligible")
        & (pl.col("exclusion_reason").is_null() | (pl.col("exclusion_reason") == ""))
    ).height
    retained_without_id = crosswalk.filter(
        pl.col("confirmatory_eligible") & pl.col("economy_id").is_null()
    ).height
    metrics = {
        "duplicate_source_keys": duplicates,
        "unresolved_codes": unresolved.height,
        "exclusions_without_reason": exclusions_without_reason,
        "retained_without_economy_id": retained_without_id,
    }
    rows = [
        {
            "metric": metric,
            "source_id": "",
            "source_code": "",
            "source_label": "",
            "value": str(value),
            "expected": "0",
            "status": "pass" if value == 0 else "fail",
        }
        for metric, value in metrics.items()
    ]
    for row in unresolved.select("source_id", "source_code", "source_label").iter_rows(
        named=True
    ):
        rows.append(
            {
                "metric": "unresolved_source_code",
                "source_id": row["source_id"],
                "source_code": row["source_code"],
                "source_label": row["source_label"],
                "value": "1",
                "expected": "review_required",
                "status": "fail",
            }
        )
    rows.extend(
        [
            {
                "metric": "reexport_hub_rows",
                "source_id": "",
                "source_code": "",
                "source_label": "",
                "value": str(crosswalk.filter(pl.col("reexport_hub")).height),
                "expected": "reported",
                "status": "reported",
            },
            {
                "metric": "micro_robustness_rows",
                "source_id": "",
                "source_code": "",
                "source_label": "",
                "value": str(
                    crosswalk.filter(pl.col("sample_version") == "micro_robustness").height
                ),
                "expected": "reported",
                "status": "reported",
            },
            {
                "metric": "taiwan_robustness_rows",
                "source_id": "",
                "source_code": "",
                "source_label": "",
                "value": str(
                    crosswalk.filter(pl.col("sample_version") == "taiwan_robustness").height
                ),
                "expected": "reported",
                "status": "reported",
            },
        ]
    )
    return rows, metrics


def build_production_economy_crosswalk(
    *,
    code_root: Path,
    data_root: Path,
    overrides_path: Path,
    contract_path: Path,
    minimum_population: int,
) -> EconomyBuildReport:
    source = collect_production_source_rows(data_root)
    overrides = pl.read_csv(
        overrides_path,
        schema_overrides={
            "source_id": pl.String,
            "source_code": pl.String,
            "source_label": pl.String,
            "economy_id": pl.String,
        },
        null_values="",
    )
    crosswalk = build_economy_crosswalk(source, overrides=overrides)
    population = _population_reference(
        data_root / "04_原始数据",
        crosswalk.filter(pl.col("source_id") == "wdi"),
    )
    crosswalk = (
        crosswalk.drop("population", "population_year")
        .join(population, on="economy_id", how="left")
        .pipe(apply_sample_flags, minimum_population=minimum_population)
        .select(OUTPUT_COLUMNS)
        .sort(["source_id", "source_code"])
    )
    audit_rows, metrics = _audit_rows(crosswalk)
    audit_path = code_root / "06_结果/经济体映射审计_v1.csv"
    _atomic_write_audit(audit_rows, audit_path)
    if any(metrics.values()):
        raise RuntimeError(
            "economy crosswalk requires explicit review: "
            + ", ".join(f"{key}={value}" for key, value in metrics.items() if value)
        )
    contract = _load_contract(contract_path)
    destination = code_root / "02_数据字典/economy_crosswalk_v1.csv"
    _atomic_write_crosswalk(crosswalk, contract, destination)
    return EconomyBuildReport(
        source_rows=crosswalk.height,
        canonical_economies=crosswalk.get_column("economy_id").drop_nulls().n_unique(),
        duplicate_source_keys=0,
        unresolved_codes=0,
        exclusions_without_reason=0,
        retained_without_economy_id=0,
        reexport_hub_rows=crosswalk.filter(pl.col("reexport_hub")).height,
        micro_robustness_rows=crosswalk.filter(
            pl.col("sample_version") == "micro_robustness"
        ).height,
        taiwan_robustness_rows=crosswalk.filter(
            pl.col("sample_version") == "taiwan_robustness"
        ).height,
        crosswalk_path=str(destination),
        audit_path=str(audit_path),
    )


def audit_economies(
    *,
    code_root: Path,
    contract_path: Path,
) -> dict[str, object]:
    contract = _load_contract(contract_path)
    path = code_root / "02_数据字典/economy_crosswalk_v1.csv"
    frame = pl.read_csv(
        path,
        schema_overrides={
            name: _dtype_map()[dtype] for name, dtype in contract.columns.items()
        },
        null_values="",
    )
    _validate_crosswalk(frame, contract)
    audit_rows, metrics = _audit_rows(frame)
    audit_path = code_root / "06_结果/经济体映射审计_v1.csv"
    _atomic_write_audit(audit_rows, audit_path)
    if any(metrics.values()):
        raise RuntimeError(f"economy audit failed: {metrics}")
    hubs_excluded_for_nonpopulation = frame.filter(
        pl.col("reexport_hub")
        & ~pl.col("confirmatory_eligible")
        & (pl.col("exclusion_reason") != "population_below_1m")
    ).height
    if hubs_excluded_for_nonpopulation:
        raise RuntimeError("a known re-export hub was excluded")
    return {
        "rows": frame.height,
        "canonical_economies": frame.get_column("economy_id").drop_nulls().n_unique(),
        "duplicate_source_keys": 0,
        "unresolved_codes": 0,
        "exclusions_without_reason": 0,
        "retained_without_economy_id": 0,
        "reexport_hub_rows": frame.filter(pl.col("reexport_hub")).height,
        "micro_robustness_rows": frame.filter(
            pl.col("sample_version") == "micro_robustness"
        ).height,
        "taiwan_robustness_rows": frame.filter(
            pl.col("sample_version") == "taiwan_robustness"
        ).height,
        "audit_path": str(audit_path),
        "status": "valid",
    }
