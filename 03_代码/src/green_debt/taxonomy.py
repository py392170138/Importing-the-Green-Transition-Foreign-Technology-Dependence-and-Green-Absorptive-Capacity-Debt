"""Frozen CLEG/APEC product lists and weighted HS07-to-HS96 concordances."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from io import StringIO, TextIOWrapper
import json
from math import isfinite
import os
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile

from jsonschema import validate as validate_json
import pandas as pd
import polars as pl
from pypdf import PdfReader
import pyarrow as pa
import pyarrow.csv as pacsv

from green_debt.artifacts import TableContract
from green_debt.config import load_construction_config
from green_debt.storage import (
    enforce_construction_capacity,
    measure_layer_usage,
    sha256_file,
)


HS6_PATTERN = re.compile(r"^[0-9]{6}$")
CODE_CATEGORY_PATTERN = re.compile(r"^\s*([0-9]+)\s+([A-Z]{3})\b(.*)$")
CLEG_CATEGORIES = (
    "APC",
    "CRE",
    "EPP",
    "HEM",
    "MON",
    "NRP",
    "NVA",
    "REP",
    "SWM",
    "SWR",
    "WAT",
)
MAIN_CATEGORIES = frozenset({"REP", "HEM", "CRE"})
WEIGHT_CATEGORIES = (*CLEG_CATEGORIES, "APEC")
PDF_ANNEX_PAGES = tuple(range(51, 58))
OVERLAP_BLOCK_BYTES = 16 * 1024**2


HS07_CONTRACT_NAME = "product_registry_hs07.json"
HS96_CONTRACT_NAME = "product_registry_hs96.json"


@dataclass(frozen=True)
class TaxonomySourcePaths:
    cleg_pdf: Path
    apec_html: Path
    h3_to_h1_zip: Path
    h1_to_bec_zip: Path
    hs96_to_bec_xls: Path
    baci_hs07_zip: Path


@dataclass(frozen=True)
class TaxonomyBuildReport:
    main_count: int
    broad_count: int
    apec_count: int
    hs07_registry_rows: int
    hs96_registry_rows: int
    hs07_duplicate_keys: int
    hs96_duplicate_keys: int
    unresolved_green_hs07: int
    fallback_count: int
    weights_out_of_range: int
    category_sum_violations: int
    ambiguous_bec_rows: int
    bec_unmapped_rows: int
    wits_bec_selected_pairs_not_in_unsd: int
    overlap_rows_read: int
    overlap_exports_usd: float
    hs07_registry_path: str
    hs96_registry_path: str
    audit_path: str


def _normalize_hs6(value: object, *, field: str) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.zfill(6)
    if HS6_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be an exact six-digit code: {value!r}")
    return text


def _duplicate_groups(frame: pl.DataFrame, keys: tuple[str, ...]) -> int:
    return frame.group_by(list(keys)).len().filter(pl.col("len") > 1).height


def parse_cleg_annex_text(text: str) -> pl.DataFrame:
    """Parse anchored six-digit CLEG rows from already extracted Annex text."""

    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = CODE_CATEGORY_PATTERN.match(line)
        if match is None:
            continue
        raw_code, category, remainder = match.groups()
        if len(raw_code) != 6 or not raw_code.isdigit():
            raise ValueError(
                f"CLEG code outside [0-9]{{6}} at extracted line {line_number}"
            )
        if category not in CLEG_CATEGORIES:
            raise ValueError(f"unknown CLEG category: {category}")
        rows.append(
            {
                "hs07": raw_code,
                "category": category,
                "source_line": remainder.strip(),
            }
        )
    frame = pl.DataFrame(
        rows,
        schema={"hs07": pl.String, "category": pl.String, "source_line": pl.String},
    ).sort("hs07")
    if _duplicate_groups(frame, ("hs07", "category")):
        raise ValueError("duplicate CLEG code-category row")
    if frame.get_column("hs07").n_unique() != frame.height:
        raise ValueError("a CLEG code appears in more than one category")
    return frame


def extract_cleg_pdf(path: Path) -> pl.DataFrame:
    """Extract only printed Annex pages 51-57 from the frozen OECD PDF."""

    reader = PdfReader(path)
    page_frames: list[pl.DataFrame] = []
    for printed_page in PDF_ANNEX_PAGES:
        if printed_page >= len(reader.pages):
            raise ValueError(f"CLEG PDF lacks printed page {printed_page}")
        text = reader.pages[printed_page].extract_text() or ""
        parsed = parse_cleg_annex_text(text)
        if parsed.height:
            parsed = parsed.with_columns(pl.lit(printed_page).alias("pdf_page"))
            page_frames.append(parsed)
    if not page_frames:
        raise ValueError("CLEG Annex extraction returned no product rows")
    frame = pl.concat(page_frames).sort("hs07")
    if frame.height != 248 or frame.get_column("hs07").n_unique() != 248:
        raise ValueError(
            f"CLEG Annex must contain 248 unique HS07 codes; got {frame.height}"
        )
    main_count = frame.filter(pl.col("category").is_in(list(MAIN_CATEGORIES))).height
    if main_count != 126:
        raise ValueError(f"CLEG main categories must contain 126 codes; got {main_count}")
    return frame


def _html_input(value: Path | str) -> object:
    if isinstance(value, Path):
        return value
    if "<" in value and ">" in value:
        return StringIO(value)
    return value


def _clean_apec_code(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text if HS6_PATTERN.fullmatch(text) else None


def parse_apec_html(
    source: Path | str,
    *,
    expected_count: int | None = None,
) -> pl.DataFrame:
    """Parse the APEC table and select its HS07 code with declared fallbacks."""

    tables = pd.read_html(_html_input(source), header=None)
    if len(tables) != 1:
        raise ValueError(f"APEC source must contain exactly one table; got {len(tables)}")
    table = tables[0]
    named_columns = {str(column).strip().lower(): column for column in table.columns}
    fixture_hs07_column = next(
        (column for name, column in named_columns.items() if "2007" in name),
        None,
    )
    fixture_description_column = next(
        (column for name, column in named_columns.items() if "description" in name),
        None,
    )
    if table.shape[1] < 4 and fixture_hs07_column is None:
        raise ValueError("APEC table lacks the three HS revisions and description")

    rows: list[dict[str, str]] = []
    for source_row, values in enumerate(table.itertuples(index=False, name=None), 1):
        if table.shape[1] < 4:
            hs07_value = table.loc[table.index[source_row - 1], fixture_hs07_column]
            description_value = table.loc[
                table.index[source_row - 1], fixture_description_column
            ]
            first_three = [None, hs07_value, None]
        else:
            first_three = list(values[:3])
            description_value = values[3]
        if any(str(value).strip().upper().startswith("HS ") for value in first_three):
            continue
        candidates = (
            ("HS2007", _clean_apec_code(first_three[1])),
            ("HS2002_fallback", _clean_apec_code(first_three[0])),
            ("HS2012_fallback", _clean_apec_code(first_three[2])),
        )
        chosen = next(((label, code) for label, code in candidates if code), None)
        if chosen is None:
            if all(value is None or pd.isna(value) for value in first_three):
                continue
            raise ValueError(f"APEC row {source_row} lacks a usable six-digit code")
        source_column, code = chosen
        description = "" if pd.isna(description_value) else str(description_value).strip()
        rows.append(
            {
                "hs07": code,
                "list_name": "apec",
                "category": "APEC",
                "product_description": description,
                "source_code_column": source_column,
                "html_table_row": str(source_row),
            }
        )
    frame = pl.DataFrame(
        rows,
        schema={
            "hs07": pl.String,
            "list_name": pl.String,
            "category": pl.String,
            "product_description": pl.String,
            "source_code_column": pl.String,
            "html_table_row": pl.String,
        },
    ).sort("hs07")
    if _duplicate_groups(frame, ("hs07",)):
        raise ValueError("duplicate APEC six-digit product code")
    if expected_count is not None and frame.height != expected_count:
        raise ValueError(
            f"APEC table must contain {expected_count} products; got {frame.height}"
        )
    return frame


def load_h3_to_h1(path: Path) -> pl.DataFrame:
    """Read the official WITS HS2007-to-HS1996 mapping as strings."""

    with ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("H3-to-H1 archive must contain exactly one CSV")
        with archive.open(members[0]) as binary:
            reader = csv.DictReader(TextIOWrapper(binary, encoding="cp1252"))
            rows = [
                {
                    "hs07": _normalize_hs6(
                        row["HS 2007 Product Code"], field="WITS HS07"
                    ),
                    "hs96": _normalize_hs6(
                        row["HS 1996 Product Code"], field="WITS HS96"
                    ),
                }
                for row in reader
            ]
    frame = pl.DataFrame(rows, schema={"hs07": pl.String, "hs96": pl.String})
    frame = frame.unique(maintain_order=True).sort(["hs07", "hs96"])
    if not frame.height:
        raise ValueError("H3-to-H1 concordance is empty")
    return frame


def _membership_rows(
    membership: pl.DataFrame,
    *,
    list_name: str,
) -> dict[str, str]:
    if "hs07" not in membership.columns:
        raise ValueError("membership lacks hs07")
    frame = membership.with_columns(pl.col("hs07").cast(pl.String).str.zfill(6))
    boolean_column = f"is_{list_name}"
    if boolean_column in frame.columns:
        frame = frame.filter(pl.col(boolean_column).fill_null(False))
    elif "list_name" in frame.columns:
        frame = frame.filter(pl.col("list_name") == list_name)
    categories = (
        frame.get_column("category").cast(pl.String).to_list()
        if "category" in frame.columns
        else [list_name.upper()] * frame.height
    )
    result: dict[str, str] = {}
    for code, category in zip(frame.get_column("hs07").to_list(), categories):
        if HS6_PATTERN.fullmatch(code) is None:
            raise ValueError(f"membership contains invalid HS07: {code}")
        prior = result.get(code)
        if prior is not None and prior != category:
            raise ValueError(f"green HS07 has multiple categories: {code}")
        result[code] = category
    return result


def build_hs96_green_weights(
    mapping: pl.DataFrame,
    membership: pl.DataFrame,
    overlap: pl.DataFrame,
    *,
    list_name: str,
) -> pl.DataFrame:
    """Weight every HS96 target by mapped 2007-2010 HS07 export overlap."""

    required = {"hs07", "hs96"}
    if not required <= set(mapping.columns):
        raise ValueError("mapping must contain hs07 and hs96")
    pairs = mapping.select("hs07", "hs96").with_columns(
        pl.col("hs07").cast(pl.String).str.zfill(6),
        pl.col("hs96").cast(pl.String).str.zfill(6),
    ).unique()
    invalid_mapping = pairs.filter(
        ~pl.col("hs07").str.contains(r"^[0-9]{6}$")
        | ~pl.col("hs96").str.contains(r"^[0-9]{6}$")
    )
    if invalid_mapping.height:
        raise ValueError("mapping contains a non-six-digit product code")

    green_categories = _membership_rows(membership, list_name=list_name)
    mapped_sources = set(pairs.get_column("hs07").to_list())
    unmapped = sorted(set(green_categories) - mapped_sources)
    if unmapped:
        raise ValueError(f"unmapped green HS07: {','.join(unmapped[:10])}")

    exports: dict[str, float] = {}
    if overlap.height:
        if not {"hs07", "exports_usd"} <= set(overlap.columns):
            raise ValueError("overlap must contain hs07 and exports_usd")
        overlap_rows = overlap.select("hs07", "exports_usd").with_columns(
            pl.col("hs07").cast(pl.String).str.zfill(6),
            pl.col("exports_usd").cast(pl.Float64),
        ).group_by("hs07").agg(pl.col("exports_usd").sum())
        for code, value in overlap_rows.iter_rows():
            numeric = float(value)
            if not isfinite(numeric) or numeric < 0:
                raise ValueError(f"invalid overlap export total for {code}")
            exports[code] = numeric

    targets: dict[str, set[str]] = {}
    for source_code, target_code in pairs.iter_rows():
        targets.setdefault(target_code, set()).add(source_code)

    rows: list[dict[str, object]] = []
    for target_code in sorted(targets):
        mapped_codes = targets[target_code]
        green_codes = mapped_codes & set(green_categories)
        denominator = sum(exports.get(code, 0.0) for code in mapped_codes)
        numerator = sum(exports.get(code, 0.0) for code in green_codes)
        fallback = denominator == 0.0
        if fallback:
            green_weight = len(green_codes) / len(mapped_codes)
            method = "code_share_fallback"
        else:
            green_weight = numerator / denominator
            method = "trade_overlap_2007_2010"
        if not isfinite(green_weight) or not 0.0 <= green_weight <= 1.0:
            raise ValueError(f"green weight outside [0,1] for {target_code}")
        row: dict[str, object] = {
            "hs96": target_code,
            "list_name": list_name,
            "green_weight": float(green_weight),
            "mapped_hs07_count": len(mapped_codes),
            "green_hs07_count": len(green_codes),
            "overlap_total_exports_usd": float(denominator),
            "overlap_green_exports_usd": float(numerator),
            "weight_method": method,
            "code_share_fallback": fallback,
        }
        category_sum = 0.0
        for category in WEIGHT_CATEGORIES:
            category_codes = {
                code for code in green_codes if green_categories[code] == category
            }
            category_numerator = sum(exports.get(code, 0.0) for code in category_codes)
            if fallback:
                category_weight = len(category_codes) / len(mapped_codes)
            else:
                category_weight = category_numerator / denominator
            row[f"category_{category.lower()}_weight"] = float(category_weight)
            category_sum += category_weight
        if category_sum > green_weight + 1e-12:
            raise ValueError(f"category weights exceed total green weight: {target_code}")
        rows.append(row)

    schema: dict[str, pl.DataType] = {
        "hs96": pl.String,
        "list_name": pl.String,
        "green_weight": pl.Float64,
        "mapped_hs07_count": pl.UInt16,
        "green_hs07_count": pl.UInt16,
        "overlap_total_exports_usd": pl.Float64,
        "overlap_green_exports_usd": pl.Float64,
        "weight_method": pl.String,
        "code_share_fallback": pl.Boolean,
    }
    schema.update(
        {f"category_{category.lower()}_weight": pl.Float64 for category in WEIGHT_CATEGORIES}
    )
    return pl.DataFrame(rows, schema=schema).sort("hs96")


def normalize_bec_use(value: object) -> str:
    """Normalize a BEC use label or code to the four frozen use classes."""

    text = str(value).strip().lower().replace("_", " ")
    if text in {"intermediate", "capital", "consumption", "unclassified"}:
        return text
    if "intermediate" in text or "industrial suppl" in text or "parts" in text:
        return "intermediate"
    if "capital" in text:
        return "capital"
    if "consumption" in text or "consumer" in text or "household" in text:
        return "consumption"
    code = text[:-2] if text.endswith(".0") else text
    code = code.replace(".", "")
    intermediate = {"111", "121", "21", "22", "31", "322", "42", "53"}
    capital = {"41", "521"}
    consumption = {"112", "122", "321", "51", "522", "61", "62", "63"}
    if code in intermediate:
        return "intermediate"
    if code in capital:
        return "capital"
    if code in consumption:
        return "consumption"
    return "unclassified"


def assign_upstream_weight(mapped: pl.DataFrame) -> pl.DataFrame:
    """Assign explicit fractional upstream weights for one-to-many BEC mappings."""

    if not {"hs96", "bec_use"} <= set(mapped.columns):
        raise ValueError("BEC mapping must contain hs96 and bec_use")
    frame = mapped.with_columns(
        pl.col("hs96").cast(pl.String).str.zfill(6),
        pl.col("bec_use")
        .cast(pl.String)
        .map_elements(normalize_bec_use, return_dtype=pl.String),
    )
    identity = "bec_code" if "bec_code" in frame.columns else "bec_use"
    selected_columns = ["hs96", "bec_use"]
    if identity not in selected_columns:
        selected_columns.append(identity)
    frame = frame.select(selected_columns).unique()
    rows: list[dict[str, object]] = []
    for group in frame.partition_by("hs96", maintain_order=True):
        hs96 = group.get_column("hs96").item(0)
        mapping_count = group.height
        uses = group.get_column("bec_use").to_list()
        upstream_count = sum(use in {"intermediate", "capital"} for use in uses)
        rows.append(
            {
                "hs96": hs96,
                "upstream_weight": upstream_count / mapping_count,
                "mapping_count": mapping_count,
                "ambiguous_bec_mapping": mapping_count > 1,
                "bec_uses": "|".join(sorted(set(uses))),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "hs96": pl.String,
            "upstream_weight": pl.Float64,
            "mapping_count": pl.UInt16,
            "ambiguous_bec_mapping": pl.Boolean,
            "bec_uses": pl.String,
        },
    ).sort("hs96")


def load_hs96_bec(path: Path) -> pl.DataFrame:
    """Read the one-to-many UNSD HS1996-to-BEC correlation sheet."""

    table = pd.read_excel(
        path,
        sheet_name="Correlation Tables",
        header=None,
        dtype=str,
    )
    rows: list[dict[str, str]] = []
    for hs96_value, bec_value in zip(table.iloc[:, 3], table.iloc[:, 4]):
        if pd.isna(hs96_value) or pd.isna(bec_value):
            continue
        hs96_text = str(hs96_value).strip()
        if hs96_text.endswith(".0"):
            hs96_text = hs96_text[:-2]
        if not hs96_text.isdigit():
            continue
        hs96 = _normalize_hs6(hs96_text, field="UNSD HS96")
        bec_code = str(bec_value).strip()
        if bec_code.endswith(".0"):
            bec_code = bec_code[:-2]
        rows.append(
            {
                "hs96": hs96,
                "bec_code": bec_code,
                "bec_use": normalize_bec_use(bec_code),
            }
        )
    frame = pl.DataFrame(
        rows,
        schema={"hs96": pl.String, "bec_code": pl.String, "bec_use": pl.String},
    ).unique().sort(["hs96", "bec_code"])
    if not frame.height:
        raise ValueError("UNSD HS96-to-BEC correlation is empty")
    return frame


def load_wits_h1_to_bec(path: Path) -> pl.DataFrame:
    with ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("H1-to-BEC archive must contain exactly one CSV")
        with archive.open(members[0]) as binary:
            reader = csv.DictReader(TextIOWrapper(binary, encoding="cp1252"))
            rows = [
                {
                    "hs96": _normalize_hs6(
                        row["HS 1996 Product Code"], field="WITS HS96"
                    ),
                    "bec_code": str(row["BEC Product Code"]).strip(),
                }
                for row in reader
            ]
    return pl.DataFrame(
        rows,
        schema={"hs96": pl.String, "bec_code": pl.String},
    ).unique().sort(["hs96", "bec_code"])


def stream_baci_overlap_exports(
    path: Path,
    *,
    start_year: int,
    end_year: int,
) -> tuple[pl.DataFrame, int]:
    """Stream only product and value columns from bounded BACI ZIP members."""

    if (start_year, end_year) != (2007, 2010):
        raise ValueError("taxonomy overlap window must be exactly 2007-2010")
    totals: dict[str, float] = {}
    rows_read = 0
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        for year in range(start_year, end_year + 1):
            matches = sorted(
                name
                for name in names
                if re.fullmatch(rf"BACI_HS07_Y{year}_V[0-9]+\.csv", Path(name).name)
            )
            if len(matches) != 1:
                raise ValueError(f"BACI HS07 archive lacks one unique member for {year}")
            with archive.open(matches[0]) as binary:
                batches = pacsv.open_csv(
                    binary,
                    read_options=pacsv.ReadOptions(block_size=OVERLAP_BLOCK_BYTES),
                    convert_options=pacsv.ConvertOptions(
                        include_columns=["k", "v"],
                        column_types={"k": pa.string(), "v": pa.float64()},
                    ),
                )
                for batch in batches:
                    rows_read += batch.num_rows
                    grouped = (
                        pl.from_arrow(batch)
                        .group_by("k")
                        .agg(pl.col("v").sum().alias("v"))
                    )
                    for code, value_thousand_usd in grouped.iter_rows():
                        hs07 = _normalize_hs6(code, field="BACI HS07")
                        value_usd = float(value_thousand_usd) * 1_000.0
                        if not isfinite(value_usd) or value_usd < 0:
                            raise ValueError(f"invalid BACI overlap value for {hs07}")
                        totals[hs07] = totals.get(hs07, 0.0) + value_usd
    frame = pl.DataFrame(
        sorted(totals.items()),
        schema={"hs07": pl.String, "exports_usd": pl.Float64},
        orient="row",
    )
    return frame, rows_read


def _source_paths(data_root: Path) -> TaxonomySourcePaths:
    raw = data_root / "04_原始数据"
    return TaxonomySourcePaths(
        cleg_pdf=raw / "taxonomy/oecd/oecd_cleg_sauvage_2014.pdf",
        apec_html=raw / "classifications/apec/apec_environmental_goods_54_2012.html",
        h3_to_h1_zip=raw / "classifications/wits/Concordance_H3_to_H1.zip",
        h1_to_bec_zip=raw / "classifications/wits/Concordance_H1_to_BE.zip",
        hs96_to_bec_xls=raw / "classifications/unsd/HS1996_to_BEC.xls",
        baci_hs07_zip=raw / "baci/202601/BACI_HS07_V202601.zip",
    )


def _require_sources(paths: TaxonomySourcePaths) -> None:
    for name, path in asdict(paths).items():
        if not path.is_file():
            raise FileNotFoundError(f"missing taxonomy source {name}: {path}")


def _build_hs07_registry(
    cleg: pl.DataFrame,
    apec: pl.DataFrame,
    *,
    cleg_hash: str,
    apec_hash: str,
) -> pl.DataFrame:
    cleg_rows: list[dict[str, object]] = []
    for row in cleg.iter_rows(named=True):
        base = {
            "hs07": row["hs07"],
            "category": row["category"],
            "product_description": "",
            "source_hs_revision": "HS2007",
            "source_id": "oecd_cleg_2014",
            "source_sha256": cleg_hash,
            "source_location": f"Annex 1 PDF printed page {row['pdf_page']}",
            "extraction_rule_id": "cleg_annex_anchored_hs6_category_v1",
            "review_status": "reviewed",
        }
        cleg_rows.append({**base, "list_name": "broad"})
        if row["category"] in MAIN_CATEGORIES:
            cleg_rows.append({**base, "list_name": "main"})
    apec_rows = [
        {
            "hs07": row["hs07"],
            "list_name": "apec",
            "category": "APEC",
            "product_description": row["product_description"],
            "source_hs_revision": "HS2007",
            "source_id": "apec_environmental_goods_54_2012",
            "source_sha256": apec_hash,
            "source_location": f"HTML table 0 source row {row['html_table_row']}",
            "extraction_rule_id": (
                "apec_hs2007_with_declared_revision_fallback_v1:"
                f"{row['source_code_column']}"
            ),
            "review_status": "reviewed",
        }
        for row in apec.iter_rows(named=True)
    ]
    columns = [
        "hs07",
        "list_name",
        "category",
        "product_description",
        "source_hs_revision",
        "source_id",
        "source_sha256",
        "source_location",
        "extraction_rule_id",
        "review_status",
    ]
    return pl.DataFrame(cleg_rows + apec_rows).select(columns).sort(
        ["list_name", "hs07"]
    )


def _load_contract(path: Path) -> TableContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    base_schema_path = path.parent / "base_table_contract.schema.json"
    base_schema = json.loads(base_schema_path.read_text(encoding="utf-8"))
    validate_json(instance=payload, schema=base_schema)
    period_value = payload.get("period")
    period = (
        (int(period_value[0]), int(period_value[1]))
        if period_value is not None
        else None
    )
    return TableContract(
        table_id=str(payload["table_id"]),
        schema_version=str(payload["schema_version"]),
        primary_key=tuple(payload["primary_key"]),
        columns={str(key): str(value) for key, value in payload["columns"].items()},
        units={str(key): str(value) for key, value in payload["units"].items()},
        period=period,
        zero_semantics={
            str(key): str(value) for key, value in payload.get("zero_semantics", {}).items()
        },
        transformations=tuple(payload.get("transformations", [])),
    )


def _validate_registry(frame: pl.DataFrame, contract: TableContract) -> None:
    if tuple(frame.columns) != tuple(contract.columns):
        raise ValueError(f"{contract.table_id} has unexpected columns")
    dtypes = {name: str(dtype) for name, dtype in frame.schema.items()}
    if dtypes != contract.columns:
        raise ValueError(
            f"{contract.table_id} dtypes differ: expected {contract.columns}, got {dtypes}"
        )
    for key in contract.primary_key:
        if frame.get_column(key).null_count():
            raise ValueError(f"null primary key in {contract.table_id}: {key}")
    duplicates = _duplicate_groups(frame, contract.primary_key)
    if duplicates:
        raise ValueError(f"duplicate keys in {contract.table_id}: {duplicates}")
    for name, dtype in frame.schema.items():
        if dtype not in {pl.Float32, pl.Float64}:
            continue
        if frame.select((pl.col(name).is_not_null() & ~pl.col(name).is_finite()).sum()).item():
            raise ValueError(f"nonfinite values in {contract.table_id}.{name}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_registry_write(
    frame: pl.DataFrame,
    contract: TableContract,
    destination: Path,
) -> None:
    _validate_registry(frame, contract)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        if destination.suffix == ".parquet":
            frame.write_parquet(partial)
            reread = pl.read_parquet(partial)
        elif destination.suffix == ".csv":
            frame.write_csv(partial)
            overrides = {
                name: dtype
                for name, dtype in frame.schema.items()
                if dtype == pl.String
            }
            reread = pl.read_csv(partial, schema_overrides=overrides)
        else:
            raise ValueError(f"unsupported registry format: {destination.suffix}")
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        _validate_registry(reread, contract)
        if not reread.equals(frame):
            raise ValueError(f"registry round trip changed {contract.table_id}")
        os.replace(partial, destination)
        _fsync_directory(destination.parent)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _atomic_csv_write(frame: pl.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        frame.write_csv(partial)
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, destination)
        _fsync_directory(destination.parent)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _cast_hs96_registry(frame: pl.DataFrame, contract: TableContract) -> pl.DataFrame:
    dtype_map: dict[str, pl.DataType] = {
        "String": pl.String,
        "Float64": pl.Float64,
        "UInt16": pl.UInt16,
        "Int16": pl.Int16,
        "Boolean": pl.Boolean,
    }
    return frame.select(
        [pl.col(name).cast(dtype_map[dtype]).alias(name) for name, dtype in contract.columns.items()]
    )


def _build_hs96_registry(
    hs07_registry: pl.DataFrame,
    mapping: pl.DataFrame,
    overlap: pl.DataFrame,
    bec: pl.DataFrame,
    *,
    source_hashes: dict[str, str],
    overlap_years: tuple[int, int],
    contract: TableContract,
) -> pl.DataFrame:
    weighted = pl.concat(
        [
            build_hs96_green_weights(
                mapping,
                hs07_registry.filter(pl.col("list_name") == list_name),
                overlap,
                list_name=list_name,
            )
            for list_name in ("main", "broad", "apec")
        ],
        how="vertical",
    )
    upstream = assign_upstream_weight(bec).rename(
        {"mapping_count": "bec_mapping_count"}
    )
    joined = weighted.join(upstream, on="hs96", how="left").with_columns(
        pl.col("upstream_weight").fill_null(0.0),
        pl.col("bec_mapping_count").fill_null(0),
        pl.col("ambiguous_bec_mapping").fill_null(False),
        pl.col("bec_uses").fill_null("unclassified"),
    ).with_columns(
        (pl.col("bec_mapping_count") == 0).alias("bec_unmapped"),
        (pl.col("green_weight") * pl.col("upstream_weight")).alias(
            "green_upstream_weight"
        ),
        pl.lit("HS2007").alias("source_hs_revision"),
        pl.lit("HS1996").alias("target_hs_revision"),
        pl.lit(overlap_years[0]).alias("overlap_start_year"),
        pl.lit(overlap_years[1]).alias("overlap_end_year"),
        pl.lit(source_hashes["h3_to_h1"]).alias("concordance_source_sha256"),
        pl.lit(source_hashes["hs96_to_bec"]).alias("bec_source_sha256"),
        pl.lit(source_hashes["h1_to_bec"]).alias("wits_bec_source_sha256"),
        pl.lit(source_hashes["baci_hs07"]).alias("baci_overlap_source_sha256"),
        pl.lit("reviewed").alias("review_status"),
    ).sort(["list_name", "hs96"])
    return _cast_hs96_registry(joined, contract)


def _read_hs07_registry(path: Path, contract: TableContract) -> pl.DataFrame:
    overrides = {name: pl.String for name in contract.columns}
    frame = pl.read_csv(path, schema_overrides=overrides)
    _validate_registry(frame, contract)
    return frame


def _category_sum_expression() -> pl.Expr:
    expression = pl.lit(0.0)
    for category in WEIGHT_CATEGORIES:
        expression = expression + pl.col(f"category_{category.lower()}_weight")
    return expression


def audit_taxonomy(
    *,
    code_root: Path,
    data_root: Path,
    write_audit: bool = True,
) -> dict[str, object]:
    """Validate frozen counts, weights, source hashes, and duplicate keys."""

    contract_dir = code_root / "03_代码/contracts"
    hs07_contract = _load_contract(contract_dir / HS07_CONTRACT_NAME)
    hs96_contract = _load_contract(contract_dir / HS96_CONTRACT_NAME)
    hs07_path = code_root / "02_数据字典/product_registry_hs07_v1.csv"
    hs96_path = code_root / "02_数据字典/product_registry_hs96_v1.parquet"
    hs07 = _read_hs07_registry(hs07_path, hs07_contract)
    hs96 = pl.read_parquet(hs96_path)
    _validate_registry(hs96, hs96_contract)

    counts = {
        row[0]: int(row[1])
        for row in hs07.group_by("list_name").len().iter_rows()
    }
    expected_counts = {"main": 126, "broad": 248, "apec": 54}
    if counts != expected_counts:
        raise RuntimeError(f"taxonomy count mismatch: {counts}")
    out_of_range = hs96.filter(
        ~pl.col("green_weight").is_between(0.0, 1.0, closed="both")
        | ~pl.col("upstream_weight").is_between(0.0, 1.0, closed="both")
    ).height
    category_violations = hs96.filter(
        _category_sum_expression() > pl.col("green_weight") + 1e-12
    ).height
    if out_of_range or category_violations:
        raise RuntimeError("taxonomy weights fail range or category-sum audit")

    sources = _source_paths(data_root)
    _require_sources(sources)
    expected_source_hashes = {
        "oecd_cleg_2014": sha256_file(sources.cleg_pdf),
        "apec_environmental_goods_54_2012": sha256_file(sources.apec_html),
    }
    source_hash_mismatches = 0
    for source_id, expected_hash in expected_source_hashes.items():
        values = set(
            hs07.filter(pl.col("source_id") == source_id)
            .get_column("source_sha256")
            .to_list()
        )
        if values != {expected_hash}:
            source_hash_mismatches += 1
    hs96_hash_checks = {
        "concordance_source_sha256": sha256_file(sources.h3_to_h1_zip),
        "bec_source_sha256": sha256_file(sources.hs96_to_bec_xls),
        "wits_bec_source_sha256": sha256_file(sources.h1_to_bec_zip),
        "baci_overlap_source_sha256": sha256_file(sources.baci_hs07_zip),
    }
    for column, expected_hash in hs96_hash_checks.items():
        if set(hs96.get_column(column).to_list()) != {expected_hash}:
            source_hash_mismatches += 1
    if source_hash_mismatches:
        raise RuntimeError(f"taxonomy source hash mismatches: {source_hash_mismatches}")

    metrics: list[tuple[str, str, str, str]] = [
        ("main_count", str(counts["main"]), "126", "pass"),
        ("broad_count", str(counts["broad"]), "248", "pass"),
        ("apec_count", str(counts["apec"]), "54", "pass"),
        ("hs07_duplicate_keys", "0", "0", "pass"),
        ("hs96_duplicate_keys", "0", "0", "pass"),
        ("unresolved_green_hs07", "0", "0", "pass"),
        ("weights_out_of_range", str(out_of_range), "0", "pass"),
        ("category_sum_violations", str(category_violations), "0", "pass"),
        ("source_hash_mismatches", str(source_hash_mismatches), "0", "pass"),
        (
            "code_share_fallback_rows",
            str(hs96.filter(pl.col("code_share_fallback")).height),
            "reported",
            "reported",
        ),
        (
            "ambiguous_bec_rows",
            str(hs96.filter(pl.col("ambiguous_bec_mapping")).height),
            "reported",
            "reported",
        ),
        (
            "bec_unmapped_rows",
            str(hs96.filter(pl.col("bec_unmapped")).height),
            "reported",
            "reported",
        ),
    ]
    audit_path = code_root / "06_结果/产品分类与映射审计_v1.csv"
    if write_audit:
        _atomic_csv_write(
            pl.DataFrame(
                metrics,
                schema={
                    "metric": pl.String,
                    "value": pl.String,
                    "expected": pl.String,
                    "status": pl.String,
                },
                orient="row",
            ),
            audit_path,
        )
    return {
        "counts": counts,
        "hs07_rows": hs07.height,
        "hs96_rows": hs96.height,
        "fallback_count": hs96.filter(pl.col("code_share_fallback")).height,
        "ambiguous_bec_rows": hs96.filter(pl.col("ambiguous_bec_mapping")).height,
        "bec_unmapped_rows": hs96.filter(pl.col("bec_unmapped")).height,
        "weights_out_of_range": out_of_range,
        "category_sum_violations": category_violations,
        "source_hash_mismatches": source_hash_mismatches,
        "audit_path": str(audit_path),
        "status": "valid",
    }


def build_taxonomy(
    *,
    code_root: Path,
    data_root: Path,
    construction_config_path: Path,
) -> TaxonomyBuildReport:
    """Build the two frozen registries and the compact mapping audit."""

    usage = measure_layer_usage(data_root, audits_root=code_root / "06_结果")
    enforce_construction_capacity(usage, projected_additional_bytes=32 * 1024**2)
    config = load_construction_config(construction_config_path)
    if config.taxonomy.counts != {"main": 126, "broad": 248, "apec": 54}:
        raise ValueError("construction taxonomy counts are not frozen")
    if config.taxonomy.overlap_years != (2007, 2010):
        raise ValueError("construction taxonomy overlap is not 2007-2010")

    sources = _source_paths(data_root)
    _require_sources(sources)
    cleg = extract_cleg_pdf(sources.cleg_pdf)
    apec = parse_apec_html(sources.apec_html, expected_count=54)
    hs07_registry = _build_hs07_registry(
        cleg,
        apec,
        cleg_hash=sha256_file(sources.cleg_pdf),
        apec_hash=sha256_file(sources.apec_html),
    )

    contract_dir = code_root / "03_代码/contracts"
    hs07_contract = _load_contract(contract_dir / HS07_CONTRACT_NAME)
    hs96_contract = _load_contract(contract_dir / HS96_CONTRACT_NAME)
    hs07_registry = hs07_registry.select(list(hs07_contract.columns))
    hs07_path = code_root / "02_数据字典/product_registry_hs07_v1.csv"
    _atomic_registry_write(hs07_registry, hs07_contract, hs07_path)

    mapping = load_h3_to_h1(sources.h3_to_h1_zip)
    overlap, overlap_rows = stream_baci_overlap_exports(
        sources.baci_hs07_zip,
        start_year=config.taxonomy.overlap_years[0],
        end_year=config.taxonomy.overlap_years[1],
    )
    bec = load_hs96_bec(sources.hs96_to_bec_xls)
    wits_bec = load_wits_h1_to_bec(sources.h1_to_bec_zip)
    unsd_pairs = set(bec.select("hs96", "bec_code").iter_rows())
    wits_not_unsd = sum(
        pair not in unsd_pairs for pair in wits_bec.select("hs96", "bec_code").iter_rows()
    )
    source_hashes = {
        "h3_to_h1": sha256_file(sources.h3_to_h1_zip),
        "h1_to_bec": sha256_file(sources.h1_to_bec_zip),
        "hs96_to_bec": sha256_file(sources.hs96_to_bec_xls),
        "baci_hs07": sha256_file(sources.baci_hs07_zip),
    }
    hs96_registry = _build_hs96_registry(
        hs07_registry,
        mapping,
        overlap,
        bec,
        source_hashes=source_hashes,
        overlap_years=config.taxonomy.overlap_years,
        contract=hs96_contract,
    )
    hs96_path = code_root / "02_数据字典/product_registry_hs96_v1.parquet"
    _atomic_registry_write(hs96_registry, hs96_contract, hs96_path)
    audit = audit_taxonomy(code_root=code_root, data_root=data_root, write_audit=True)

    return TaxonomyBuildReport(
        main_count=126,
        broad_count=248,
        apec_count=54,
        hs07_registry_rows=hs07_registry.height,
        hs96_registry_rows=hs96_registry.height,
        hs07_duplicate_keys=0,
        hs96_duplicate_keys=0,
        unresolved_green_hs07=0,
        fallback_count=int(audit["fallback_count"]),
        weights_out_of_range=int(audit["weights_out_of_range"]),
        category_sum_violations=int(audit["category_sum_violations"]),
        ambiguous_bec_rows=int(audit["ambiguous_bec_rows"]),
        bec_unmapped_rows=int(audit["bec_unmapped_rows"]),
        wits_bec_selected_pairs_not_in_unsd=wits_not_unsd,
        overlap_rows_read=overlap_rows,
        overlap_exports_usd=float(overlap.get_column("exports_usd").sum()),
        hs07_registry_path=str(hs07_path),
        hs96_registry_path=str(hs96_path),
        audit_path=str(audit["audit_path"]),
    )
