"""Annual RCA/PCI construction and unstandardized green-trade inputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh

from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    verify_manifest,
    write_authoritative_table,
)
from green_debt.paths import ProjectPaths, resolve_project_paths


_CONTRACT_ROOT = Path(__file__).resolve().parents[2] / "contracts"
_GDP_INDICATOR = "NY.GDP.MKTP.CD"
_YEARS = tuple(range(1996, 2025))
_TAXONOMY_SOURCE_COUNTS = {"main": 126, "broad": 248, "apec": 54}
_TRADE_COMPONENT_NUMERIC_COLUMNS = (
    "green_imports_usd",
    "green_exports_usd",
    "gdp_current_usd",
    "green_import_intensity_raw",
    "green_import_complexity_raw",
    "gnir_raw",
)
_SUPPLIER_NUMERIC_COLUMNS = ("gud_raw", "grd_raw")


@dataclass(frozen=True)
class ComplexityBuildReport:
    years: tuple[int, ...]
    gpci_rows: int
    zero_diversity_economies: int
    zero_ubiquity_products: int
    output_path: str
    audit_path: str


@dataclass(frozen=True)
class TradeComponentsBuildReport:
    rows: int
    import_complexity_rows: int
    output_path: str


@dataclass(frozen=True)
class ComplexityAuditReport:
    gpci_years: tuple[int, ...]
    trade_component_years: tuple[int, ...]
    orientation_violations: int
    current_year_gpci_leakage_count: int
    nonfinite_values: int
    status: str


@dataclass(frozen=True)
class SupplierBuildReport:
    rows: int
    years: tuple[int, ...]
    gud_nonmissing_rows: int
    grd_nonmissing_rows: int
    nonfinite_values: int
    output_path: str
    audit_path: str


def _load_contract(name: str) -> TableContract:
    payload = json.loads((_CONTRACT_ROOT / name).read_text(encoding="utf-8"))
    period_value = payload.get("period")
    return TableContract(
        table_id=str(payload["table_id"]),
        schema_version=str(payload["schema_version"]),
        primary_key=tuple(str(value) for value in payload["primary_key"]),
        columns={str(key): str(value) for key, value in payload["columns"].items()},
        units={str(key): str(value) for key, value in payload["units"].items()},
        period=(int(period_value[0]), int(period_value[1]))
        if period_value is not None
        else None,
        zero_semantics={
            str(key): str(value)
            for key, value in payload.get("zero_semantics", {}).items()
        },
        transformations=tuple(str(value) for value in payload.get("transformations", [])),
    )


def _require_columns(frame: pl.DataFrame, columns: set[str], *, name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _reject_invalid_amount(frame: pl.DataFrame, column: str, *, label: str) -> None:
    invalid = frame.select(
        (
            pl.col(column).is_null()
            | ~pl.col(column).is_finite().fill_null(False)
            | (pl.col(column) < 0.0).fill_null(True)
        ).any()
    ).item()
    if invalid:
        raise ValueError(f"invalid {label}")


def assert_finite_trade_components(frame: pl.DataFrame) -> None:
    """Fail the pre-write boundary if any retained numeric component is nonfinite."""

    missing = [
        column for column in _TRADE_COMPONENT_NUMERIC_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"trade-component frame lacks numeric columns: {missing}")
    nonfinite = {
        column: int(
            frame.select(
                (pl.col(column).is_not_null() & ~pl.col(column).is_finite()).sum()
            ).item()
        )
        for column in _TRADE_COMPONENT_NUMERIC_COLUMNS
    }
    violations = {column: count for column, count in nonfinite.items() if count}
    if violations:
        raise ValueError(
            "nonfinite trade-component values before write: " f"{violations}"
        )


def compute_rca(exports: pl.DataFrame) -> pl.DataFrame:
    """Calculate annual Balassa RCA from all-product economy exports."""

    _require_columns(
        exports,
        {"economy_id", "hs6", "year", "export_usd"},
        name="exports",
    )
    source = exports.select("economy_id", "hs6", "year", "export_usd").with_columns(
        pl.col("economy_id").cast(pl.String),
        pl.col("hs6").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("export_usd").cast(pl.Float64),
    )
    _reject_invalid_amount(source, "export_usd", label="export trade value")
    prepared = (
        source
        .filter(
            pl.col("economy_id").is_not_null()
            & pl.col("hs6").is_not_null()
            & pl.col("year").is_not_null()
        )
        .group_by("economy_id", "hs6", "year")
        .agg(pl.col("export_usd").sum())
    )
    economy_totals = prepared.group_by("economy_id", "year").agg(
        pl.col("export_usd").sum().alias("_economy_total")
    )
    product_totals = prepared.group_by("hs6", "year").agg(
        pl.col("export_usd").sum().alias("_product_total")
    )
    world_totals = prepared.group_by("year").agg(
        pl.col("export_usd").sum().alias("_world_total")
    )
    return (
        prepared.join(economy_totals, on=["economy_id", "year"])
        .join(product_totals, on=["hs6", "year"])
        .join(world_totals, on="year")
        .with_columns(
            pl.when(
                (pl.col("_economy_total") > 0.0)
                & (pl.col("_product_total") > 0.0)
                & (pl.col("_world_total") > 0.0)
            )
            .then(
                (pl.col("export_usd") / pl.col("_economy_total"))
                / (pl.col("_product_total") / pl.col("_world_total"))
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("rca")
        )
        .select("economy_id", "hs6", "year", "export_usd", "rca")
        .sort(["year", "economy_id", "hs6"])
    )


def compute_product_proximity(
    incidence: pl.DataFrame, *, anchor_products: set[str] | None = None
) -> pl.DataFrame:
    """Build one sparse, annual, symmetric product-space edge list in memory."""

    _require_columns(
        incidence, {"economy_id", "hs6", "year", "rca_present"}, name="incidence"
    )
    present = (
        incidence.select("economy_id", "hs6", "year", "rca_present")
        .with_columns(
            pl.col("economy_id").cast(pl.String),
            pl.col("hs6").cast(pl.String),
            pl.col("year").cast(pl.Int16),
            pl.col("rca_present").cast(pl.Boolean).fill_null(False),
        )
        .filter(pl.col("rca_present"))
        .select("economy_id", "hs6", "year")
        .unique()
    )
    anchors = None if anchor_products is None else {str(value) for value in anchor_products}
    edge_years: list[int] = []
    edge_left: list[str] = []
    edge_right: list[str] = []
    edge_values: list[float] = []
    for year, annual in present.partition_by("year", as_dict=True).items():
        product_counts: Counter[str] = Counter()
        pair_counts: Counter[tuple[str, str]] = Counter()
        for group in annual.partition_by("economy_id", maintain_order=False):
            products = sorted(group.get_column("hs6").to_list())
            product_counts.update(products)
            for left_index, left in enumerate(products):
                for right in products[left_index + 1 :]:
                    if anchors is None or left in anchors or right in anchors:
                        pair_counts[(left, right)] += 1
        year_value = int(year[0] if isinstance(year, tuple) else year)
        for (left, right), coexporters in pair_counts.items():
            value = min(
                coexporters / product_counts[left], coexporters / product_counts[right]
            )
            edge_years.extend((year_value, year_value))
            edge_left.extend((left, right))
            edge_right.extend((right, left))
            edge_values.extend((value, value))
    return pl.DataFrame(
        {"year": edge_years, "hs6_a": edge_left, "hs6_b": edge_right, "proximity": edge_values},
        schema={"year": pl.Int16, "hs6_a": pl.String, "hs6_b": pl.String, "proximity": pl.Float64},
    ).sort(["year", "hs6_a", "hs6_b"])


def _supplier_output_from_adjacency(
    incidence: pl.DataFrame,
    weights: dict[str, float],
    coverage: pl.DataFrame,
    adjacency: dict[tuple[int, str], list[tuple[str, float]]],
) -> pl.DataFrame:
    capabilities: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in incidence.filter(pl.col("rca_present")).iter_rows(named=True):
        capabilities[(str(row["economy_id"]), int(row["year"]))].add(str(row["hs6"]))
    output: list[dict[str, object]] = []
    for row in coverage.sort(["year", "economy_id"]).iter_rows(named=True):
        economy, year = str(row["economy_id"]), int(row["year"])
        if not bool(row["export_coverage_normal"]):
            output.append({"economy_id": economy, "year": year, "gud_raw": None, "gud_raw_reason": "no_export_coverage", "grd_raw": None, "grd_raw_reason": "no_export_coverage"})
            continue
        present = capabilities[(economy, year)]
        gud = sum(weight for product, weight in weights.items() if product in present)
        opportunities = [(product, weight) for product, weight in weights.items() if product not in present]
        if not opportunities:
            grd, grd_reason = None, "no_upstream_opportunities"
        else:
            weighted_density, invalid_density = 0.0, False
            for product, weight in opportunities:
                neighbors = adjacency.get((year, product), [])
                denominator = sum(value for neighbor, value in neighbors if neighbor != product)
                if denominator <= 0.0:
                    invalid_density = True
                    break
                numerator = sum(value for neighbor, value in neighbors if neighbor in present and neighbor != product)
                weighted_density += weight * (numerator / denominator)
            total_weight = sum(weight for _, weight in opportunities)
            if invalid_density:
                grd, grd_reason = None, "zero_proximity_denominator"
            elif total_weight <= 0.0:
                grd, grd_reason = None, "no_upstream_opportunities"
            else:
                grd, grd_reason = weighted_density / total_weight, None
        output.append({"economy_id": economy, "year": year, "gud_raw": gud, "gud_raw_reason": None, "grd_raw": grd, "grd_raw_reason": grd_reason})
    return pl.DataFrame(
        output,
        schema={"economy_id": pl.String, "year": pl.Int16, "gud_raw": pl.Float64, "gud_raw_reason": pl.String, "grd_raw": pl.Float64, "grd_raw_reason": pl.String},
    ).sort(["year", "economy_id"])


def _annual_upstream_adjacency(
    incidence: pl.DataFrame, anchors: set[str]
) -> tuple[dict[tuple[int, str], list[tuple[str, float]]], int]:
    """Compute only anchor-to-product proximity rows with sparse multiplication."""

    present = incidence.filter(pl.col("rca_present")).select("economy_id", "hs6", "year").unique()
    if present.is_empty():
        return {}, 0
    years = present.get_column("year").unique().to_list()
    if len(years) != 1:
        raise ValueError("annual upstream adjacency requires exactly one year")
    year = int(years[0])
    economies = present.get_column("economy_id").unique().sort().to_list()
    products = present.get_column("hs6").unique().sort().to_list()
    product_index = {product: index for index, product in enumerate(products)}
    anchor_indices = [product_index[product] for product in sorted(anchors) if product in product_index]
    if not anchor_indices:
        return {}, 0
    economy_index = {economy: index for index, economy in enumerate(economies)}
    rows = np.fromiter((economy_index[row["economy_id"]] for row in present.iter_rows(named=True)), dtype=np.int32, count=present.height)
    columns = np.fromiter((product_index[row["hs6"]] for row in present.iter_rows(named=True)), dtype=np.int32, count=present.height)
    matrix = csr_matrix((np.ones(present.height), (rows, columns)), shape=(len(economies), len(products)))
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    coexport = (matrix[:, anchor_indices].T @ matrix).tocsr()
    adjacency: dict[tuple[int, str], list[tuple[str, float]]] = defaultdict(list)
    for anchor_row, anchor_index in enumerate(anchor_indices):
        anchor = products[anchor_index]
        for pointer in range(coexport.indptr[anchor_row], coexport.indptr[anchor_row + 1]):
            product_index_value = int(coexport.indices[pointer])
            if product_index_value == anchor_index:
                continue
            overlap = float(coexport.data[pointer])
            value = min(overlap / counts[anchor_index], overlap / counts[product_index_value])
            if value > 0.0:
                adjacency[(year, anchor)].append((products[product_index_value], value))
    return adjacency, int(sum(len(value) for value in adjacency.values()))


def _annual_supplier_raw_sparse(
    incidence: pl.DataFrame,
    weights: dict[str, float],
    coverage: pl.DataFrame,
) -> tuple[pl.DataFrame, int]:
    """Vectorize anchor-product density without materializing a product-pair table."""

    economies = coverage.get_column("economy_id").to_list()
    year = int(coverage.get_column("year").unique().item())
    anchors = sorted(weights)
    present = incidence.filter(pl.col("rca_present")).select("economy_id", "hs6").unique()
    if present.is_empty():
        blank = coverage.select("economy_id", "year").with_columns(
            pl.lit(None, dtype=pl.Float64).alias("gud_raw"),
            pl.lit("no_export_coverage", dtype=pl.String).alias("gud_raw_reason"),
            pl.lit(None, dtype=pl.Float64).alias("grd_raw"),
            pl.lit("no_export_coverage", dtype=pl.String).alias("grd_raw_reason"),
        )
        return blank, 0
    products = present.get_column("hs6").unique().sort().to_list()
    economy_index = {economy: index for index, economy in enumerate(economies)}
    product_index = {product: index for index, product in enumerate(products)}
    rows = np.fromiter((economy_index[row["economy_id"]] for row in present.iter_rows(named=True)), dtype=np.int32, count=present.height)
    columns = np.fromiter((product_index[row["hs6"]] for row in present.iter_rows(named=True)), dtype=np.int32, count=present.height)
    matrix = csr_matrix((np.ones(present.height), (rows, columns)), shape=(len(economies), len(products)))
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    anchor_indices = [product_index.get(anchor) for anchor in anchors]
    densities = np.full((len(economies), len(anchors)), np.nan, dtype=np.float64)
    has_anchor = np.zeros((len(economies), len(anchors)), dtype=np.float64)
    temporary_edges = 0
    for anchor_position, anchor_index in enumerate(anchor_indices):
        if anchor_index is None:
            continue
        has_anchor[:, anchor_position] = matrix[:, anchor_index].toarray().ravel()
        coexport = (matrix[:, anchor_index].T @ matrix).tocsr()
        proximity = np.zeros(len(products), dtype=np.float64)
        for pointer in range(coexport.indptr[0], coexport.indptr[1]):
            product_position = int(coexport.indices[pointer])
            if product_position == anchor_index:
                continue
            overlap = float(coexport.data[pointer])
            value = min(overlap / counts[anchor_index], overlap / counts[product_position])
            if value > 0.0:
                proximity[product_position] = value
                temporary_edges += 1
        denominator = float(proximity.sum())
        if denominator > 0.0:
            densities[:, anchor_position] = (matrix @ proximity) / denominator
    weight_values = np.asarray([weights[anchor] for anchor in anchors], dtype=np.float64)
    opportunities = 1.0 - has_anchor
    invalid_density = (opportunities.astype(bool) & ~np.isfinite(densities)).any(axis=1)
    total_opportunity_weight = opportunities @ weight_values
    grd = (np.nan_to_num(densities, nan=0.0) * opportunities) @ weight_values
    valid_grd = (~invalid_density) & (total_opportunity_weight > 0.0)
    raw_grd = np.where(valid_grd, grd / np.where(total_opportunity_weight > 0.0, total_opportunity_weight, 1.0), np.nan)
    coverage_values = coverage.get_column("export_coverage_normal").to_numpy().astype(bool)
    gud = has_anchor @ weight_values
    output = pl.DataFrame({
        "economy_id": economies,
        "year": [year] * len(economies),
        "gud_raw": [float(value) if covered else None for value, covered in zip(gud, coverage_values, strict=True)],
        "gud_raw_reason": [None if covered else "no_export_coverage" for covered in coverage_values],
        "grd_raw": [float(value) if covered and math.isfinite(float(value)) else None for value, covered in zip(raw_grd, coverage_values, strict=True)],
        "grd_raw_reason": [
            None if covered and valid else (
                "no_export_coverage" if not covered else (
                    "no_upstream_opportunities" if total <= 0.0 else "zero_proximity_denominator"
                )
            )
            for covered, valid, total in zip(coverage_values, valid_grd, total_opportunity_weight, strict=True)
        ],
    }, schema={"economy_id": pl.String, "year": pl.Int16, "gud_raw": pl.Float64, "gud_raw_reason": pl.String, "grd_raw": pl.Float64, "grd_raw_reason": pl.String})
    return output.sort(["year", "economy_id"]), temporary_edges


def compute_supplier_raw(
    incidence: pl.DataFrame,
    upstream: pl.DataFrame,
    proximity: pl.DataFrame,
    *,
    economy_years: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute unstandardized upstream diversity and opportunity relatedness."""

    _require_columns(
        incidence, {"economy_id", "hs6", "year", "rca_present"}, name="incidence"
    )
    _require_columns(upstream, {"hs6", "upstream_weight"}, name="upstream")
    _require_columns(proximity, {"year", "hs6_a", "hs6_b", "proximity"}, name="proximity")
    incidence_clean = incidence.select("economy_id", "hs6", "year", "rca_present").with_columns(
        pl.col("economy_id").cast(pl.String),
        pl.col("hs6").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("rca_present").cast(pl.Boolean).fill_null(False),
    ).unique()
    upstream_clean = upstream.select("hs6", "upstream_weight").with_columns(
        pl.col("hs6").cast(pl.String), pl.col("upstream_weight").cast(pl.Float64)
    )
    invalid_weights = upstream_clean.filter(
        pl.col("upstream_weight").is_null()
        | ~pl.col("upstream_weight").is_finite().fill_null(False)
        | (pl.col("upstream_weight") < 0.0).fill_null(True)
    ).height
    if invalid_weights or upstream_clean.group_by("hs6").len().filter(pl.col("len") > 1).height:
        raise ValueError("upstream weights must be unique, finite, and nonnegative")
    weights = {
        str(row["hs6"]): float(row["upstream_weight"])
        for row in upstream_clean.filter(pl.col("upstream_weight") > 0.0).iter_rows(named=True)
    }
    if economy_years is None:
        coverage = incidence_clean.select("economy_id", "year").unique().with_columns(
            pl.lit(True).alias("export_coverage_normal")
        )
    else:
        _require_columns(economy_years, {"economy_id", "year"}, name="economy_years")
        coverage = economy_years.select(
            "economy_id", "year",
            *(["export_coverage_normal"] if "export_coverage_normal" in economy_years.columns else []),
        ).with_columns(
            pl.col("economy_id").cast(pl.String), pl.col("year").cast(pl.Int16),
        )
        if "export_coverage_normal" not in coverage.columns:
            coverage = coverage.with_columns(pl.lit(False).alias("export_coverage_normal"))
        else:
            coverage = coverage.with_columns(
                pl.col("export_coverage_normal").cast(pl.Boolean).fill_null(False)
            )
        if coverage.group_by(["economy_id", "year"]).len().filter(pl.col("len") > 1).height:
            raise ValueError("economy_years has duplicate keys")
    invalid_proximity = proximity.filter(
        pl.col("proximity").is_null()
        | ~pl.col("proximity").cast(pl.Float64).is_finite().fill_null(False)
        | ~pl.col("proximity").cast(pl.Float64).is_between(0.0, 1.0, closed="both").fill_null(False)
    ).height
    if invalid_proximity:
        raise ValueError("product proximity must be finite and in [0, 1]")
    adjacency: dict[tuple[int, str], list[tuple[str, float]]] = defaultdict(list)
    for row in proximity.select("year", "hs6_a", "hs6_b", "proximity").iter_rows(named=True):
        year = int(row["year"])
        left, right, value = str(row["hs6_a"]), str(row["hs6_b"]), float(row["proximity"])
        if left != right:
            adjacency[(year, left)].append((right, value))
    return _supplier_output_from_adjacency(incidence_clean, weights, coverage, adjacency)


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = float(values.mean())
    deviation = float(values.std(ddof=0))
    if not np.isfinite(deviation) or deviation <= 0.0:
        raise ValueError("PCI solution is non-identifiable because its variance is zero")
    return (values - mean) / deviation


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size < 2:
        raise ValueError("PCI orientation is non-identifiable")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("PCI orientation is non-identifiable")
    value = float(np.dot(left_centered, right_centered) / denominator)
    if not np.isfinite(value):
        raise ValueError("PCI orientation is non-real")
    return value


def _annual_gpci(
    rca: pl.DataFrame,
    year: int,
    *,
    economy_universe: tuple[str, ...] = (),
    product_universe: tuple[str, ...] = (),
) -> tuple[pl.DataFrame, dict[str, object]]:
    annual = rca.filter(pl.col("year") == year)
    economies = sorted(
        set(annual.get_column("economy_id").unique().to_list()).union(economy_universe)
    )
    products = sorted(
        set(annual.get_column("hs6").unique().to_list()).union(product_universe)
    )
    economy_index = {value: index for index, value in enumerate(economies)}
    product_index = {value: index for index, value in enumerate(products)}
    incidence_rows = annual.filter(pl.col("rca") >= 1.0).select("economy_id", "hs6")
    row = np.fromiter(
        (economy_index[value] for value in incidence_rows.get_column("economy_id")),
        dtype=np.int32,
        count=incidence_rows.height,
    )
    column = np.fromiter(
        (product_index[value] for value in incidence_rows.get_column("hs6")),
        dtype=np.int32,
        count=incidence_rows.height,
    )
    matrix = csr_matrix(
        (np.ones(incidence_rows.height, dtype=np.float64), (row, column)),
        shape=(len(economies), len(products)),
    )
    diversity = np.asarray(matrix.sum(axis=1)).ravel()
    ubiquity = np.asarray(matrix.sum(axis=0)).ravel()
    retained_economies = diversity > 0.0
    retained_products = ubiquity > 0.0
    zero_diversity = int((~retained_economies).sum())
    zero_ubiquity = int((~retained_products).sum())
    zero_diversity_codes = np.asarray(economies, dtype=object)[
        ~retained_economies
    ].tolist()
    zero_ubiquity_codes = np.asarray(products, dtype=object)[
        ~retained_products
    ].tolist()
    matrix = matrix[retained_economies][:, retained_products].tocsr()
    diversity = diversity[retained_economies]
    ubiquity = ubiquity[retained_products]
    retained_product_codes = np.asarray(products, dtype=object)[retained_products]
    if matrix.shape[0] < 2 or matrix.shape[1] < 3:
        raise ValueError(f"PCI network is non-identifiable in {year}")

    inverse_sqrt_ubiquity = 1.0 / np.sqrt(ubiquity)
    inverse_diversity = 1.0 / diversity

    def multiply(vector: np.ndarray) -> np.ndarray:
        scaled = inverse_sqrt_ubiquity * vector
        country = matrix @ scaled
        return inverse_sqrt_ubiquity * (matrix.T @ (inverse_diversity * country))

    if matrix.shape[1] <= 128:
        dense_incidence = matrix.toarray()
        symmetric_product_matrix = (
            inverse_sqrt_ubiquity[:, None]
            * dense_incidence.T
            @ (inverse_diversity[:, None] * dense_incidence)
            * inverse_sqrt_ubiquity[None, :]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric_product_matrix)
    else:
        operator = LinearOperator(
            (matrix.shape[1], matrix.shape[1]), matvec=multiply, dtype=np.float64
        )
        try:
            eigenvalues, eigenvectors = eigsh(
                operator,
                k=3,
                which="LA",
                v0=np.linspace(1.0, 2.0, matrix.shape[1]),
                tol=1e-10,
                maxiter=max(500, matrix.shape[1] * 10),
            )
        except Exception as exc:  # scipy exposes several solver-specific failures.
            raise ValueError(f"PCI eigensolver failed in {year}: {exc}") from exc
    order = np.argsort(eigenvalues)[::-1]
    top, second, third = (eigenvalues[order[index]] for index in range(3))
    scale = max(1.0, abs(float(top)), abs(float(second)), abs(float(third)))
    tolerance = 1e-10 * scale
    if not np.isfinite(top) or not np.isfinite(second) or abs(top - second) <= tolerance:
        raise ValueError(f"PCI eigenvector is non-identifiable in {year}")
    if not np.isfinite(third) or abs(second - third) <= tolerance:
        raise ValueError(f"PCI second eigengap is non-identifiable in {year}")
    raw_pci = inverse_sqrt_ubiquity * eigenvectors[:, order[1]]
    if not np.isrealobj(raw_pci) or not np.isfinite(raw_pci).all():
        raise ValueError(f"PCI eigenvector is non-real in {year}")
    gpci = _zscore(np.asarray(raw_pci, dtype=np.float64))

    exporter_diversity = np.asarray(matrix.T @ diversity).ravel() / ubiquity
    anchor = _zscore(-ubiquity) + _zscore(exporter_diversity)
    orientation_score = _pearson(gpci, anchor)
    if orientation_score < 0.0:
        gpci = -gpci
        orientation_score = -orientation_score
    if orientation_score < 0.0:
        raise ValueError(f"PCI orientation remains negative in {year}")
    frame = pl.DataFrame(
        {
            "hs6": retained_product_codes.tolist(),
            "year": [year] * len(retained_product_codes),
            "gpci": gpci.tolist(),
            "orientation_score": [orientation_score] * len(retained_product_codes),
            "network_economies": [int(matrix.shape[0])] * len(retained_product_codes),
            "network_products": [int(matrix.shape[1])] * len(retained_product_codes),
            "zero_diversity_economies": [zero_diversity] * len(retained_product_codes),
            "zero_ubiquity_products": [zero_ubiquity] * len(retained_product_codes),
        },
        schema={
            "hs6": pl.String,
            "year": pl.Int16,
            "gpci": pl.Float64,
            "orientation_score": pl.Float64,
            "network_economies": pl.UInt32,
            "network_products": pl.UInt32,
            "zero_diversity_economies": pl.UInt32,
            "zero_ubiquity_products": pl.UInt32,
        },
    ).sort("hs6")
    return frame, {
        "year": year,
        "network_economies": int(matrix.shape[0]),
        "network_products": int(matrix.shape[1]),
        "zero_diversity_economies": zero_diversity,
        "zero_ubiquity_products": zero_ubiquity,
        "zero_diversity_economy_codes": zero_diversity_codes,
        "zero_ubiquity_product_codes": zero_ubiquity_codes,
    }


def compute_gpci_with_audit(
    exports: pl.DataFrame,
    *,
    economy_universe: tuple[str, ...] = (),
    product_universe: tuple[str, ...] = (),
) -> tuple[pl.DataFrame, list[dict[str, object]]]:
    """Construct PCI and retain sparse-universe network exclusions by year."""

    rca = compute_rca(exports)
    years = rca.get_column("year").unique().sort().to_list()
    if not years:
        raise ValueError("cannot construct PCI from empty exports")
    annual = [
        _annual_gpci(
            rca,
            int(year),
            economy_universe=economy_universe,
            product_universe=product_universe,
        )
        for year in years
    ]
    return pl.concat([item[0] for item in annual]).sort(["year", "hs6"]), [
        item[1] for item in annual
    ]


def compute_gpci(
    exports: pl.DataFrame,
    *,
    economy_universe: tuple[str, ...] = (),
    product_universe: tuple[str, ...] = (),
) -> pl.DataFrame:
    """Construct annual product complexity on the complete all-product network."""

    return compute_gpci_with_audit(
        exports,
        economy_universe=economy_universe,
        product_universe=product_universe,
    )[0]


def compute_trade_components(
    imports: pl.DataFrame,
    gpci: pl.DataFrame,
    gdp: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate raw GIMC inputs with a mandatory product-complexity lag."""

    _require_columns(
        imports,
        {
            "economy_id",
            "hs6",
            "year",
            "weighted_green_import_usd",
            "trade_coverage_normal",
        },
        name="imports",
    )
    _require_columns(gpci, {"hs6", "year", "gpci"}, name="gpci")
    _require_columns(gdp, {"economy_id", "year", "gdp_current_usd"}, name="gdp")
    prepared = imports.select(
        "economy_id",
        "hs6",
        "year",
        "weighted_green_import_usd",
        "trade_coverage_normal",
        *(
            ["weighted_green_export_usd"]
            if "weighted_green_export_usd" in imports.columns
            else []
        ),
    ).with_columns(
        pl.col("economy_id").cast(pl.String),
        pl.col("hs6").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("weighted_green_import_usd").cast(pl.Float64),
        pl.col("trade_coverage_normal").cast(pl.Boolean).fill_null(False),
    )
    _reject_invalid_amount(
        prepared, "weighted_green_import_usd", label="green import trade value"
    )
    if "weighted_green_export_usd" not in prepared.columns:
        prepared = prepared.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("weighted_green_export_usd"),
            pl.lit(False).alias("_exports_available"),
        )
    else:
        prepared = prepared.with_columns(
            pl.col("weighted_green_export_usd").cast(pl.Float64),
            pl.lit(True).alias("_exports_available"),
        )
        _reject_invalid_amount(
            prepared, "weighted_green_export_usd", label="green export trade value"
        )
    invalid_gpci = gpci.select(
        (
            pl.col("gpci").is_not_null()
            & ~pl.col("gpci").cast(pl.Float64).is_finite().fill_null(False)
        ).any()
    ).item()
    if invalid_gpci:
        raise ValueError("invalid GPCI value")
    lagged_gpci = gpci.select(
        pl.col("hs6").cast(pl.String),
        pl.col("year").cast(pl.Int16).alias("_gpci_source_year"),
        (pl.col("year").cast(pl.Int16) + 1).alias("year"),
        pl.col("gpci").cast(pl.Float64).alias("_lagged_gpci"),
    )
    weighted = prepared.join(lagged_gpci, on=["hs6", "year"], how="left")
    by_economy_year = weighted.group_by("economy_id", "year").agg(
        pl.col("weighted_green_import_usd").sum().alias("green_imports_usd"),
        pl.col("weighted_green_export_usd").sum().alias("green_exports_usd"),
        pl.col("_exports_available").all().alias("_exports_available"),
        pl.col("trade_coverage_normal").all().alias("trade_coverage_normal"),
        (pl.col("weighted_green_import_usd") > 0.0)
        .sum()
        .alias("_positive_import_product_rows"),
        (
            (pl.col("weighted_green_import_usd") > 0.0)
            & pl.col("_lagged_gpci").is_not_null()
        )
        .sum()
        .alias("_lagged_gpci_positive_import_rows"),
        pl.when(pl.col("weighted_green_import_usd") > 0.0)
        .then(pl.col("_gpci_source_year"))
        .otherwise(pl.lit(None, dtype=pl.Int16))
        .min()
        .alias("gpci_source_year"),
        (pl.col("weighted_green_import_usd") * pl.col("_lagged_gpci"))
        .sum()
        .alias("_complexity_numerator"),
    )
    gdp_prepared = gdp.select("economy_id", "year", "gdp_current_usd").with_columns(
        pl.col("economy_id").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("gdp_current_usd").cast(pl.Float64),
    )
    combined = by_economy_year.join(
        gdp_prepared, on=["economy_id", "year"], how="left"
    )
    combined = combined.with_columns(
        pl.when(pl.col("_exports_available"))
        .then(pl.col("green_exports_usd"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("green_exports_usd")
    ).with_columns(
        (
            pl.col("gdp_current_usd").is_not_null()
            & ~pl.col("gdp_current_usd").is_finite().fill_null(False)
        ).alias("_nonfinite_gdp"),
        pl.when(
            pl.col("gdp_current_usd").is_not_null()
            & pl.col("gdp_current_usd").is_finite().fill_null(False)
        )
        .then(pl.col("gdp_current_usd"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("gdp_current_usd"),
    )
    return (
        combined.with_columns(
            pl.when(~pl.col("trade_coverage_normal"))
            .then(pl.lit("abnormal_trade_coverage"))
            .when(pl.col("_nonfinite_gdp"))
            .then(pl.lit("nonfinite_gdp"))
            .when(pl.col("gdp_current_usd").is_null())
            .then(pl.lit("missing_gdp"))
            .when(pl.col("gdp_current_usd") <= 0.0)
            .then(pl.lit("nonpositive_gdp"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("green_import_intensity_reason"),
            pl.when(~pl.col("trade_coverage_normal"))
            .then(pl.lit("abnormal_trade_coverage"))
            .when(pl.col("green_imports_usd") <= 0.0)
            .then(pl.lit("nonpositive_green_imports"))
            .when(
                pl.col("_lagged_gpci_positive_import_rows")
                != pl.col("_positive_import_product_rows")
            )
            .then(pl.lit("missing_lagged_gpci"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("green_import_complexity_reason"),
            pl.when(~pl.col("trade_coverage_normal"))
            .then(pl.lit("abnormal_trade_coverage"))
            .when(pl.col("green_exports_usd").is_null())
            .then(pl.lit("missing_green_exports"))
            .when((pl.col("green_imports_usd") + pl.col("green_exports_usd")) <= 0.0)
            .then(pl.lit("nonpositive_green_trade"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("gnir_reason"),
        )
        .with_columns(
            pl.when(pl.col("green_import_intensity_reason").is_null())
            .then((1.0 + pl.col("green_imports_usd") / pl.col("gdp_current_usd")).log())
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("green_import_intensity_raw"),
            pl.when(pl.col("green_import_complexity_reason").is_null())
            .then(pl.col("_complexity_numerator") / pl.col("green_imports_usd"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("green_import_complexity_raw"),
            pl.when(pl.col("gnir_reason").is_null())
            .then(
                pl.col("green_imports_usd")
                / (pl.col("green_imports_usd") + pl.col("green_exports_usd"))
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("gnir_raw"),
        )
        .with_columns(
            (pl.col("year") - 1).cast(pl.Int16).alias("gpci_lag_year"),
            pl.col("gpci_source_year").cast(pl.Int16),
        )
        .select(
            "economy_id", "year", "green_imports_usd", "green_exports_usd",
            "gdp_current_usd", "trade_coverage_normal", "green_import_intensity_raw",
            "green_import_intensity_reason", "green_import_complexity_raw",
            "green_import_complexity_reason", "gnir_raw", "gnir_reason", "gpci_lag_year",
            "gpci_source_year",
        )
        .sort(["year", "economy_id"])
    )


def _green_registry(
    code_root: Path, taxonomy: str
) -> tuple[pl.DataFrame, tuple[str, ...], Path]:
    if not taxonomy.endswith("_hs96"):
        raise ValueError("complexity construction requires an HS96 taxonomy")
    list_name = taxonomy.removesuffix("_hs96")
    path = code_root / "02_数据字典/product_registry_hs96_v1.parquet"
    universe = (
        pl.read_parquet(path)
        .filter(pl.col("list_name") == list_name)
        .select(pl.col("hs96").alias("hs6").cast(pl.String))
        .unique()
        .sort("hs6")
    )
    registry = (
        pl.read_parquet(path)
        .filter((pl.col("list_name") == list_name) & (pl.col("green_weight") > 0.0))
        .select(pl.col("hs96").alias("hs6").cast(pl.String))
        .unique()
        .sort("hs6")
    )
    if registry.is_empty():
        raise ValueError(f"taxonomy has no positive HS96 green products: {taxonomy}")
    return registry, tuple(universe.get_column("hs6").to_list()), path


def _supplier_upstream_weights(code_root: Path, taxonomy: str) -> tuple[pl.DataFrame, Path]:
    if not taxonomy.endswith("_hs96"):
        raise ValueError("supplier construction requires an HS96 taxonomy")
    list_name = taxonomy.removesuffix("_hs96")
    path = code_root / "02_数据字典/product_registry_hs96_v1.parquet"
    weights = (
        pl.read_parquet(path)
        .filter(pl.col("list_name") == list_name)
        .select(
            pl.col("hs96").cast(pl.String).alias("hs6"),
            pl.col("green_upstream_weight").cast(pl.Float64).alias("upstream_weight"),
        )
        .filter(pl.col("upstream_weight") > 0.0)
        .sort("hs6")
    )
    if weights.is_empty():
        raise ValueError(f"taxonomy has no positive green-upstream weights: {taxonomy}")
    return weights, path


def _supplier_export_coverage(paths: ProjectPaths, year: int) -> tuple[pl.DataFrame, Path]:
    """Retain the annual BACI economy skeleton, including non-exporter cells."""

    source = (
        paths.normalized / "baci/economy_year_totals" / f"year={year}"
        / "taxonomy_version=all_hs96.parquet"
    )
    totals = pl.read_parquet(source)
    _require_columns(
        totals, {"economy_id", "year", "reported_as_exporter"}, name="BACI totals"
    )
    coverage = totals.select(
        pl.col("economy_id").cast(pl.String),
        pl.col("year").cast(pl.Int16),
        pl.col("reported_as_exporter").cast(pl.Boolean).fill_null(False).alias(
            "export_coverage_normal"
        ),
    )
    if coverage.group_by(["economy_id", "year"]).len().filter(pl.col("len") > 1).height:
        raise ValueError("BACI supplier coverage has duplicate economy-year keys")
    return coverage.sort(["year", "economy_id"]), source


def _baci_economy_universe(code_root: Path) -> tuple[tuple[str, ...], Path]:
    path = code_root / "02_数据字典/economy_crosswalk_v1.csv"
    economies = (
        pl.read_csv(path, schema_overrides={"source_code": pl.String}, null_values="")
        .filter((pl.col("source_id") == "baci") & pl.col("economy_id").is_not_null())
        .select(pl.col("economy_id").cast(pl.String))
        .unique()
        .sort("economy_id")
    )
    if economies.is_empty():
        raise ValueError("BACI economy universe is empty")
    return tuple(economies.get_column("economy_id").to_list()), path


def _validate_taxonomy_source_counts(code_root: Path) -> tuple[dict[str, int], Path]:
    """Verify frozen source-list memberships before using their HS96 weights."""

    path = code_root / "02_数据字典/product_registry_hs07_v1.csv"
    registry = pl.read_csv(path, schema_overrides={"hs07": pl.String}, null_values="")
    counts = {
        name: registry.filter(pl.col("list_name") == name)
        .select("hs07")
        .n_unique()
        for name in _TAXONOMY_SOURCE_COUNTS
    }
    if counts != _TAXONOMY_SOURCE_COUNTS:
        raise ValueError(
            "frozen HS07 registry counts differ: "
            f"expected {_TAXONOMY_SOURCE_COUNTS}, got {counts}"
        )
    return counts, path


def build_complexity(
    paths: ProjectPaths,
    *,
    taxonomy: str,
    build: BuildIdentity,
) -> ComplexityBuildReport:
    """Build GPCI only after each annual all-product network is solved."""

    registry, product_universe, registry_path = _green_registry(paths.code_root, taxonomy)
    economy_universe, economy_path = _baci_economy_universe(paths.code_root)
    source_registry_counts, source_registry_path = _validate_taxonomy_source_counts(
        paths.code_root
    )
    annual_inputs: list[InputArtifact] = [
        InputArtifact.from_path(registry_path),
        InputArtifact.from_path(economy_path),
        InputArtifact.from_path(source_registry_path),
    ]
    frames: list[pl.DataFrame] = []
    annual_audits: list[dict[str, object]] = []
    for year in _YEARS:
        source = (
            paths.normalized
            / "baci/exporter_product"
            / f"year={year}"
            / "taxonomy_version=all_hs96.parquet"
        )
        annual_inputs.append(InputArtifact.from_path(source))
        exports = pl.read_parquet(source).select(
            "economy_id", "hs6", "year", pl.col("trade_value_usd").alias("export_usd")
        )
        pci, audits = compute_gpci_with_audit(
            exports,
            economy_universe=economy_universe,
            product_universe=product_universe,
        )
        annual_audits.extend(audits)
        frames.append(
            pci.join(registry, on="hs6", how="inner")
            .with_columns(pl.lit(taxonomy, dtype=pl.String).alias("taxonomy_version"))
            .select(
                "taxonomy_version", "year", "hs6", "gpci", "orientation_score",
                "network_economies", "network_products", "zero_diversity_economies",
                "zero_ubiquity_products",
            )
        )
    output = pl.concat(frames).with_columns(
        pl.col("year").cast(pl.Int16),
        pl.col("gpci").cast(pl.Float64),
        pl.col("orientation_score").cast(pl.Float64),
        pl.col("network_economies").cast(pl.UInt32),
        pl.col("network_products").cast(pl.UInt32),
        pl.col("zero_diversity_economies").cast(pl.UInt32),
        pl.col("zero_ubiquity_products").cast(pl.UInt32),
    ).sort(["year", "hs6"])
    destination = paths.measures / "trade/gpci_product_year.parquet"
    write_authoritative_table(
        output,
        _load_contract("gpci_product_year.json"),
        destination,
        tuple(annual_inputs),
        build,
    )
    audit_path = paths.measures / "trade/complexity_network_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    orientation = (
        output.select("year", "orientation_score")
        .unique()
        .sort("year")
        .to_dicts()
    )
    orientation_by_year = {
        int(item["year"]): float(item["orientation_score"]) for item in orientation
    }
    year_records = [
        {
            **audit,
            "orientation_score": orientation_by_year[int(audit["year"])],
        }
        for audit in annual_audits
    ]
    audit_path.write_text(
        json.dumps(
            {
                "taxonomy_version": taxonomy,
                "source_registry_counts": source_registry_counts,
                "positive_hs96_weight_count": registry.height,
                "full_hs96_universe_count": len(product_universe),
                "economy_universe_count": len(economy_universe),
                "years": year_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ComplexityBuildReport(
        years=_YEARS,
        gpci_rows=output.height,
        zero_diversity_economies=sum(
            int(item["zero_diversity_economies"]) for item in year_records
        ),
        zero_ubiquity_products=sum(
            int(item["zero_ubiquity_products"]) for item in year_records
        ),
        output_path=str(destination),
        audit_path=str(audit_path),
    )


def _trade_component_input(paths: ProjectPaths, taxonomy: str) -> tuple[pl.DataFrame, tuple[Path, ...]]:
    frames: list[pl.DataFrame] = []
    input_paths: list[Path] = []
    for year in _YEARS:
        green_path = (
            paths.normalized / "baci/green_economy_product" / f"year={year}"
            / f"taxonomy_version={taxonomy}.parquet"
        )
        totals_path = (
            paths.normalized / "baci/economy_year_totals" / f"year={year}"
            / "taxonomy_version=all_hs96.parquet"
        )
        input_paths.extend((green_path, totals_path))
        green = pl.read_parquet(green_path)
        imports = green.filter(pl.col("flow_role") == "importer").select(
            "economy_id", "hs6", "year",
            pl.col("weighted_green_trade_usd").alias("weighted_green_import_usd"),
        )
        exports = green.filter(pl.col("flow_role") == "exporter").select(
            "economy_id", "hs6", "year",
            pl.col("weighted_green_trade_usd").alias("weighted_green_export_usd"),
        )
        products = imports.join(exports, on=["economy_id", "hs6", "year"], how="full", coalesce=True)
        coverage = pl.read_parquet(totals_path).select(
            "economy_id", "year",
            (pl.col("reported_as_importer") & pl.col("reported_as_exporter"))
            .alias("trade_coverage_normal"),
        )
        covered_products = coverage.join(products, on=["economy_id", "year"], how="left").with_columns(
            pl.col("weighted_green_import_usd").fill_null(0.0),
            pl.col("weighted_green_export_usd").fill_null(0.0),
        )
        frames.append(covered_products)
    return pl.concat(frames), tuple(input_paths)


def build_trade_components(
    paths: ProjectPaths,
    *,
    taxonomy: str,
    build: BuildIdentity,
) -> TradeComponentsBuildReport:
    """Build raw GIMC and GNIR inputs from normalized annual aggregates."""

    trade, input_paths = _trade_component_input(paths, taxonomy)
    gpci_path = paths.measures / "trade/gpci_product_year.parquet"
    wdi_path = paths.normalized / "wdi/wdi_country_year.parquet"
    gpci = pl.read_parquet(gpci_path).filter(pl.col("taxonomy_version") == taxonomy)
    gdp = pl.read_parquet(wdi_path).filter(pl.col("indicator_id") == _GDP_INDICATOR).select(
        "economy_id", "year", pl.col("value").alias("gdp_current_usd")
    )
    components = compute_trade_components(trade, gpci, gdp).with_columns(
        pl.lit(taxonomy, dtype=pl.String).alias("taxonomy_version")
    ).select(
        "taxonomy_version", "economy_id", "year", "green_imports_usd", "green_exports_usd",
        "gdp_current_usd", "trade_coverage_normal", "green_import_intensity_raw",
        "green_import_intensity_reason", "green_import_complexity_raw",
        "green_import_complexity_reason", "gnir_raw", "gnir_reason", "gpci_lag_year",
        "gpci_source_year",
    ).with_columns(
        pl.col("year").cast(pl.Int16),
        pl.col("gpci_lag_year").cast(pl.Int16),
        pl.col("gpci_source_year").cast(pl.Int16),
        pl.col("green_imports_usd").cast(pl.Float64),
        pl.col("green_exports_usd").cast(pl.Float64),
        pl.col("gdp_current_usd").cast(pl.Float64),
        pl.col("green_import_intensity_raw").cast(pl.Float64),
        pl.col("green_import_complexity_raw").cast(pl.Float64),
        pl.col("gnir_raw").cast(pl.Float64),
    ).sort(["year", "economy_id"])
    destination = paths.measures / "trade/trade_components_raw.parquet"
    assert_finite_trade_components(components)
    write_authoritative_table(
        components,
        _load_contract("trade_components_raw.json"),
        destination,
        tuple(InputArtifact.from_path(path) for path in (*input_paths, gpci_path, wdi_path)),
        build,
    )
    return TradeComponentsBuildReport(
        rows=components.height,
        import_complexity_rows=components.filter(
            pl.col("green_import_complexity_raw").is_not_null()
        ).height,
        output_path=str(destination),
    )


def build_supplier_raw(
    paths: ProjectPaths,
    *,
    taxonomy: str,
    build: BuildIdentity,
) -> SupplierBuildReport:
    """Build annual supplier capability without persisting product-product edges."""

    upstream, registry_path = _supplier_upstream_weights(paths.code_root, taxonomy)
    upstream_weights = {
        str(row["hs6"]): float(row["upstream_weight"])
        for row in upstream.iter_rows(named=True)
    }
    inputs: list[InputArtifact] = [InputArtifact.from_path(registry_path)]
    frames: list[pl.DataFrame] = []
    annual_audits: list[dict[str, object]] = []
    for year in _YEARS:
        source = (
            paths.normalized / "baci/exporter_product" / f"year={year}"
            / "taxonomy_version=all_hs96.parquet"
        )
        inputs.append(InputArtifact.from_path(source))
        exports = pl.read_parquet(source).select(
            "economy_id", "hs6", "year", pl.col("trade_value_usd").alias("export_usd")
        )
        rca = compute_rca(exports)
        incidence = rca.select(
            "economy_id", "hs6", "year", (pl.col("rca") >= 1.0).fill_null(False).alias("rca_present")
        )
        coverage, coverage_source = _supplier_export_coverage(paths, year)
        inputs.append(InputArtifact.from_path(coverage_source))
        supplier, temporary_edges = _annual_supplier_raw_sparse(
            incidence, upstream_weights, coverage
        )
        supplier = supplier.with_columns(pl.lit(taxonomy, dtype=pl.String).alias("taxonomy_version")).select(
            "taxonomy_version", "economy_id", "year", "gud_raw", "gud_raw_reason", "grd_raw", "grd_raw_reason"
        )
        missing_export_rows = coverage.filter(~pl.col("export_coverage_normal")).height
        supplier_missing_gud = supplier.filter(
            pl.col("gud_raw_reason") == "no_export_coverage"
        ).height
        supplier_missing_grd = supplier.filter(
            pl.col("grd_raw_reason") == "no_export_coverage"
        ).height
        if (
            supplier.height != coverage.height
            or supplier_missing_gud != missing_export_rows
            or supplier_missing_grd != missing_export_rows
        ):
            raise RuntimeError("supplier output does not preserve BACI coverage skeleton")
        frames.append(supplier)
        annual_audits.append(
            {
                "year": year,
                "skeleton_rows": coverage.height,
                "output_rows": supplier.height,
                "normal_export_coverage_rows": coverage.filter(pl.col("export_coverage_normal")).height,
                "no_export_coverage_rows": missing_export_rows,
                "rca_incidence_rows": incidence.filter(pl.col("rca_present")).height,
                "temporary_proximity_edges": temporary_edges,
                "gud_nonmissing": supplier.filter(pl.col("gud_raw").is_not_null()).height,
                "grd_nonmissing": supplier.filter(pl.col("grd_raw").is_not_null()).height,
                "gud_no_export_coverage_reason_rows": supplier_missing_gud,
                "grd_no_export_coverage_reason_rows": supplier_missing_grd,
            }
        )
    output = pl.concat(frames).with_columns(
        pl.col("year").cast(pl.Int16),
        pl.col("gud_raw").cast(pl.Float64),
        pl.col("grd_raw").cast(pl.Float64),
    ).sort(["year", "economy_id"])
    nonfinite = sum(
        int(output.select((pl.col(column).is_not_null() & ~pl.col(column).is_finite()).sum()).item())
        for column in _SUPPLIER_NUMERIC_COLUMNS
    )
    if nonfinite:
        raise ValueError("nonfinite supplier values before authoritative write")
    skeleton_rows = sum(int(record["skeleton_rows"]) for record in annual_audits)
    if output.height != skeleton_rows:
        raise RuntimeError("supplier output row count differs from BACI coverage skeleton")
    destination = paths.measures / "trade/supplier_raw.parquet"
    write_authoritative_table(
        output,
        _load_contract("supplier_raw.json"),
        destination,
        tuple(inputs),
        build,
    )
    audit_path = paths.measures / "trade/supplier_raw_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                "taxonomy_version": taxonomy,
                "years": annual_audits,
                "rows": output.height,
                "skeleton_rows": skeleton_rows,
                "duplicate_keys": int(output.group_by(["taxonomy_version", "economy_id", "year"]).len().filter(pl.col("len") > 1).height),
                "nonfinite_values": nonfinite,
                "1996_absorption_input_coverage": {
                    "gud": output.filter((pl.col("year") == 1996) & pl.col("gud_raw").is_not_null()).height,
                    "grd": output.filter((pl.col("year") == 1996) & pl.col("grd_raw").is_not_null()).height,
                },
                "missing_reasons": {
                    "gud": output.filter(pl.col("gud_raw_reason").is_not_null()).group_by("gud_raw_reason").len().sort("gud_raw_reason").to_dicts(),
                    "grd": output.filter(pl.col("grd_raw_reason").is_not_null()).group_by("grd_raw_reason").len().sort("grd_raw_reason").to_dicts(),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return SupplierBuildReport(
        rows=output.height,
        years=_YEARS,
        gud_nonmissing_rows=output.filter(pl.col("gud_raw").is_not_null()).height,
        grd_nonmissing_rows=output.filter(pl.col("grd_raw").is_not_null()).height,
        nonfinite_values=nonfinite,
        output_path=str(destination),
        audit_path=str(audit_path),
    )


def audit_complexity(*, data_root: Path, code_root: Path) -> ComplexityAuditReport:
    """Audit spectral orientation, lag discipline, and finite production values."""

    paths = resolve_project_paths(code_root, data_root)
    gpci_path = paths.measures / "trade/gpci_product_year.parquet"
    components_path = paths.measures / "trade/trade_components_raw.parquet"
    network_audit_path = paths.measures / "trade/complexity_network_audit.json"
    verify_manifest(gpci_path.with_name(f"{gpci_path.name}.manifest.json"))
    verify_manifest(components_path.with_name(f"{components_path.name}.manifest.json"))
    gpci = pl.read_parquet(gpci_path)
    components = pl.read_parquet(components_path)
    network_audit = json.loads(network_audit_path.read_text(encoding="utf-8"))
    network_years = network_audit.get("years")
    if not isinstance(network_years, list):
        raise RuntimeError("complexity network audit has no annual records")
    network_record_years = tuple(
        sorted(int(record["year"]) for record in network_years)
    )
    if network_record_years != _YEARS:
        raise RuntimeError(f"complexity network-audit period differs: {network_record_years}")
    for record in network_years:
        zero_diversity_codes = record.get("zero_diversity_economy_codes")
        zero_ubiquity_codes = record.get("zero_ubiquity_product_codes")
        if (
            not isinstance(zero_diversity_codes, list)
            or not isinstance(zero_ubiquity_codes, list)
            or len(zero_diversity_codes) != int(record["zero_diversity_economies"])
            or len(zero_ubiquity_codes) != int(record["zero_ubiquity_products"])
        ):
            raise RuntimeError("complexity network-audit exclusion codes are incomplete")
    numeric_gpci = ("gpci", "orientation_score")
    numeric_components = _TRADE_COMPONENT_NUMERIC_COLUMNS
    nonfinite = sum(
        int(
            frame.select(
                sum(
                    (pl.col(column).is_not_null() & ~pl.col(column).is_finite()).sum()
                    for column in columns
                )
            ).item()
        )
        for frame, columns in ((gpci, numeric_gpci), (components, numeric_components))
    )
    orientation_violations = gpci.filter(pl.col("orientation_score") < 0.0).height
    leakage = components.filter(
        pl.col("green_import_complexity_raw").is_not_null()
        & (
            pl.col("gpci_source_year").is_null()
            | (pl.col("gpci_source_year") != pl.col("year") - 1)
        )
    ).height
    gpci_years = tuple(gpci.get_column("year").unique().sort().to_list())
    component_years = tuple(components.get_column("year").unique().sort().to_list())
    if gpci_years != _YEARS:
        raise RuntimeError(f"GPCI period differs: {gpci_years}")
    if component_years != _YEARS:
        raise RuntimeError(f"trade-component period differs: {component_years}")
    if orientation_violations or leakage or nonfinite:
        raise RuntimeError(
            "complexity audit failed: "
            f"orientation={orientation_violations}, leakage={leakage}, nonfinite={nonfinite}"
        )
    return ComplexityAuditReport(
        gpci_years=gpci_years,
        trade_component_years=component_years,
        orientation_violations=orientation_violations,
        current_year_gpci_leakage_count=leakage,
        nonfinite_values=nonfinite,
        status="valid",
    )
