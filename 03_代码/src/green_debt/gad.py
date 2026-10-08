"""Frozen green absorptive-debt (GAD) component and recursion construction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import polars as pl


FROZEN_SCALER_HASH = "d44dfe2bd8023e6b9c695b91dc73026a061e5f9b64198b41cf642979de8d612e"
_TAXONOMY_VERSION = "main_hs96"
_SAMPLE_VERSION = "confirmatory"

GAD_OUTPUT_COLUMNS = (
    "economy_id", "year", "specification_id", "formula_id", "family", "half_life",
    "taxonomy_version", "sample_version", "scaler_hash", "gimc", "supp",
    "external_exposure", "absorption", "lagged_absorption", "gap", "gad", "rho", "uses_lagged_debt",
    "run_id", "run_position", "gap_valid", "warmup_ineligible", "initialization_run",
    "initialization_complete", "core_economy_eligible", "core_eligibility_reason",
    "eligible", "confirmatory_eligible", "descriptive_only", "gimc_present", "supp_present",
    "core_exposure_present", "core_absorption_present", "provisional_core",
    "positive_green_import_baseline_eligible", "external_exposure_present",
    "absorption_present", "lagged_absorption_present", "gap_complete",
)


@dataclass(frozen=True)
class GADSpecification:
    """One immutable formula/decay/taxonomy/sample GAD definition."""

    specification_id: str
    formula_id: str
    family: str
    half_life: int | None
    rho: float
    uses_lagged_debt: bool
    taxonomy_version: str = _TAXONOMY_VERSION
    sample_version: str = _SAMPLE_VERSION

    @classmethod
    def core(cls, *, half_life: int = 5) -> "GADSpecification":
        if half_life <= 0:
            raise ValueError("GAD half-life must be positive")
        identifier = "gad_core" if half_life == 5 else f"gad_core_hl{half_life}"
        return cls(
            specification_id=identifier,
            formula_id="core",
            family="half_life",
            half_life=half_life,
            rho=2 ** (-1 / half_life),
            uses_lagged_debt=True,
        )

    @classmethod
    def static(cls) -> "GADSpecification":
        return cls(
            specification_id="gad_static",
            formula_id="core_static",
            family="static",
            half_life=None,
            rho=0.0,
            uses_lagged_debt=False,
        )


def registered_gad_specifications() -> tuple[GADSpecification, ...]:
    """Return the complete, non-duplicated immutable GAD registry."""

    core = GADSpecification.core(half_life=5)
    registry = (
        core,
        GADSpecification("gad_lite", "lite", "formula", 5, core.rho, True),
        GADSpecification("gad_no_gfvad", "no_gfvad", "formula", 5, core.rho, True),
        GADSpecification("gad_no_supp", "no_supp", "formula", 5, core.rho, True),
        GADSpecification("gad_no_gsci", "no_gsci", "formula", 5, core.rho, True),
        GADSpecification.core(half_life=3),
        GADSpecification.core(half_life=8),
        GADSpecification.core(half_life=10),
        GADSpecification("gad_no_decay", "core_no_decay", "no_decay", None, 1.0, True),
        GADSpecification.static(),
    )
    ids = [specification.specification_id for specification in registry]
    keys = [
        (
            specification.formula_id,
            specification.half_life,
            specification.taxonomy_version,
            specification.sample_version,
        )
        for specification in registry
    ]
    if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
        raise ValueError("GAD registry contains duplicate specification combinations")
    return registry


def _require_columns(frame: pl.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"GAD input lacks columns: {missing}")


def build_component_indices(frame: pl.DataFrame) -> pl.DataFrame:
    """Append exact equal-weighted component indices without changing raw/z0 data."""

    output = frame
    if "gimc" not in output.columns:
        _require_columns(
            output,
            ("z0_green_import_intensity_raw", "z0_green_import_complexity_raw"),
        )
        output = output.with_columns(
            (
                0.5 * pl.col("z0_green_import_intensity_raw")
                + 0.5 * pl.col("z0_green_import_complexity_raw")
            ).alias("gimc")
        )
    if "supp" not in output.columns:
        _require_columns(output, ("z0_gud_raw", "z0_grd_raw"))
        output = output.with_columns(
            (0.5 * pl.col("z0_gud_raw") + 0.5 * pl.col("z0_grd_raw")).alias("supp")
        )
    _require_columns(
        output,
        ("gimc", "supp", "z0_gfvad_raw", "z0_gnir_raw", "z0_gsci_raw"),
    )
    return output.with_columns(
        (
            0.5 * pl.col("gimc") + 0.5 * pl.col("z0_gfvad_raw")
        ).alias("external_exposure_core"),
        (0.5 * pl.col("gimc") + 0.5 * pl.col("z0_gnir_raw")).alias(
            "external_exposure_lite"
        ),
        pl.col("gimc").alias("external_exposure_no_gfvad"),
        (0.5 * pl.col("z0_gsci_raw") + 0.5 * pl.col("supp")).alias(
            "absorption_core"
        ),
        pl.col("z0_gsci_raw").alias("absorption_no_supp"),
        pl.col("supp").alias("absorption_no_gsci"),
    )


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("GAD component values must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("GAD component values must be finite")
    return numeric


def compute_gad(frame: pl.DataFrame, specification: GADSpecification) -> pl.DataFrame:
    """Compute strictly lagged, run-reset GAD for one economy-year panel."""

    _require_columns(frame, ("economy_id", "year", "external_exposure", "absorption"))
    duplicate_keys = frame.group_by("economy_id", "year").len().filter(pl.col("len") > 1).height
    if duplicate_keys:
        raise ValueError(f"GAD input has duplicate economy-year keys: {duplicate_keys}")
    ordered = frame.sort("economy_id", "year")
    rows: list[dict[str, Any]] = []
    current_economy: str | None = None
    prior_year: int | None = None
    prior_absorption: float | None = None
    prior_gap_year: int | None = None
    prior_gad: float | None = None
    current_run_id = 0
    current_run_position = 0
    first_run_start_year: int | None = None

    for source in ordered.iter_rows(named=True):
        economy = str(source["economy_id"])
        year = int(source["year"])
        if economy != current_economy:
            current_economy = economy
            prior_year = None
            prior_absorption = None
            prior_gap_year = None
            prior_gad = None
            current_run_id = 0
            current_run_position = 0
            first_run_start_year = None
        exposure = _finite_or_none(source["external_exposure"])
        absorption = _finite_or_none(source["absorption"])
        lagged_absorption = prior_absorption if prior_year == year - 1 else None
        gap = (
            exposure - lagged_absorption
            if exposure is not None and lagged_absorption is not None
            else None
        )
        consecutive = gap is not None and prior_gap_year == year - 1 and prior_gad is not None
        if gap is None:
            gad = None
            run_id = None
            run_position = None
            warmup = False
            eligible = False
            prior_gap_year = None
            prior_gad = None
            current_run_position = 0
        else:
            if not consecutive:
                current_run_id += 1
                current_run_position = 1
                if current_run_id == 1:
                    first_run_start_year = year
                prior_for_recursion = 0.0
            else:
                current_run_position += 1
                prior_for_recursion = float(prior_gad)
            if specification.uses_lagged_debt:
                gad = max(0.0, specification.rho * prior_for_recursion + gap)
            else:
                gad = max(0.0, gap)
            run_id = current_run_id
            run_position = current_run_position
            warmup = run_position <= 3
            eligible = not warmup
            prior_gap_year = year
            prior_gad = gad
        rows.append(
            {
                **source,
                "lagged_absorption": lagged_absorption,
                "gap": gap,
                "gad": gad,
                "rho": specification.rho,
                "run_id": run_id,
                "run_position": run_position,
                "gap_valid": gap is not None,
                "warmup_ineligible": warmup,
                "initialization_run": bool(
                    current_run_id == 1 and first_run_start_year == 1997 and 1997 <= year <= 1999 and gap is not None
                ),
                "eligible": eligible,
            }
        )
        prior_year = year
        prior_absorption = absorption
    return pl.DataFrame(rows).with_columns(
        pl.col("year").cast(pl.Int16),
        pl.col("external_exposure").cast(pl.Float64),
        pl.col("absorption").cast(pl.Float64),
        pl.col("lagged_absorption").cast(pl.Float64),
        pl.col("gap").cast(pl.Float64),
        pl.col("gad").cast(pl.Float64),
        pl.col("rho").cast(pl.Float64),
        pl.col("run_id").cast(pl.Int16),
        pl.col("run_position").cast(pl.Int16),
    )


def _formula_columns(specification: GADSpecification) -> tuple[str, str]:
    formula = specification.formula_id
    if formula in {"core", "core_no_decay", "core_static"}:
        return "external_exposure_core", "absorption_core"
    if formula == "lite":
        return "external_exposure_lite", "absorption_core"
    if formula == "no_gfvad":
        return "external_exposure_no_gfvad", "absorption_core"
    if formula == "no_supp":
        return "external_exposure_core", "absorption_no_supp"
    if formula == "no_gsci":
        return "external_exposure_core", "absorption_no_gsci"
    raise ValueError(f"unknown GAD formula: {formula}")


def derive_core_eligibility(
    components: pl.DataFrame,
    provisional_sample: pl.DataFrame,
    *,
    implementation_commit: str | None = None,
) -> pl.DataFrame:
    """Recompute core initialization and eligibility from source components only."""

    _require_columns(
        components,
        ("economy_id", "year", "external_exposure_core", "absorption_core"),
    )
    _require_columns(
        provisional_sample,
        (
            "economy_id", "sample_version", "provisional_core",
            "positive_green_import_baseline_eligible",
        ),
    )
    samples = provisional_sample.filter(pl.col("provisional_core")).select(
        "economy_id", "sample_version", "provisional_core",
        "positive_green_import_baseline_eligible",
    )
    if samples.group_by("economy_id").len().filter(pl.col("len") > 1).height:
        raise ValueError("provisional core sample has duplicate economy ids")
    source = components.select(
        "economy_id", "year", "external_exposure_core", "absorption_core"
    )
    if source.group_by("economy_id", "year").len().filter(pl.col("len") > 1).height:
        raise ValueError("scaled components have duplicate economy-year keys")
    by_key = {
        (str(row["economy_id"]), int(row["year"])): row
        for row in source.iter_rows(named=True)
    }
    rows: list[dict[str, Any]] = []
    for sample in samples.sort("economy_id").iter_rows(named=True):
        economy = str(sample["economy_id"])
        a1996 = _finite_or_none(by_key.get((economy, 1996), {}).get("absorption_core"))
        gap_valid: dict[int, bool] = {}
        for year in range(1997, 2023):
            current = by_key.get((economy, year))
            previous = by_key.get((economy, year - 1))
            exposure = _finite_or_none(current.get("external_exposure_core")) if current else None
            lag = _finite_or_none(previous.get("absorption_core")) if previous else None
            gap_valid[year] = exposure is not None and lag is not None
        initialization_complete = all(gap_valid[year] for year in (1997, 1998, 1999))
        valid_main_years = sum(gap_valid[year] for year in range(2000, 2023))
        baseline = bool(sample["positive_green_import_baseline_eligible"])
        if not baseline:
            reason = "baseline_sample_ineligible"
        elif a1996 is None:
            reason = "missing_absorption_1996"
        elif not initialization_complete:
            reason = "initialization_gap_1997_1999"
        elif valid_main_years < 12:
            reason = "fewer_than_12_valid_main_years"
        else:
            reason = "eligible"
        rows.append(
            {
                "economy_id": economy,
                "sample_version": str(sample["sample_version"]),
                "provisional_core": True,
                "positive_green_import_baseline_eligible": baseline,
                "absorption_1996": a1996,
                "absorption_1996_present": a1996 is not None,
                "gap_1997_valid": gap_valid[1997],
                "gap_1998_valid": gap_valid[1998],
                "gap_1999_valid": gap_valid[1999],
                "initialization_complete": initialization_complete,
                "valid_main_years": valid_main_years,
                "core_economy_eligible": reason == "eligible",
                "core_eligibility_reason": reason,
            }
        )
    output = pl.DataFrame(rows).with_columns(pl.col("valid_main_years").cast(pl.Int16))
    if implementation_commit is None:
        return output
    if len(implementation_commit) != 40 or any(character not in "0123456789abcdef" for character in implementation_commit):
        raise ValueError("initialization audit implementation commit must be a lowercase git SHA")
    return output.with_columns(pl.lit(implementation_commit).alias("implementation_commit"))


def build_gad_variants(
    scaled_components: pl.DataFrame,
    provisional_sample: pl.DataFrame,
    *,
    scaler_hash: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build every registry variant plus one row per economy initialization audit."""

    if scaler_hash != FROZEN_SCALER_HASH:
        raise ValueError("frozen GAD scaler hash differs from the approved anchor")
    _require_columns(scaled_components, ("economy_id", "year"))
    _require_columns(
        provisional_sample,
        (
            "economy_id",
            "sample_version",
            "provisional_core",
            "positive_green_import_baseline_eligible",
        ),
    )
    components = build_component_indices(scaled_components)
    component_presence = components.with_columns(
        pl.col("gimc").is_not_null().alias("gimc_present"),
        pl.col("supp").is_not_null().alias("supp_present"),
        pl.col("external_exposure_core").is_not_null().alias("core_exposure_present"),
        pl.col("absorption_core").is_not_null().alias("core_absorption_present"),
    )
    sample = provisional_sample.select(
        "economy_id",
        "sample_version",
        "provisional_core",
        "positive_green_import_baseline_eligible",
    )
    rows: list[pl.DataFrame] = []
    for specification in registered_gad_specifications():
        exposure, absorption = _formula_columns(specification)
        cutoff = 2024 if specification.specification_id == "gad_lite" else 2022
        panel = component_presence.filter(pl.col("year") <= cutoff).with_columns(
            pl.col(exposure).alias("external_exposure"),
            pl.col(absorption).alias("absorption"),
        )
        computed = compute_gad(panel, specification).join(sample, on="economy_id", how="left")
        rows.append(
            computed.with_columns(
                pl.lit(specification.specification_id).alias("specification_id"),
                pl.lit(specification.formula_id).alias("formula_id"),
                pl.lit(specification.family).alias("family"),
                pl.lit(specification.half_life).cast(pl.Int16).alias("half_life"),
                pl.lit(specification.taxonomy_version).alias("taxonomy_version"),
                pl.lit(scaler_hash).alias("scaler_hash"),
                pl.lit(specification.uses_lagged_debt).alias("uses_lagged_debt"),
                ((pl.col("year") >= 2023) & (specification.specification_id == "gad_lite")).alias("descriptive_only"),
            )
        )
    output = pl.concat(rows, how="vertical_relaxed").sort("economy_id", "year", "specification_id")

    initialization_audit = derive_core_eligibility(components, provisional_sample)
    output = output.join(
        initialization_audit.select("economy_id", "initialization_complete", "core_economy_eligible", "core_eligibility_reason"),
        on="economy_id", how="left",
    ).with_columns(
        (pl.col("eligible") & pl.col("core_economy_eligible") & ~pl.col("descriptive_only")).alias("confirmatory_eligible"),
        pl.col("external_exposure").is_not_null().alias("external_exposure_present"),
        pl.col("absorption").is_not_null().alias("absorption_present"),
        pl.col("lagged_absorption").is_not_null().alias("lagged_absorption_present"),
        pl.col("gap").is_not_null().alias("gap_complete"),
    )
    return output, initialization_audit


def gad_output_table(frame: pl.DataFrame) -> pl.DataFrame:
    """Select the fixed long-form authority columns in contract order."""

    _require_columns(frame, GAD_OUTPUT_COLUMNS)
    return frame.select(*GAD_OUTPUT_COLUMNS).sort("economy_id", "year", "specification_id")


def build_construction_audit(
    frame: pl.DataFrame,
    *,
    implementation_commit: str,
) -> pl.DataFrame:
    """Summarize constructed authority coverage after source-level recomputation."""

    _require_columns(
        frame,
        (
            "specification_id", "formula_id", "taxonomy_version", "sample_version", "year",
            "economy_id", "gap_valid", "gad", "run_position", "warmup_ineligible",
            "confirmatory_eligible", "descriptive_only",
        ),
    )
    if len(implementation_commit) != 40 or any(character not in "0123456789abcdef" for character in implementation_commit):
        raise ValueError("construction audit implementation commit must be a lowercase git SHA")
    return (
        frame.group_by("specification_id", "formula_id", "taxonomy_version", "sample_version", "year")
        .agg(
            pl.len().alias("rows"),
            pl.col("economy_id").n_unique().alias("economies"),
            pl.col("gap_valid").sum().alias("valid_gap_rows"),
            pl.col("gad").is_not_null().sum().alias("gad_rows"),
            (pl.col("run_position") == 1).sum().alias("restart_rows"),
            pl.col("warmup_ineligible").sum().alias("warmup_rows"),
            pl.col("confirmatory_eligible").sum().alias("confirmatory_eligible_rows"),
            pl.col("descriptive_only").sum().alias("descriptive_rows"),
        )
        .with_columns(
            pl.lit(True).alias("source_recomputed_from_scaled"),
            pl.lit(implementation_commit).alias("implementation_commit"),
        )
        .sort("specification_id", "year")
    )


def _same_float(actual: object, expected: float | None) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)


def _exact_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"GAD never-null Boolean field is null or invalid: {name}")
    return value


def audit_gad_frame(
    frame: pl.DataFrame,
    *,
    scaled_components: pl.DataFrame | None = None,
    provisional_sample: pl.DataFrame | None = None,
) -> dict[str, int]:
    """Fail closed by independently rebuilding formulas, runs, and sample gates."""

    required = (
        "economy_id", "year", "specification_id", "formula_id", "family", "half_life",
        "taxonomy_version", "sample_version", "scaler_hash", "external_exposure", "absorption",
        "lagged_absorption", "gap", "gad", "rho", "uses_lagged_debt", "run_id", "run_position", "gap_valid",
        "warmup_ineligible", "initialization_run", "eligible", "descriptive_only",
    )
    _require_columns(frame, required)
    if provisional_sample is not None:
        _require_columns(frame, GAD_OUTPUT_COLUMNS)
        never_null = (
            "formula_id", "family", "taxonomy_version", "sample_version", "scaler_hash",
            "rho", "uses_lagged_debt", "gap_valid", "warmup_ineligible", "initialization_run",
            "initialization_complete", "core_economy_eligible", "core_eligibility_reason", "eligible",
            "confirmatory_eligible", "descriptive_only", "gimc_present", "supp_present",
            "core_exposure_present", "core_absorption_present", "provisional_core",
            "positive_green_import_baseline_eligible", "external_exposure_present", "absorption_present",
            "lagged_absorption_present", "gap_complete",
        )
        null_counts = frame.select([pl.col(name).is_null().sum().alias(name) for name in never_null]).row(0, named=True)
        nulls = [name for name in never_null if int(null_counts[name])]
        if nulls:
            raise ValueError(f"GAD never-null fields contain nulls: {nulls}")
    duplicates = frame.group_by("economy_id", "year", "specification_id").len().filter(pl.col("len") > 1).height
    if duplicates:
        raise ValueError("GAD authority has duplicate primary keys")
    for name, dtype in frame.schema.items():
        if str(dtype) in {"Float32", "Float64"} and frame.select((pl.col(name).is_not_null() & ~pl.col(name).is_finite()).any()).item():
            raise ValueError(f"GAD authority contains nonfinite values: {name}")
    if frame.get_column("scaler_hash").unique().to_list() != [FROZEN_SCALER_HASH]:
        raise ValueError("GAD variants do not share the approved frozen scaler hash")
    registered = {specification.specification_id: specification for specification in registered_gad_specifications()}
    actual_ids = set(frame.get_column("specification_id").unique().to_list())
    if not actual_ids <= set(registered):
        raise ValueError("GAD authority has an unregistered specification")

    detailed: dict[tuple[str, int], dict[str, Any]] = {}
    eligibility: dict[str, dict[str, Any]] = {}
    if scaled_components is not None:
        components = build_component_indices(scaled_components)
        detailed = {(str(row["economy_id"]), int(row["year"])): row for row in components.iter_rows(named=True)}
        if provisional_sample is not None:
            initialization = derive_core_eligibility(components, provisional_sample)
            eligibility = {str(row["economy_id"]): row for row in initialization.iter_rows(named=True)}
            expected_economies = set(eligibility)
            scaled_economies = set(scaled_components.get_column("economy_id").unique().to_list())
            gad_economies = set(frame.get_column("economy_id").unique().to_list())
            if expected_economies != scaled_economies or expected_economies != gad_economies:
                raise ValueError("provisional, scaled, and GAD economy sets differ")
            expected_scaled_keys = {(economy, year) for economy in expected_economies for year in range(1996, 2025)}
            if set(detailed) != expected_scaled_keys:
                raise ValueError("scaled authority primary-key matrix differs from the frozen sample calendar")
            expected_keys = {
                (economy, year, specification.specification_id)
                for economy in expected_economies
                for specification in registered.values()
                for year in range(1996, 2025 if specification.specification_id == "gad_lite" else 2023)
            }
            actual_keys = {(str(row["economy_id"]), int(row["year"]), str(row["specification_id"])) for row in frame.select("economy_id", "year", "specification_id").iter_rows(named=True)}
            if actual_keys != expected_keys:
                raise ValueError("GAD primary-key matrix differs from the registered sample calendar")
            if actual_ids != set(registered):
                raise ValueError("GAD authority does not contain every registered variant")

    row_count = 0
    restart_count = 0
    warmup_count = 0
    for (economy, specification_id), group in frame.sort("economy_id", "specification_id", "year").group_by("economy_id", "specification_id", maintain_order=True):
        specification = registered[str(specification_id)]
        prior_year: int | None = None
        prior_absorption: float | None = None
        prior_gad: float | None = None
        prior_valid = False
        expected_run_id = 0
        expected_position = 0
        first_run_start: int | None = None
        for row in group.iter_rows(named=True):
            row_count += 1
            year = int(row["year"])
            expected_half_life = specification.half_life
            if (
                str(row["formula_id"]) != specification.formula_id
                or str(row["family"]) != specification.family
                or row["half_life"] != expected_half_life
                or str(row["taxonomy_version"]) != specification.taxonomy_version
                or str(row["sample_version"]) != specification.sample_version
                or _exact_bool(row["uses_lagged_debt"], "uses_lagged_debt")
                != specification.uses_lagged_debt
                or not _same_float(row["rho"], specification.rho)
            ):
                raise ValueError("GAD registry metadata differs from its immutable specification")
            descriptive = _exact_bool(row["descriptive_only"], "descriptive_only") if provisional_sample is not None else bool(row["descriptive_only"])
            expected_descriptive = specification_id == "gad_lite" and year in {2023, 2024}
            if descriptive != expected_descriptive:
                raise ValueError("GAD descriptive_only flag differs from the exact declared boundary")
            if specification_id != "gad_lite" and year > 2022:
                raise ValueError("only gad_lite may contain 2023-2024 rows")
            if year >= 2023 and not expected_descriptive:
                raise ValueError("2023-2024 GAD rows must be descriptive Lite only")
            if detailed:
                source = detailed.get((str(economy), year))
                if source is None:
                    raise ValueError("GAD output has no scaled component source row")
                exposure_column, absorption_column = _formula_columns(specification)
                expected_e = _finite_or_none(source[exposure_column])
                expected_a = _finite_or_none(source[absorption_column])
                if not _same_float(row["external_exposure"], expected_e) or not _same_float(row["absorption"], expected_a):
                    raise ValueError("GAD component formula is not reproducible from scaled inputs")
                if "gimc" in frame.columns and not _same_float(row["gimc"], _finite_or_none(source["gimc"])):
                    raise ValueError("GIMC formula is not reproducible from scaled inputs")
                if "supp" in frame.columns and not _same_float(row["supp"], _finite_or_none(source["supp"])):
                    raise ValueError("SUPP formula is not reproducible from scaled inputs")
            exposure = _finite_or_none(row["external_exposure"])
            absorption = _finite_or_none(row["absorption"])
            expected_lag = prior_absorption if prior_year == year - 1 else None
            expected_gap = exposure - expected_lag if exposure is not None and expected_lag is not None else None
            if not _same_float(row["lagged_absorption"], expected_lag):
                raise ValueError("GAD strict lagged absorption is not the prior calendar-year A")
            gap_valid = _exact_bool(row["gap_valid"], "gap_valid") if provisional_sample is not None else bool(row["gap_valid"])
            if not _same_float(row["gap"], expected_gap) or gap_valid != (expected_gap is not None):
                raise ValueError("GAD gap does not equal E_t minus A_t_minus_1")
            if provisional_sample is not None:
                presence = {
                    "external_exposure_present": exposure is not None,
                    "absorption_present": absorption is not None,
                    "lagged_absorption_present": expected_lag is not None,
                    "gap_complete": expected_gap is not None,
                }
                for name, expected_presence in presence.items():
                    if name not in row or _exact_bool(row[name], name) != expected_presence:
                        raise ValueError("GAD formula-specific presence flags differ from source recomputation")
            if expected_gap is None:
                if any(row[name] is not None for name in ("gad", "run_id", "run_position")):
                    raise ValueError("invalid GAD gap must have null debt and run fields")
                warmup = _exact_bool(row["warmup_ineligible"], "warmup_ineligible") if provisional_sample is not None else bool(row["warmup_ineligible"])
                eligible_value = _exact_bool(row["eligible"], "eligible") if provisional_sample is not None else bool(row["eligible"])
                init_run = _exact_bool(row["initialization_run"], "initialization_run") if provisional_sample is not None else bool(row["initialization_run"])
                if warmup or eligible_value or init_run:
                    raise ValueError("invalid GAD gap has non-null run flags")
                prior_valid = False
                prior_gad = None
                expected_position = 0
            else:
                if not (prior_valid and prior_year == year - 1 and prior_gad is not None):
                    expected_run_id += 1
                    expected_position = 1
                    prior_debt = 0.0
                    if expected_run_id == 1:
                        first_run_start = year
                    restart_count += 1
                else:
                    expected_position += 1
                    prior_debt = prior_gad
                expected_gad = max(0.0, specification.rho * prior_debt + expected_gap) if specification.uses_lagged_debt else max(0.0, expected_gap)
                expected_warmup = expected_position <= 3
                expected_initialization_run = bool(
                    expected_run_id == 1 and first_run_start == 1997 and 1997 <= year <= 1999
                )
                if row["run_id"] != expected_run_id or row["run_position"] != expected_position:
                    raise ValueError("GAD run identifiers or positions differ from independent recursion")
                if not _same_float(row["gad"], expected_gad):
                    raise ValueError("GAD recursion crosses a gap or uses the wrong debt rule")
                warmup = _exact_bool(row["warmup_ineligible"], "warmup_ineligible") if provisional_sample is not None else bool(row["warmup_ineligible"])
                eligible_value = _exact_bool(row["eligible"], "eligible") if provisional_sample is not None else bool(row["eligible"])
                init_run = _exact_bool(row["initialization_run"], "initialization_run") if provisional_sample is not None else bool(row["initialization_run"])
                if warmup != expected_warmup or eligible_value != (not expected_warmup):
                    raise ValueError("GAD restart warm-up flags are invalid")
                if init_run != expected_initialization_run:
                    raise ValueError("GAD initialization_run flag differs from independent recursion")
                prior_valid = True
                prior_gad = expected_gad
                warmup_count += int(expected_warmup)
            if eligibility:
                expected = eligibility[str(economy)]
                for name in ("provisional_core", "positive_green_import_baseline_eligible", "initialization_complete", "core_economy_eligible", "core_eligibility_reason"):
                    if row.get(name) != expected[name]:
                        raise ValueError("stored core eligibility fields differ from the source recomputation")
                expected_confirmatory = expected_gap is not None and expected_position > 3 and _exact_bool(expected["core_economy_eligible"], "source_core_economy_eligible") and not expected_descriptive
                if _exact_bool(row["confirmatory_eligible"], "confirmatory_eligible") != expected_confirmatory:
                    raise ValueError("stored confirmatory eligibility differs from the source recomputation")
            prior_year = year
            prior_absorption = absorption
    return {"rows": row_count, "restarts": restart_count, "warmup_rows": warmup_count, "specifications": len(registered)}
