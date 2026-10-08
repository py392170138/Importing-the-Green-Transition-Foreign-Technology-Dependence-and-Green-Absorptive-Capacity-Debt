import csv
from pathlib import Path

import pytest

from green_debt.config import (
    load_construction_config,
    load_outcome_gad_map,
    load_project_config,
)


ROOT = Path(__file__).resolve().parents[2]


def test_initialization_and_capacity_are_frozen() -> None:
    project = load_project_config(ROOT / "config" / "project.yaml")
    build = load_construction_config(ROOT / "config" / "construction.yaml")

    assert project.period.openalex_history == (1992, 1995)
    assert project.period.tiva_history == (1995, 1999)
    assert project.period.initialization == (1997, 1999)
    assert project.period.main == (2000, 2022)
    assert project.storage.intermediate_quota_gb == 25
    assert project.network.allow_logged_user_route_exception is True
    assert build.taxonomy.counts == {"main": 126, "broad": 248, "apec": 54}
    assert build.taxonomy.overlap_years == (2007, 2010)
    assert build.openalex.coverage_floor_ratio == pytest.approx(0.10)
    assert build.gad.warmup_observations == 3
    assert build.gad.scaler_years == (2000, 2004)
    assert build.iv.baseline_years == (1996, 1999)
    assert build.iv.minimum_retained_coverage == pytest.approx(0.95)


def test_every_construction_block_is_typed_and_exact() -> None:
    build = load_construction_config(ROOT / "config" / "construction.yaml")

    assert build.schema_version == 1
    assert build.taxonomy.source_hs == "HS07"
    assert build.taxonomy.main_hs == "HS96"
    assert build.taxonomy.upstream_bec_uses == ("intermediate", "capital")
    assert build.openalex.counting_method == "full_country_participation"
    assert build.openalex.include_xpac is False
    assert build.scaling.center == "median"
    assert build.scaling.scale == "mad_1_4826"
    assert build.scaling.fallback == "iqr_div_1_349"
    assert build.scaling.regression_percentiles == (1, 99)
    assert build.gad.start_year == 1997
    assert build.gad.implicit_prior_debt == pytest.approx(0.0)
    assert build.gad.half_lives == (3, 5, 8, 10)
    assert build.tiva.weight_years == (2000, 2004)
    assert build.sample.minimum_population == 1_000_000
    assert build.sample.minimum_valid_main_years == 12
    assert build.sample.minimum_positive_import_baseline_years == 2
    assert build.iv.minimum_cell_share == pytest.approx(0.0001)
    assert build.iv.robustness_cell_share == pytest.approx(0.0005)
    assert build.iv.cmz_lags == 5


def test_outcome_mapping_fails_closed() -> None:
    mapping = load_outcome_gad_map(ROOT / "config" / "outcome_gad_map.yaml")

    assert mapping.variant_for("domestic_value_added_share") == "gad_no_gfvad"
    assert mapping.variant_for("green_export_complexity") == "gad_no_supp"
    assert mapping.variant_for("green_science_output") == "gad_no_gsci"
    assert mapping.variant_for("co2_tonnes_per_million_current_usd") == "gad_core"
    with pytest.raises(KeyError, match="unregistered outcome"):
        mapping.variant_for("mystery")


def test_indicator_registry_freezes_downloaded_and_rejected_wdi_codes() -> None:
    path = ROOT / "02_数据字典" / "indicator_registry_v1.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert set(rows[0]) == {
        "source_id",
        "source_field",
        "project_field",
        "unit",
        "start_year",
        "end_year",
        "role",
        "zero_semantics",
        "missing_rule",
        "status",
    }
    wdi = {row["source_field"]: row for row in rows if row["source_id"] == "wdi"}
    assert {
        "EG.EGY.PRIM.PP.KD",
        "EG.FEC.RNEW.ZS",
        "EN.GHG.CO2.MT.CE.AR5",
        "NE.TRD.GNFS.ZS",
        "NV.IND.TOTL.ZS",
        "NY.GDP.MKTP.CD",
        "NY.GDP.PCAP.CD",
        "SP.POP.TOTL",
    } <= set(wdi)
    assert wdi["EN.GHG.CO2.MT.CE.AR5"]["unit"] == "Mt_CO2e_excluding_LULUCF_AR5"
    assert wdi["EN.ATM.CO2E.KT"]["status"] == "rejected_retired_code"


def test_economy_override_registry_has_review_columns() -> None:
    path = ROOT / "02_数据字典" / "economy_overrides_v1.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert set(rows[0]) == {
        "source_id",
        "source_code",
        "source_label",
        "economy_id",
        "confirmatory_eligible",
        "exclusion_reason",
        "reexport_hub",
        "review_status",
        "notes",
    }
    assert all(row["review_status"] == "reviewed" for row in rows)
