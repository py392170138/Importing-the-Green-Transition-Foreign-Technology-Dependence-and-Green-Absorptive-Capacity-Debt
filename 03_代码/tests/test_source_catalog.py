from pathlib import Path
import csv

from green_debt.sources import load_source_catalog
from green_debt.storage import GIB


ROOT = Path(__file__).resolve().parents[2]


def test_catalog_covers_every_registered_public_source() -> None:
    catalog = load_source_catalog(ROOT / "config" / "sources.yaml", ROOT)
    registry_path = ROOT / "07_文献与日志" / "公共数据源登记表_v0.1.csv"
    with registry_path.open(encoding="utf-8", newline="") as handle:
        registered = {row["source_id"] for row in csv.DictReader(handle)}

    assert set(catalog.sources) == registered
    assert catalog.sources["openalex_snapshot_reference"].enabled is False
    assert (
        catalog.sources["openalex_snapshot_reference"].reason
        == "exceeds_150GB_absolute_limit"
    )


def test_night_core_batch_freezes_two_official_baci_202601_files() -> None:
    catalog = load_source_catalog(ROOT / "config" / "sources.yaml", ROOT)

    specs = catalog.batch_specs("night_core")

    assert [spec.source_id for spec in specs] == ["baci_hs96", "baci_hs07"]
    assert [spec.source_version for spec in specs] == ["202601", "202601"]
    assert [spec.url for spec in specs] == [
        "https://www.cepii.fr/DATA_DOWNLOAD/baci/data/BACI_HS96_V202601.zip",
        "https://www.cepii.fr/DATA_DOWNLOAD/baci/data/BACI_HS07_V202601.zip",
    ]
    assert all(spec.allowed_hosts == ("www.cepii.fr",) for spec in specs)
    assert sum(spec.projected_working_bytes for spec in specs) == 7 * GIB
    assert all(
        spec.destination.is_relative_to(ROOT / "04_原始数据")
        for spec in specs
    )


def test_unfrozen_sources_and_openalex_api_cannot_enter_baci_batch() -> None:
    catalog = load_source_catalog(ROOT / "config" / "sources.yaml", ROOT)

    assert catalog.sources["oecd_icio_2025"].enabled is False
    assert catalog.sources["irena_downloads"].enabled is False
    assert "openalex_works_api" not in catalog.batches["night_core"]


def test_frozen_openalex_api_is_enabled_with_optional_free_key() -> None:
    catalog = load_source_catalog(ROOT / "config" / "sources.yaml", ROOT)

    source = catalog.sources["openalex_works_api"]
    assert source.enabled is True
    assert source.reason is None
    assert source.authentication == "free_api_key_optional"
    assert source.api_base_url == "https://api.openalex.org/works"
