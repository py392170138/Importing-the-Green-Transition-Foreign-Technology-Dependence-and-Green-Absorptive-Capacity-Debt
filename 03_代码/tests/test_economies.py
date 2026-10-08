import json
import polars as pl
from zipfile import ZipFile

from green_debt import economies
from green_debt.economies import apply_sample_flags, build_economy_crosswalk


def test_aggregates_and_special_economies_are_explicit() -> None:
    source = pl.DataFrame(
        {
            "source_id": ["baci", "baci", "oecd", "wdi", "wdi"],
            "source_code": ["490", "999", "ROW", "TWN", "HKG"],
            "source_label": [
                "Asia, nes",
                "World",
                "Rest of world",
                "Taiwan",
                "Hong Kong SAR, China",
            ],
        }
    )

    out = build_economy_crosswalk(source, overrides=pl.DataFrame())

    assert (
        out.filter(pl.col("source_code") == "490")["exclusion_reason"].item()
        == "asia_nes"
    )
    assert (
        out.filter(pl.col("source_code") == "ROW")["exclusion_reason"].item()
        == "aggregate_or_row"
    )
    assert (
        out.filter(pl.col("source_code") == "TWN")["confirmatory_eligible"].item()
        is False
    )
    assert out.filter(pl.col("source_code") == "HKG")["reexport_hub"].item() is True


def test_micro_economy_flag_uses_population_floor() -> None:
    frame = pl.DataFrame(
        {"economy_id": ["AAA", "BBB"], "population": [999_999, 1_000_000]}
    )

    out = apply_sample_flags(frame, minimum_population=1_000_000)

    assert out["confirmatory_eligible"].to_list() == [False, True]
    assert out["sample_version"].to_list() == ["micro_robustness", "confirmatory"]


def test_frozen_override_is_exact_and_takes_precedence() -> None:
    source = pl.DataFrame(
        {
            "source_id": ["custom"],
            "source_code": ["ZZ1"],
            "source_label": ["Reviewed statistical economy"],
        }
    )
    overrides = pl.DataFrame(
        {
            "source_id": ["custom"],
            "source_code": ["ZZ1"],
            "source_label": ["Reviewed statistical economy"],
            "economy_id": ["ZZZ"],
            "confirmatory_eligible": [True],
            "exclusion_reason": [None],
            "reexport_hub": [False],
            "review_status": ["reviewed"],
            "notes": ["fixture"],
        }
    )

    out = build_economy_crosswalk(source, overrides=overrides)

    assert out["economy_id"].item() == "ZZZ"
    assert out["mapping_method"].item() == "frozen_override"


def test_near_match_is_not_fuzzy_mapped() -> None:
    source = pl.DataFrame(
        {
            "source_id": ["custom"],
            "source_code": ["??"],
            "source_label": ["Unted Stats"],
        }
    )

    out = build_economy_crosswalk(source, overrides=pl.DataFrame())

    assert out["economy_id"].item() is None
    assert out["exclusion_reason"].item() == "unresolved_source_code"


def test_baci_dictionary_hash_is_computed_once(tmp_path, monkeypatch) -> None:
    archive = tmp_path / "baci/202601/BACI_HS96_V202601.zip"
    archive.parent.mkdir(parents=True)
    body = (
        "country_code,country_name,country_iso2,country_iso3\n"
        "4,Afghanistan,AF,AFG\n"
        "8,Albania,AL,ALB\n"
    )
    with ZipFile(archive, "w") as handle:
        handle.writestr("country_codes_V202601.csv", body)
    calls = []

    def fake_hash(path):
        calls.append(path)
        return "a" * 64

    monkeypatch.setattr(economies, "sha256_file", fake_hash)

    rows = economies._baci_rows(tmp_path)

    assert len(rows) == 2
    assert calls == [archive]


def test_wdi_non_iso_code_is_aggregate_unless_overridden() -> None:
    source = pl.DataFrame(
        {
            "source_id": ["wdi"],
            "source_code": ["CEB"],
            "source_label": ["Central Europe and the Baltics"],
        }
    )

    out = build_economy_crosswalk(source, overrides=pl.DataFrame())

    assert out["economy_id"].item() is None
    assert out["exclusion_reason"].item() == "aggregate_or_row"


def test_aggregate_token_does_not_match_inside_trinidad() -> None:
    source = pl.DataFrame(
        {
            "source_id": ["wdi"],
            "source_code": ["TTO"],
            "source_label": ["Trinidad and Tobago"],
        }
    )

    out = build_economy_crosswalk(source, overrides=pl.DataFrame())

    assert out["economy_id"].item() == "TTO"
    assert out["exclusion_reason"].item() is None


def test_irena_dictionary_unions_capacity_generation_and_share_areas(
    tmp_path,
) -> None:
    data = tmp_path / "irena/20260821/data"
    data.mkdir(parents=True)

    def payload(dimension, codes):
        return {
            "dimension": {
                dimension: {
                    "category": {
                        "index": {code: index for index, code in enumerate(codes)},
                        "label": {code: label for code, label in codes.items()},
                    }
                }
            }
        }

    files = {
        "Country_ELECCAP_2026_H1.2000-2007.jsonstat2.json": payload(
            "Country/area", {"AFG": "Afghanistan", "REA": "Eurasia"}
        ),
        "Country_ELECGEN_2025_H2.2000-2006.jsonstat2.json": payload(
            "Country/area", {"ALB": "Albania"}
        ),
        "RE-SHARE_2026_H1.2000-2025.jsonstat2.json": payload(
            "Region/country/area", {"GLO": "World", "RAF": "Africa"}
        ),
    }
    for name, body in files.items():
        (data / name).write_text(json.dumps(body), encoding="utf-8")

    rows = economies._irena_rows(tmp_path)

    assert {row["source_code"] for row in rows} == {
        "AFG",
        "ALB",
        "GLO",
        "RAF",
        "REA",
    }


def test_ifcma_country_codes_enter_the_source_specific_crosswalk(tmp_path) -> None:
    path = (
        tmp_path
        / "oecd_ifcma/202604/IFCMA_ClimatePolicyDatabase_Data_April_2026.csv"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        "Country ISO,Country,Policy Instrument ID,Instrument / subscheme\n"
        "ARG,Argentina,ARG1,Instrument\n"
        "SGP,Singapore,SGP1,Subscheme\n"
        "Country ISO,Country,Policy Instrument ID,Instrument / subscheme\n",
        encoding="utf-8",
    )

    rows = economies._ifcma_rows(tmp_path)

    assert {(row["source_id"], row["source_code"]) for row in rows} == {
        ("oecd_ifcma", "ARG"),
        ("oecd_ifcma", "SGP"),
    }
