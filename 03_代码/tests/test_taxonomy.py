from pathlib import Path

import polars as pl
import pytest

from green_debt.taxonomy import (
    assign_upstream_weight,
    build_hs96_green_weights,
    parse_apec_html,
    parse_cleg_annex_text,
)


FIXTURES = Path(__file__).parent / "fixtures" / "taxonomy"


def test_hs_codes_keep_six_digits_and_category_membership() -> None:
    out = parse_cleg_annex_text("840290 REP boiler parts\n010101 HEM example")

    assert out.schema["hs07"] == pl.String
    assert out["hs07"].to_list() == ["010101", "840290"]
    assert out.filter(pl.col("category") == "REP").height == 1


def test_cleg_duplicate_code_category_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_cleg_annex_text("840290 REP boiler parts\n840290 REP repeated")


def test_apec_repeated_header_is_removed_and_codes_remain_strings() -> None:
    out = parse_apec_html(FIXTURES / "apec_table_excerpt.html")

    assert out.schema["hs07"] == pl.String
    assert out["hs07"].to_list() == ["441872", "850231"]
    assert out["list_name"].unique().to_list() == ["apec"]


def test_hs96_weight_uses_trade_overlap_and_fallback() -> None:
    mapping = pl.DataFrame(
        {
            "hs07": ["100001", "100002", "100003"],
            "hs96": ["900001", "900001", "900002"],
        }
    )
    membership = pl.DataFrame({"hs07": ["100001"], "is_main": [True]})
    overlap = pl.DataFrame(
        {"hs07": ["100001", "100002"], "exports_usd": [30.0, 70.0]}
    )

    out = build_hs96_green_weights(
        mapping,
        membership,
        overlap,
        list_name="main",
    )

    assert (
        out.filter(pl.col("hs96") == "900001")["green_weight"].item()
        == pytest.approx(0.3)
    )
    fallback = out.filter(pl.col("hs96") == "900002")
    assert fallback["green_weight"].item() == pytest.approx(0.0)
    assert fallback["weight_method"].item() == "code_share_fallback"


def test_unmapped_green_hs07_is_rejected() -> None:
    mapping = pl.read_csv(
        FIXTURES / "h3_to_h1_excerpt.csv",
        schema_overrides={"hs07": pl.String, "hs96": pl.String},
    )
    membership = pl.DataFrame({"hs07": ["999999"], "is_main": [True]})

    with pytest.raises(ValueError, match="unmapped green HS07"):
        build_hs96_green_weights(
            mapping,
            membership,
            pl.DataFrame({"hs07": [], "exports_usd": []}),
            list_name="main",
        )


def test_bec_fraction_is_explicit() -> None:
    mapped = assign_upstream_weight(
        pl.DataFrame(
            {"hs96": ["850001", "850001"], "bec_use": ["capital", "consumption"]}
        )
    )

    assert mapped["upstream_weight"].item() == pytest.approx(0.5)
    assert mapped["ambiguous_bec_mapping"].item() is True


def test_bec_unclassified_does_not_enter_upstream_numerator() -> None:
    source = pl.read_csv(
        FIXTURES / "hs96_to_bec_excerpt.csv",
        schema_overrides={"hs96": pl.String},
    )

    mapped = assign_upstream_weight(source)

    row = mapped.filter(pl.col("hs96") == "850002")
    assert row["upstream_weight"].item() == pytest.approx(0.0)
    assert row["mapping_count"].item() == 1
