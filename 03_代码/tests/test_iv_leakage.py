from pathlib import Path

import polars as pl
import pytest

from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    verify_manifest,
    write_authoritative_table,
)
from green_debt import instruments as instruments_module
from green_debt.cli import build_parser
from green_debt.instruments import (
    aggregate_shift_share,
    build_baseline_shares,
    build_cmz,
    build_partner_shocks,
    build_z_gad_interactions,
    independent_country_instrument_audit,
    independent_partner_shock_audit,
    validate_iv_input_columns,
)


@pytest.mark.parametrize("column", ["current_import_share", "future_shock", "outcome_value"])
def test_prohibited_input_columns_fail_before_aggregation(column: str) -> None:
    with pytest.raises(ValueError, match=column):
        validate_iv_input_columns(("importer", "exporter", "hs6", "year", column))


def test_baseline_share_table_is_time_invariant() -> None:
    baseline = pl.DataFrame(
        {
            "importer": ["C"],
            "exporter": ["J"],
            "hs6": ["p"],
            "mean_weighted_import_usd": [1.0],
        }
    )
    out = build_baseline_shares(baseline, minimum_share=0.0001)
    assert "year" not in out.retained.columns
    assert out.retained["baseline_share"].item() == pytest.approx(1.0)


def test_missing_retained_shock_never_renormalizes_weights_over_time() -> None:
    weights = pl.DataFrame(
        {
            "importer": ["C", "C"],
            "exporter": ["J1", "J2"],
            "hs6": ["p1", "p2"],
            "baseline_share": [0.75, 0.25],
            "confirmatory_iv_eligible": [True, True],
        }
    )
    shocks = pl.DataFrame(
        {
            "destination_excluded": ["C"],
            "exporter": ["J1"],
            "hs6": ["p1"],
            "year": [2001],
            "partner_shock": [0.2],
            "shock_missing_reason": [None],
        }
    )
    out = aggregate_shift_share(weights, shocks, expected_years=(2001,))
    row = out.country_year.row(0, named=True)
    assert row["z"] is None
    assert row["z_missing_reason"] == "retained_shock_missing"
    assert row["observed_baseline_share"] == pytest.approx(0.75)
    assert row["confirmatory_iv_eligible"] is False
    assert out.contributions["baseline_share"].to_list() == pytest.approx([0.75, 0.25])


def test_z_is_exact_sum_of_long_contributions_and_destination_matches_importer() -> None:
    weights = pl.DataFrame(
        {
            "importer": ["C", "C"],
            "exporter": ["J1", "J2"],
            "hs6": ["p1", "p2"],
            "baseline_share": [0.75, 0.25],
            "confirmatory_iv_eligible": [True, True],
        }
    )
    shocks = pl.DataFrame(
        {
            "destination_excluded": ["C", "C"],
            "exporter": ["J1", "J2"],
            "hs6": ["p1", "p2"],
            "year": [2001, 2001],
            "partner_shock": [0.2, -0.4],
            "shock_missing_reason": [None, None],
        }
    )
    out = aggregate_shift_share(weights, shocks, expected_years=(2001,))
    assert out.country_year["z"].item() == pytest.approx(0.05)
    assert out.contributions["contribution"].sum() == pytest.approx(0.05)
    assert out.contributions.filter(
        pl.col("destination_excluded") != pl.col("importer")
    ).is_empty()


def test_z_gad_interactions_keep_version_in_key_and_use_lagged_gad() -> None:
    z = pl.DataFrame({"importer": ["C"], "year": [2001], "z": [2.0]})
    gad = pl.DataFrame(
        {
            "economy_id": ["C", "C", "C", "C"],
            "year": [2000, 2001, 2000, 2001],
            "specification_id": ["gad_core", "gad_core", "gad_no_supp", "gad_no_supp"],
            "gad": [3.0, 30.0, 5.0, 50.0],
        }
    )
    out = build_z_gad_interactions(z, gad, gad_versions=("gad_core", "gad_no_supp"))
    assert out.select("importer", "year", "gad_version").is_duplicated().sum() == 0
    assert dict(zip(out["gad_version"], out["z_gad"], strict=True)) == pytest.approx(
        {"gad_core": 6.0, "gad_no_supp": 10.0}
    )
    assert out["gad_time"].to_list() == [2000, 2000]


def test_cmz_requires_five_complete_terms_and_uses_lagged_absorption_midrank() -> None:
    z = pl.DataFrame(
        {
            "importer": ["C"] * 5 + ["D"] * 5,
            "year": list(range(2001, 2006)) * 2,
            "z": [1.0] * 10,
        }
    )
    gad = pl.DataFrame(
        {
            "economy_id": ["C"] * 6 + ["D"] * 6,
            "year": list(range(2000, 2006)) * 2,
            "specification_id": ["gad_core"] * 12,
            "gad": [2.0] * 12,
            "absorption": [1.0, 1.0, 1.0, 1.0, 1.0, 999.0] + [2.0] * 6,
            "rho": [0.5] * 12,
            "confirmatory_eligible": [True] * 12,
        }
    )
    out = build_cmz(z, gad, gad_versions=("gad_core",), lags=5)
    c = out.filter((pl.col("importer") == "C") & (pl.col("year") == 2005)).row(0, named=True)
    d = out.filter((pl.col("importer") == "D") & (pl.col("year") == 2005)).row(0, named=True)
    assert c["cmz"] == pytest.approx(1.9375)
    assert c["cmz_complete_terms"] == 5
    assert c["absorption_latest_time"] == 2004
    assert d["cmz"] == pytest.approx(0.0)
    assert out.filter(pl.col("year") == 2004)["cmz"].null_count() == 2
    assert set(out.filter(pl.col("year") == 2004)["cmz_missing_reason"]) == {"fewer_than_five_complete_terms"}


def test_cmz_schema_does_not_depend_on_first_hundred_rows_being_complete() -> None:
    incomplete = [f"A{number:02d}" for number in range(30)]
    economies = [*incomplete, "ZZY", "ZZZ"]
    z_rows = [
        {"importer": economy, "year": year, "z": 1.0}
        for economy in incomplete
        for year in range(2001, 2005)
    ] + [
        {"importer": economy, "year": year, "z": 1.0}
        for economy in ("ZZY", "ZZZ")
        for year in range(2001, 2006)
    ]
    gad_rows = [
        {
            "economy_id": economy,
            "year": year,
            "specification_id": "gad_core",
            "absorption": float(position),
            "rho": 0.5,
            "confirmatory_eligible": True,
        }
        for position, economy in enumerate(economies)
        for year in range(2000, 2005)
    ]
    out = build_cmz(
        pl.DataFrame(z_rows),
        pl.DataFrame(gad_rows),
        gad_versions=("gad_core",),
        lags=5,
    )
    assert out.filter(pl.col("cmz").is_not_null()).height == 2
    assert out.schema["cmz"] == pl.Float64


def test_instrument_commands_are_registered() -> None:
    parser = build_parser()
    build = parser.parse_args(["build-instruments", "--taxonomy", "main_hs96"])
    audit = parser.parse_args(["audit-instruments"])
    assert build.command == "build-instruments"
    assert audit.command == "audit-instruments"


def test_parent_lineage_accepts_the_current_bundle_manifest_destination(
    tmp_path: Path,
) -> None:
    authority = tmp_path / "registry/current/bundle/data/parent.parquet"
    compatibility = tmp_path / "data/parent.parquet"
    child = tmp_path / "data/child.parquet"
    contract = TableContract(
        table_id="parent",
        schema_version="1.0.0",
        primary_key=("id",),
        columns={"id": "String"},
        units={},
    )
    write_authoritative_table(
        pl.DataFrame({"id": ["parent"]}),
        contract,
        authority,
        (),
        BuildIdentity(command="parent", code_commit="a" * 40),
    )
    compatibility.parent.mkdir(parents=True)
    compatibility.write_bytes(authority.read_bytes())
    compatibility.with_name(f"{compatibility.name}.manifest.json").write_bytes(
        authority.with_name(f"{authority.name}.manifest.json").read_bytes()
    )
    write_authoritative_table(
        pl.DataFrame({"id": ["child"]}),
        TableContract(
            table_id="child",
            schema_version="1.0.0",
            primary_key=("id",),
            columns={"id": "String"},
            units={},
        ),
        child,
        (InputArtifact.from_path(authority),),
        BuildIdentity(command="child", code_commit="a" * 40),
    )
    child_manifest = verify_manifest(
        child.with_name(f"{child.name}.manifest.json")
    )

    lineage_audit = getattr(
        instruments_module, "_direct_parent_lineage_failures", None
    )
    assert lineage_audit is not None
    assert lineage_audit((child_manifest,), ((compatibility,),)) == 0


def test_parent_based_shock_audit_rejects_internally_plausible_tampering() -> None:
    bilateral = pl.DataFrame(
        {
            "year": [2000, 2001, 2000, 2001, 2000, 2001],
            "exporter": ["J", "J", "J", "J", "K", "K"],
            "importer": ["C", "C", "D", "D", "D", "D"],
            "hs6": ["p"] * 6,
            "weighted_green_trade_usd": [10.0, 20.0, 30.0, 60.0, 30.0, 30.0],
        }
    )
    totals = pl.DataFrame(
        {
            "year": [2000, 2001, 2000, 2001],
            "exporter": ["J", "J", "K", "K"],
            "hs6": ["p"] * 4,
            "weighted_green_trade_usd": [40.0, 80.0, 30.0, 30.0],
        }
    )
    cells = pl.DataFrame({"importer": ["C"], "exporter": ["J"], "hs6": ["p"]})
    published = build_partner_shocks(
        bilateral,
        cells=cells,
        expected_years=(2000, 2001),
        exporter_product_totals=totals,
    ).filter(pl.col("year") == 2001).with_columns(
        pl.lit("C").alias("importer"),
        pl.lit("main_0.0001").alias("share_version"),
        pl.lit(1.0).alias("baseline_share"),
        pl.col("partner_shock").alias("contribution"),
        pl.lit(None, dtype=pl.String).alias("contribution_missing_reason"),
    )
    assert independent_partner_shock_audit(published, bilateral, totals)[
        "shock_parent_reconstruction_failures"
    ] == 0

    fake_exporter_growth = 2.0 * (40.0 - 30.0) / 70.0
    fake_global_growth = published["global_product_growth_excluding_destination"].item()
    fake_shock = fake_exporter_growth - fake_global_growth
    tampered = published.with_columns(
        pl.lit(40.0).alias("exports_excluding_destination"),
        pl.lit(fake_exporter_growth).alias("exporter_growth_excluding_destination"),
        pl.lit(fake_shock).alias("partner_shock"),
        pl.lit(fake_shock).alias("contribution"),
    )
    assert independent_partner_shock_audit(tampered, bilateral, totals)[
        "shock_parent_reconstruction_failures"
    ] > 0


def test_parent_based_shock_audit_accepts_machine_precision_level_roundoff() -> None:
    scale = 1_000_000_000.0
    bilateral = pl.DataFrame(
        {
            "year": [2000, 2001, 2000, 2001, 2000, 2001],
            "exporter": ["J", "J", "J", "J", "K", "K"],
            "importer": ["C", "C", "D", "D", "D", "D"],
            "hs6": ["p"] * 6,
            "weighted_green_trade_usd": [
                value * scale for value in (10.0, 20.0, 30.0, 60.0, 30.0, 30.0)
            ],
        }
    )
    totals = pl.DataFrame(
        {
            "year": [2000, 2001, 2000, 2001],
            "exporter": ["J", "J", "K", "K"],
            "hs6": ["p"] * 4,
            "weighted_green_trade_usd": [
                value * scale for value in (40.0, 80.0, 30.0, 30.0)
            ],
        }
    )
    cells = pl.DataFrame(
        {"importer": ["C"], "exporter": ["J"], "hs6": ["p"]}
    )
    published = (
        build_partner_shocks(
            bilateral,
            cells=cells,
            expected_years=(2000, 2001),
            exporter_product_totals=totals,
        )
        .filter(pl.col("year") == 2001)
        .with_columns(
            pl.lit("C").alias("importer"),
            pl.lit("main_0.0001").alias("share_version"),
            pl.lit(1.0).alias("baseline_share"),
            pl.col("partner_shock").alias("contribution"),
            pl.lit(None, dtype=pl.String).alias("contribution_missing_reason"),
        )
        .with_columns(
            (
                pl.col("global_product_exports_excluding_destination_lag")
                + 1e-5
            ).alias("global_product_exports_excluding_destination_lag"),
            (pl.col("global_product_exports_excluding_destination") + 1e-5).alias(
                "global_product_exports_excluding_destination"
            ),
        )
    )

    assert independent_partner_shock_audit(published, bilateral, totals)[
        "shock_parent_reconstruction_failures"
    ] == 0


def _published_country_fixture() -> tuple[pl.DataFrame, pl.DataFrame]:
    z = pl.DataFrame(
        {
            "importer": ["C"] * 5 + ["D"] * 5,
            "year": list(range(2001, 2006)) * 2,
            "z": [1.0] * 10,
        }
    )
    gad = pl.DataFrame(
        {
            "economy_id": ["C"] * 6 + ["D"] * 6,
            "year": list(range(2000, 2006)) * 2,
            "specification_id": ["gad_core"] * 12,
            "gad": [float(year) for year in range(2000, 2006)] * 2,
            "absorption": [1.0] * 6 + [2.0] * 6,
            "rho": [0.5] * 12,
            "confirmatory_eligible": [True] * 12,
        }
    )
    interaction = build_z_gad_interactions(z, gad, gad_versions=("gad_core",))
    cmz = build_cmz(z, gad, gad_versions=("gad_core",), lags=5)
    return interaction.join(
        cmz,
        on=("importer", "share_version", "year", "gad_version"),
        how="left",
    ), gad


def test_parent_based_gad_audit_rejects_contemporaneous_self_consistent_values() -> None:
    published, gad = _published_country_fixture()
    assert sum(independent_country_instrument_audit(published, gad).values()) == 0
    tampered = published.with_columns(
        pl.when((pl.col("importer") == "C") & (pl.col("year") == 2005))
        .then(pl.lit(2005))
        .otherwise(pl.col("gad_time"))
        .alias("gad_time"),
        pl.when((pl.col("importer") == "C") & (pl.col("year") == 2005))
        .then(pl.lit(2005.0))
        .otherwise(pl.col("gad_lag"))
        .alias("gad_lag"),
    ).with_columns((pl.col("z") * pl.col("gad_lag")).alias("z_gad"))
    assert independent_country_instrument_audit(tampered, gad)[
        "gad_parent_reconstruction_failures"
    ] > 0


@pytest.mark.parametrize(
    ("updates", "metric"),
    [
        ({"cmz": 999.0}, "cmz_value_reconstruction_failures"),
        (
            {"cmz": None, "cmz_complete_terms": 4, "cmz_missing_reason": "fewer_than_five_complete_terms"},
            "cmz_term_reason_reconstruction_failures",
        ),
        ({"absorption_latest_time": 2005}, "cmz_time_reconstruction_failures"),
    ],
)
def test_parent_based_cmz_audit_rejects_rank_term_and_time_tampering(
    updates: dict[str, object], metric: str
) -> None:
    published, gad = _published_country_fixture()
    condition = (pl.col("importer") == "C") & (pl.col("year") == 2005)
    tampered = published
    for column, value in updates.items():
        tampered = tampered.with_columns(
            pl.when(condition).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
        )
    assert independent_country_instrument_audit(tampered, gad)[metric] > 0
