from pathlib import Path
import json
import os
import subprocess
import sys
from zipfile import ZipFile

import polars as pl
import pytest

from green_debt.sources.baci import stream_baci_aggregates
from green_debt.storage import sha256_file


def _fixture_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    taxonomy = pl.DataFrame(
        {
            "hs6": ["010101"],
            "green_weight": [0.4],
            "taxonomy_version": ["main_hs96"],
        },
        schema={
            "hs6": pl.String,
            "green_weight": pl.Float64,
            "taxonomy_version": pl.String,
        },
    )
    economies = pl.DataFrame(
        {"source_code": ["1", "2"], "economy_id": ["AAA", "BBB"]},
        schema={"source_code": pl.String, "economy_id": pl.String},
    )
    return taxonomy, economies


def _write_archive(path: Path, rows: str, *, member: str = "BACI_HS96_Y2000_V.csv") -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr(member, "t,i,j,k,v,q\n" + rows)


def test_baci_stream_preserves_hs_zeroes_and_converts_thousand_usd(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "fixture.zip"
    _write_archive(
        archive,
        "2000,1,2,010101,1.5,2\n"
        "2000,1,2,999999,2.0,3\n"
        "2000,2,1,999999,0,0\n",
    )
    taxonomy, economies = _fixture_inputs()

    report = stream_baci_aggregates(
        archive, "HS96", taxonomy, economies, tmp_path / "out"
    )

    bilateral = pl.read_parquet(
        tmp_path / "out/green_bilateral/year=2000/*.parquet"
    )
    assert bilateral["hs6"].item() == "010101"
    assert bilateral["trade_value_usd"].item() == 1500.0
    assert bilateral["quantity_tonnes"].item() == 2.0
    assert bilateral["weighted_green_trade_usd"].item() == 600.0

    exports = pl.read_parquet(
        tmp_path / "out/exporter_product/year=2000/*.parquet"
    )
    assert set(exports["hs6"].to_list()) == {"010101", "999999"}
    zero = exports.filter(
        (pl.col("economy_id") == "BBB") & (pl.col("hs6") == "999999")
    )
    assert zero["trade_value_usd"].item() == 0.0

    green_products = pl.read_parquet(
        tmp_path / "out/green_economy_product/year=2000/*.parquet"
    )
    assert set(green_products["flow_role"].to_list()) == {"exporter", "importer"}
    assert green_products["weighted_green_trade_usd"].sum() == 1200.0

    totals = pl.read_parquet(
        tmp_path / "out/economy_year_totals/year=2000/*.parquet"
    )
    aaa = totals.filter(pl.col("economy_id") == "AAA")
    assert aaa["exports_usd"].item() == 3500.0
    assert aaa["imports_usd"].item() == 0.0
    assert aaa["export_source_rows"].item() == 2
    assert aaa["import_source_rows"].item() == 1

    assert report.years == (2000,)
    assert report.source_rows == 3
    assert report.valid_rows == 3
    assert report.quarantined_rows == 0
    assert report.expanded_csv_files_written == 0
    assert report.max_csv_block_size_bytes <= 64 * 1024**2


def test_baci_rejects_ambiguous_annual_members_before_output(tmp_path: Path) -> None:
    archive = tmp_path / "ambiguous.zip"
    with ZipFile(archive, "w") as zf:
        body = "t,i,j,k,v,q\n2000,1,2,010101,1,1\n"
        zf.writestr("BACI_HS96_Y2000_V1.csv", body)
        zf.writestr("nested/BACI_HS96_Y2000_V2.csv", body)
    taxonomy, economies = _fixture_inputs()
    output = tmp_path / "out"

    with pytest.raises(RuntimeError, match="ambiguous BACI member"):
        stream_baci_aggregates(archive, "HS96", taxonomy, economies, output)

    assert not output.exists()


def test_baci_requires_exact_source_columns_before_output(tmp_path: Path) -> None:
    archive = tmp_path / "changed-header.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr(
            "BACI_HS96_Y2000_V.csv",
            "t,i,j,k,value,q\n2000,1,2,010101,1,1\n",
        )
    taxonomy, economies = _fixture_inputs()
    output = tmp_path / "out"

    with pytest.raises(RuntimeError, match="exact columns"):
        stream_baci_aggregates(archive, "HS96", taxonomy, economies, output)

    assert not output.exists()


def test_baci_quarantines_unmapped_and_malformed_rows_with_stable_reasons(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "invalid.zip"
    _write_archive(
        archive,
        "2000,1,999,010101,1,1\n"
        "2000,1,2,12345,1,1\n"
        "2000,1,2,010101,-1,1\n",
    )
    taxonomy, economies = _fixture_inputs()

    report = stream_baci_aggregates(
        archive, "HS96", taxonomy, economies, tmp_path / "out"
    )

    quarantined = pl.read_parquet(
        tmp_path / "out/quarantine/revision=hs96/year=2000/quarantine.parquet"
    )
    assert set(quarantined["quarantine_reason"].to_list()) == {
        "invalid_hs6",
        "negative_trade_value",
        "unmapped_importer",
    }
    assert report.valid_rows == 0
    assert report.quarantined_rows == 3


def test_baci_distinguishes_frozen_exclusion_from_absent_mapping(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "excluded.zip"
    _write_archive(
        archive,
        "2000,1,999,010101,1,1\n"
        "2000,1,998,010101,1,1\n",
    )
    taxonomy, _ = _fixture_inputs()
    economies = pl.DataFrame(
        {
            "source_code": ["1", "2", "999"],
            "economy_id": ["AAA", "BBB", None],
            "exclusion_reason": [None, None, "aggregate_or_row"],
        },
        schema={
            "source_code": pl.String,
            "economy_id": pl.String,
            "exclusion_reason": pl.String,
        },
    )

    stream_baci_aggregates(
        archive, "HS96", taxonomy, economies, tmp_path / "out"
    )

    quarantined = pl.read_parquet(
        tmp_path / "out/quarantine/revision=hs96/year=2000/quarantine.parquet"
    ).sort("source_row_number")
    assert quarantined["quarantine_reason"].to_list() == [
        "excluded_importer",
        "unmapped_importer",
    ]
    assert quarantined["importer_exclusion_reason"].to_list() == [
        "aggregate_or_row",
        None,
    ]


def test_hs96_and_hs07_quarantines_cannot_overwrite_each_other(
    tmp_path: Path,
) -> None:
    hs96 = tmp_path / "hs96.zip"
    hs07 = tmp_path / "hs07.zip"
    _write_archive(
        hs96,
        "2007,1,999,010101,1,1\n",
        member="BACI_HS96_Y2007_V.csv",
    )
    _write_archive(
        hs07,
        "2007,1,999,010101,1,1\n",
        member="BACI_HS07_Y2007_V.csv",
    )
    hs96_taxonomy, economies = _fixture_inputs()
    hs07_taxonomy = hs96_taxonomy.with_columns(
        pl.lit("hs07_native").alias("taxonomy_version")
    )
    output = tmp_path / "out"

    stream_baci_aggregates(hs96, "HS96", hs96_taxonomy, economies, output)
    stream_baci_aggregates(hs07, "HS07", hs07_taxonomy, economies, output)

    assert (
        output / "quarantine/revision=hs96/year=2007/quarantine.parquet"
    ).is_file()
    assert (
        output / "quarantine/revision=hs07/year=2007/quarantine.parquet"
    ).is_file()


def test_baci_decimal_aggregation_is_exact_across_row_chunk_and_reader_thread_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import green_debt.sources.baci as baci
    from green_debt.build import canonical_parquet_fingerprint

    rows = [
        "2000,1,2,010101,100000000,100000000",
        "2000,1,2,010101,0.0000000000000001,0.00000000000000001",
        "2000,1,2,010101,0.0000000000000001,0.00000000000000001",
        "2000,2,1,010101,1.2345678901234567,2.34567890123456789",
    ]
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _write_archive(first, "\n".join(rows) + "\n")
    _write_archive(second, "\n".join(reversed(rows)) + "\n")
    taxonomy, economies = _fixture_inputs()

    monkeypatch.setattr(baci, "MAX_CSV_BLOCK_SIZE_BYTES", 96)
    monkeypatch.setattr(baci, "CSV_READER_USE_THREADS", False)
    stream_baci_aggregates(first, "HS96", taxonomy, economies, tmp_path / "out-one")
    monkeypatch.setattr(baci, "MAX_CSV_BLOCK_SIZE_BYTES", 1024 * 1024)
    monkeypatch.setattr(baci, "CSV_READER_USE_THREADS", True)
    stream_baci_aggregates(second, "HS96", taxonomy, economies, tmp_path / "out-two")

    for relative in (
        "importer_product/year=2000/taxonomy_version=all_hs96.parquet",
        "economy_year_totals/year=2000/taxonomy_version=all_hs96.parquet",
        "green_economy_product/year=2000/taxonomy_version=main_hs96.parquet",
    ):
        one = tmp_path / "out-one" / relative
        two = tmp_path / "out-two" / relative
        manifest = json.loads(one.with_name(f"{one.name}.manifest.json").read_text())
        assert pl.read_parquet(one).equals(pl.read_parquet(two), null_equal=True)
        assert canonical_parquet_fingerprint(
            one, primary_key=tuple(manifest["primary_key"])
        ) == canonical_parquet_fingerprint(
            two, primary_key=tuple(manifest["primary_key"])
        )
        assert sha256_file(one) == sha256_file(two)


def test_baci_decimal_aggregation_is_exact_across_polars_thread_counts(
    tmp_path: Path,
) -> None:
    rows = [
        "2000,1,2,010101,100000000,100000000",
        "2000,1,2,010101,0.0000000000000001,0.00000000000000001",
        "2000,1,2,010101,0.0000000000000001,0.00000000000000001",
    ]
    archives = (tmp_path / "one.zip", tmp_path / "four.zip")
    _write_archive(archives[0], "\n".join(rows) + "\n")
    _write_archive(archives[1], "\n".join(reversed(rows)) + "\n")
    program = """
from pathlib import Path
import polars as pl
from green_debt.sources.baci import stream_baci_aggregates
taxonomy = pl.DataFrame({'hs6':['010101'],'green_weight':[0.4],'taxonomy_version':['main_hs96']})
economies = pl.DataFrame({'source_code':['1','2'],'economy_id':['AAA','BBB']})
stream_baci_aggregates(Path(__import__('sys').argv[1]), 'HS96', taxonomy, economies, Path(__import__('sys').argv[2]))
"""
    outputs = (tmp_path / "threads-one", tmp_path / "threads-four")
    for threads, archive, output in zip(("1", "4"), archives, outputs, strict=True):
        environment = os.environ.copy()
        environment["POLARS_MAX_THREADS"] = threads
        subprocess.run(
            [sys.executable, "-c", program, str(archive), str(output)],
            check=True,
            env=environment,
        )
    for relative in (
        "importer_product/year=2000/taxonomy_version=all_hs96.parquet",
        "economy_year_totals/year=2000/taxonomy_version=all_hs96.parquet",
    ):
        assert sha256_file(outputs[0] / relative) == sha256_file(outputs[1] / relative)


@pytest.mark.parametrize(
    ("value_column", "value", "reason"),
    (
        ("v", "0.1234567890123456789", "invalid_trade_value"),
        ("v", "1000000000", "invalid_trade_value"),
        ("q", "100000000000", "invalid_quantity"),
    ),
)
def test_baci_decimal_precision_and_overflow_gates_fail_closed(
    tmp_path: Path, value_column: str, value: str, reason: str
) -> None:
    archive = tmp_path / f"{value_column}.zip"
    trade = value if value_column == "v" else "1"
    quantity = value if value_column == "q" else "1"
    _write_archive(archive, f"2000,1,2,010101,{trade},{quantity}\n")
    taxonomy, economies = _fixture_inputs()

    report = stream_baci_aggregates(
        archive, "HS96", taxonomy, economies, tmp_path / "out"
    )

    assert report.valid_rows == 0
    quarantined = pl.read_parquet(
        tmp_path / "out/quarantine/revision=hs96/year=2000/quarantine.parquet"
    )
    assert quarantined["quarantine_reason"].to_list() == [reason]
