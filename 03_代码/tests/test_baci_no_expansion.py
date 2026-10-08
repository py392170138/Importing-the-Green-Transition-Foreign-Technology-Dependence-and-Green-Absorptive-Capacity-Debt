from pathlib import Path
from zipfile import ZipFile

import polars as pl

from green_debt.sources.baci import stream_baci_aggregates


def test_baci_stream_never_extracts_zip_or_writes_expanded_csv(
    tmp_path: Path, monkeypatch
) -> None:
    archive = tmp_path / "fixture.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr(
            "BACI_HS96_Y2000_V.csv",
            "t,i,j,k,v,q\n2000,1,2,010101,1,1\n",
        )
    taxonomy = pl.DataFrame(
        {
            "hs6": ["010101"],
            "green_weight": [1.0],
            "taxonomy_version": ["main_hs96"],
        }
    )
    economies = pl.DataFrame(
        {"source_code": ["1", "2"], "economy_id": ["AAA", "BBB"]}
    )

    def extraction_is_forbidden(*_args, **_kwargs):
        raise AssertionError("BACI ZIP extraction is forbidden")

    monkeypatch.setattr(ZipFile, "extract", extraction_is_forbidden)
    monkeypatch.setattr(ZipFile, "extractall", extraction_is_forbidden)

    report = stream_baci_aggregates(
        archive, "HS96", taxonomy, economies, tmp_path / "out"
    )

    assert report.expanded_csv_files_written == 0
    assert not list((tmp_path / "out").rglob("*.csv"))
    assert not any("expanded" in path.name for path in (tmp_path / "out").rglob("*"))
