from __future__ import annotations

from pathlib import Path
import subprocess

import polars as pl
import pytest

from green_debt.sources.ilostat import extract_ilostat_rds, normalize_ilostat_skill


ROOT = Path(__file__).resolve().parents[2]


TECHNICAL_CODES = [
    "OC2_ISCO08_31",
    "OC2_ISCO08_35",
    "OC2_ISCO08_71",
    "OC2_ISCO08_72",
    "OC2_ISCO08_74",
    "OC2_ISCO08_81",
    "OC2_ISCO08_82",
]


def _row(
    classif1: str,
    value: float,
    *,
    source: str = "SURVEY_A",
    sex: str = "SEX_T",
    best_source: int = 1,
) -> dict[str, object]:
    return {
        "ref_area": "AAA",
        "source": source,
        "indicator": "EMP_TEMP_SEX_OC2_NB",
        "sex": sex,
        "classif1": classif1,
        "time": 2020,
        "obs_value": value,
        "obs_status": None,
        "best_source": best_source,
    }


def test_ilostat_skill_uses_only_approved_total_sex_best_source_rows() -> None:
    rows = [_row(code, float(index)) for index, code in enumerate(TECHNICAL_CODES, 1)]
    rows.append(_row("OC2_ISCO08_TOTAL", 100.0))
    rows.extend(
        [
            _row("OC2_ISCO08_31", 900.0, sex="SEX_M"),
            _row("OC2_ISCO08_35", 900.0, best_source=0),
            _row("OC2_ISCO08_21", 900.0),
        ]
    )

    out = normalize_ilostat_skill(
        pl.DataFrame(rows),
        indicator="EMP_TEMP_SEX_OC2_NB",
        series_id="technical_skill_share_primary",
        robustness_only=False,
    )

    row = out.row(0, named=True)
    assert row["numerator_thousands"] == pytest.approx(28.0)
    assert row["denominator_thousands"] == pytest.approx(100.0)
    assert row["skill_share"] == pytest.approx(0.28)
    assert row["robustness_only"] is False
    assert row["missing_reason"] is None


def test_ilostat_never_uses_a_cross_source_denominator() -> None:
    rows = [_row(code, 1.0, source="SURVEY_A") for code in TECHNICAL_CODES]
    rows.append(_row("OC2_ISCO08_TOTAL", 100.0, source="SURVEY_B"))

    out = normalize_ilostat_skill(
        pl.DataFrame(rows),
        indicator="EMP_TEMP_SEX_OC2_NB",
        series_id="technical_skill_share_primary",
        robustness_only=False,
    )

    assert out.height == 1
    assert out["source_ref"].item() == "SURVEY_A"
    assert out["denominator_thousands"].item() is None
    assert out["skill_share"].item() is None
    assert out["missing_reason"].item() == "same_source_denominator_missing"


def test_rds_extractor_exits_nonzero_for_a_non_data_frame(tmp_path) -> None:
    source = tmp_path / "not_a_frame.rds"
    completed = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            "saveRDS(1:3, commandArgs(trailingOnly=TRUE)[1])",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0

    with pytest.raises(RuntimeError, match="not a data frame"):
        extract_ilostat_rds(
            r_script=ROOT / "03_代码/R/extract_ilostat_rds.R",
            source=source,
            destination=tmp_path / "out.csv",
        )
