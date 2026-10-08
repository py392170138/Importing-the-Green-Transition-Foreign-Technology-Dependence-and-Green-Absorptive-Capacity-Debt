from pathlib import Path

import polars as pl
import pytest

from analysis_fixtures import (
    AnalysisProjectFixture,
    baseline_shares_fixture,
    build_analysis_project_fixture,
    model_panel_fixture,
    synthetic_python_run_context,
)
from green_debt.analysis_io import RunContext
from green_debt.analysis_spec import AnalysisSpec, load_analysis_spec


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def project_fixture(tmp_path: Path) -> AnalysisProjectFixture:
    return build_analysis_project_fixture(tmp_path)


@pytest.fixture
def spec() -> AnalysisSpec:
    return load_analysis_spec(ROOT / "config/analysis.yaml")


@pytest.fixture
def run_context(spec: AnalysisSpec) -> RunContext:
    return synthetic_python_run_context(spec)


@pytest.fixture
def stage_a_fixture() -> pl.DataFrame:
    return model_panel_fixture()


@pytest.fixture
def baseline_share_fixture() -> pl.DataFrame:
    return baseline_shares_fixture()
