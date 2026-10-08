import math
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import green_debt.analysis_io as analysis_io
import green_debt.cli as cli
import green_debt.diagnostics as diagnostics
from analysis_fixtures import (
    baseline_share_contract_frame,
    full_registered_contract_panel,
    load_test_contract,
    replace_authority_table,
    synthetic_python_run_context,
)
from green_debt.analysis_io import (
    AnalysisPaths,
    DiagnosticBundle,
    canonical_authority_hash,
    validate_analysis_authority,
)
from green_debt.analysis_spec import ModelCell
from green_debt.artifacts import verify_manifest
from green_debt.diagnostics import (
    classify_first_stage,
    classify_model_gate,
    exposure_concentration,
    first_stage_screen,
    run_stage_a_diagnostics,
)


def _cell() -> ModelCell:
    return ModelCell(
        "green_export_complexity",
        3,
        "gad_no_supp",
        "core_complete_case",
        "h2_h3_primary",
    )


def test_first_stage_screen_uses_no_outcome_values(stage_a_fixture, spec) -> None:
    changed = stage_a_fixture.with_columns(
        (pl.col("delta_outcome") * 1000 + 77).alias("delta_outcome")
    )

    left = first_stage_screen(stage_a_fixture.lazy(), spec, cells=(_cell(),))
    right = first_stage_screen(changed.lazy(), spec, cells=(_cell(),))

    assert left.equals(right)
    assert left.height == 1
    assert left["n"].item() == 40
    assert left["economies"].item() == 20
    assert left["clusters"].item() == 20
    assert left["instrument_rank"].item() == 2
    assert left["cross_moment_rank"].item() == 2
    assert math.isfinite(left["condition_number"].item())
    assert left["gate_status"].item() == "wild_bootstrap_required"


def test_first_stage_nuisance_design_partials_out_gad(
    stage_a_fixture, spec
) -> None:
    changed = (
        stage_a_fixture.with_row_index("_row")
        .with_columns(
            ((pl.col("_row") ** 2 * 0.017).sin()).alias(
                "gad_lag_p01_p99"
            )
        )
        .drop("_row")
    )
    sample = diagnostics.exact_model_sample(
        changed.lazy(), _cell(), spec
    )
    residualized_gad = diagnostics._qr_residualize(
        sample.select("gad_a").to_numpy(),
        diagnostics._fixed_effect_design(sample),
    )

    assert np.max(np.abs(residualized_gad)) < 1e-12


def test_first_stage_screen_fails_rank_deficient_cell(stage_a_fixture, spec) -> None:
    deficient = stage_a_fixture.with_columns(
        pl.col("Z_p01_p99").alias("Z_GAD_p01_p99")
    )

    out = first_stage_screen(deficient.lazy(), spec, cells=(_cell(),))

    assert out["instrument_rank"].item() == 1
    assert out["rank"].item() == 1
    assert out["gate_status"].item() == "fail_rank_deficient"
    assert out["partial_r2_gimc"].item() is None
    assert out["effective_f_interaction"].item() is None


def test_cluster_and_first_stage_gates_are_literal() -> None:
    assert (
        classify_model_gate(rank=2, clusters=19, condition_number=12.0)
        == "exploratory_lt20_clusters"
    )
    assert (
        classify_model_gate(rank=2, clusters=20, condition_number=12.0)
        == "wild_bootstrap_required"
    )
    assert (
        classify_model_gate(rank=2, clusters=30, condition_number=12.0)
        == "ready_cluster_robust"
    )
    assert (
        classify_model_gate(rank=1, clusters=52, condition_number=12.0)
        == "fail_rank_deficient"
    )
    assert (
        classify_model_gate(rank=2, clusters=52, condition_number=float("inf"))
        == "fail_rank_deficient"
    )
    assert (
        classify_first_stage(rank=2, effective_f=(9.9, 12.0))
        == "weak_reference_below_10"
    )
    assert (
        classify_first_stage(rank=2, effective_f=(10.0, 12.0))
        == "adequate_reference_10"
    )
    assert (
        classify_first_stage(rank=1, effective_f=(100.0, 100.0))
        == "fail_rank_deficient"
    )


def test_exposure_concentration_has_exact_identities(
    baseline_share_fixture,
) -> None:
    out = exposure_concentration(baseline_share_fixture.lazy())

    assert out["effective_units"].to_list() == pytest.approx(
        (1 / out["hhi"]).to_list()
    )
    assert (out["top1_share"] <= out["top5_share"]).all()
    assert (out["top5_share"] <= 1.0 + 1e-12).all()
    row_a = out.filter(pl.col("importer") == "A").row(0, named=True)
    assert row_a["hhi"] == pytest.approx(0.75**2 + 0.25**2)
    assert row_a["top1_share"] == pytest.approx(0.75)
    assert row_a["top5_share"] == pytest.approx(1.0)
    assert row_a["exposure_units"] == 2


def test_exposure_concentration_rejects_non_normalized_or_negative_shares(
    baseline_share_fixture,
) -> None:
    imbalanced = baseline_share_fixture.with_columns(
        pl.when(pl.col("importer") == "A")
        .then(pl.col("baseline_share") * 0.5)
        .otherwise(pl.col("baseline_share"))
        .alias("baseline_share")
    )
    with pytest.raises(ValueError, match="sum to one"):
        exposure_concentration(imbalanced.lazy())

    negative = baseline_share_fixture.with_columns(
        pl.when(
            (pl.col("importer") == "A") & (pl.col("exporter") == "Y")
        )
        .then(-0.25)
        .otherwise(pl.col("baseline_share"))
        .alias("baseline_share")
    )
    with pytest.raises(ValueError, match="nonnegative"):
        exposure_concentration(negative.lazy())


def _tree_metadata(root: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (
            str(path.relative_to(root)),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def test_stage_a_bundle_writes_seven_verified_tables_and_85_gates(
    project_fixture, spec, monkeypatch
) -> None:
    model_contract = load_test_contract(project_fixture.code_root, "model_panel")
    panel = full_registered_contract_panel(spec, model_contract)
    share_contract = load_test_contract(
        project_fixture.code_root, "iv_baseline_shares"
    )
    shares = baseline_share_contract_frame(share_contract)
    replace_authority_table(project_fixture, "model_panel", panel)
    replace_authority_table(project_fixture, "iv_baseline_shares", shares)
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", panel.height)
    authority = validate_analysis_authority(
        project_fixture.code_root, project_fixture.data_root
    )
    context = replace(
        synthetic_python_run_context(spec),
        input_authority_hash=canonical_authority_hash(authority),
    )
    paths = AnalysisPaths(
        code_root=project_fixture.code_root,
        data_root=project_fixture.data_root,
        output_root=project_fixture.code_root / "06_结果/analysis",
    )
    paths.output_root.mkdir(parents=True)
    (paths.output_root / "run_manifest.json").write_text(
        '{"status":"stale-success"}\n', encoding="utf-8"
    )
    before = _tree_metadata(project_fixture.data_root / "05_中间数据")

    bundle = run_stage_a_diagnostics(paths, spec, context)

    assert bundle.cell_count == 85
    assert len(bundle.table_paths) == 7
    assert {path.name for path in bundle.table_paths} == {
        "sample_cells.parquet",
        "missingness.parquet",
        "distributions.parquet",
        "correlations.parquet",
        "first_stage_screen.parquet",
        "exposure_concentration.parquet",
        "descriptive_paths.parquet",
    }
    for path in bundle.table_paths:
        assert verify_manifest(
            path.with_name(f"{path.name}.manifest.json")
        ).output_sha256
        frame = pl.read_parquet(path)
        assert frame.get_column("run_id").unique().to_list() == [context.run_id]
        assert frame.get_column("input_authority_hash").unique().to_list() == [
            context.input_authority_hash
        ]
    assert pl.read_parquet(
        paths.output_root / "diagnostics/sample_cells.parquet"
    ).height == 85
    assert pl.read_parquet(
        paths.output_root / "diagnostics/first_stage_screen.parquet"
    ).height == 85
    distributions = pl.read_parquet(
        paths.output_root / "diagnostics/distributions.parquet"
    )
    assert distributions.get_column("n").unique().to_list() == [40]
    correlations = pl.read_parquet(
        paths.output_root / "diagnostics/correlations.parquet"
    )
    diagonal = correlations.filter(pl.col("variable_x") == pl.col("variable_y"))
    assert diagonal.get_column("correlation").to_list() == pytest.approx(
        [1.0] * diagonal.height
    )
    descriptive = pl.read_parquet(
        paths.output_root / "diagnostics/descriptive_paths.parquet"
    )
    assert 0 in descriptive.get_column("horizon").to_list()
    assert descriptive.get_column("interpretation_status").unique().to_list() == [
        "noncausal_descriptive"
    ]
    gate = json.loads(bundle.gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == "frozen"
    assert gate["run_id"] == context.run_id
    assert gate["input_authority_hash"] == context.input_authority_hash
    assert len(gate["cells"]) == 85
    assert all(cell["horizon"] != 0 for cell in gate["cells"])
    summary = json.loads(
        (paths.output_root / "diagnostics/stage_a_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "valid"
    assert summary["registered_cells"] == 85
    assert not (paths.output_root / "run_manifest.json").exists()
    assert not list(paths.output_root.rglob("*.partial"))
    assert _tree_metadata(project_fixture.data_root / "05_中间数据") == before


def test_stage_a_rejects_context_authority_mismatch_before_writing(
    project_fixture, spec, monkeypatch
) -> None:
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)
    paths = AnalysisPaths(
        code_root=project_fixture.code_root,
        data_root=project_fixture.data_root,
        output_root=project_fixture.code_root / "06_结果/analysis",
    )
    context = replace(
        synthetic_python_run_context(spec), input_authority_hash="0" * 64
    )

    with pytest.raises(ValueError, match="authority hash mismatch"):
        run_stage_a_diagnostics(paths, spec, context)

    assert not paths.output_root.exists()


def test_stage_a_rejects_output_inside_read_only_intermediate(
    project_fixture, spec, monkeypatch
) -> None:
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)
    authority = validate_analysis_authority(
        project_fixture.code_root, project_fixture.data_root
    )
    context = replace(
        synthetic_python_run_context(spec),
        input_authority_hash=canonical_authority_hash(authority),
    )
    intermediate = project_fixture.data_root / "05_中间数据"
    before = _tree_metadata(intermediate)

    with pytest.raises(ValueError, match="05_中间数据"):
        run_stage_a_diagnostics(
            AnalysisPaths(
                code_root=project_fixture.code_root,
                data_root=project_fixture.data_root,
                output_root=intermediate / "forbidden-analysis-output",
            ),
            spec,
            context,
        )

    assert _tree_metadata(intermediate) == before


def test_analysis_diagnostics_cli_requires_explicit_input_and_output_roots(
    tmp_path: Path,
) -> None:
    args = cli.build_parser().parse_args(
        [
            "analysis-diagnostics",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "output"),
        ]
    )

    assert args.command == "analysis-diagnostics"
    assert args.data_root == tmp_path / "data"
    assert args.output_root == tmp_path / "output"


def test_analysis_diagnostics_cli_builds_bound_context_and_reports_bundle(
    project_fixture, spec, monkeypatch, capsys
) -> None:
    authority = object()
    context = synthetic_python_run_context(spec)
    output_root = project_fixture.code_root / "06_结果/analysis"
    expected_bundle = DiagnosticBundle(
        table_paths=(
            output_root / "diagnostics/sample_cells.parquet",
            output_root / "diagnostics/first_stage_screen.parquet",
        ),
        gate_path=output_root / "registries/analysis_gate_v1.json",
        cell_count=85,
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(cli, "PROJECT_ROOT", project_fixture.code_root)
    monkeypatch.setattr(cli, "analysis_preflight", lambda **_kwargs: None)
    monkeypatch.setattr(
        cli, "load_analysis_spec", lambda _path: spec, raising=False
    )
    monkeypatch.setattr(
        cli,
        "validate_analysis_authority",
        lambda _code_root, _data_root: authority,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "build_run_context",
        lambda observed_spec, observed_authority, _code_root: context,
        raising=False,
    )

    def fake_run(paths, observed_spec, observed_context):
        observed.update(
            paths=paths,
            spec=observed_spec,
            context=observed_context,
        )
        return expected_bundle

    monkeypatch.setattr(
        cli, "run_stage_a_diagnostics", fake_run, raising=False
    )

    exit_code = cli.main(
        [
            "analysis-diagnostics",
            "--data-root",
            str(project_fixture.data_root),
            "--output-root",
            str(output_root),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "valid"
    assert payload["run_id"] == context.run_id
    assert payload["cell_count"] == 85
    assert len(payload["table_paths"]) == 2
    assert observed["spec"] is spec
    assert observed["context"] is context
    assert observed["paths"] == AnalysisPaths(
        code_root=project_fixture.code_root,
        data_root=project_fixture.data_root,
        output_root=output_root,
    )
