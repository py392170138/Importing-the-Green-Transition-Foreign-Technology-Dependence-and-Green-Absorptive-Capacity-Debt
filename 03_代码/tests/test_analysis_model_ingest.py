from pathlib import Path
import tomllib

import polars as pl
import pytest
from scipy import stats

import green_debt.analysis_io as analysis_io
import green_debt.cli as cli
from analysis_fixtures import (
    write_staging_estimates,
    write_threshold_audit_staging,
)
from green_debt.analysis_io import (
    _validate_threshold_estimates,
    ingest_model_bundle,
    load_stage_a_run_context,
)
from green_debt.artifacts import verify_manifest
from green_debt.cli import build_parser


def test_model_ingest_requires_complete_provenance(
    tmp_path: Path, run_context, spec
) -> None:
    staging = write_staging_estimates(tmp_path, omit={"renv_lock_sha256"})
    with pytest.raises(ValueError, match="renv_lock_sha256"):
        ingest_model_bundle(staging, tmp_path / "models", run_context, spec)


def test_model_ingest_rejects_duplicate_term_key(
    tmp_path: Path, run_context, spec
) -> None:
    staging = write_staging_estimates(tmp_path, duplicate_first_row=True)
    with pytest.raises(ValueError, match="duplicate"):
        ingest_model_bundle(staging, tmp_path / "models", run_context, spec)


@pytest.mark.parametrize(
    ("fixture_option", "message"),
    [
        ({"covariance_asymmetry": True}, "symmetric"),
        ({"covariance_negative": True}, "positive semidefinite"),
        ({"n_offset": 1}, "stage A gate"),
        ({"bad_confidence": True}, "confidence arithmetic"),
    ],
)
def test_model_ingest_fails_closed_on_invalid_model_bundle(
    tmp_path: Path, run_context, spec, fixture_option, message
) -> None:
    staging = write_staging_estimates(tmp_path, **fixture_option)
    with pytest.raises(ValueError, match=message):
        ingest_model_bundle(staging, tmp_path / "models", run_context, spec)


def test_model_ingest_requires_wild_fields_for_20_to_29_clusters(
    tmp_path: Path, run_context, spec
) -> None:
    staging = write_staging_estimates(tmp_path, clusters=20, omit_wild=True)
    with pytest.raises(ValueError, match="wild-bootstrap fields"):
        ingest_model_bundle(staging, tmp_path / "models", run_context, spec)


def test_model_ingest_preserves_unbounded_wild_intervals(
    tmp_path: Path, run_context, spec
) -> None:
    staging = write_staging_estimates(
        tmp_path, clusters=20, wild_unbounded=True
    )

    published = ingest_model_bundle(
        staging, tmp_path / "models", run_context, spec
    )

    fe = pl.read_parquet(
        next(path for path in published if path.name == "lp_fe.parquet")
    )
    assert fe.get_column("inference_status").unique().to_list() == [
        "wild_bootstrap_unbounded"
    ]
    assert fe.get_column("wild_conf_low").null_count() == fe.height
    assert fe.get_column("wild_p_value").null_count() == 0


def test_model_ingest_publishes_four_contract_checked_tables(
    tmp_path: Path, run_context, spec
) -> None:
    staging = write_staging_estimates(tmp_path)

    published = ingest_model_bundle(
        staging, tmp_path / "models", run_context, spec
    )

    assert not staging.exists()
    assert {path.name for path in published} == {
        "lp_fe.parquet",
        "lp_iv.parquet",
        "model_covariance.parquet",
        "marginal_effects.parquet",
    }
    expected_rows = {
        "lp_fe.parquet": 84 * 2,
        "lp_iv.parquet": 84 * 2,
        "model_covariance.parquet": 84 * 2 * 4,
        "marginal_effects.parquet": 84 * 2 * 3,
    }
    for path in published:
        assert pl.read_parquet(path).height == expected_rows[path.name]
        assert verify_manifest(
            path.with_name(f"{path.name}.manifest.json")
        ).output_sha256


def test_model_ingest_cli_and_stage_a_context_are_explicit(
    tmp_path: Path, run_context
) -> None:
    write_staging_estimates(tmp_path)
    args = build_parser().parse_args(
        [
            "analysis-ingest-models",
            "--kind",
            "lp",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path),
        ]
    )

    assert args.command == "analysis-ingest-models"
    assert args.kind == "lp"
    assert args.data_root == tmp_path / "data"
    assert load_stage_a_run_context(tmp_path) == run_context


def test_threshold_audit_ingest_rejects_result_q_retargeting(
    tmp_path: Path, run_context, spec
) -> None:
    staging = write_threshold_audit_staging(
        tmp_path, threshold_q_offset=0.01
    )
    with pytest.raises(ValueError, match="frozen registry"):
        ingest_model_bundle(staging, tmp_path / "models", run_context, spec)


def test_threshold_audit_ingest_publishes_four_tables_and_preserves_hs6(
    tmp_path: Path, run_context, spec
) -> None:
    staging = write_threshold_audit_staging(tmp_path)

    published = ingest_model_bundle(
        staging, tmp_path / "models", run_context, spec
    )

    assert not staging.exists()
    assert {path.name for path in published} == {
        "threshold_estimates.parquet",
        "weak_iv_sets.parquet",
        "shift_share_summary.parquet",
        "shift_share_weights.parquet",
    }
    threshold = pl.read_parquet(
        next(path for path in published if path.name == "threshold_estimates.parquet")
    )
    weak = pl.read_parquet(
        next(path for path in published if path.name == "weak_iv_sets.parquet")
    )
    weights = pl.read_parquet(
        next(path for path in published if path.name == "shift_share_weights.parquet")
    )
    assert threshold.height == 39 * 2
    assert weak.height == 45
    assert weights.height == 45
    assert weights.get_column("hs6").unique().to_list() == ["000001"]
    for path in published:
        assert verify_manifest(
            path.with_name(f"{path.name}.manifest.json")
        ).output_sha256


def test_threshold_audit_preserves_directional_ar_bounds_and_weights(
    tmp_path: Path, run_context, spec
) -> None:
    staging = write_threshold_audit_staging(
        tmp_path,
        partial_ar_bounds=True,
        unavailable_shock_covariance=True,
    )

    published = ingest_model_bundle(
        staging, tmp_path / "models", run_context, spec
    )

    weak = pl.read_parquet(
        next(path for path in published if path.name == "weak_iv_sets.parquet")
    )
    summaries = pl.read_parquet(
        next(
            path
            for path in published
            if path.name == "shift_share_summary.parquet"
        )
    )
    weights = pl.read_parquet(
        next(
            path
            for path in published
            if path.name == "shift_share_weights.parquet"
        )
    )
    assert weak.get_column("beta_low").null_count() == weak.height
    assert weak.get_column("beta_high").null_count() == 0
    assert weak.get_column("theta_low").null_count() == 0
    assert weak.get_column("theta_high").null_count() == 0
    assert summaries.get_column("shock_inference_status").unique().to_list() == [
        "unavailable"
    ]
    assert summaries.get_column("absolute_weight_sum").null_count() == 0
    assert summaries.get_column("shock_std_error_gimc").null_count() == (
        summaries.height
    )
    assert weights.height == 45


def test_threshold_audit_kind_is_exposed_by_cli(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "analysis-ingest-models",
            "--kind",
            "threshold-and-iv-audit",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path),
        ]
    )
    assert args.kind == "threshold-and-iv-audit"


def _threshold_validation_inputs(*, clusters: int = 30):
    key = (
        "confirmatory",
        "green_export_complexity",
        1,
        "gad_no_supp",
        "core_complete_case",
    )
    reference_df = float(clusters - 1)
    critical = 2.045229642132703 if clusters == 30 else 2.063898561628021
    rows = []
    difference_estimate = -1.2
    difference_std_error = 0.2
    for regime, estimate in (("low", 1.0), ("high", -0.2)):
        std_error = 0.1
        rows.append(
            {
                "analysis_family": key[0],
                "outcome_id": key[1],
                "horizon": key[2],
                "gad_version": key[3],
                "sample_version": key[4],
                "estimator": "threshold_iv",
                "selection_outcome": "green_industrial_upgrading_index",
                "registry_hash": "a" * 64,
                "registry_sample_hash": "b" * 64,
                "q": 1.4777372886633888,
                "regime": regime,
                "estimate": estimate,
                "std_error": std_error,
                "conf_low": estimate - critical * std_error,
                "conf_high": estimate + critical * std_error,
                "p_value": float(
                    2 * stats.t.sf(abs(estimate / std_error), reference_df)
                ),
                "wild_estimate": estimate if clusters == 25 else None,
                "wild_std_error": std_error if clusters == 25 else None,
                "wild_conf_low": estimate - 0.3 if clusters == 25 else None,
                "wild_conf_high": estimate + 0.3 if clusters == 25 else None,
                "wild_p_value": 0.2 if clusters == 25 else None,
                "wild_inference_status": (
                    "wild_bootstrap_bounded" if clusters == 25 else "not_required"
                ),
                "wild_draws": 9999 if clusters == 25 else None,
                "wild_seed": 20260820 if clusters == 25 else None,
                "difference_estimate": difference_estimate,
                "low_high_covariance": -0.01,
                "difference_std_error": difference_std_error,
                "difference_conf_low": difference_estimate - critical * difference_std_error,
                "difference_conf_high": difference_estimate + critical * difference_std_error,
                "difference_p_value": float(
                    2
                    * stats.t.sf(
                        abs(difference_estimate / difference_std_error),
                        reference_df,
                    )
                ),
                "difference_wild_estimate": (
                    difference_estimate if clusters == 25 else None
                ),
                "difference_wild_std_error": (
                    difference_std_error if clusters == 25 else None
                ),
                "difference_wild_conf_low": -1.6 if clusters == 25 else None,
                "difference_wild_conf_high": -0.8 if clusters == 25 else None,
                "difference_wild_p_value": 0.1 if clusters == 25 else None,
                "difference_wild_inference_status": (
                    "wild_bootstrap_bounded" if clusters == 25 else "not_required"
                ),
                "difference_wild_draws": 9999 if clusters == 25 else None,
                "difference_wild_seed": 20260820 if clusters == 25 else None,
                "regime_n": 20,
                "regime_share": 0.5,
                "n": 40,
                "economies": clusters,
                "clusters": clusters,
                "first_stage_status": "adequate_reference_10",
                "inference_status": (
                    "wild_bootstrap_required" if clusters == 25 else "cluster_robust"
                ),
                "reference_distribution": "cluster_t",
                "reference_df": reference_df,
            }
        )
    frame = pl.DataFrame(rows)
    gates = {
        key: {
            "n": 40,
            "economies": clusters,
            "clusters": clusters,
            "rank": 2,
            "gate_status": (
                "wild_bootstrap_required" if clusters < 30 else "ready_cluster_robust"
            ),
        }
    }
    registry = {
        "selection_outcome": "green_industrial_upgrading_index",
        "registry_hash": "a" * 64,
        "sample_hash": "b" * 64,
        "q": 1.4777372886633888,
    }
    return key, frame, gates, registry


def test_threshold_validation_rejects_paired_contrast_mismatch() -> None:
    key, frame, gates, registry = _threshold_validation_inputs()
    frame = frame.with_columns(
        pl.when(pl.col("regime") == "high")
        .then(pl.col("difference_estimate") + 0.01)
        .otherwise(pl.col("difference_estimate"))
        .alias("difference_estimate")
    )
    with pytest.raises(ValueError, match="paired high-minus-low contrast"):
        _validate_threshold_estimates(frame, {key}, gates, registry)


def test_threshold_validation_rejects_wrong_contrast_variance() -> None:
    key, frame, gates, registry = _threshold_validation_inputs()
    frame = frame.with_columns(pl.lit(0.21).alias("difference_std_error"))
    with pytest.raises(ValueError, match="contrast variance"):
        _validate_threshold_estimates(frame, {key}, gates, registry)


def test_threshold_validation_requires_all_three_wild_tests_at_20_to_29() -> None:
    key, frame, gates, registry = _threshold_validation_inputs(clusters=25)
    frame = frame.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("difference_wild_conf_low"),
        pl.lit("not_required").alias("difference_wild_inference_status"),
    )
    with pytest.raises(ValueError, match="20-29.*difference wild bootstrap"):
        _validate_threshold_estimates(frame, {key}, gates, registry)


def test_threshold_validation_rejects_fit_failure_or_changed_wild_point() -> None:
    key, frame, gates, registry = _threshold_validation_inputs(clusters=25)
    failed = frame.with_columns(pl.lit("fit_failed").alias("inference_status"))
    with pytest.raises(ValueError, match="all 39 threshold cells"):
        _validate_threshold_estimates(failed, {key}, gates, registry)

    changed = frame.with_columns(
        pl.when(pl.col("regime") == "low")
        .then(pl.col("wild_estimate") + 0.1)
        .otherwise(pl.col("wild_estimate"))
        .alias("wild_estimate")
    )
    with pytest.raises(ValueError, match="same-sample conventional estimate"):
        _validate_threshold_estimates(changed, {key}, gates, registry)


def test_threshold_validation_accepts_unbounded_wild_only_with_null_bounds() -> None:
    key, frame, gates, registry = _threshold_validation_inputs(clusters=25)
    frame = frame.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("difference_wild_conf_low"),
        pl.lit(None, dtype=pl.Float64).alias("difference_wild_conf_high"),
        pl.lit("wild_bootstrap_unbounded").alias(
            "difference_wild_inference_status"
        ),
    )
    _validate_threshold_estimates(frame, {key}, gates, registry)

    fallback = frame.with_columns(pl.lit(-1.6).alias("difference_wild_conf_low"))
    with pytest.raises(ValueError, match="unbounded.*null bounds"):
        _validate_threshold_estimates(fallback, {key}, gates, registry)


def test_model_ingest_rejects_unauthorized_root_before_touching_it(
    project_fixture, tmp_path: Path
) -> None:
    unauthorized = tmp_path / "unauthorized-output"
    unauthorized.mkdir()
    sentinel = unauthorized / "run_manifest.json"
    sentinel.write_text("prior success\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen outputs.root"):
        cli.main(
            [
                "analysis-ingest-models",
                "--kind",
                "lp",
                "--data-root",
                str(project_fixture.data_root),
                "--output-root",
                str(unauthorized),
            ]
        )
    assert sentinel.read_text(encoding="utf-8") == "prior success\n"


def test_model_ingest_rejects_symlink_output_alias(
    project_fixture
) -> None:
    frozen = project_fixture.code_root / "06_结果/analysis"
    frozen.mkdir(parents=True)
    alias = project_fixture.code_root / "analysis-alias"
    alias.symlink_to(frozen, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        cli.main(
            [
                "analysis-ingest-models",
                "--kind",
                "lp",
                "--data-root",
                str(project_fixture.data_root),
                "--output-root",
                str(alias),
            ]
        )


def test_model_ingest_runs_capacity_preflight_before_reading_staging(
    project_fixture, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", project_fixture.code_root)
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)
    monkeypatch.setattr(
        analysis_io, "directory_usage_bytes", lambda _path: 10 * 1024**3
    )
    output = project_fixture.code_root / "06_结果/analysis"

    with pytest.raises(RuntimeError, match="10 GB analysis output quota"):
        cli.main(
            [
                "analysis-ingest-models",
                "--kind",
                "lp",
                "--data-root",
                str(project_fixture.data_root),
                "--output-root",
                str(output),
            ]
        )
    assert not (output / "models").exists()


def test_julia_iv_bootstrap_environment_is_manifest_locked() -> None:
    root = Path(__file__).resolve().parents[2]
    with (root / "julia/Project.toml").open("rb") as handle:
        project = tomllib.load(handle)
    with (root / "julia/Manifest.toml").open("rb") as handle:
        manifest = tomllib.load(handle)

    assert manifest["julia_version"] == "1.12.7"
    assert project["deps"] == {
        "StableRNGs": "860ef19b-820b-49d6-a774-d7a799459cd3",
        "Tables": "bd369af6-aec1-5ad0-b16a-f7cc5008161c",
        "WildBootTests": "65c2e505-86ba-4c19-93f1-95506c1443d5",
    }
    assert project["compat"] == {
        "StableRNGs": "=1.0.4",
        "Tables": "=1.14.0",
        "WildBootTests": "=0.9.8",
        "julia": "=1.12.7",
    }
    wild = manifest["deps"]["WildBootTests"]
    if isinstance(wild, list):
        assert len(wild) == 1
        wild = wild[0]
    assert wild["version"] == "0.9.8"
