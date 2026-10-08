import json
from pathlib import Path
import shutil

import polars as pl
import pytest

import green_debt.analysis_io as analysis_io
from analysis_fixtures import complete_result_frames
from green_debt.analysis_audit import (
    _audit_publication_set,
    _registered_interval,
    _publication_hashes,
    _evidence_scorecard_contract,
    _manifest_git_identities,
    ar_directionally_supports,
    audit_analysis_frames,
    audit_analysis_outputs,
    direction_matches,
    grade_evidence,
    publish_analysis_audit,
    registered_inference_status,
    summarize_hypotheses,
    write_evidence_scorecard,
)
from green_debt.analysis_io import load_stage_a_run_context
from green_debt.artifacts import verify_manifest
from green_debt.cli import build_parser


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT.parents[1] if ROOT.parent.name == ".worktrees" else ROOT


def test_evidence_grade_follows_registered_inference_status() -> None:
    assert grade_evidence(
        rank_ok=True,
        clusters=52,
        ar_status="bounded",
        iv_direction_matches_ar=True,
        shock_status="available",
    ) == "conditional_causal"
    assert grade_evidence(
        rank_ok=True,
        clusters=52,
        ar_status="unbounded",
        iv_direction_matches_ar=False,
        shock_status="available",
    ) == "conditional_association"
    assert grade_evidence(
        rank_ok=True,
        clusters=18,
        ar_status="bounded",
        iv_direction_matches_ar=True,
        shock_status="available",
    ) == "exploratory"
    assert grade_evidence(
        rank_ok=False,
        clusters=52,
        ar_status="empty",
        iv_direction_matches_ar=False,
        shock_status="unavailable",
    ) == "not_estimated"


def test_unresolved_concentration_policy_forces_estimated_cells_exploratory() -> None:
    assert grade_evidence(
        rank_ok=True,
        clusters=52,
        ar_status="bounded",
        iv_direction_matches_ar=True,
        shock_status="available",
        concentration_status="unresolved_no_preregistered_cutoff",
    ) == "exploratory"


def test_unbounded_registered_interval_never_falls_back_to_conventional() -> None:
    assert _registered_interval(
        {
            "inference_status": "wild_bootstrap_unbounded",
            "conf_low": -0.4,
            "conf_high": 0.8,
            "wild_conf_low": None,
            "wild_conf_high": None,
        }
    ) == (None, None)


def test_scorecard_contract_distinguishes_model_and_audit_code_identity() -> None:
    columns = _evidence_scorecard_contract().columns
    assert columns["upstream_model_git_commit"] == "String"
    assert columns["audit_git_commit"] == "String"
    assert columns["evidence_policy_sha256"] == "String"
    assert columns["concentration_status"] == "String"
    assert columns["concentration_cutoff_hhi"] == "Float64"
    assert columns["concentration_cutoff_top1_share"] == "Float64"


def test_manifest_keeps_verified_reporting_identity_distinct_from_audit() -> None:
    identities = _manifest_git_identities(
        upstream_model_git_commit="a" * 40,
        reporting_git_commit="b" * 40,
        audit_git_commit="c" * 40,
    )
    assert identities == {
        "upstream_model_git_commit": "a" * 40,
        "reporting_git_commit": "b" * 40,
        "audit_git_commit": "c" * 40,
        "git_commit": "c" * 40,
    }


def test_audit_requires_one_status_for_every_confirmatory_cell(
    spec, run_context
) -> None:
    frames = complete_result_frames(spec, run_context)
    frames["lp_iv_status"] = frames["lp_iv_status"].filter(
        ~(
            (pl.col("outcome_id") == "green_export_complexity")
            & (pl.col("horizon") == 5)
            & (pl.col("analysis_family") == "confirmatory")
        )
    )
    report = audit_analysis_frames(frames, spec, run_context)
    assert report.status == "failed"
    assert report.missing_confirmatory_cells == (
        ("green_export_complexity", 5),
    )


def test_direction_matches_rejects_zero_and_unknown_sign() -> None:
    assert direction_matches(0.2, "positive")
    assert direction_matches(-0.2, "negative")
    assert not direction_matches(0.0, "positive")
    assert not direction_matches(0.0, "negative")
    try:
        direction_matches(1.0, "sideways")
    except ValueError as error:
        assert str(error) == "unknown expected sign: sideways"
    else:
        raise AssertionError("unknown expected signs must fail closed")


def test_ar_direction_requires_bounded_interval_wholly_beyond_zero() -> None:
    assert ar_directionally_supports(
        ar_status="bounded",
        conventional_point_accepted=True,
        target_term="gimc_gad_a",
        expected_sign="negative",
        beta_low=-1.0,
        beta_high=1.0,
        theta_low=-0.8,
        theta_high=-0.1,
    )
    assert not ar_directionally_supports(
        ar_status="bounded",
        conventional_point_accepted=True,
        target_term="gimc_gad_a",
        expected_sign="negative",
        beta_low=-1.0,
        beta_high=1.0,
        theta_low=-0.8,
        theta_high=0.1,
    )
    assert not ar_directionally_supports(
        ar_status="unbounded",
        conventional_point_accepted=True,
        target_term="gimc_gad_a",
        expected_sign="negative",
        beta_low=None,
        beta_high=None,
        theta_low=None,
        theta_high=None,
    )
    assert not ar_directionally_supports(
        ar_status="bounded",
        conventional_point_accepted=False,
        target_term="gimc_gad_a",
        expected_sign="negative",
        beta_low=-1.0,
        beta_high=1.0,
        theta_low=-0.8,
        theta_high=-0.1,
    )


def test_registered_cluster_rule_has_no_unregistered_fallback() -> None:
    assert registered_inference_status(rank=1, clusters=52) == "fail_rank_deficient"
    assert registered_inference_status(rank=2, clusters=18) == "exploratory_lt20_clusters"
    assert registered_inference_status(rank=2, clusters=20) == "wild_bootstrap_required"
    assert registered_inference_status(rank=2, clusters=29) == "wild_bootstrap_required"
    assert registered_inference_status(rank=2, clusters=30) == "cluster_robust"


def test_hypothesis_breadth_uses_only_admissible_registered_cells() -> None:
    rows = []
    for outcome in (
        "co2_tonnes_per_million_current_usd",
        "renewable_capacity_additions_mw_per_million",
    ):
        for horizon in (1, 2):
            rows.append(
                {
                    "outcome_id": outcome,
                    "horizon": horizon,
                    "expected_sign": "negative" if outcome.startswith("co2") else "positive",
                    "interval_low": -0.8 if outcome.startswith("co2") else 0.1,
                    "interval_high": -0.1 if outcome.startswith("co2") else 0.8,
                    "evidence_grade": "conditional_association",
                }
            )
    scorecard = pl.DataFrame(rows)
    labels = summarize_hypotheses(scorecard)
    assert labels["H1"] == "conditional_association: broadly_consistent"
    assert labels["H2"] == "mixed_or_inconclusive"
    assert labels["H3"] == "mixed_or_inconclusive"
    corroborating = pl.concat(
        [
            scorecard,
            pl.DataFrame(
                [
                    {
                        "outcome_id": "foreign_value_added_dependence",
                        "horizon": horizon,
                        "expected_sign": "positive",
                        "interval_low": 0.1,
                        "interval_high": 0.8,
                        "evidence_grade": "conditional_association",
                    }
                    for horizon in (3, 4, 5, 6)
                ]
            ),
        ],
        how="vertical",
    )
    assert summarize_hypotheses(corroborating)["H3_corroborating"] == (
        "conditional_association: corroborating"
    )


def test_authoritative_output_audit_covers_every_registered_claim() -> None:
    report = audit_analysis_outputs(
        code_root=ROOT,
        data_root=DATA_ROOT,
        output_root=ROOT / "06_结果/analysis",
    )
    assert report.status == "passed"
    assert report.continuous_cells == 84
    assert report.ar_cells == 45
    assert report.threshold_cells == 39
    assert report.scorecard.height == 39
    assert report.output_bytes <= 10 * 1024**3
    assert report.project_bytes < 150 * 1024**3


def test_output_audit_cli_requires_both_roots() -> None:
    args = build_parser().parse_args(
        [
            "analysis-output-audit",
            "--data-root",
            str(ROOT),
            "--output-root",
            str(ROOT / "06_结果/analysis"),
        ]
    )
    assert args.command == "analysis-output-audit"
    assert args.data_root == ROOT


def test_audit_rejects_unauthorized_root_before_revoking_manifest(
    project_fixture, tmp_path: Path
) -> None:
    unauthorized = tmp_path / "unauthorized-analysis"
    unauthorized.mkdir()
    manifest = unauthorized / "run_manifest.json"
    manifest.write_text("prior success\n", encoding="utf-8")

    try:
        publish_analysis_audit(
            code_root=project_fixture.code_root,
            data_root=project_fixture.data_root,
            output_root=unauthorized,
        )
    except ValueError as error:
        assert "frozen outputs.root" in str(error)
    else:
        raise AssertionError("unauthorized analysis root must fail closed")
    assert manifest.read_text(encoding="utf-8") == "prior success\n"


def test_audit_capacity_preflight_precedes_manifest_revocation(
    project_fixture, monkeypatch
) -> None:
    output = project_fixture.code_root / "06_结果/analysis"
    output.mkdir(parents=True)
    manifest = output / "run_manifest.json"
    manifest.write_text("prior success\n", encoding="utf-8")
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)
    monkeypatch.setattr(
        analysis_io, "directory_usage_bytes", lambda _path: 10 * 1024**3
    )

    with pytest.raises(RuntimeError, match="10 GB analysis output quota"):
        publish_analysis_audit(
            code_root=project_fixture.code_root,
            data_root=project_fixture.data_root,
            output_root=output,
        )
    assert manifest.read_text(encoding="utf-8") == "prior success\n"


def test_publication_audit_rejects_files_outside_fixed_set(tmp_path: Path) -> None:
    source = ROOT / "06_结果/analysis"
    shutil.copytree(source / "tables", tmp_path / "tables")
    shutil.copytree(source / "figures", tmp_path / "figures")
    source_hashes: set[str] = set()
    for path in (tmp_path / "figures").glob("*.provenance.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_hashes.update(payload["source_table_hashes"].values())
    assert len(_publication_hashes(tmp_path, source_hashes)) == 45

    (tmp_path / "tables/stale.csv").write_text("stale\n", encoding="utf-8")
    try:
        _publication_hashes(tmp_path, source_hashes)
    except ValueError as error:
        assert "unexpected publication files" in str(error)
    else:
        raise AssertionError("publication audit must reject stale files")


def _copied_publication(
    tmp_path: Path,
) -> tuple[set[str], str, str]:
    source = ROOT / "06_结果/analysis"
    shutil.copytree(source / "tables", tmp_path / "tables")
    shutil.copytree(source / "figures", tmp_path / "figures")
    source_hashes: set[str] = set()
    for path in (tmp_path / "figures").glob("*.provenance.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_hashes.update(payload["source_table_hashes"].values())
    context = load_stage_a_run_context(source)
    provenance = json.loads(
        (tmp_path / "figures/figure_1.provenance.json").read_text(
            encoding="utf-8"
        )
    )
    return source_hashes, context.git_commit, provenance["reporting_git_commit"]


def test_publication_identity_accepts_one_fully_bound_reporting_set(
    tmp_path: Path,
) -> None:
    source_hashes, upstream_commit, reporting_commit = _copied_publication(
        tmp_path
    )
    audited = _audit_publication_set(
        tmp_path,
        source_hashes,
        upstream_model_git_commit=upstream_commit,
        expected_reporting_git_commit=reporting_commit,
    )
    assert audited.reporting_git_commit == reporting_commit
    assert len(audited.hashes) == 45


def test_publication_audit_rejects_missing_submission_format(tmp_path: Path) -> None:
    source_hashes, _, _ = _copied_publication(tmp_path)
    (tmp_path / "figures/figure_4.svg").unlink()

    with pytest.raises(ValueError, match="unexpected publication files"):
        _publication_hashes(tmp_path, source_hashes)


def test_publication_audit_rejects_rendered_file_hash_mismatch(
    tmp_path: Path,
) -> None:
    source_hashes, _, _ = _copied_publication(tmp_path)
    png = tmp_path / "figures/figure_2.png"
    png.write_bytes(png.read_bytes() + b"stale")

    with pytest.raises(ValueError, match="rendered-file hash mismatch"):
        _publication_hashes(tmp_path, source_hashes)


def test_publication_identity_rejects_stale_reports_under_new_audit_head(
    tmp_path: Path,
) -> None:
    source_hashes, upstream_commit, _ = _copied_publication(tmp_path)
    with pytest.raises(ValueError, match="stale reporting Git commit"):
        _audit_publication_set(
            tmp_path,
            source_hashes,
            upstream_model_git_commit=upstream_commit,
            expected_reporting_git_commit="f" * 40,
        )


def test_publication_identity_rejects_mixed_reporting_commits(
    tmp_path: Path,
) -> None:
    source_hashes, upstream_commit, reporting_commit = _copied_publication(
        tmp_path
    )
    path = tmp_path / "figures/figure_7.provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reporting_git_commit"] = "e" * 40
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mixed reporting Git commits"):
        _audit_publication_set(
            tmp_path,
            source_hashes,
            upstream_model_git_commit=upstream_commit,
            expected_reporting_git_commit=reporting_commit,
        )


def test_scorecard_is_contract_checked_and_bound_to_audited_inputs(tmp_path) -> None:
    output_root = ROOT / "06_结果/analysis"
    report = audit_analysis_outputs(
        code_root=ROOT, data_root=DATA_ROOT, output_root=output_root
    )
    context = load_stage_a_run_context(output_root)
    path = write_evidence_scorecard(
        report=report,
        destination=tmp_path / "evidence_scorecard.parquet",
        source_root=output_root,
        context=context,
    )
    manifest = verify_manifest(path.with_name(f"{path.name}.manifest.json"))
    assert manifest.table_id == "evidence_scorecard"
    assert manifest.rows == 39
    assert set(manifest.input_hashes) == {value for _, value in report.audited_hashes}
