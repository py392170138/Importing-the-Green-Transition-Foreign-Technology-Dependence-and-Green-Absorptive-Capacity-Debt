import re
import subprocess
from pathlib import Path

import pytest

import green_debt.analysis_io as analysis_io
from green_debt.analysis_io import build_run_context, validate_analysis_authority
from green_debt.analysis_spec import load_analysis_spec


def _git(code_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=code_root,
        check=True,
        capture_output=True,
        text=True,
    )


def _commit_fixture(code_root: Path, *, include_lock: bool = True) -> None:
    _git(code_root, "init")
    _git(code_root, "config", "user.name", "Analysis Test")
    _git(code_root, "config", "user.email", "analysis-test@example.invalid")
    if include_lock:
        (code_root / "renv.lock").write_text(
            '{"R":{"Version":"4.6.1"}}\n', encoding="utf-8"
        )
    _git(code_root, "add", ".")
    _git(code_root, "commit", "-m", "fixture")


def _inputs(project_fixture, monkeypatch):
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)
    spec = load_analysis_spec(project_fixture.code_root / "config/analysis.yaml")
    authority = validate_analysis_authority(
        project_fixture.code_root, project_fixture.data_root
    )
    return spec, authority


def test_run_context_binds_clean_tracked_lock_and_is_logically_deterministic(
    project_fixture, monkeypatch
) -> None:
    _commit_fixture(project_fixture.code_root)
    spec, authority = _inputs(project_fixture, monkeypatch)

    first = build_run_context(spec, authority, project_fixture.code_root)
    second = build_run_context(spec, authority, project_fixture.code_root)

    assert first.run_id == second.run_id
    assert re.fullmatch(r"[0-9a-f]{16}", first.run_id)
    assert re.fullmatch(r"[0-9a-f]{40}", first.git_commit)
    assert re.fullmatch(r"[0-9a-f]{64}", first.input_authority_hash)
    assert re.fullmatch(r"[0-9a-f]{64}", first.renv_lock_sha256)
    assert first.spec_id == "gad_lp_iv_v1"
    assert first.seed == 20260820
    assert first.created_at_utc.endswith("Z")


def test_run_context_rejects_dirty_renv_lock(project_fixture, monkeypatch) -> None:
    _commit_fixture(project_fixture.code_root)
    spec, authority = _inputs(project_fixture, monkeypatch)
    (project_fixture.code_root / "renv.lock").write_text(
        '{"R":{"Version":"changed"}}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="renv.lock.*dirty"):
        build_run_context(spec, authority, project_fixture.code_root)


def test_run_context_rejects_missing_renv_lock(project_fixture, monkeypatch) -> None:
    _commit_fixture(project_fixture.code_root, include_lock=False)
    spec, authority = _inputs(project_fixture, monkeypatch)

    with pytest.raises(ValueError, match="renv.lock.*missing"):
        build_run_context(spec, authority, project_fixture.code_root)
