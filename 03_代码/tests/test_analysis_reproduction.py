from __future__ import annotations

import json
import hashlib
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import socket
import threading
import time
import signal

import polars as pl
import pytest
import green_debt.cli as cli
import green_debt.analysis_io as analysis_io

from green_debt.analysis_io import (
    AnalysisReproductionMismatch,
    RunContext,
    analysis_reproduction_audited_at,
    analysis_tree_snapshot,
    bind_analysis_reproduction_context,
    canonical_json_for_analysis_reproduction,
    cleanup_analysis_reproduction,
    compare_analysis_output_trees,
    compare_analysis_figure_sources,
    compare_logical_parquet,
    create_failed_analysis_reproduction_receipt,
    create_analysis_reproduction_receipt,
    create_pending_analysis_reproduction_receipt,
    mint_analysis_reproduction_capability,
    resolve_authorized_analysis_output,
    run_analysis_reproduction_check,
    validate_analysis_reproduction_capability,
)
from green_debt.storage import sha256_file
from green_debt.cli import build_parser, main


GIT_COMMIT = "a" * 40
RUN_ID = "b" * 16


class _CapturedReceiptWrite(RuntimeError):
    pass


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def test_logical_parquet_uses_keys_not_row_order_and_bounds_scaled_floats(
    tmp_path: Path,
) -> None:
    """Catch bytewise or row-order comparison of logically equal tables."""

    formal = tmp_path / "formal.parquet"
    reproduced = tmp_path / "reproduced.parquet"
    _write_parquet(
        formal,
        [
            {"id": 1, "label": "a", "value": 1.0, "special": math.nan},
            {"id": 2, "label": "b", "value": -2.0, "special": math.inf},
            {"id": 3, "label": "c", "value": None, "special": -math.inf},
        ],
    )
    _write_parquet(
        reproduced,
        [
            {"id": 3, "label": "c", "value": None, "special": -math.inf},
            {"id": 1, "label": "a", "value": 1.0 + 5e-13, "special": math.nan},
            {"id": 2, "label": "b", "value": -2.0 - 1e-12, "special": math.inf},
        ],
    )
    result = compare_logical_parquet(formal, reproduced, primary_key=("id",))
    assert result.matched is True
    assert result.rows == 3
    assert 0 < result.max_scaled_float_error <= 1e-12


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"id": 1, "label": "changed", "value": 1.0}, "non-float"),
        ({"id": 9, "label": "a", "value": 1.0}, "row keys"),
        ({"id": 1, "label": "a", "value": 1.0 + 2e-12}, "float tolerance"),
    ],
)
def test_logical_parquet_fails_closed_on_key_nonfloat_or_float_mismatch(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    """Catch a comparator that ignores a material cell or row-identity change."""

    formal = tmp_path / "formal.parquet"
    reproduced = tmp_path / "reproduced.parquet"
    _write_parquet(formal, [{"id": 1, "label": "a", "value": 1.0}])
    _write_parquet(reproduced, [mutation])
    with pytest.raises(AnalysisReproductionMismatch, match=message):
        compare_logical_parquet(formal, reproduced, primary_key=("id",))


def test_logical_parquet_rejects_schema_nan_null_and_infinity_mismatches(
    tmp_path: Path,
) -> None:
    """Catch coercion that equates null, NaN, finite values, or opposite infinities."""

    formal = tmp_path / "formal.parquet"
    reproduced = tmp_path / "reproduced.parquet"
    _write_parquet(formal, [{"id": 1, "x": math.nan, "y": math.inf}])
    _write_parquet(reproduced, [{"id": 1, "x": None, "y": -math.inf}])
    with pytest.raises(AnalysisReproductionMismatch, match="schema|float"):
        compare_logical_parquet(formal, reproduced, primary_key=("id",))
    _write_parquet(reproduced, [{"id": "1", "x": math.nan, "y": math.inf}])
    with pytest.raises(AnalysisReproductionMismatch, match="schema"):
        compare_logical_parquet(formal, reproduced, primary_key=("id",))


def test_canonical_json_excludes_only_created_at_utc_and_execution_id() -> None:
    """Catch an expanding ignore list that hides a changed scientific identity."""

    left = {
        "created_at_utc": "old",
        "execution_id": "one",
        "run_id": RUN_ID,
        "nested": {"created_at_utc": "old", "value": 3},
    }
    right = {
        "created_at_utc": "new",
        "execution_id": "two",
        "run_id": RUN_ID,
        "nested": {"created_at_utc": "new", "value": 3},
    }
    assert canonical_json_for_analysis_reproduction(left) == (
        canonical_json_for_analysis_reproduction(right)
    )
    for field in ("run_id", "audited_at_utc", "created_at"):
        changed = dict(right)
        changed[field] = "changed"
        assert canonical_json_for_analysis_reproduction(left) != (
            canonical_json_for_analysis_reproduction(changed)
        )
    with pytest.raises(ValueError, match="nonfinite"):
        canonical_json_for_analysis_reproduction({"value": math.nan})


def test_figure_comparison_uses_source_hashes_and_never_pdf_bytes(
    tmp_path: Path,
) -> None:
    """Catch treating nondeterministic PDF metadata as scientific content."""

    formal = tmp_path / "formal"
    reproduced = tmp_path / "reproduced"
    for root, pdf in ((formal, b"pdf-one"), (reproduced, b"pdf-two")):
        (root / "figures").mkdir(parents=True)
        (root / "figures/figure_1.pdf").write_bytes(pdf)
        (root / "figures/figure_1.provenance.json").write_text(
            json.dumps(
                {
                    "figure_id": "figure_1",
                    "plotted_source_data_hash": "c" * 64,
                    "source_table_hashes": {"model": "d" * 64},
                }
            ),
            encoding="utf-8",
        )
    result = compare_analysis_figure_sources(formal, reproduced)
    assert result == (("figure_1", "c" * 64),)
    payload = json.loads(
        (reproduced / "figures/figure_1.provenance.json").read_text()
    )
    payload["source_table_hashes"] = {"model": "e" * 64}
    (reproduced / "figures/figure_1.provenance.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    assert compare_analysis_figure_sources(formal, reproduced) == result
    payload["plotted_source_data_hash"] = "e" * 64
    (reproduced / "figures/figure_1.provenance.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(AnalysisReproductionMismatch, match="figure source"):
        compare_analysis_figure_sources(formal, reproduced)


def _mint(tmp_path: Path, execution_id: str = "20260829T010203Z-deadbeef"):
    data = tmp_path / "data"
    formal = tmp_path / "formal"
    data.mkdir()
    formal.mkdir()
    capability = mint_analysis_reproduction_capability(
        data_root=data,
        formal_output_root=formal,
        execution_id=execution_id,
        git_commit=GIT_COMMIT,
        formal_run_id=RUN_ID,
        formal_created_at_utc="2026-08-29T01:00:00Z",
        formal_audited_at_utc="2026-08-29T02:00:00Z",
    )
    return data, formal, capability


def _formal_manifest_for_receipt(formal: Path) -> Path:
    manifest = formal / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "success",
                "run_id": RUN_ID,
                "git_commit": GIT_COMMIT,
                "spec_id": "test-spec",
                "input_authority_hash": "1" * 64,
                "renv_lock_sha256": "2" * 64,
                "evidence_policy_sha256": "4" * 64,
                "threshold_registry_hash": "3" * 64,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _receipt_transition(
    tmp_path: Path, state: str
):
    data, formal, capability = _mint(tmp_path)
    manifest = _formal_manifest_for_receipt(formal)
    receipt = analysis_io._analysis_reproduction_journal_path(
        formal, capability.execution_id
    )
    if state != "pending":
        create_pending_analysis_reproduction_receipt(
            capability=capability, formal_manifest_path=manifest
        )

    if state == "pending":
        def transition() -> Path:
            return create_pending_analysis_reproduction_receipt(
                capability=capability, formal_manifest_path=manifest
            )
    elif state == "matched":
        def transition() -> Path:
            return create_analysis_reproduction_receipt(
                capability=capability,
                formal_manifest_path=manifest,
                data_snapshot_before="4" * 64,
                data_snapshot_after="4" * 64,
                manifest_snapshot_before="5" * 64,
                manifest_snapshot_after="5" * 64,
                compared_parquet=23,
                compared_json=1,
                compared_csv=8,
                compared_figures=7,
                max_scaled_float_error=0.0,
                threshold_registry_hash="3" * 64,
            )
    elif state == "failed":
        def transition() -> Path:
            return create_failed_analysis_reproduction_receipt(
                formal_output_root=formal,
                scratch_root=capability.scratch_root,
                failure="InjectedStageError: quota boundary",
            )
    else:
        raise AssertionError(f"unsupported test receipt state: {state}")
    return data, formal, capability, receipt, transition


def _marker_environment(capability) -> dict[str, str]:
    return {
        "GREEN_DEBT_ANALYSIS_REPRO_MARKER": str(capability.marker_path),
        "GREEN_DEBT_ANALYSIS_REPRO_SCRATCH": str(capability.scratch_root),
        "GREEN_DEBT_ANALYSIS_REPRO_EXECUTION_ID": capability.execution_id,
    }


def _test_stage_authority(capability, stage: str):
    authority = capability.open_stage_authority(stage)
    authority.set_stage_root_pid(os.getpid())
    return authority


def test_capability_is_bound_to_data_formal_scratch_execution_marker_and_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a reusable boolean or token that authorizes another output path."""

    data, formal, capability = _mint(tmp_path)
    monkeypatch.setattr(
        analysis_io, "_validate_orchestrator_peer", lambda _fd, _marker: os.getpid()
    )
    authority = _test_stage_authority(capability, "preflight")
    environment = authority.environment
    assert "CAPABILITY" not in environment
    marker = validate_analysis_reproduction_capability(
        data_root=data,
        formal_output_root=formal,
        scratch_root=capability.scratch_root,
        expected_git_commit=GIT_COMMIT,
        environment=environment,
    )
    assert marker["execution_id"] == capability.execution_id
    hostile = dict(environment)
    hostile["GREEN_DEBT_ANALYSIS_REPRO_BROKER_FD"] = "999999"
    with pytest.raises(ValueError, match="broker|capability"):
        validate_analysis_reproduction_capability(
            data_root=data,
            formal_output_root=formal,
            scratch_root=capability.scratch_root,
            expected_git_commit=GIT_COMMIT,
            environment=hostile,
        )
    with pytest.raises(ValueError, match="data root"):
        validate_analysis_reproduction_capability(
            data_root=tmp_path / "other-data",
            formal_output_root=formal,
            scratch_root=capability.scratch_root,
            expected_git_commit=GIT_COMMIT,
            environment=environment,
        )
    with pytest.raises(ValueError, match="Git commit"):
        validate_analysis_reproduction_capability(
            data_root=data,
            formal_output_root=formal,
            scratch_root=capability.scratch_root,
            expected_git_commit="f" * 40,
            environment=environment,
        )
    authority.close()


def test_minted_capability_has_no_persistent_stage_fd_or_secret(
    tmp_path: Path,
) -> None:
    """Catch one authority endpoint living across deps and all write stages."""

    _, _, capability = _mint(tmp_path)
    marker = json.loads(capability.marker_path.read_text())
    assert not hasattr(capability, "broker")
    assert not hasattr(capability, "environment")
    assert all("secret" not in key.lower() for key in marker)
    assert all("private" not in key.lower() for key in marker)


def test_stage_interrupt_kills_and_reaps_child_process_group(
    tmp_path: Path,
) -> None:
    """Catch returning control while a stage grandchild can still write scratch."""

    assert hasattr(analysis_io, "_run_analysis_stage_process")
    writer = tmp_path / "writer.py"
    counter = tmp_path / "counter.txt"
    grandchild_pid = tmp_path / "grandchild.pid"
    writer.write_text(
        "import os, subprocess, sys, time\n"
        "counter, pid_path = sys.argv[1:]\n"
        "program = \"import os,sys,time; p=sys.argv[1]; open(sys.argv[2],'w').write(str(os.getpid())); \" \\\n"
        "          \"[(open(p,'a').write('x'), time.sleep(.02)) for _ in iter(int,1)]\"\n"
        "subprocess.Popen([sys.executable, '-c', program, counter, pid_path])\n"
        "while not os.path.exists(pid_path): time.sleep(.01)\n"
        "while True: time.sleep(1)\n",
        encoding="utf-8",
    )
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt(_signum: int, _frame) -> None:
        raise analysis_io.AnalysisReproductionInterrupted("SIGTERM")

    signal.signal(signal.SIGTERM, interrupt)
    timer = threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        with pytest.raises(analysis_io.AnalysisReproductionInterrupted):
            analysis_io._run_analysis_stage_process(
                (sys.executable, str(writer), str(counter), str(grandchild_pid)),
                dict(os.environ),
                (),
                None,
            )
    finally:
        timer.cancel()
        signal.signal(signal.SIGTERM, previous)
    pid = int(grandchild_pid.read_text())
    size = counter.stat().st_size
    time.sleep(0.15)
    assert counter.stat().st_size == size
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize(
    "window",
    ("after_popen", "on_started_signal", "on_started_exception", "restore_handler"),
)
def test_spawn_registration_failures_kill_and_reap_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: str
) -> None:
    """Cover the spawn-to-registration race with a real writing grandchild."""

    writer = tmp_path / "spawn_writer.py"
    counter = tmp_path / "spawn-counter.txt"
    grandchild_pid = tmp_path / "spawn-grandchild.pid"
    writer.write_text(
        "import os, subprocess, sys, time\n"
        "counter, pid_path = sys.argv[1:]\n"
        "program = \"import os,sys,time; p=sys.argv[1]; open(sys.argv[2],'w').write(str(os.getpid())); \" \\\n"
        "          \"[(open(p,'a').write('x'), time.sleep(.02)) for _ in iter(int,1)]\"\n"
        "subprocess.Popen([sys.executable, '-c', program, counter, pid_path])\n"
        "while True: time.sleep(1)\n",
        encoding="utf-8",
    )
    real_popen = subprocess.Popen
    spawned: list[subprocess.Popen[bytes]] = []

    def spawn_then_signal(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        deadline = time.monotonic() + 2
        while not grandchild_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert grandchild_pid.exists()
        if window == "after_popen":
            os.kill(os.getpid(), signal.SIGTERM)
        return process

    monkeypatch.setattr(analysis_io.subprocess, "Popen", spawn_then_signal)
    previous = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def interrupt(_signum: int, _frame) -> None:
        raise analysis_io.AnalysisReproductionInterrupted("spawn-window SIGTERM")

    real_signal = signal.signal
    real_signal(signal.SIGTERM, interrupt)
    real_signal(signal.SIGINT, interrupt)
    restore_injected = False

    def signal_race_hook(stage: str) -> None:
        nonlocal restore_injected
        if (
            window == "restore_handler"
            and stage == "restore_before_unmask"
            and not restore_injected
        ):
            restore_injected = True
            raise analysis_io.AnalysisReproductionInterrupted(
                "spawn-window SIGTERM"
            )

    def on_started(_pid: int) -> None:
        if window == "on_started_signal":
            os.kill(os.getpid(), signal.SIGINT)
        elif window == "on_started_exception":
            raise RuntimeError("injected registration failure")

    expected_error = (
        RuntimeError
        if window == "on_started_exception"
        else analysis_io.AnalysisReproductionInterrupted
    )
    expected_match = "registration failure|spawn-window"
    try:
        with pytest.raises(expected_error, match=expected_match):
            analysis_io._run_analysis_stage_process(
                (sys.executable, str(writer), str(counter), str(grandchild_pid)),
                dict(os.environ),
                (),
                on_started,
                _signal_race_hook=signal_race_hook,
            )
        size = counter.stat().st_size
        time.sleep(0.15)
        assert counter.stat().st_size == size
        with pytest.raises(ProcessLookupError):
            os.kill(int(grandchild_pid.read_text()), 0)
        assert spawned[0].poll() is not None
    finally:
        real_signal(signal.SIGTERM, previous)
        real_signal(signal.SIGINT, previous_int)
        monkeypatch.undo()
        for process in spawned:
            if process.poll() is None:
                analysis_io._terminate_analysis_process_group(process)


def test_stage_popen_failure_has_no_process_group_to_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spawn failure restores handlers without inventing a PID to terminate."""

    terminated: list[object] = []

    def fail_popen(*_args, **_kwargs):
        raise OSError("injected spawn failure")

    monkeypatch.setattr(analysis_io.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(
        analysis_io,
        "_terminate_analysis_process_group",
        lambda process: terminated.append(process),
    )
    with pytest.raises(OSError, match="spawn failure"):
        analysis_io._run_analysis_stage_process(
            ("/does/not/exist",), dict(os.environ), (), None
        )
    assert terminated == []


def _current_signal_mask() -> set[signal.Signals]:
    return set(signal.pthread_sigmask(signal.SIG_BLOCK, set()))


def test_stage_signal_installation_is_grouped_and_restores_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a second signal escaping after only the first handler is installed."""

    class InstalledSignal(RuntimeError):
        pass

    real_signal = signal.signal
    original_int = signal.getsignal(signal.SIGINT)
    original_term = signal.getsignal(signal.SIGTERM)
    original_mask = _current_signal_mask()

    def old_int(_signum: int, _frame) -> None:
        return None

    def old_term(_signum: int, _frame) -> None:
        raise InstalledSignal("signal injected during grouped installation")

    real_signal(signal.SIGINT, old_int)
    real_signal(signal.SIGTERM, old_term)

    def inject_after_first_install(stage: str) -> None:
        if stage == "install_after_sigint":
            os.kill(os.getpid(), signal.SIGTERM)
    try:
        with pytest.raises(InstalledSignal, match="grouped installation"):
            analysis_io._run_analysis_stage_process(
                (sys.executable, "-c", "raise SystemExit(0)"),
                dict(os.environ),
                (),
                None,
                _signal_race_hook=inject_after_first_install,
            )
        assert signal.getsignal(signal.SIGINT) is old_int
        assert signal.getsignal(signal.SIGTERM) is old_term
        assert _current_signal_mask() == original_mask
    finally:
        monkeypatch.undo()
        signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)
        real_signal(signal.SIGINT, original_int)
        real_signal(signal.SIGTERM, original_term)


def test_stage_signal_restoration_is_grouped_before_pending_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch an old handler throwing after SIGINT restore but before SIGTERM restore."""

    class RestoredSignal(RuntimeError):
        pass

    writer = tmp_path / "restore-writer.py"
    counter = tmp_path / "restore-counter.txt"
    grandchild_pid = tmp_path / "restore-grandchild.pid"
    writer.write_text(
        "import os, subprocess, sys, time\n"
        "counter, pid_path = sys.argv[1:]\n"
        "program = \"import os,sys,time; p=sys.argv[1]; open(sys.argv[2],'w').write(str(os.getpid())); \" \\\n"
        "          \"[(open(p,'a').write('x'), time.sleep(.02)) for _ in iter(int,1)]\"\n"
        "subprocess.Popen([sys.executable, '-c', program, counter, pid_path])\n"
        "while not os.path.exists(pid_path): time.sleep(.01)\n"
        "while True: time.sleep(1)\n",
        encoding="utf-8",
    )
    real_signal = signal.signal
    original_int = signal.getsignal(signal.SIGINT)
    original_term = signal.getsignal(signal.SIGTERM)
    original_mask = _current_signal_mask()

    def old_int(_signum: int, _frame) -> None:
        raise RestoredSignal("signal injected during grouped restoration")

    def old_term(_signum: int, _frame) -> None:
        return None

    real_signal(signal.SIGINT, old_int)
    real_signal(signal.SIGTERM, old_term)
    restore_injected = False

    def inject_after_first_restore(stage: str) -> None:
        nonlocal restore_injected
        if stage == "restore_after_sigint" and not restore_injected:
            restore_injected = True
            os.kill(os.getpid(), signal.SIGINT)
            raise RestoredSignal("signal injected during grouped restoration")

    def await_grandchild(_pid: int) -> None:
        deadline = time.monotonic() + 2
        while not grandchild_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert grandchild_pid.exists()

    try:
        with pytest.raises(RestoredSignal, match="grouped restoration"):
            analysis_io._run_analysis_stage_process(
                (sys.executable, str(writer), str(counter), str(grandchild_pid)),
                dict(os.environ),
                (),
                await_grandchild,
                _signal_race_hook=inject_after_first_restore,
            )
        size = counter.stat().st_size
        time.sleep(0.15)
        assert counter.stat().st_size == size
        with pytest.raises(ProcessLookupError):
            os.kill(int(grandchild_pid.read_text()), 0)
        assert signal.getsignal(signal.SIGINT) is old_int
        assert signal.getsignal(signal.SIGTERM) is old_term
        assert _current_signal_mask() == original_mask
    finally:
        monkeypatch.undo()
        signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)
        real_signal(signal.SIGINT, original_int)
        real_signal(signal.SIGTERM, original_term)


def test_stage_pending_signal_at_install_unmask_prevents_spawn_and_restores_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch pending delivery at unmask being ignored before a PID exists."""

    original_int = signal.getsignal(signal.SIGINT)
    original_term = signal.getsignal(signal.SIGTERM)
    original_mask = _current_signal_mask()
    delivered: list[int] = []
    spawned = tmp_path / "spawned.txt"

    def old_term(signum: int, _frame) -> None:
        delivered.append(signum)

    signal.signal(signal.SIGTERM, old_term)

    def inject_at_first_unmask(stage: str) -> None:
        if stage == "install_before_unmask":
            os.kill(os.getpid(), signal.SIGTERM)
    try:
        with pytest.raises(
            analysis_io.AnalysisReproductionInterrupted, match="SIGTERM"
        ):
            analysis_io._run_analysis_stage_process(
                (
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(spawned)!r}).write_text('bad')",
                ),
                dict(os.environ),
                (),
                None,
                _signal_race_hook=inject_at_first_unmask,
            )
        assert delivered == [signal.SIGTERM]
        assert not spawned.exists()
        assert signal.getsignal(signal.SIGINT) == original_int
        assert signal.getsignal(signal.SIGTERM) is old_term
        assert _current_signal_mask() == original_mask
    finally:
        monkeypatch.undo()
        signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)
        signal.signal(signal.SIGINT, original_int)
        signal.signal(signal.SIGTERM, original_term)


@pytest.mark.parametrize("window", ("install_after_sigint", "restore_after_sigint"))
def test_reproduction_lock_signal_lifecycle_is_grouped(
    project_fixture,
    monkeypatch: pytest.MonkeyPatch,
    window: str,
) -> None:
    """The outer journal lock cannot retain one deferred handler after interruption."""

    class OuterSignal(RuntimeError):
        pass

    formal = project_fixture.code_root / "06_结果/analysis"
    formal.mkdir(parents=True)
    original_int = signal.getsignal(signal.SIGINT)
    original_term = signal.getsignal(signal.SIGTERM)
    original_mask = _current_signal_mask()
    injected = False

    def old_int(_signum: int, _frame) -> None:
        raise OuterSignal("outer SIGINT")

    def old_term(_signum: int, _frame) -> None:
        raise OuterSignal("outer SIGTERM")

    signal.signal(signal.SIGINT, old_int)
    signal.signal(signal.SIGTERM, old_term)
    monkeypatch.setattr(
        analysis_io,
        "_run_analysis_reproduction_check_locked",
        lambda **_kwargs: formal / "fake-receipt.json",
    )

    def inject(stage: str) -> None:
        nonlocal injected
        if stage == window and not injected:
            injected = True
            if window == "restore_after_sigint":
                raise OuterSignal("outer SIGINT")
            os.kill(os.getpid(), signal.SIGTERM)

    try:
        expected_error = (
            OuterSignal
            if window == "restore_after_sigint"
            else analysis_io.AnalysisReproductionInterrupted
        )
        with pytest.raises(expected_error):
            run_analysis_reproduction_check(
                code_root=project_fixture.code_root,
                data_root=project_fixture.data_root,
                formal_output_root=formal,
                expected_git_commit=GIT_COMMIT,
                _signal_race_hook=inject,
            )
        assert injected
        assert signal.getsignal(signal.SIGINT) is old_int
        assert signal.getsignal(signal.SIGTERM) is old_term
        assert _current_signal_mask() == original_mask
        lock_fd = os.open(
            formal / "_tmp/.analysis-reproduction.lock", os.O_RDWR
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)
        signal.signal(signal.SIGINT, original_int)
        signal.signal(signal.SIGTERM, original_term)


def test_stage_waits_for_short_lived_descendant_before_declaring_a_leak(
    tmp_path: Path,
) -> None:
    """Catch a normal dependency helper being killed during brief finalization."""

    program = tmp_path / "short_descendant.py"
    program.write_text(
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(0.2)'])\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    analysis_io._run_analysis_stage_process(
        (sys.executable, str(program)), dict(os.environ), (), None
    )
    assert time.monotonic() - started >= 0.15


def test_broker_rejects_wrong_stage_dead_channel_and_replayed_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch stage-free, persistent, or transcript-replay authorization."""

    monkeypatch.setattr(
        analysis_io, "_validate_orchestrator_peer", lambda _fd, _marker: os.getpid()
    )
    (tmp_path / "wrong").mkdir()
    data, formal, wrong_stage = _mint(tmp_path / "wrong")
    wrong_authority = _test_stage_authority(wrong_stage, "preflight")
    wrong_environment = wrong_authority.environment
    wrong_environment["GREEN_DEBT_ANALYSIS_REPRO_STAGE"] = "threshold"
    with pytest.raises(ValueError, match="broker|capability"):
        validate_analysis_reproduction_capability(
            data_root=data,
            formal_output_root=formal,
            scratch_root=wrong_stage.scratch_root,
            expected_git_commit=GIT_COMMIT,
            environment=wrong_environment,
        )
    wrong_authority.close()

    (tmp_path / "dead").mkdir()
    data, formal, dead = _mint(tmp_path / "dead")
    dead_authority = _test_stage_authority(dead, "preflight")
    dead_environment = dead_authority.environment
    dead_authority.close()
    with pytest.raises(ValueError, match="broker|capability"):
        validate_analysis_reproduction_capability(
            data_root=data,
            formal_output_root=formal,
            scratch_root=dead.scratch_root,
            expected_git_commit=GIT_COMMIT,
            environment=dead_environment,
        )

    (tmp_path / "replay").mkdir()
    data, formal, replay = _mint(tmp_path / "replay")
    replay_authority = _test_stage_authority(replay, "preflight")
    first = replay_authority.environment
    monkeypatch.setattr(analysis_io.secrets, "token_hex", lambda _size: "1" * 64)
    validate_analysis_reproduction_capability(
        data_root=data,
        formal_output_root=formal,
        scratch_root=replay.scratch_root,
        expected_git_commit=GIT_COMMIT,
        environment=first,
    )
    with pytest.raises(ValueError, match="broker|capability"):
        validate_analysis_reproduction_capability(
            data_root=data,
            formal_output_root=formal,
            scratch_root=replay.scratch_root,
            expected_git_commit=GIT_COMMIT,
            environment=first,
        )
    replay_authority.close()


def test_responsive_fake_broker_cannot_authorize_a_public_marker(
    tmp_path: Path,
) -> None:
    """Catch accepting any socket peer that can echo the public request fields."""

    data, formal, capability = _mint(tmp_path)
    server, client = socket.socketpair()
    secret_read_fd, secret_write_fd = os.pipe()
    os.write(secret_write_fd, b"x" * 32)
    os.close(secret_write_fd)
    environment = {
        **_marker_environment(capability),
        "GREEN_DEBT_ANALYSIS_REPRO_BROKER_FD": str(client.fileno()),
        "GREEN_DEBT_ANALYSIS_REPRO_SECRET_FD": str(secret_read_fd),
        "GREEN_DEBT_ANALYSIS_REPRO_STAGE": "preflight",
    }

    def echo_public_request() -> None:
        with server, server.makefile("rwb", buffering=0) as stream:
            line = stream.readline()
            if not line:
                return
            request = json.loads(line)
            stream.write(
                json.dumps(
                    {
                        "status": "authorized",
                        "nonce": request["nonce"],
                        "stage": request["stage"],
                        "execution_id": request["execution_id"],
                        "marker_instance_id": request["marker_instance_id"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                + b"\n"
            )

    responder = threading.Thread(target=echo_public_request)
    responder.start()
    environment["GREEN_DEBT_ANALYSIS_REPRO_BROKER_FD"] = str(client.fileno())
    with pytest.raises(ValueError, match="orchestrator|peer|broker"):
        validate_analysis_reproduction_capability(
            data_root=data,
            formal_output_root=formal,
            scratch_root=capability.scratch_root,
            expected_git_commit=GIT_COMMIT,
            environment=environment,
        )
    client.close()
    os.close(secret_read_fd)
    responder.join(timeout=2)


def test_broker_peer_requires_bound_pid_ancestry_and_exact_cli_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a responsive same-UID peer that is not the live CLI orchestrator."""

    server, client = socket.socketpair()
    try:
        data = tmp_path / "data"
        formal = tmp_path / "formal"
        data.mkdir()
        formal.mkdir()
        executable = os.path.realpath(sys.executable)
        arguments = (
            sys.executable,
            "-m",
            "green_debt.cli",
            "analysis-reproduce-check",
            "--data-root",
            str(data),
            "--output-root",
            str(formal),
        )
        started = "Sat Aug 29 01:02:03 2026"
        marker = {
            "orchestrator_pid": os.getpid(),
            "orchestrator_executable_realpath": executable,
            "orchestrator_argv_sha256": hashlib.sha256(
                analysis_io._canonical_json_bytes(list(arguments))
            ).hexdigest(),
            "orchestrator_start_sha256": hashlib.sha256(
                started.encode()
            ).hexdigest(),
            "data_root": str(data.resolve()),
            "formal_output_root": str(formal.resolve()),
        }
        monkeypatch.setattr(
            analysis_io, "_process_identity", lambda _pid: ("unused", started)
        )
        monkeypatch.setattr(
            analysis_io,
            "_darwin_process_arguments",
            lambda _pid: (executable, arguments),
        )
        monkeypatch.setattr(
            analysis_io, "_is_process_ancestor", lambda _ancestor, _child: True
        )
        assert analysis_io._validate_orchestrator_peer(client.fileno(), marker) == (
            os.getpid()
        )

        with pytest.raises(ValueError, match="PID mismatch"):
            analysis_io._validate_orchestrator_peer(
                client.fileno(), dict(marker, orchestrator_pid=os.getpid() + 1)
            )

        monkeypatch.setattr(
            analysis_io, "_is_process_ancestor", lambda _ancestor, _child: False
        )
        with pytest.raises(ValueError, match="not an ancestor"):
            analysis_io._validate_orchestrator_peer(client.fileno(), marker)

        wrong_arguments = list(arguments)
        wrong_arguments[3] = "config-check"
        wrong_arguments = tuple(wrong_arguments)
        wrong_identity = dict(
            marker,
            orchestrator_argv_sha256=hashlib.sha256(
                analysis_io._canonical_json_bytes(list(wrong_arguments))
            ).hexdigest(),
        )
        monkeypatch.setattr(
            analysis_io, "_is_process_ancestor", lambda _ancestor, _child: True
        )
        monkeypatch.setattr(
            analysis_io,
            "_darwin_process_arguments",
            lambda _pid: (executable, wrong_arguments),
        )
        with pytest.raises(ValueError, match="not analysis-reproduce-check"):
            analysis_io._validate_orchestrator_peer(
                client.fileno(), wrong_identity
            )
    finally:
        client.close()
        server.close()


def test_broker_peer_rejects_script_prefix_with_embedded_cli_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch accepting a fake script because its trailing argv contains CLI tokens."""

    server, client = socket.socketpair()
    try:
        data = tmp_path / "data"
        formal = tmp_path / "formal"
        data.mkdir()
        formal.mkdir()
        executable = os.path.realpath(sys.executable)
        arguments = (
            sys.executable,
            "fake_orchestrator.py",
            "-m",
            "green_debt.cli",
            "analysis-reproduce-check",
            "--data-root",
            str(data),
            "--output-root",
            str(formal),
        )
        started = "Sat Aug 29 01:02:03 2026"
        marker = {
            "orchestrator_pid": os.getpid(),
            "orchestrator_executable_realpath": executable,
            "orchestrator_argv_sha256": hashlib.sha256(
                analysis_io._canonical_json_bytes(list(arguments))
            ).hexdigest(),
            "orchestrator_start_sha256": hashlib.sha256(
                started.encode()
            ).hexdigest(),
            "data_root": str(data.resolve()),
            "formal_output_root": str(formal.resolve()),
        }
        monkeypatch.setattr(
            analysis_io, "_process_identity", lambda _pid: ("unused", started)
        )
        monkeypatch.setattr(
            analysis_io,
            "_darwin_process_arguments",
            lambda _pid: (executable, arguments),
        )
        monkeypatch.setattr(
            analysis_io, "_is_process_ancestor", lambda _ancestor, _child: True
        )
        with pytest.raises(ValueError, match="analysis-reproduce-check|argv|identity"):
            analysis_io._validate_orchestrator_peer(client.fileno(), marker)
    finally:
        client.close()
        server.close()


@pytest.mark.parametrize(
    "case",
    (
        "argument_value",
        "shell_c",
        "other_module",
        "extra_token",
        "wrong_root",
        "wrong_python",
    ),
)
def test_kernel_argv_validation_rejects_noncanonical_cli_forms(
    tmp_path: Path, case: str
) -> None:
    """Kernel argv boundaries, executable identity, and root binding are exact."""

    data = tmp_path / "data"
    formal = tmp_path / "formal"
    data.mkdir()
    formal.mkdir()
    executable = os.path.realpath(sys.executable)
    arguments = [
        sys.executable,
        "-m",
        "green_debt.cli",
        "analysis-reproduce-check",
        "--data-root",
        str(data),
        "--output-root",
        str(formal),
    ]
    observed_executable = executable
    if case == "argument_value":
        arguments = [
            sys.executable,
            "fake.py",
            "-m green_debt.cli analysis-reproduce-check",
        ]
    elif case == "shell_c":
        arguments = [
            sys.executable,
            "-c",
            "-m green_debt.cli analysis-reproduce-check",
        ]
    elif case == "other_module":
        arguments[2] = "other.cli"
    elif case == "extra_token":
        arguments.append("--forged")
    elif case == "wrong_root":
        arguments[5] = str(tmp_path)
    else:
        observed_executable = str(tmp_path / "fake-python")
    marker = {
        "orchestrator_executable_realpath": executable,
        "data_root": str(data.resolve()),
        "formal_output_root": str(formal.resolve()),
    }
    with pytest.raises(ValueError, match="executable|argv|extra|binding"):
        analysis_io._validate_reproduction_cli_arguments(
            observed_executable, tuple(arguments), marker
        )


def test_broker_death_invalidates_even_an_already_authorized_same_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a process-global cache bypassing the required live authorization."""

    data, formal, capability = _mint(tmp_path)
    monkeypatch.setattr(
        analysis_io, "_validate_orchestrator_peer", lambda _fd, _marker: os.getpid()
    )
    authority = _test_stage_authority(capability, "preflight")
    environment = authority.environment
    validate_analysis_reproduction_capability(
        data_root=data,
        formal_output_root=formal,
        scratch_root=capability.scratch_root,
        expected_git_commit=GIT_COMMIT,
        environment=environment,
    )
    validate_analysis_reproduction_capability(
        data_root=data,
        formal_output_root=formal,
        scratch_root=capability.scratch_root,
        expected_git_commit=GIT_COMMIT,
        environment=environment,
    )
    authority.close()
    with pytest.raises(ValueError, match="broker|orchestrator"):
        validate_analysis_reproduction_capability(
            data_root=data,
            formal_output_root=formal,
            scratch_root=capability.scratch_root,
            expected_git_commit=GIT_COMMIT,
            environment=environment,
        )


def test_capability_rejects_traversal_symlink_sibling_and_unmarked_scratch(
    tmp_path: Path,
) -> None:
    """Catch widening the one exact non-symlink scratch child authorization."""

    data, formal, capability = _mint(tmp_path)
    sibling = formal / "_tmp/reproduce.sibling"
    sibling.mkdir()
    for target in (
        formal / "_tmp/../outside",
        sibling,
        formal / "_tmp",
        formal,
    ):
        with pytest.raises(ValueError):
            validate_analysis_reproduction_capability(
                data_root=data,
                formal_output_root=formal,
                scratch_root=target,
                expected_git_commit=GIT_COMMIT,
                environment=_marker_environment(capability),
            )
    link = formal / "_tmp/reproduce.link"
    link.symlink_to(capability.scratch_root, target_is_directory=True)
    hostile = _marker_environment(capability)
    hostile["GREEN_DEBT_ANALYSIS_REPRO_SCRATCH"] = str(link)
    with pytest.raises(ValueError, match="symbolic link|scratch"):
        validate_analysis_reproduction_capability(
            data_root=data,
            formal_output_root=formal,
            scratch_root=link,
            expected_git_commit=GIT_COMMIT,
            environment=hostile,
        )


def test_authorized_output_keeps_formal_root_strict_and_allows_only_live_capability(
    project_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch a global scratch exception leaking into ordinary production commands."""

    formal = project_fixture.code_root / "06_结果/analysis"
    formal.mkdir(parents=True)
    capability = mint_analysis_reproduction_capability(
        data_root=project_fixture.data_root,
        formal_output_root=formal,
        execution_id="20260829T010203Z-cafebabe",
        git_commit=GIT_COMMIT,
        formal_run_id=RUN_ID,
        formal_created_at_utc="2026-08-29T01:00:00Z",
        formal_audited_at_utc="2026-08-29T02:00:00Z",
    )
    monkeypatch.setattr(
        analysis_io, "_validate_orchestrator_peer", lambda _fd, _marker: os.getpid()
    )
    authority = _test_stage_authority(capability, "preflight")
    assert resolve_authorized_analysis_output(
        project_fixture.code_root,
        project_fixture.data_root,
        formal,
        expected_git_commit=GIT_COMMIT,
        environment={},
    ) == formal.resolve()
    with pytest.raises(ValueError, match="frozen outputs.root"):
        resolve_authorized_analysis_output(
            project_fixture.code_root,
            project_fixture.data_root,
            capability.scratch_root,
            expected_git_commit=GIT_COMMIT,
            environment={},
        )
    assert resolve_authorized_analysis_output(
        project_fixture.code_root,
        project_fixture.data_root,
        capability.scratch_root,
        expected_git_commit=GIT_COMMIT,
        environment=authority.environment,
    ) == capability.scratch_root
    authority.close()


def test_complete_static_marker_and_token_forgery_cannot_authorize_scratch(
    project_fixture,
) -> None:
    """Catch a public self-authentication algorithm acting as write authority."""

    formal = project_fixture.code_root / "06_结果/analysis"
    scratch = formal / "_tmp/reproduce.20260829T010203Z-deadbeef"
    scratch.mkdir(parents=True)
    token = "9" * 64
    observed = scratch.stat()
    core = {
        "schema_version": 2,
        "execution_id": "20260829T010203Z-deadbeef",
        "instance_id": "8" * 32,
        "data_root": str(project_fixture.data_root.resolve()),
        "formal_output_root": str(formal.resolve()),
        "scratch_root": str(scratch.resolve()),
        "git_commit": GIT_COMMIT,
        "formal_run_id": RUN_ID,
        "formal_created_at_utc": "2026-08-29T01:00:00Z",
        "formal_audited_at_utc": "2026-08-29T02:00:00Z",
        "scratch_device": observed.st_dev,
        "scratch_inode": observed.st_ino,
        "created_at_utc": "2026-08-29T00:00:00Z",
    }
    marker = scratch / ".analysis-reproduction-capability.json"
    marker.write_text(json.dumps(core), encoding="utf-8")
    forged = {
        "GREEN_DEBT_ANALYSIS_REPRO_BROKER_FD": "999999",
        "GREEN_DEBT_ANALYSIS_REPRO_STAGE": "preflight",
        "GREEN_DEBT_ANALYSIS_REPRO_CAPABILITY": token,
        "GREEN_DEBT_ANALYSIS_REPRO_MARKER": str(marker),
        "GREEN_DEBT_ANALYSIS_REPRO_SCRATCH": str(scratch.resolve()),
        "GREEN_DEBT_ANALYSIS_REPRO_EXECUTION_ID": core["execution_id"],
    }

    with pytest.raises(ValueError, match="capability"):
        resolve_authorized_analysis_output(
            project_fixture.code_root,
            project_fixture.data_root,
            scratch,
            expected_git_commit=GIT_COMMIT,
            environment=forged,
        )


def test_reproduction_reuses_formal_context_time_and_audit_time(
    project_fixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch execution timestamps contaminating logical table and manifest equality."""

    formal = project_fixture.code_root / "06_结果/analysis"
    formal.mkdir(parents=True)
    capability = mint_analysis_reproduction_capability(
        data_root=project_fixture.data_root,
        formal_output_root=formal,
        execution_id="20260829T010203Z-facefeed",
        git_commit=GIT_COMMIT,
        formal_run_id=RUN_ID,
        formal_created_at_utc="2026-08-29T01:00:00Z",
        formal_audited_at_utc="2026-08-29T02:00:00Z",
    )
    monkeypatch.setattr(
        analysis_io, "_validate_orchestrator_peer", lambda _fd, _marker: os.getpid()
    )
    context = RunContext(
        run_id=RUN_ID,
        spec_id="spec",
        input_authority_hash="1" * 64,
        git_commit=GIT_COMMIT,
        renv_lock_sha256="2" * 64,
        evidence_policy_sha256="3" * 64,
        seed=7,
        created_at_utc="2026-08-29T03:00:00Z",
    )
    first_authority = _test_stage_authority(capability, "preflight")
    rebound = bind_analysis_reproduction_context(
        code_root=project_fixture.code_root,
        data_root=project_fixture.data_root,
        output_root=capability.scratch_root,
        context=context,
        environment=first_authority.environment,
    )
    first_authority.close()
    assert rebound.created_at_utc == "2026-08-29T01:00:00Z"
    assert rebound.run_id == context.run_id
    second_authority = _test_stage_authority(capability, "preflight")
    assert analysis_reproduction_audited_at(
        code_root=project_fixture.code_root,
        data_root=project_fixture.data_root,
        output_root=capability.scratch_root,
        default="2026-08-29T04:00:00Z",
        expected_git_commit=GIT_COMMIT,
        environment=second_authority.environment,
    ) == "2026-08-29T02:00:00Z"
    second_authority.close()


def test_tree_snapshot_binds_paths_bytes_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    """Catch a read-only proof that watches names but not file content."""

    root = tmp_path / "05_中间数据"
    (root / "analysis").mkdir(parents=True)
    authority = root / "analysis/table.parquet"
    authority.write_bytes(b"one")
    first = analysis_tree_snapshot(root)
    authority.write_bytes(b"two")
    second = analysis_tree_snapshot(root)
    assert first.sha256 != second.sha256
    assert first.files == second.files == 1
    link = root / "analysis/link"
    link.symlink_to(authority)
    with pytest.raises(ValueError, match="symbolic link"):
        analysis_tree_snapshot(root)


def test_output_tree_comparison_covers_tables_json_figures_and_threshold(
    tmp_path: Path,
) -> None:
    """Catch a reproduction receipt that compares only a subset of outputs."""

    formal = tmp_path / "formal"
    reproduced = tmp_path / "reproduced"
    for root in (formal, reproduced):
        _write_parquet(
            root / "models/model.parquet",
            [{"id": 1, "label": "x", "estimate": 2.0}],
        )
        (root / "models/model.parquet.manifest.json").write_text(
            json.dumps({"primary_key": ["id"]}), encoding="utf-8"
        )
        (root / "tables").mkdir(parents=True)
        (root / "tables/table_1.csv").write_text(
            "id,label,value\n1,x,2.0\n", encoding="utf-8"
        )
        (root / "registries").mkdir(parents=True)
        (root / "registries/threshold_registry_v1.json").write_text(
            json.dumps(
                {
                    "registry_hash": "3" * 64,
                    "created_at_utc": "formal" if root == formal else "scratch",
                    "q": 0.5,
                }
            ),
            encoding="utf-8",
        )
        (root / "figures").mkdir(parents=True)
        (root / "figures/figure_1.pdf").write_bytes(
            b"formal-pdf" if root == formal else b"scratch-pdf"
        )
        (root / "figures/figure_1.provenance.json").write_text(
            json.dumps(
                {
                    "figure_id": "figure_1",
                    "plotted_source_data_hash": "4" * 64,
                    "source_table_hashes": {"model": "5" * 64},
                }
            ),
            encoding="utf-8",
        )
    comparison = compare_analysis_output_trees(formal, reproduced)
    assert comparison.compared_parquet == 1
    assert comparison.compared_csv == 1
    assert comparison.compared_json == 1
    assert comparison.compared_figures == 1
    assert comparison.threshold_registry_hash == "3" * 64
    assert comparison.max_scaled_float_error == 0
    (reproduced / "tables/table_1.csv").write_text(
        "id,label,value\n1,x,2.1\n", encoding="utf-8"
    )
    with pytest.raises(AnalysisReproductionMismatch, match="float tolerance"):
        compare_analysis_output_trees(formal, reproduced)


def _receipt(tmp_path: Path):
    data, formal, capability = _mint(tmp_path)
    (capability.scratch_root / "nested").mkdir()
    (capability.scratch_root / "nested/result.txt").write_text("result")
    formal_manifest = formal / "run_manifest.json"
    formal_manifest.write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "git_commit": GIT_COMMIT,
                "input_authority_hash": "1" * 64,
                "renv_lock_sha256": "2" * 64,
                "threshold_registry_hash": "3" * 64,
                "status": "success",
            }
        ),
        encoding="utf-8",
    )
    receipt = create_analysis_reproduction_receipt(
        capability=capability,
        formal_manifest_path=formal_manifest,
        data_snapshot_before="4" * 64,
        data_snapshot_after="4" * 64,
        manifest_snapshot_before="5" * 64,
        manifest_snapshot_after="5" * 64,
        compared_parquet=2,
        compared_json=3,
        compared_csv=4,
        compared_figures=7,
        max_scaled_float_error=5e-13,
        threshold_registry_hash="3" * 64,
    )
    return data, formal, capability, receipt


def test_cleanup_removes_only_receipt_bound_scratch_and_audits_cleaned_state(
    tmp_path: Path,
) -> None:
    """Catch broad cleanup or deletion without an auditable terminal receipt."""

    _, formal, capability, receipt = _receipt(tmp_path)
    sibling = formal / "_tmp/reproduce.keep"
    sibling.mkdir()
    removed = cleanup_analysis_reproduction(receipt)
    assert removed == capability.scratch_root
    assert not capability.scratch_root.exists()
    assert sibling.is_dir()
    payload = json.loads(receipt.read_text())
    assert payload["comparison_scope"]["figures"] == (
        "plotted_source_hash_and_source_labels; "
        "underlying_source_tables_logically_compared_separately"
    )
    assert payload["cleanup_status"] == "cleaned"
    assert payload["status"] == "cleaned"
    assert payload["pre_cleanup_status"] == "matched"
    assert payload["cleaned_at_utc"].endswith("Z")
    assert payload["receipt_sha256"] != sha256_file(receipt)


def test_cleanup_rejects_forged_wrong_sibling_broad_and_symlink_receipts(
    tmp_path: Path,
) -> None:
    """Catch receipt edits that retarget cleanup outside the minted directory."""

    _, formal, capability, receipt = _receipt(tmp_path)
    original = json.loads(receipt.read_text())
    targets = (
        str(formal),
        str(formal / "_tmp"),
        str(formal / "_tmp/reproduce.sibling"),
        str(formal / "_tmp/../outside"),
    )
    for index, target in enumerate(targets):
        payload = dict(original)
        payload["scratch_root"] = target
        hostile = tmp_path / f"hostile-{index}.json"
        hostile.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            cleanup_analysis_reproduction(hostile)
    link = formal / "_tmp/reproduce.cleanup-link"
    link.symlink_to(capability.scratch_root, target_is_directory=True)
    payload = dict(original)
    payload["scratch_root"] = str(link)
    hostile = tmp_path / "hostile-link.json"
    hostile.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        cleanup_analysis_reproduction(hostile)
    assert capability.scratch_root.is_dir()


def test_old_receipt_cannot_delete_new_scratch_reusing_the_same_name(
    tmp_path: Path,
) -> None:
    """Catch ABA path reuse where an old receipt deletes a later execution."""

    _, _, old_capability, receipt = _receipt(tmp_path)
    marker = old_capability.marker_path
    marker.unlink()
    (old_capability.scratch_root / "nested/result.txt").unlink()
    (old_capability.scratch_root / "nested").rmdir()
    old_capability.scratch_root.rmdir()
    old_capability.scratch_root.mkdir()
    marker.write_text(json.dumps({"instance_id": "new"}), encoding="utf-8")
    with pytest.raises(ValueError, match="marker|identity"):
        cleanup_analysis_reproduction(receipt)
    assert old_capability.scratch_root.is_dir()


def test_cleanup_rejects_receipt_after_formal_run_manifest_changes(
    tmp_path: Path,
) -> None:
    """Catch an old receipt cleaning scratch after a newer formal run replaces it."""

    _, formal, capability, receipt = _receipt(tmp_path)
    manifest = formal / "run_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["run_id"] = "c" * 16
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="formal run manifest"):
        cleanup_analysis_reproduction(receipt)
    assert capability.scratch_root.is_dir()


def test_cleanup_safely_unlinks_inventoried_fifo_without_opening_it(
    tmp_path: Path,
) -> None:
    """FIFO entries are inventoried and unlinked without blocking or following."""

    _, _, capability, receipt = _receipt(tmp_path)
    fifo = capability.scratch_root / "unsafe.fifo"
    os.mkfifo(fifo)
    cleanup_analysis_reproduction(receipt)
    assert not capability.scratch_root.exists()
    assert json.loads(receipt.read_text())["status"] == "cleaned"


def test_cleanup_refuses_active_reproduction_lock_before_quarantine(
    tmp_path: Path,
) -> None:
    """Catch cleanup racing an active writer that still owns the journal lock."""

    _, formal, capability, receipt = _receipt(tmp_path)
    lock_path = formal / "_tmp/.analysis-reproduction.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active|lock"):
            cleanup_analysis_reproduction(receipt)
        assert capability.scratch_root.is_dir()
        assert capability.marker_path.is_file()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    cleanup_analysis_reproduction(receipt)
    assert not capability.scratch_root.exists()


def test_cleanup_final_receipt_write_failure_recovers_from_cleaning_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch deleted scratch plus an old receipt permanently blocking future work."""

    _, _, capability, receipt = _receipt(tmp_path)
    original_writer = analysis_io._atomic_write_json_object
    failed_once = False

    def fail_first_cleaned_write(path: Path, payload: dict[str, object]) -> None:
        nonlocal failed_once
        if payload.get("status") == "cleaned" and not failed_once:
            failed_once = True
            raise OSError("injected final receipt write failure")
        original_writer(path, payload)

    monkeypatch.setattr(
        analysis_io, "_atomic_write_json_object", fail_first_cleaned_write
    )
    with pytest.raises(OSError, match="final receipt"):
        cleanup_analysis_reproduction(receipt)
    interrupted = json.loads(receipt.read_text())
    assert interrupted["status"] == "cleaning"
    assert interrupted["cleanup_status"] == "cleaning"
    assert interrupted["pre_cleanup_status"] == "matched"

    monkeypatch.setattr(analysis_io, "_atomic_write_json_object", original_writer)
    assert cleanup_analysis_reproduction(receipt) == capability.scratch_root
    recovered = json.loads(receipt.read_text())
    assert recovered["status"] == "cleaned"
    assert recovered["cleanup_status"] == "cleaned"
    assert not capability.scratch_root.exists()


def test_cleanup_resumes_exact_persisted_quarantine_after_interruption(
    tmp_path: Path,
) -> None:
    """Catch a crash after rename stranding an unbound quarantine forever."""

    _data, formal, capability, receipt = _receipt(tmp_path)

    def interrupt_after_rename(
        stage: str, _scratch: Path, _quarantine: Path | None
    ) -> None:
        if stage == "rename_to_walk":
            raise KeyboardInterrupt("injected cleanup interruption")

    with pytest.raises(KeyboardInterrupt, match="cleanup interruption"):
        cleanup_analysis_reproduction(receipt, _race_hook=interrupt_after_rename)
    interrupted = json.loads(receipt.read_text())
    assert interrupted["status"] == "cleaning"
    assert not capability.scratch_root.exists()
    quarantine = formal / "_tmp" / interrupted["cleaning_quarantine_name"]
    assert quarantine.is_dir()

    assert cleanup_analysis_reproduction(receipt) == capability.scratch_root
    assert not quarantine.exists()
    cleaned = json.loads(receipt.read_text())
    assert cleaned["status"] == "cleaned"
    assert cleaned["pre_cleanup_status"] == "matched"


def test_cleanup_resumes_after_marker_and_partial_tree_were_deleted(
    tmp_path: Path,
) -> None:
    """A cleaning receipt remains authoritative after real recursive progress."""

    _data, formal, capability, receipt = _receipt(tmp_path)
    result = capability.scratch_root / "nested/result.txt"

    def interrupt_during_walk(
        stage: str, _scratch: Path, quarantine: Path | None
    ) -> None:
        if (
            stage == "walk_progress"
            and not capability.marker_path.exists()
            and quarantine is not None
            and not (quarantine / "nested/result.txt").exists()
        ):
            raise KeyboardInterrupt("injected partial walk interruption")

    with pytest.raises(KeyboardInterrupt, match="partial walk"):
        cleanup_analysis_reproduction(receipt, _race_hook=interrupt_during_walk)
    interrupted = json.loads(receipt.read_text())
    quarantine = formal / "_tmp" / interrupted["cleaning_quarantine_name"]
    assert interrupted["status"] == "cleaning"
    assert interrupted["cleaning_inventory"]
    assert quarantine.is_dir()
    assert not (quarantine / analysis_io._REPRODUCTION_MARKER_NAME).exists()

    assert cleanup_analysis_reproduction(receipt) == capability.scratch_root
    assert not quarantine.exists()
    assert json.loads(receipt.read_text())["status"] == "cleaned"


def test_failed_cleanup_unlinks_bad_internal_entries_without_following(
    tmp_path: Path,
) -> None:
    """Failed scratch cleanup treats internal data as untrusted directory entries."""

    _data, formal, capability = _mint(tmp_path)
    receipt = create_failed_analysis_reproduction_receipt(
        formal_output_root=formal,
        scratch_root=capability.scratch_root,
        failure="injected failed stage",
    )
    outside_file = tmp_path / "outside-file.txt"
    outside_file.write_text("survive", encoding="utf-8")
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    outside_nested = outside_dir / "survive.txt"
    outside_nested.write_text("survive", encoding="utf-8")
    capability.marker_path.unlink()
    capability.marker_path.symlink_to(outside_file)
    (capability.scratch_root / "nested-link").symlink_to(
        outside_dir, target_is_directory=True
    )
    unreadable = capability.scratch_root / "unreadable.bin"
    unreadable.write_bytes(b"opaque")
    unreadable.chmod(0)
    fifo = capability.scratch_root / "failed.fifo"
    os.mkfifo(fifo)
    unix_socket_path = capability.scratch_root / "failed.socket"
    unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    prior_cwd = Path.cwd()
    try:
        os.chdir(capability.scratch_root)
        unix_socket.bind(unix_socket_path.name)
    finally:
        os.chdir(prior_cwd)
    unix_socket.close()
    locked_directory = capability.scratch_root / "locked"
    locked_directory.mkdir()
    (locked_directory / "opaque.bin").write_bytes(b"opaque")
    locked_directory.chmod(0)

    cleanup_analysis_reproduction(receipt)

    assert json.loads(receipt.read_text())["status"] == "cleaned"
    assert not capability.scratch_root.exists()
    assert outside_file.read_text(encoding="utf-8") == "survive"
    assert outside_nested.read_text(encoding="utf-8") == "survive"


def test_failed_transition_uses_pending_authority_after_marker_corruption(
    tmp_path: Path,
) -> None:
    """A stage-corrupted marker cannot suppress failed journaling or cleanup."""

    _data, formal, capability = _mint(tmp_path)
    formal_manifest = formal / "run_manifest.json"
    formal_manifest.write_text(
        json.dumps(
            {
                "status": "success",
                "run_id": RUN_ID,
                "git_commit": GIT_COMMIT,
                "spec_id": "spec",
                "input_authority_hash": "1" * 64,
                "renv_lock_sha256": "2" * 64,
                "threshold_registry_hash": "3" * 64,
            }
        ),
        encoding="utf-8",
    )
    pending = create_pending_analysis_reproduction_receipt(
        capability=capability, formal_manifest_path=formal_manifest
    )
    outside = tmp_path / "outside-marker.txt"
    outside.write_text("survive", encoding="utf-8")
    capability.marker_path.unlink()
    capability.marker_path.symlink_to(outside)

    failed = create_failed_analysis_reproduction_receipt(
        formal_output_root=formal,
        scratch_root=capability.scratch_root,
        failure="OriginalStageError: marker damaged",
    )

    payload = json.loads(failed.read_text())
    assert failed == pending
    assert payload["status"] == "failed"
    assert payload["failure"] == "OriginalStageError: marker damaged"
    cleanup_analysis_reproduction(failed)
    assert outside.read_text(encoding="utf-8") == "survive"
    assert json.loads(failed.read_text())["status"] == "cleaned"


@pytest.mark.parametrize("mutation", ("addition", "replacement", "symlink"))
def test_partial_cleanup_rejects_entries_not_in_persisted_inventory(
    tmp_path: Path, mutation: str
) -> None:
    """Partial retry rejects additions and inode/type substitutions."""

    _data, formal, capability, receipt = _receipt(tmp_path)
    result = capability.scratch_root / "nested/result.txt"

    def interrupt_during_walk(
        stage: str, _scratch: Path, quarantine: Path | None
    ) -> None:
        if (
            stage == "walk_progress"
            and not capability.marker_path.exists()
            and quarantine is not None
            and not (quarantine / "nested/result.txt").exists()
        ):
            raise KeyboardInterrupt("partial")

    with pytest.raises(KeyboardInterrupt, match="partial"):
        cleanup_analysis_reproduction(receipt, _race_hook=interrupt_during_walk)
    payload = json.loads(receipt.read_text())
    quarantine = formal / "_tmp" / payload["cleaning_quarantine_name"]
    remaining_directory = quarantine / "nested"
    outside = tmp_path / "outside.txt"
    outside.write_text("survive", encoding="utf-8")
    if mutation == "addition":
        hostile = quarantine / "new.txt"
        hostile.write_text("new", encoding="utf-8")
    elif mutation == "replacement":
        remaining_directory.rmdir()
        hostile = remaining_directory
        hostile.mkdir()
    else:
        remaining_directory.rmdir()
        hostile = remaining_directory
        hostile.symlink_to(outside)

    with pytest.raises(ValueError, match="inventory|changed|symbolic"):
        cleanup_analysis_reproduction(receipt)
    assert os.path.lexists(hostile)
    assert outside.read_text(encoding="utf-8") == "survive"
    if mutation == "addition":
        hostile.unlink()
        cleanup_analysis_reproduction(receipt)
        assert not quarantine.exists()


@pytest.mark.parametrize(
    "race_stage",
    ("validation_to_rename", "rename_to_walk", "walk_to_rmdir"),
)
def test_cleanup_live_entry_exchange_never_deletes_replacement(
    tmp_path: Path, race_stage: str
) -> None:
    """Catch path-based cleanup deleting a sibling after a live ABA exchange."""

    _, formal, capability, receipt = _receipt(tmp_path)
    sibling = formal / "_tmp/reproduce.sibling"
    sibling.mkdir()
    survivor = sibling / "must-survive.txt"
    survivor.write_text("keep", encoding="utf-8")
    held = formal / f"_tmp/held-{race_stage}"

    def exchange(stage: str, scratch: Path, quarantine: Path | None) -> None:
        if stage != race_stage:
            return
        target = scratch if quarantine is None else quarantine
        target.rename(held)
        target.symlink_to(sibling, target_is_directory=True)

    with pytest.raises(ValueError, match="changed|exchange|symbolic|identity"):
        cleanup_analysis_reproduction(receipt, _race_hook=exchange)

    assert survivor.read_text(encoding="utf-8") == "keep"
    assert held.exists()


@pytest.mark.parametrize(
    ("replacement_type", "race_stage", "relative"),
    (
        ("regular", "entry_stat_to_rename", "nested/result.txt"),
        ("replaced_inode", "entry_stat_to_rename", "nested/result.txt"),
        ("symlink", "entry_stat_to_rename", "nested/result.txt"),
        ("fifo", "entry_rename_to_unlink", "nested/result.txt"),
        ("socket", "entry_rename_to_unlink", "nested/result.txt"),
        ("empty_directory", "entry_stat_to_open", "nested"),
        ("empty_directory", "entry_walk_to_rmdir", "nested"),
    ),
)
def test_cleanup_revalidates_persisted_entry_at_each_destructive_window(
    tmp_path: Path,
    replacement_type: str,
    race_stage: str,
    relative: str,
) -> None:
    """A live replacement is never removed after the one-time subset precheck."""

    _data, formal, capability, receipt = _receipt(tmp_path)
    sibling = formal / "_tmp/reproduce.sibling"
    sibling.mkdir()
    sibling_file = sibling / "survive.txt"
    sibling_file.write_text("sibling", encoding="utf-8")
    outside = tmp_path / "outside-target.txt"
    outside.write_text("outside", encoding="utf-8")
    saved = tmp_path / "saved-expected-entry"
    replacement_path: Path | None = None
    injected = False

    def inject_replacement(
        stage: str, entry_relative: str, renamed_leaf: str | None
    ) -> None:
        nonlocal injected, replacement_path
        if injected or stage != race_stage or entry_relative != relative:
            return
        injected = True
        payload = json.loads(receipt.read_text())
        quarantine = formal / "_tmp" / payload["cleaning_quarantine_name"]
        original = quarantine / relative
        replacement_path = (
            original.parent / renamed_leaf
            if renamed_leaf is not None
            else original
        )
        replacement_path.rename(saved)
        if replacement_type in {"regular", "replaced_inode"}:
            replacement_path.write_text("hostile replacement", encoding="utf-8")
        elif replacement_type == "symlink":
            replacement_path.symlink_to(outside)
        elif replacement_type == "fifo":
            os.mkfifo(replacement_path)
        elif replacement_type == "socket":
            replacement_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            prior_cwd = Path.cwd()
            try:
                os.chdir(replacement_path.parent)
                replacement_socket.bind(replacement_path.name)
            finally:
                os.chdir(prior_cwd)
                replacement_socket.close()
        else:
            replacement_path.mkdir()

    with pytest.raises(ValueError, match="inventory|changed|symbolic|identity"):
        cleanup_analysis_reproduction(
            receipt, _entry_race_hook=inject_replacement
        )

    assert injected
    assert replacement_path is not None
    assert os.path.lexists(replacement_path)
    assert outside.read_text(encoding="utf-8") == "outside"
    assert sibling_file.read_text(encoding="utf-8") == "sibling"
    if replacement_path.is_dir() and not replacement_path.is_symlink():
        replacement_path.rmdir()
    else:
        replacement_path.unlink()
    payload = json.loads(receipt.read_text())
    quarantine = formal / "_tmp" / payload["cleaning_quarantine_name"]
    restored = quarantine / relative
    saved.rename(restored)

    cleanup_analysis_reproduction(receipt)

    assert json.loads(receipt.read_text())["status"] == "cleaned"
    assert not quarantine.exists()
    assert sibling_file.read_text(encoding="utf-8") == "sibling"
    assert outside.read_text(encoding="utf-8") == "outside"


def _collect_test_inventory(root: Path) -> list[dict[str, object]]:
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        return analysis_io._inventory_tree_at(root_fd, os.fstat(root_fd).st_dev)
    finally:
        os.close(root_fd)


def _make_descriptor_relative_chain(
    root: Path, *, depth: int, external_target: Path
) -> None:
    """Build a chain without asking the kernel to resolve its complete path."""

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current_fd = os.open(root, flags)
    try:
        for _index in range(depth):
            os.mkdir("d", dir_fd=current_fd)
            child_fd = os.open("d", flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = child_fd
        os.symlink(str(external_target), "external-link", dir_fd=current_fd)
        os.mkfifo("deep.fifo", dir_fd=current_fd)
    finally:
        os.close(current_fd)


@pytest.mark.parametrize("limit_kind", ("depth", "entries", "canonical_bytes"))
def test_cleanup_inventory_collection_has_explicit_resource_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit_kind: str
) -> None:
    """Catch recursive, count-unbounded, or oversized inventory collection."""

    root = tmp_path / "inventory-root"
    root.mkdir()
    if limit_kind == "depth":
        monkeypatch.setattr(
            analysis_io, "_CLEANUP_INVENTORY_MAX_DEPTH", 3, raising=False
        )
        current = root
        for index in range(4):
            current = current / f"d{index}"
            current.mkdir()
    elif limit_kind == "entries":
        monkeypatch.setattr(
            analysis_io, "_CLEANUP_INVENTORY_MAX_ENTRIES", 4, raising=False
        )
        for index in range(5):
            (root / f"f{index}").touch()
    else:
        monkeypatch.setattr(
            analysis_io,
            "_CLEANUP_INVENTORY_MAX_CANONICAL_BYTES",
            96,
            raising=False,
        )
        (root / ("x" * 120)).touch()

    with pytest.raises(ValueError, match=f"{limit_kind}.*limit|limit.*{limit_kind}"):
        _collect_test_inventory(root)


def test_cleanup_inventory_default_limits_accept_real_92_entry_shape(
    tmp_path: Path,
) -> None:
    """The frozen cleanup gate remains wider than the real 92-entry output tree."""

    root = tmp_path / "normal-inventory"
    root.mkdir()
    for index in range(92):
        (root / f"result-{index:03d}.json").write_text("{}", encoding="utf-8")

    inventory = _collect_test_inventory(root)

    assert len(inventory) == 92
    assert len(analysis_io._canonical_json_bytes(inventory)) < (
        analysis_io._CLEANUP_INVENTORY_MAX_CANONICAL_BYTES
    )


@pytest.mark.parametrize(
    "batch",
    (
        {"ancestors": [{}] * 4_097, "entries": [{}]},
        {
            "schema_version": 2,
            "operation": "remove",
            "destination": ".cleanup-node." + "a" * 32,
            "entries": [{}] * 33,
        },
    ),
)
def test_bounded_batch_count_gate_runs_before_json_serialization(
    monkeypatch: pytest.MonkeyPatch, batch: dict[str, object]
) -> None:
    """Hostile batch lists are count-rejected before any whole-list JSON allocation."""

    def reject_serialization(_payload: object) -> bytes:
        raise AssertionError("unbounded batch reached JSON serialization")

    monkeypatch.setattr(
        analysis_io, "_canonical_json_bytes", reject_serialization
    )
    with pytest.raises(ValueError, match="cleanup batch is invalid"):
        analysis_io._validate_bounded_cleanup_batch(batch)


@pytest.mark.parametrize("limit_kind", ("depth", "entries", "canonical_bytes"))
def test_overlimit_tree_uses_bounded_journal_and_remains_safely_cleanable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit_kind: str
) -> None:
    """An inventory limit failure never strands the exact receipt-bound scratch."""

    _data, formal, capability, receipt = _receipt(tmp_path)
    sibling = formal / "_tmp/reproduce.sibling"
    sibling.mkdir()
    survivor = sibling / "survive.txt"
    survivor.write_text("keep", encoding="utf-8")
    if limit_kind == "depth":
        monkeypatch.setattr(
            analysis_io, "_CLEANUP_INVENTORY_MAX_DEPTH", 3, raising=False
        )
        current = capability.scratch_root
        for index in range(4):
            current = current / f"deep-{index}"
            current.mkdir()
    elif limit_kind == "entries":
        monkeypatch.setattr(
            analysis_io, "_CLEANUP_INVENTORY_MAX_ENTRIES", 16, raising=False
        )
        for index in range(80):
            (capability.scratch_root / f"zero-{index:03d}").touch()
    else:
        monkeypatch.setattr(
            analysis_io,
            "_CLEANUP_INVENTORY_MAX_CANONICAL_BYTES",
            384,
            raising=False,
        )
        for index in range(6):
            (capability.scratch_root / (f"long-{index}-" + "x" * 90)).touch()

    cleanup_analysis_reproduction(receipt)

    payload = json.loads(receipt.read_text())
    assert payload["status"] == "cleaned"
    assert payload["cleaning_strategy"] == "bounded_batches_v1"
    assert payload["cleaning_limit"]["kind"] == limit_kind
    assert payload["cleaning_inventory"] is None
    assert receipt.stat().st_size < 64 * 1024
    assert len(list(receipt.parent.glob("reproduction.*.json"))) == 1
    assert survivor.read_text(encoding="utf-8") == "keep"
    assert not capability.scratch_root.exists()
    assert not list((formal / "_tmp").glob(".cleanup-*"))


def test_overlimit_cleanup_never_retries_an_unserializable_ancestor_chain_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A >262144-byte ancestor chain is flattened with bounded journal authority."""

    _data, formal, capability, receipt = _receipt(tmp_path)
    sibling = formal / "_tmp/reproduce.sibling"
    sibling.mkdir()
    sibling_survivor = sibling / "survive.txt"
    sibling_survivor.write_text("sibling", encoding="utf-8")
    outside = tmp_path / "outside-target.txt"
    outside.write_text("outside", encoding="utf-8")
    _make_descriptor_relative_chain(
        capability.scratch_root, depth=500, external_target=outside
    )
    root_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    prior_cwd = Path.cwd()
    try:
        os.chdir(capability.scratch_root)
        root_socket.bind("bounded.socket")
    finally:
        os.chdir(prior_cwd)
        root_socket.close()
    # Isolate the cleanup walker: quota accounting itself is covered separately,
    # and a descriptor-only chain intentionally cannot be measured by a long path.
    monkeypatch.setattr(
        analysis_io,
        "directory_usage_bytes",
        lambda _path: receipt.stat().st_size,
    )
    monkeypatch.setattr(
        analysis_io,
        "combined_project_usage_bytes",
        lambda _data, _formal: receipt.stat().st_size,
    )

    blocked: list[str] = []
    for _attempt in range(2):
        try:
            cleanup_analysis_reproduction(receipt)
        except ValueError as exc:
            blocked.append(str(exc))
        if json.loads(receipt.read_text())["status"] == "cleaned":
            break

    assert len(blocked) < 2, (
        "bounded cleanup permanently retried the same overlarge ancestor journal: "
        f"{blocked}"
    )
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "cleaned"
    assert payload["cleaning_strategy"] == "bounded_batches_v1"
    assert payload["cleaning_batch"] is None
    assert receipt.stat().st_size < 64 * 1024
    assert not capability.scratch_root.exists()
    assert sibling_survivor.read_text(encoding="utf-8") == "sibling"
    assert outside.read_text(encoding="utf-8") == "outside"


def test_bounded_frontier_resume_and_directory_identity_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A promoted directory is journaled, resumable, and never replaced by path."""

    _data, formal, capability, receipt = _receipt(tmp_path)
    outside = tmp_path / "outside-target.txt"
    outside.write_text("outside", encoding="utf-8")
    parent = capability.scratch_root / "frontier"
    child = parent / "child"
    child.mkdir(parents=True)
    (child / "external-link").symlink_to(outside)
    sibling = formal / "_tmp/reproduce.sibling"
    sibling.mkdir()
    sibling_survivor = sibling / "survive.txt"
    sibling_survivor.write_text("sibling", encoding="utf-8")
    monkeypatch.setattr(analysis_io, "_CLEANUP_INVENTORY_MAX_DEPTH", 1)
    injected = False
    saved = tmp_path / "saved-expected-directory"
    replacement: Path | None = None

    def replace_before_promote(
        stage: str, relative: str, _renamed: str | None
    ) -> None:
        nonlocal injected, replacement
        if injected or stage != "bounded_entry_stat_to_rename":
            return
        if not relative.endswith("/child"):
            return
        injected = True
        payload = json.loads(receipt.read_text())
        quarantine = formal / "_tmp" / payload["cleaning_quarantine_name"]
        replacement = quarantine / relative
        replacement.rename(saved)
        replacement.mkdir()

    with pytest.raises(ValueError, match="inventory|changed|identity"):
        cleanup_analysis_reproduction(
            receipt, _entry_race_hook=replace_before_promote
        )

    assert injected
    assert replacement is not None and replacement.is_dir()
    assert outside.read_text(encoding="utf-8") == "outside"
    assert sibling_survivor.read_text(encoding="utf-8") == "sibling"
    replacement.rmdir()
    saved.rename(replacement)

    interrupted = False

    def interrupt_after_promote(
        stage: str, relative: str, _renamed: str | None
    ) -> None:
        nonlocal interrupted
        if (
            not interrupted
            and stage == "bounded_rename_to_journal_clear"
            and relative.endswith("/child")
        ):
            interrupted = True
            raise KeyboardInterrupt("injected frontier promotion interruption")

    with pytest.raises(KeyboardInterrupt, match="frontier promotion"):
        cleanup_analysis_reproduction(
            receipt, _entry_race_hook=interrupt_after_promote
        )
    persisted = json.loads(receipt.read_text())
    assert persisted["status"] == "cleaning"
    assert persisted["cleaning_batch"]["operation"] == "promote"
    assert persisted["cleaning_batch"]["entries"]

    cleanup_analysis_reproduction(receipt)

    assert json.loads(receipt.read_text())["status"] == "cleaned"
    assert outside.read_text(encoding="utf-8") == "outside"
    assert sibling_survivor.read_text(encoding="utf-8") == "sibling"


def test_bounded_cleanup_resumes_the_persisted_batch_after_journal_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after batch deletion reuses the expected batch and missing subset."""

    _data, formal, capability, receipt = _receipt(tmp_path)
    monkeypatch.setattr(analysis_io, "_CLEANUP_INVENTORY_MAX_ENTRIES", 8)
    for index in range(80):
        (capability.scratch_root / f"zero-{index:03d}").touch()
    original_writer = analysis_io._write_cleanup_reproduction_receipt
    bounded_writes = 0

    def interrupt_first_batch_clear(
        path: Path, payload: dict[str, object]
    ) -> None:
        nonlocal bounded_writes
        if payload.get("cleaning_strategy") == "bounded_batches_v1":
            bounded_writes += 1
            if bounded_writes == 3:
                raise OSError("injected bounded journal interruption")
        original_writer(path, payload)

    monkeypatch.setattr(
        analysis_io,
        "_write_cleanup_reproduction_receipt",
        interrupt_first_batch_clear,
    )
    with pytest.raises(OSError, match="bounded journal interruption"):
        cleanup_analysis_reproduction(receipt)
    interrupted = json.loads(receipt.read_text())
    quarantine = formal / "_tmp" / interrupted["cleaning_quarantine_name"]
    assert interrupted["status"] == "cleaning"
    assert interrupted["cleaning_batch"]["entries"]
    assert quarantine.is_dir()
    assert len(list(quarantine.glob("zero-*"))) < 80

    monkeypatch.setattr(
        analysis_io, "_write_cleanup_reproduction_receipt", original_writer
    )
    cleanup_analysis_reproduction(receipt)

    assert json.loads(receipt.read_text())["status"] == "cleaned"
    assert not quarantine.exists()
    assert not capability.scratch_root.exists()


@pytest.mark.parametrize("budget", ("output", "project"))
def test_cleanup_journal_projected_quota_fails_before_quarantine_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: str
) -> None:
    """Count a journal write once against both frozen quotas before destruction."""

    _data, formal, capability, receipt = _receipt(tmp_path)
    with monkeypatch.context() as patcher:
        if budget == "output":
            patcher.setattr(
                analysis_io,
                "directory_usage_bytes",
                lambda _path: 10 * 1024**3 - 1,
            )
        else:
            patcher.setattr(
                analysis_io,
                "combined_project_usage_bytes",
                lambda _data, _formal: 150 * 1024**3 - 1,
            )
        with pytest.raises(RuntimeError, match=f"{budget}.*quota|{budget}.*limit"):
            cleanup_analysis_reproduction(receipt)

    unchanged = json.loads(receipt.read_text())
    assert unchanged["status"] == "matched"
    assert unchanged["cleanup_status"] == "pending"
    assert capability.scratch_root.is_dir()
    assert not list((formal / "_tmp").glob(".cleanup-*"))

    cleanup_analysis_reproduction(receipt)

    assert json.loads(receipt.read_text())["status"] == "cleaned"
    assert not capability.scratch_root.exists()


@pytest.mark.parametrize("state", ("pending", "matched", "failed"))
@pytest.mark.parametrize("budget", ("output", "project"))
@pytest.mark.parametrize("boundary", ("last_allowed_byte", "first_blocked_byte"))
def test_every_reproduction_receipt_transition_uses_exact_replacement_quota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    budget: str,
    boundary: str,
) -> None:
    """Catch a direct receipt write, a missing final newline, or old-byte double count."""

    data, formal, capability, receipt, transition = _receipt_transition(
        tmp_path, state
    )
    prior_document = receipt.read_bytes() if receipt.exists() else None
    captured: dict[str, object] = {}

    def capture_atomic_write(path: Path, payload: dict[str, object]) -> None:
        captured["path"] = path
        captured["payload"] = payload
        raise _CapturedReceiptWrite("capture exact receipt document")

    with monkeypatch.context() as patcher:
        patcher.setattr(
            analysis_io, "_atomic_write_json_object", capture_atomic_write
        )
        with pytest.raises(_CapturedReceiptWrite, match="exact receipt document"):
            transition()

    assert captured["path"] == receipt
    document = (
        json.dumps(
            captured["payload"],
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    old_bytes = 0 if prior_document is None else len(prior_document)
    if budget == "output":
        limit = 10 * 1024**3
        last_allowed = limit
    else:
        limit = 150 * 1024**3
        last_allowed = limit - 1
    projected = last_allowed + (boundary == "first_blocked_byte")
    usage_before = projected - len(document) + old_bytes
    assert usage_before >= 0

    with monkeypatch.context() as patcher:
        if budget == "output":
            patcher.setattr(
                analysis_io,
                "directory_usage_bytes",
                lambda _path: usage_before,
            )
        else:
            patcher.setattr(
                analysis_io,
                "combined_project_usage_bytes",
                lambda _data, _formal: usage_before,
            )
        if boundary == "first_blocked_byte":
            with pytest.raises(
                RuntimeError,
                match=(
                    "analysis output quota"
                    if budget == "output"
                    else "absolute project limit"
                ),
            ):
                transition()
        else:
            assert transition() == receipt

    if boundary == "first_blocked_byte":
        if prior_document is None:
            assert not receipt.exists()
        else:
            assert receipt.read_bytes() == prior_document
    else:
        assert len(receipt.read_bytes()) == len(document)
    assert capability.scratch_root.is_dir()
    assert not list((formal / "_tmp").glob(".cleanup-*"))
    assert data.is_dir()


@pytest.mark.parametrize("budget", ("output", "project"))
def test_receipt_quota_replaces_a_stale_atomic_partial_without_double_counting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget: str,
) -> None:
    """A recoverable stale partial is removed before the exact replacement write."""

    _data, _formal, _capability, receipt, transition = _receipt_transition(
        tmp_path, "matched"
    )
    captured: dict[str, object] = {}

    def capture_atomic_write(_path: Path, payload: dict[str, object]) -> None:
        captured["payload"] = payload
        raise _CapturedReceiptWrite("capture stale-partial replacement")

    with monkeypatch.context() as patcher:
        patcher.setattr(
            analysis_io, "_atomic_write_json_object", capture_atomic_write
        )
        with pytest.raises(_CapturedReceiptWrite, match="stale-partial"):
            transition()
    document = (
        json.dumps(
            captured["payload"],
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    old_bytes = receipt.stat().st_size
    partial = receipt.with_name(f"{receipt.name}.partial")
    partial.write_bytes(b"stale" * 100)
    last_allowed = (10 * 1024**3) if budget == "output" else (150 * 1024**3 - 1)
    usage_before = (
        last_allowed - len(document) + old_bytes + partial.stat().st_size
    )

    with monkeypatch.context() as patcher:
        if budget == "output":
            patcher.setattr(
                analysis_io,
                "directory_usage_bytes",
                lambda _path: usage_before,
            )
        else:
            patcher.setattr(
                analysis_io,
                "combined_project_usage_bytes",
                lambda _data, _formal: usage_before,
            )
        assert transition() == receipt

    assert not partial.exists()
    assert len(receipt.read_bytes()) == len(document)


def test_failed_attempt_receipt_safely_cleans_historical_exact_scratch(
    tmp_path: Path,
) -> None:
    """A failed run remains cleanable after formal output advances, but only exactly."""

    _, formal, capability = _mint(tmp_path)
    scratch_manifest = capability.scratch_root / "run_manifest.json"
    scratch_manifest.write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "git_commit": GIT_COMMIT,
                "input_authority_hash": "1" * 64,
                "renv_lock_sha256": "2" * 64,
                "threshold_registry_hash": "3" * 64,
                "status": "success",
            }
        ),
        encoding="utf-8",
    )
    receipt = create_failed_analysis_reproduction_receipt(
        formal_output_root=formal,
        scratch_root=capability.scratch_root,
        failure="logical comparison failed",
    )
    (formal / "run_manifest.json").write_text(
        json.dumps({"run_id": "c" * 16, "status": "success"}),
        encoding="utf-8",
    )
    sibling = formal / "_tmp/reproduce.keep"
    sibling.mkdir()

    removed = cleanup_analysis_reproduction(receipt)

    assert removed == capability.scratch_root
    assert not capability.scratch_root.exists()
    assert sibling.is_dir()
    assert json.loads(receipt.read_text())["cleanup_status"] == "cleaned"


@pytest.mark.parametrize(
    ("manifest_case", "expected_observation"),
    (
        ("missing", "missing"),
        ("invalid_json", "invalid_json"),
        ("wrong_identity", "mismatch"),
        ("symlink", "symlink"),
        ("permission", "permission_error"),
    ),
)
def test_failed_transition_preserves_original_error_for_bad_scratch_manifest(
    tmp_path: Path, manifest_case: str, expected_observation: str
) -> None:
    """Catch failure journaling being replaced by a manifest parse/identity error."""

    data, formal, capability = _mint(tmp_path)
    formal_manifest = formal / "run_manifest.json"
    formal_manifest.write_text(
        json.dumps(
            {
                "status": "success",
                "run_id": RUN_ID,
                "git_commit": GIT_COMMIT,
                "spec_id": "spec",
                "input_authority_hash": "1" * 64,
                "renv_lock_sha256": "2" * 64,
                "threshold_registry_hash": "3" * 64,
            }
        ),
        encoding="utf-8",
    )
    receipt = create_pending_analysis_reproduction_receipt(
        capability=capability, formal_manifest_path=formal_manifest
    )
    scratch_manifest = capability.scratch_root / "run_manifest.json"
    outside = tmp_path / "outside.json"
    if manifest_case == "invalid_json":
        scratch_manifest.write_text("{broken", encoding="utf-8")
    elif manifest_case == "wrong_identity":
        scratch_manifest.write_text(
            json.dumps({"run_id": "c" * 16, "git_commit": GIT_COMMIT}),
            encoding="utf-8",
        )
    elif manifest_case == "symlink":
        outside.write_text("{}", encoding="utf-8")
        scratch_manifest.symlink_to(outside)
    elif manifest_case == "permission":
        scratch_manifest.write_text("{}", encoding="utf-8")
        scratch_manifest.chmod(0)

    result = create_failed_analysis_reproduction_receipt(
        formal_output_root=formal,
        scratch_root=capability.scratch_root,
        failure="OriginalStageError: keep me",
    )
    payload = json.loads(result.read_text())
    assert result == receipt
    assert payload["status"] == "failed"
    assert payload["failure"] == "OriginalStageError: keep me"
    assert payload["scratch_run_manifest_observation"]["state"] == expected_observation
    try:
        cleanup_analysis_reproduction(result)
    finally:
        if manifest_case == "permission" and scratch_manifest.exists():
            scratch_manifest.chmod(0o600)
    assert json.loads(result.read_text())["status"] == "cleaned"
    assert not capability.scratch_root.exists()
    if manifest_case == "symlink":
        assert outside.is_file()


def test_analysis_reproduction_cli_requires_real_data_and_output_roots(
    tmp_path: Path,
) -> None:
    """Catch a stale CLI that accepts a caller-selected scratch root."""

    args = build_parser().parse_args(
        [
            "analysis-reproduce-check",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "formal"),
        ]
    )
    assert args.command == "analysis-reproduce-check"
    assert args.data_root == tmp_path / "data"
    assert args.output_root == tmp_path / "formal"
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "analysis-reproduce-check",
                "--data-root",
                str(tmp_path / "data"),
                "--output-root",
                str(tmp_path / "formal"),
                "--scratch-root",
                str(tmp_path / "attacker-selected"),
            ]
        )


def test_cleanup_analysis_reproduction_cli_uses_the_formal_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catch routing the analysis cleanup command to construction cleanup."""

    _, _, capability, receipt = _receipt(tmp_path)
    assert main(
        ["cleanup-analysis-reproduction", "--receipt", str(receipt)]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "cleaned"
    assert payload["removed_path"] == str(capability.scratch_root)


def test_reproduction_orchestrator_mints_scratch_copies_registry_and_receipts_scope(
    project_fixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch caller-selected scratch, threshold reselection, or a partial receipt."""

    formal = project_fixture.code_root / "06_结果/analysis"
    _write_parquet(
        formal / "models/model.parquet",
        [{"id": 1, "label": "x", "estimate": 2.0}],
    )
    (formal / "models/model.parquet.manifest.json").write_text(
        json.dumps({"primary_key": ["id"]}), encoding="utf-8"
    )
    (formal / "tables").mkdir()
    (formal / "tables/table_1.csv").write_text(
        "id,label,value\n1,x,2.0\n", encoding="utf-8"
    )
    (formal / "registries").mkdir()
    registry = formal / "registries/threshold_registry_v1.json"
    registry.write_text(
        json.dumps({"registry_hash": "3" * 64, "q": 0.5}),
        encoding="utf-8",
    )
    (formal / "figures").mkdir()
    (formal / "figures/figure_1.pdf").write_bytes(b"formal")
    (formal / "figures/figure_1.provenance.json").write_text(
        json.dumps(
            {
                "figure_id": "figure_1",
                "plotted_source_data_hash": "4" * 64,
                "source_table_hashes": {"model": "5" * 64},
            }
        ),
        encoding="utf-8",
    )
    spec_id = "spec"
    input_hash = "1" * 64
    lock_hash = "2" * 64
    policy_hash = "4" * 64
    run_id = hashlib.sha256(
        json.dumps(
            [spec_id, input_hash, GIT_COMMIT, lock_hash, policy_hash],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    manifest = formal / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "success",
                "run_id": run_id,
                "spec_id": spec_id,
                "input_authority_hash": input_hash,
                "git_commit": GIT_COMMIT,
                "renv_lock_sha256": lock_hash,
                "evidence_policy_sha256": policy_hash,
                "threshold_registry_hash": "3" * 64,
                "created_at_utc": "2026-08-29T01:00:00Z",
                "audited_at_utc": "2026-08-29T02:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    observed: list[tuple[tuple[str, ...], dict[str, str]]] = []
    retained_authorities: list[dict[str, str]] = []
    monkeypatch.setattr(
        analysis_io, "_validate_orchestrator_peer", lambda _fd, _marker: os.getpid()
    )

    def fake_runner(command: tuple[str, ...], environment: dict[str, str]) -> None:
        observed.append((command, dict(environment)))
        if "GREEN_DEBT_ANALYSIS_REPRO_STAGE" not in environment:
            assert command[-2:] == ("analysis-deps", "--verify")
            assert not any(
                key.startswith("GREEN_DEBT_ANALYSIS_REPRO_")
                for key in environment
            )
            return
        scratch = Path(environment["GREEN_DEBT_ANALYSIS_REPRO_SCRATCH"])
        if (
            retained_authorities
            and retained_authorities[-1]["GREEN_DEBT_ANALYSIS_REPRO_MARKER"]
            != environment["GREEN_DEBT_ANALYSIS_REPRO_MARKER"]
        ):
            for stale in retained_authorities:
                os.close(int(stale["GREEN_DEBT_ANALYSIS_REPRO_BROKER_FD"]))
                os.close(int(stale["GREEN_DEBT_ANALYSIS_REPRO_SECRET_FD"]))
            retained_authorities.clear()
        for stale in retained_authorities[-1:]:
            with pytest.raises(ValueError, match="broker|secret|orchestrator"):
                validate_analysis_reproduction_capability(
                    data_root=project_fixture.data_root,
                    formal_output_root=formal,
                    scratch_root=scratch,
                    expected_git_commit=GIT_COMMIT,
                    environment=stale,
                )
        retained = dict(environment)
        retained["GREEN_DEBT_ANALYSIS_REPRO_BROKER_FD"] = str(
            os.dup(int(environment["GREEN_DEBT_ANALYSIS_REPRO_BROKER_FD"]))
        )
        retained["GREEN_DEBT_ANALYSIS_REPRO_SECRET_FD"] = str(
            os.dup(int(environment["GREEN_DEBT_ANALYSIS_REPRO_SECRET_FD"]))
        )
        validate_analysis_reproduction_capability(
            data_root=project_fixture.data_root,
            formal_output_root=formal,
            scratch_root=scratch,
            expected_git_commit=GIT_COMMIT,
            environment=environment,
        )
        retained_authorities.append(retained)
        for relative in (
            "models/model.parquet",
            "models/model.parquet.manifest.json",
            "tables/table_1.csv",
            "figures/figure_1.pdf",
            "figures/figure_1.provenance.json",
        ):
            destination = scratch / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((formal / relative).read_bytes())
        (scratch / "run_manifest.json").write_text(
            manifest.read_text(), encoding="utf-8"
        )

    receipt = run_analysis_reproduction_check(
        code_root=project_fixture.code_root,
        data_root=project_fixture.data_root,
        formal_output_root=formal,
        expected_git_commit=GIT_COMMIT,
        command_runner=fake_runner,
        enforce_production_counts=False,
    )
    payload = json.loads(receipt.read_text())
    scratch = Path(payload["scratch_root"])
    assert receipt == (
        formal / "reproduction_receipts" /
        f"reproduction.{payload['execution_id']}.json"
    )
    assert len(observed) == 11
    assert [
        environment.get("GREEN_DEBT_ANALYSIS_REPRO_STAGE")
        for _, environment in observed
    ] == [None, *analysis_io.planned_analysis_stages()]
    assert all(
        command[-4:] == (
            "--data-root",
            str(project_fixture.data_root.resolve()),
            "--output-root",
            str(scratch),
        )
        for command, _environment in observed[1:]
    )
    assert all(
        "GREEN_DEBT_ANALYSIS_REPRO_BROKER_FD" in environment
        and "GREEN_DEBT_ANALYSIS_REPRO_SECRET_FD" in environment
        for _command, environment in observed[1:]
    )
    for retained in retained_authorities:
        os.close(int(retained["GREEN_DEBT_ANALYSIS_REPRO_BROKER_FD"]))
        os.close(int(retained["GREEN_DEBT_ANALYSIS_REPRO_SECRET_FD"]))
    retained_authorities.clear()
    assert (scratch / "registries/threshold_registry_v1.json").read_bytes() == (
        registry.read_bytes()
    )
    assert payload["formal_run_id"] == run_id
    assert payload["evidence_policy_sha256"] == policy_hash
    assert payload["comparison_results"] == {
        "all_matched": True,
        "csv": 1,
        "figures": 1,
        "json": 1,
        "max_scaled_float_error": 0.0,
        "parquet": 1,
    }
    assert payload["data_snapshot_before"] == payload["data_snapshot_after"]
    assert payload["manifest_snapshot_before"] == payload["manifest_snapshot_after"]
    with pytest.raises(ValueError, match="unclean|pending|active"):
        run_analysis_reproduction_check(
            code_root=project_fixture.code_root,
            data_root=project_fixture.data_root,
            formal_output_root=formal,
            expected_git_commit=GIT_COMMIT,
            command_runner=fake_runner,
            enforce_production_counts=False,
        )
    cleanup_analysis_reproduction(receipt)

    with monkeypatch.context() as patcher:
        def fail_compare(_formal: Path, _scratch: Path):
            raise AnalysisReproductionMismatch("injected compare failure")

        patcher.setattr(analysis_io, "compare_analysis_output_trees", fail_compare)
        with pytest.raises(AnalysisReproductionMismatch, match="injected compare"):
            run_analysis_reproduction_check(
                code_root=project_fixture.code_root,
                data_root=project_fixture.data_root,
                formal_output_root=formal,
                expected_git_commit=GIT_COMMIT,
                command_runner=fake_runner,
                enforce_production_counts=False,
            )
    unclean = [
        path for path in (formal / "reproduction_receipts").glob("*.json")
        if json.loads(path.read_text())["cleanup_status"] == "pending"
    ]
    assert len(unclean) == 1
    assert json.loads(unclean[0].read_text())["status"] == "failed"
    cleanup_analysis_reproduction(unclean[0])

    with monkeypatch.context() as patcher:
        patcher.setattr(
            analysis_io, "directory_usage_bytes", lambda _path: 11 * 1024**3
        )
        with pytest.raises(RuntimeError, match="10 GB"):
            run_analysis_reproduction_check(
                code_root=project_fixture.code_root,
                data_root=project_fixture.data_root,
                formal_output_root=formal,
                expected_git_commit=GIT_COMMIT,
                command_runner=fake_runner,
                enforce_production_counts=False,
            )
    unclean = [
        path for path in (formal / "reproduction_receipts").glob("*.json")
        if json.loads(path.read_text())["cleanup_status"] == "pending"
    ]
    assert unclean == []
    assert not list((formal / "_tmp").glob("reproduce.*"))
    for retained in retained_authorities:
        os.close(int(retained["GREEN_DEBT_ANALYSIS_REPRO_BROKER_FD"]))
        os.close(int(retained["GREEN_DEBT_ANALYSIS_REPRO_SECRET_FD"]))


def test_analysis_reproduce_cli_runs_internal_orchestrator_not_public_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catch a parsed reproduction command that never reaches the gated runner."""

    receipt = tmp_path / "formal/reproduction_receipt.json"
    receipt.parent.mkdir()
    receipt.write_text(
        json.dumps(
            {
                "execution_id": "20260829T010203Z-feedface",
                "formal_run_id": RUN_ID,
                "scratch_root": str(tmp_path / "formal/_tmp/reproduce.x"),
                "comparison_results": {
                    "max_scaled_float_error": 0.0,
                    "all_matched": True,
                },
            }
        ),
        encoding="utf-8",
    )
    observed: dict[str, Path] = {}

    def fake_reproduce(**kwargs):
        observed.update(kwargs)
        return receipt

    monkeypatch.setattr(
        cli, "run_analysis_reproduction_check", fake_reproduce, raising=False
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path / "code")
    assert main(
        [
            "analysis-reproduce-check",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "formal"),
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "matched"
    assert payload["execution_id"] == "20260829T010203Z-feedface"
    assert observed == {
        "code_root": tmp_path / "code",
        "data_root": tmp_path / "data",
        "formal_output_root": tmp_path / "formal",
    }


def test_runner_failure_is_journaled_and_blocks_repeat_until_cleanup(
    project_fixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a stage exception leaving an orphan or letting a repeat overwrite it."""

    formal = project_fixture.code_root / "06_结果/analysis"
    (formal / "registries").mkdir(parents=True)
    registry = formal / "registries/threshold_registry_v1.json"
    registry.write_text(
        json.dumps({"registry_hash": "3" * 64, "q": 0.5}), encoding="utf-8"
    )
    spec_id = "spec"
    input_hash = "1" * 64
    lock_hash = "2" * 64
    policy_hash = "4" * 64
    run_id = hashlib.sha256(
        json.dumps(
            [spec_id, input_hash, GIT_COMMIT, lock_hash, policy_hash],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    manifest = formal / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "success",
                "run_id": run_id,
                "spec_id": spec_id,
                "input_authority_hash": input_hash,
                "git_commit": GIT_COMMIT,
                "renv_lock_sha256": lock_hash,
                "evidence_policy_sha256": policy_hash,
                "threshold_registry_hash": "3" * 64,
                "created_at_utc": "2026-08-29T01:00:00Z",
                "audited_at_utc": "2026-08-29T02:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    def interrupted(_command, environment):
        if "GREEN_DEBT_ANALYSIS_REPRO_STAGE" not in environment:
            return
        scratch = Path(environment["GREEN_DEBT_ANALYSIS_REPRO_SCRATCH"])
        (scratch / "partial.txt").write_text("partial", encoding="utf-8")
        raise KeyboardInterrupt("controlled interrupt")

    with pytest.raises(KeyboardInterrupt, match="controlled"):
        run_analysis_reproduction_check(
            code_root=project_fixture.code_root,
            data_root=project_fixture.data_root,
            formal_output_root=formal,
            expected_git_commit=GIT_COMMIT,
            command_runner=interrupted,
            enforce_production_counts=False,
        )
    receipts = tuple((formal / "reproduction_receipts").glob("reproduction.*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text())
    assert payload["status"] == "failed"
    assert payload["cleanup_status"] == "pending"
    assert Path(payload["scratch_root"]).is_dir()
    with pytest.raises(ValueError, match="unclean|active"):
        run_analysis_reproduction_check(
            code_root=project_fixture.code_root,
            data_root=project_fixture.data_root,
            formal_output_root=formal,
            expected_git_commit=GIT_COMMIT,
            command_runner=interrupted,
            enforce_production_counts=False,
        )
    cleanup_analysis_reproduction(receipts[0])
    assert not Path(payload["scratch_root"]).exists()

    lock_fd = os.open(formal / "_tmp/.analysis-reproduction.lock", os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already active"):
            run_analysis_reproduction_check(
                code_root=project_fixture.code_root,
                data_root=project_fixture.data_root,
                formal_output_root=formal,
                expected_git_commit=GIT_COMMIT,
                command_runner=interrupted,
                enforce_production_counts=False,
            )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    original_writer = analysis_io._atomic_write_json_object

    def fail_journal_write(path: Path, value: dict[str, object]) -> None:
        if path.parent.name == "reproduction_receipts":
            raise OSError("injected journal write failure")
        original_writer(path, value)

    monkeypatch.setattr(analysis_io, "_atomic_write_json_object", fail_journal_write)
    with pytest.raises(OSError, match="journal write"):
        run_analysis_reproduction_check(
            code_root=project_fixture.code_root,
            data_root=project_fixture.data_root,
            formal_output_root=formal,
            expected_git_commit=GIT_COMMIT,
            command_runner=interrupted,
            enforce_production_counts=False,
        )
    assert not tuple((formal / "_tmp").glob("reproduce.*"))


def test_sigterm_after_mint_transitions_pending_journal_to_failed(
    project_fixture,
) -> None:
    """Catch journal failure being published before a stage process tree is dead."""

    formal = project_fixture.code_root / "06_结果/analysis"
    (formal / "registries").mkdir(parents=True)
    (formal / "registries/threshold_registry_v1.json").write_text(
        json.dumps({"registry_hash": "3" * 64, "q": 0.5}), encoding="utf-8"
    )
    spec_id, input_hash, lock_hash, policy_hash = (
        "spec", "1" * 64, "2" * 64, "4" * 64
    )
    run_id = hashlib.sha256(
        json.dumps(
            [spec_id, input_hash, GIT_COMMIT, lock_hash, policy_hash],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    (formal / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "success", "run_id": run_id, "spec_id": spec_id,
                "input_authority_hash": input_hash, "git_commit": GIT_COMMIT,
                "renv_lock_sha256": lock_hash,
                "evidence_policy_sha256": policy_hash,
                "threshold_registry_hash": "3" * 64,
                "created_at_utc": "2026-08-29T01:00:00Z",
                "audited_at_utc": "2026-08-29T02:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    grandchild = project_fixture.code_root / "stage_grandchild.py"
    grandchild.write_text(
        """
import os, sys, time
from pathlib import Path
counter, child_pid = map(Path, sys.argv[1:])
child_pid.write_text(str(os.getpid()))
while True:
    counter.write_text(str(time.time()))
    time.sleep(0.02)
""",
        encoding="utf-8",
    )
    writer = project_fixture.code_root / "stage_writer.py"
    writer.write_text(
        """
import os, subprocess, sys, time
from pathlib import Path
counter, child_pid, grandchild = map(Path, sys.argv[1:])
child = subprocess.Popen([
    sys.executable, str(grandchild), str(counter), str(child_pid),
])
while True:
    counter.write_text(str(time.time()))
    time.sleep(0.02)
""",
        encoding="utf-8",
    )
    counter = project_fixture.code_root / "stage-counter.txt"
    child_pid = project_fixture.code_root / "stage-child.pid"
    program = """
import os, signal, sys, threading
from pathlib import Path
import green_debt.analysis_io as analysis_io
from green_debt.analysis_io import AnalysisCommand, run_analysis_reproduction_check
def plan(**_kwargs):
    return (
        AnalysisCommand(None, ('/usr/bin/true',)),
        AnalysisCommand('preflight', (
                sys.executable, os.environ['TEST_WRITER'],
                os.environ['TEST_COUNTER'], os.environ['TEST_CHILD_PID'],
                os.environ['TEST_GRANDCHILD'],
        )),
    )
analysis_io.planned_analysis_commands = plan
threading.Timer(0.35, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
run_analysis_reproduction_check(
    code_root=Path(os.environ['TEST_CODE_ROOT']),
    data_root=Path(os.environ['TEST_DATA_ROOT']),
    formal_output_root=Path(os.environ['TEST_FORMAL_ROOT']),
    expected_git_commit=os.environ['TEST_GIT_COMMIT'],
    enforce_production_counts=False,
)
"""
    environment = {
        **os.environ,
        "TEST_CODE_ROOT": str(project_fixture.code_root),
        "TEST_DATA_ROOT": str(project_fixture.data_root),
        "TEST_FORMAL_ROOT": str(formal),
        "TEST_GIT_COMMIT": GIT_COMMIT,
        "TEST_WRITER": str(writer),
        "TEST_COUNTER": str(counter),
        "TEST_CHILD_PID": str(child_pid),
        "TEST_GRANDCHILD": str(grandchild),
    }
    completed = subprocess.run(
        [sys.executable, "-c", program], env=environment, check=False
    )
    assert completed.returncode != 0
    receipts = tuple((formal / "reproduction_receipts").glob("reproduction.*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text())
    assert payload["status"] == "failed"
    assert "SIGTERM" in payload["failure"]
    assert child_pid.is_file()
    killed_pid = int(child_pid.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(killed_pid, 0)
    stopped = counter.read_text()
    time.sleep(0.15)
    assert counter.read_text() == stopped
    cleanup_analysis_reproduction(receipts[0])

    def second_attempt(_command, environment):
        if "GREEN_DEBT_ANALYSIS_REPRO_STAGE" in environment:
            raise RuntimeError("new run reached a write stage")

    with pytest.raises(RuntimeError, match="new run reached"):
        run_analysis_reproduction_check(
            code_root=project_fixture.code_root,
            data_root=project_fixture.data_root,
            formal_output_root=formal,
            expected_git_commit=GIT_COMMIT,
            command_runner=second_attempt,
            enforce_production_counts=False,
        )
    second_receipt = max(
        (formal / "reproduction_receipts").glob("reproduction.*.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    assert json.loads(second_receipt.read_text())["status"] == "failed"
    cleanup_analysis_reproduction(second_receipt)
