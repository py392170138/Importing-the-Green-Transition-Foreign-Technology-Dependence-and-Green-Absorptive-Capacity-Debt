from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from green_debt.analysis_io import planned_analysis_stages
import green_debt.analysis_io as analysis_io


ROOT = Path(__file__).resolve().parents[2]


def _fake_runner_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    runner = project / "03_代码/bin/run-analysis"
    runner.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "03_代码/bin/run-analysis", runner)
    runner.chmod(0o755)
    (project / "03_代码/R").mkdir(parents=True)
    (project / "03_代码/R/run_analysis.R").write_text("# fixture\n")
    python = project / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/bin/sh\n"
        "printf 'python' >> \"$ANALYSIS_LOG\"\n"
        "for arg in \"$@\"; do printf '\\t%s' \"$arg\" >> \"$ANALYSIS_LOG\"; done\n"
        "printf '\\n' >> \"$ANALYSIS_LOG\"\n"
        "case \" $* \" in *\" $FAIL_PATTERN \"*) exit 9;; esac\n"
        "if [ \"${CREATE_AUDIT_MANIFEST:-0}\" = 1 ] && "
        "[ \"${3:-}\" = analysis-output-audit ]; then\n"
        "  while [ \"$#\" -gt 0 ]; do\n"
        "    if [ \"$1\" = --output-root ]; then mkdir -p \"$2\"; "
        "printf '{}\\n' > \"$2/run_manifest.json\"; break; fi\n"
        "    shift\n"
        "  done\n"
        "fi\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    rscript = tmp_path / "bin/Rscript"
    rscript.parent.mkdir()
    rscript.write_text(
        "#!/bin/sh\n"
        "printf 'Rscript' >> \"$ANALYSIS_LOG\"\n"
        "for arg in \"$@\"; do printf '\\t%s' \"$arg\" >> \"$ANALYSIS_LOG\"; done\n"
        "printf '\\n' >> \"$ANALYSIS_LOG\"\n"
        "case \" $* \" in *\" $FAIL_PATTERN \"*) exit 9;; esac\n",
        encoding="utf-8",
    )
    rscript.chmod(0o755)
    return runner, rscript.parent


def _run_fake(
    tmp_path: Path,
    *arguments: str,
    fail_pattern: str = "__never__",
    create_manifest: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], Path]:
    runner, fake_bin = _fake_runner_project(tmp_path)
    log = tmp_path / "calls.log"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "ANALYSIS_LOG": str(log),
        "FAIL_PATTERN": fail_pattern,
        "CREATE_AUDIT_MANIFEST": "1" if create_manifest else "0",
    }
    completed = subprocess.run(
        [str(runner), *arguments],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    calls = (
        [line.split("\t") for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return completed, calls, runner.parents[2]


def test_runner_has_the_only_permitted_stage_order() -> None:
    """Catch any orchestration path that reorders or omits a frozen stage."""

    assert planned_analysis_stages() == (
        "preflight",
        "diagnostics",
        "lp",
        "lp_ingest",
        "threshold",
        "weak_iv",
        "shift_share",
        "threshold_and_iv_audit_ingest",
        "report",
        "audit",
    )


def test_shared_command_plan_covers_dependency_then_every_real_stage(
    tmp_path: Path,
) -> None:
    """Catch production and reproduction maintaining separate drifting commands."""

    assert hasattr(analysis_io, "planned_analysis_commands")
    plan = analysis_io.planned_analysis_commands(
        code_root=tmp_path / "project",
        data_root=tmp_path / "data root",
        output_root=tmp_path / "output root",
    )
    assert [item.stage for item in plan] == [None, *planned_analysis_stages()]
    assert plan[0].command[-2:] == ("analysis-deps", "--verify")
    assert plan[4].command[-7:] == (
        "analysis-ingest-models",
        "--kind",
        "lp",
        "--data-root",
        str(tmp_path / "data root"),
        "--output-root",
        str(tmp_path / "output root"),
    )
    assert plan[8].command[-7:] == (
        "analysis-ingest-models",
        "--kind",
        "threshold-and-iv-audit",
        "--data-root",
        str(tmp_path / "data root"),
        "--output-root",
        str(tmp_path / "output root"),
    )


def test_production_orchestrator_executes_shared_plan_fail_fast(
    tmp_path: Path,
) -> None:
    """Catch production bypassing the command table used by reproduction."""

    assert hasattr(analysis_io, "run_analysis_command_plan")
    observed: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def runner(command: tuple[str, ...], environment: dict[str, str]) -> None:
        observed.append((command, environment))
        if command[3:4] == ("analysis-ingest-models",) and "lp" in command:
            raise subprocess.CalledProcessError(9, command)

    with pytest.raises(subprocess.CalledProcessError):
        analysis_io.run_analysis_command_plan(
            code_root=tmp_path / "project",
            data_root=tmp_path / "data",
            output_root=tmp_path / "output",
            command_runner=runner,
        )
    assert [item[0] for item in observed] == [
        item.command
        for item in analysis_io.planned_analysis_commands(
            code_root=tmp_path / "project",
            data_root=tmp_path / "data",
            output_root=tmp_path / "output",
        )[:5]
    ]
    assert not any(
        "GREEN_DEBT_ANALYSIS_REPRO_BROKER_FD" in environment
        or "GREEN_DEBT_ANALYSIS_REPRO_SECRET_FD" in environment
        for _command, environment in observed
    )


def test_offline_runner_contains_no_network_or_dependency_action() -> None:
    """Catch an offline analysis command that installs, restores, or downloads."""

    script = (ROOT / "03_代码/bin/run-analysis").read_text(encoding="utf-8")
    assert "analysis-run" in script
    assert "analysis-deps --initialize" not in script
    assert "analysis-deps --restore" not in script
    assert "curl " not in script
    assert "wget " not in script
    assert "install.packages" not in script
    assert "renv::restore" not in script


def test_runner_executes_real_cli_arguments_in_exact_fail_fast_order(
    tmp_path: Path,
) -> None:
    """Catch the one-click shell bypassing the shared Python orchestrator."""

    data = tmp_path / "authoritative data"
    output = tmp_path / "formal output"
    completed, calls, _ = _run_fake(
        tmp_path,
        "--data-root",
        str(data),
        "--output-root",
        str(output),
    )
    assert completed.returncode == 0
    assert calls == [[
        "python",
        "-m",
        "green_debt.cli",
        "analysis-run",
        "--data-root",
        str(data),
        "--output-root",
        str(output),
    ]]


def test_runner_reaches_combined_ingest_report_and_atomic_audit_last(
    tmp_path: Path,
) -> None:
    """Catch split audit ingestion or any manifest publication before final audit."""

    data = tmp_path / "data"
    output = tmp_path / "output"
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], _environment: dict[str, str]) -> None:
        assert not (output / "run_manifest.json").exists()
        calls.append(command)
        if command[3:4] == ("analysis-output-audit",):
            output.mkdir(parents=True)
            (output / "run_manifest.json").write_text("{}\n", encoding="utf-8")

    analysis_io.run_analysis_command_plan(
        code_root=tmp_path / "project",
        data_root=data,
        output_root=output,
        command_runner=runner,
    )
    plan = analysis_io.planned_analysis_commands(
        code_root=tmp_path / "project", data_root=data, output_root=output
    )
    assert calls == [item.command for item in plan]
    assert calls[-4][3] == "shift-share"
    assert calls[-3][3:6] == (
        "analysis-ingest-models", "--kind", "threshold-and-iv-audit"
    )
    assert calls[-2][3] == "report"
    assert calls[-1][3] == "analysis-output-audit"
    assert (output / "run_manifest.json").is_file()


def test_runner_rejects_missing_values_unknown_arguments_and_missing_data_root(
    tmp_path: Path,
) -> None:
    """Catch argument shifting that silently treats an option as a path."""

    cases = (
        ((), "--data-root is required"),
        (("--data-root",), "missing --data-root value"),
        (("--output-root",), "missing --output-root value"),
        (("--wat", "x"), "unknown argument: --wat"),
    )
    for index, (arguments, expected) in enumerate(cases):
        case_root = tmp_path / str(index)
        case_root.mkdir()
        completed, calls, _ = _run_fake(case_root, *arguments)
        assert completed.returncode == 2
        assert expected in completed.stderr
        assert calls == []


def test_runner_failure_before_audit_does_not_create_success_manifest(
    tmp_path: Path,
) -> None:
    """Catch a shell runner that predeclares success before reporting passes."""

    output = tmp_path / "output"
    def runner(command: tuple[str, ...], _environment: dict[str, str]) -> None:
        if command[3:4] == ("report",):
            raise subprocess.CalledProcessError(9, command)
        if command[3:4] == ("analysis-output-audit",):
            output.mkdir(parents=True)
            (output / "run_manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(subprocess.CalledProcessError):
        analysis_io.run_analysis_command_plan(
            code_root=tmp_path / "project",
            data_root=tmp_path / "data",
            output_root=output,
            command_runner=runner,
        )
    assert not (output / "run_manifest.json").exists()
