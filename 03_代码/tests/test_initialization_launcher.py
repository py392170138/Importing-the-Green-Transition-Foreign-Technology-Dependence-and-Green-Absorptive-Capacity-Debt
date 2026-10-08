from pathlib import Path
import os
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "双击补充初始化数据.command"


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def test_launcher_executes_only_the_three_bounded_cli_steps(tmp_path: Path) -> None:
    assert LAUNCHER.is_file()
    assert os.access(LAUNCHER, os.X_OK)
    project = tmp_path / "project"
    project.mkdir()
    copied = project / LAUNCHER.name
    shutil.copy2(LAUNCHER, copied)
    call_log = tmp_path / "calls.log"
    mutation_log = tmp_path / "mutations.log"
    _write_executable(
        project / ".venv/bin/python",
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$GAD_TEST_CALL_LOG\"\nexit 0\n",
    )
    traps = tmp_path / "traps"
    for command in ("networksetup", "scutil", "osascript"):
        _write_executable(
            traps / command,
            "#!/bin/sh\nprintf '%s\\n' \"$0 $*\" >> \"$GAD_TEST_MUTATION_LOG\"\nexit 99\n",
        )
    environment = {
        **os.environ,
        "PATH": f"{traps}:{os.environ['PATH']}",
        "GAD_TEST_CALL_LOG": str(call_log),
        "GAD_TEST_MUTATION_LOG": str(mutation_log),
    }

    completed = subprocess.run(
        [str(copied)],
        input="\n",
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert completed.returncode == 0
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert calls == [
        "-m green_debt.cli acquire-initialization-supplements "
        f"--data-root {project} --route-exception-authorization "
        "user_authorized_2026-08-22",
        f"-m green_debt.cli audit-initialization-supplements --data-root {project}",
        "-m green_debt.cli raw-hash-snapshot "
        f"--data-root {project} --output "
        f"{project}/04_原始数据/SHA256SUMS_20260823.txt",
    ]
    assert not mutation_log.exists()
    assert "初始化补充数据已完成并通过审计" in completed.stdout
