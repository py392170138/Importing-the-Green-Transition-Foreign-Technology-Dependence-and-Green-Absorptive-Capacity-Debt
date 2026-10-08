from pathlib import Path
import json
import os
import subprocess


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "双击开始夜间下载.command"


def test_double_click_launcher_is_executable_and_prints_frozen_plan() -> None:
    assert LAUNCHER.is_file()
    assert os.access(LAUNCHER, os.X_OK)

    completed = subprocess.run(
        [str(LAUNCHER), "--print-plan"],
        check=True,
        capture_output=True,
        text=True,
    )

    plan = json.loads(completed.stdout.splitlines()[-1])
    assert plan["batch"] == "night_core"
    assert plan["source_ids"] == ["baci_hs96", "baci_hs07"]
    assert plan["projected_working_bytes"] == 7 * 1024**3
    assert plan["downloads_response_bodies"] is False


def test_launcher_never_changes_shadowrocket_or_macos_network_settings() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")

    forbidden_mutations = (
        "networksetup",
        "scutil --nc stop",
        "defaults write",
        "killall Shadowrocket",
        "osascript",
    )
    assert all(command not in script for command in forbidden_mutations)
    assert "03_代码/bin/direct-python" in script
    assert "acquire-batch" in script
    assert "night_core" in script
