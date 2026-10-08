import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_double_click_dependency_launcher_is_safe_and_has_offline_plan() -> None:
    launcher = ROOT / "双击安装分析依赖.command"
    assert launcher.is_file() and os.access(launcher, os.X_OK)
    text = launcher.read_text(encoding="utf-8")
    assert all(
        value not in text
        for value in (
            "networksetup",
            "scutil --nc stop",
            "killall Shadowrocket",
            "defaults write",
            "osascript",
        )
    )
    completed = subprocess.run(
        [str(launcher), "--print-plan"],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout.splitlines()[-1])
    assert plan == {
        "downloads_response_bodies": False,
        "hosts": [
            "api.github.com",
            "cloud.r-project.org",
            "codeload.github.com",
        ],
        "projected_working_bytes": 2 * 1024**3,
        "status": "plan_only",
    }


def test_launcher_has_valid_posix_shell_syntax() -> None:
    launcher = ROOT / "双击安装分析依赖.command"
    subprocess.run(["/bin/sh", "-n", str(launcher)], check=True)


def test_r_bootstrap_freezes_exact_analysis_package_set() -> None:
    bootstrap = (ROOT / "03_代码/R/bootstrap_analysis_env.R").read_text(
        encoding="utf-8"
    )
    package_block = re.search(
        r"packages <- c\((.*?)\)\n",
        bootstrap,
        flags=re.DOTALL,
    )
    assert package_block is not None
    assert re.findall(r'"([A-Za-z0-9.]+)"', package_block.group(1)) == [
        "arrow",
        "data.table",
        "fixest",
        "ivreg",
        "clubSandwich",
        "fwildclusterboot",
        "dqrng",
        "ggplot2",
        "ragg",
        "svglite",
        "jsonlite",
        "digest",
        "modelsummary",
        "testthat",
    ]
    assert "336bb574eba169ac0183317f01d0564791d8122f" in bootstrap
    assert re.search(
        r'renv::install\(\s*packages\[packages != "fwildclusterboot"\]',
        bootstrap,
    )
    assert 'renv::install("summclust@0.7.2"' in bootstrap
    assert re.search(
        r'renv::install\(\s*paste0\("s3alfisc/fwildclusterboot@", '
        r"fwildclusterboot_commit\)",
        bootstrap,
    )
    assert "dependencies = TRUE" not in bootstrap
    for dependency_reference in (
        "arrow::read_parquet",
        "data.table::data.table",
        "fixest::feols",
        "ivreg::ivreg",
        "clubSandwich::vcovCR",
        "fwildclusterboot::boottest",
        "dqrng::dqset.seed",
        "ggplot2::ggplot",
        "ragg::agg_png",
        "svglite::svglite",
        "jsonlite::toJSON",
        "digest::digest",
        "modelsummary::modelsummary",
        "testthat::test_dir",
    ):
        assert dependency_reference in bootstrap
    verify_branch = bootstrap.split("} else {", 1)[1]
    assert verify_branch.index("renv::load(project = root)") < (
        verify_branch.index("renv::status(project = root)")
    )


def test_only_local_r_libraries_are_ignored_and_autoloader_is_exact() -> None:
    ignore_lines = set(
        (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    )
    assert {
        ".r-bootstrap-library/",
        "renv/library/",
        "renv/staging/",
        "renv/python/",
    }.issubset(ignore_lines)
    assert "renv.lock" not in ignore_lines
    assert ".Rprofile" not in ignore_lines
    assert "renv/activate.R" not in ignore_lines
    assert (ROOT / ".Rprofile").read_text(encoding="utf-8") == (
        'source("renv/activate.R")\n'
    )
