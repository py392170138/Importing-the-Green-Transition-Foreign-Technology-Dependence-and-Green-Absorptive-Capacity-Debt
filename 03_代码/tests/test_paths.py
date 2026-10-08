from pathlib import Path

import pytest

from green_debt.cli import build_parser
from green_debt.paths import resolve_project_paths


def test_code_and_data_roots_are_distinct_and_bounded(tmp_path: Path) -> None:
    code = tmp_path / "worktree"
    data = tmp_path / "shared-data"
    code.mkdir()
    data.mkdir()

    paths = resolve_project_paths(code, data)

    assert paths.code_root == code.resolve()
    assert paths.data_root == data.resolve()
    assert paths.raw == data.resolve() / "04_原始数据"
    assert paths.normalized == data.resolve() / "05_中间数据/normalized"
    assert paths.harmonized == data.resolve() / "05_中间数据/harmonized"
    assert paths.measures == data.resolve() / "05_中间数据/measures"
    assert paths.analysis == data.resolve() / "05_中间数据/analysis"
    assert paths.schemas == data.resolve() / "05_中间数据/schemas"
    assert paths.manifests == data.resolve() / "05_中间数据/manifests"
    assert paths.scratch == data.resolve() / "05_中间数据/_tmp"
    assert paths.audits == code.resolve() / "06_结果"


def test_path_resolution_is_pure_and_rejects_a_file_data_root(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    missing_data = tmp_path / "not-created"

    paths = resolve_project_paths(code, missing_data)

    assert paths.data_root == missing_data.resolve()
    assert not missing_data.exists()

    data_file = tmp_path / "data.txt"
    data_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="data_root must be a directory"):
        resolve_project_paths(code, data_file)


@pytest.mark.parametrize("command", ["openalex-topic-audit", "science-audit"])
def test_audit_commands_accept_the_shared_data_root(
    command: str, tmp_path: Path
) -> None:
    argv = [command, "--data-root", str(tmp_path)]
    if command == "science-audit":
        argv.extend(["--counts-csv", str(tmp_path / "counts.csv")])

    args = build_parser().parse_args(argv)

    assert args.data_root == tmp_path
