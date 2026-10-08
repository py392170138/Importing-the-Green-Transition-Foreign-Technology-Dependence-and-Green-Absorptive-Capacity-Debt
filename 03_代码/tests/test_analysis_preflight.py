import json
from pathlib import Path

import pytest

import green_debt.analysis_io as analysis_io
import green_debt.cli as cli
from green_debt.analysis_io import analysis_preflight
from green_debt.storage import GIB


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, bytes | None], ...]:
    return tuple(
        (
            str(path.relative_to(root)),
            "directory" if path.is_dir() else "file",
            path.read_bytes() if path.is_file() else None,
        )
        for path in sorted(root.rglob("*"))
    )


def test_preflight_is_read_only_and_reports_the_complete_registered_grid(
    project_fixture, monkeypatch
) -> None:
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)
    output_root = project_fixture.code_root / "06_结果/analysis"
    before_code = _tree_snapshot(project_fixture.code_root)
    before_data = _tree_snapshot(project_fixture.data_root)

    report = analysis_preflight(
        project_fixture.code_root,
        project_fixture.data_root,
        output_root,
    )

    assert report.status == "ready"
    assert report.table_id == "model_panel"
    assert report.rows == 1
    assert report.authority_tables == 3
    assert report.confirmatory_cells == 39
    assert report.registered_cells == 85
    assert report.write_count == 0
    assert len(report.input_authority_hash) == 64
    assert len(report.evidence_policy_sha256) == 64
    assert not output_root.exists()
    assert _tree_snapshot(project_fixture.code_root) == before_code
    assert _tree_snapshot(project_fixture.data_root) == before_data


def test_evidence_policy_is_tracked_machine_readable_and_unresolved() -> None:
    loader = getattr(analysis_io, "load_evidence_policy", None)
    assert callable(loader), "analysis_io must expose the frozen evidence policy"
    policy, policy_sha256 = loader(
        Path(__file__).resolve().parents[2] / "config/evidence_policy.json"
    )
    concentration = policy["concentration"]
    assert len(policy_sha256) == 64
    assert concentration["status"] == "unresolved_no_preregistered_cutoff"
    assert concentration["unresolved_policy"] == "exploratory"
    assert concentration["cutoffs"] == {"hhi": None, "top1_share": None}
    assert concentration["aggregation"] == "diagnostic_disclosure_only"


@pytest.mark.parametrize(
    "relative_output",
    (
        Path("05_中间数据/analysis-output"),
        Path("."),
    ),
)
def test_preflight_rejects_output_overlap_with_read_only_intermediate(
    project_fixture, monkeypatch, relative_output: Path
) -> None:
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)

    with pytest.raises(ValueError, match="output root.*05_中间数据"):
        analysis_preflight(
            project_fixture.code_root,
            project_fixture.data_root,
            project_fixture.data_root / relative_output,
        )


def test_preflight_rejects_output_outside_frozen_analysis_root(
    project_fixture, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)
    with pytest.raises(ValueError, match="frozen outputs.root"):
        analysis_preflight(
            project_fixture.code_root,
            project_fixture.data_root,
            tmp_path / "unauthorized-analysis-output",
        )


def test_preflight_rejects_symlink_alias_for_frozen_output(
    project_fixture, monkeypatch
) -> None:
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)
    frozen = project_fixture.code_root / "06_结果/analysis"
    frozen.mkdir(parents=True)
    alias = project_fixture.code_root / "analysis-output-alias"
    alias.symlink_to(frozen, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        analysis_preflight(
            project_fixture.code_root,
            project_fixture.data_root,
            alias,
        )


def test_combined_usage_counts_external_output_once(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    external_output = tmp_path / "external-output"
    nested_output = data_root / "06_结果/analysis"
    data_root.mkdir()
    external_output.mkdir()
    nested_output.mkdir(parents=True)
    (data_root / "authority.bin").write_bytes(b"abc")
    (nested_output / "nested.bin").write_bytes(b"12345")
    (external_output / "external.bin").write_bytes(b"1234567")

    assert analysis_io.combined_project_usage_bytes(data_root, external_output) == 15
    assert analysis_io.combined_project_usage_bytes(data_root, nested_output) == 8


def test_preflight_fails_closed_at_output_quota(project_fixture, monkeypatch) -> None:
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)
    monkeypatch.setattr(
        analysis_io,
        "directory_usage_bytes",
        lambda _path: 10 * GIB,
    )

    with pytest.raises(RuntimeError, match="10 GB analysis output quota"):
        analysis_preflight(
            project_fixture.code_root,
            project_fixture.data_root,
            project_fixture.code_root / "06_结果/analysis",
        )


def test_preflight_fails_closed_at_absolute_project_limit(
    project_fixture, monkeypatch
) -> None:
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)
    monkeypatch.setattr(
        analysis_io,
        "project_usage_bytes",
        lambda _path: 150 * GIB,
    )

    with pytest.raises(RuntimeError, match="150 GB absolute project limit"):
        analysis_preflight(
            project_fixture.code_root,
            project_fixture.data_root,
            project_fixture.code_root / "06_结果/analysis",
        )


def test_preflight_counts_external_output_toward_absolute_limit(
    project_fixture, monkeypatch
) -> None:
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)
    monkeypatch.setattr(
        analysis_io,
        "directory_usage_bytes",
        lambda _path: 6 * GIB,
    )
    monkeypatch.setattr(
        analysis_io,
        "project_usage_bytes",
        lambda _path: 145 * GIB,
    )

    with pytest.raises(RuntimeError, match="150 GB absolute project limit"):
        analysis_preflight(
            project_fixture.code_root,
            project_fixture.data_root,
            project_fixture.code_root / "06_结果/analysis",
        )


def test_analysis_preflight_cli_prints_one_read_only_json_object(
    project_fixture, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)
    monkeypatch.setattr(cli, "PROJECT_ROOT", project_fixture.code_root)

    exit_code = cli.main(
        ["analysis-preflight", "--data-root", str(project_fixture.data_root)]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "ready"
    assert payload["rows"] == 1
    assert payload["registered_cells"] == 85
    assert payload["write_count"] == 0
