import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from green_debt.checkpoints import (
    _CHECKPOINT_REQUIREMENTS,
    _approval_gate_manifest_scope,
    CheckpointFailure,
    CheckpointReceipt,
    audit_input_artifacts,
    discover_authoritative_manifest_paths,
    evaluate_checkpoint,
    run_checkpoint,
    verify_raw_hash_snapshot,
)
from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    write_authoritative_table,
)
from green_debt.cli import build_parser, main
from green_debt.paths import resolve_project_paths
from green_debt.storage import GIB, LayerUsage, measure_layer_usage


def test_checkpoint_fails_at_intermediate_quota() -> None:
    with pytest.raises(CheckpointFailure, match="25 GB intermediate quota"):
        evaluate_checkpoint(
            number=1,
            required_manifests=("taxonomy",),
            present_manifests={"taxonomy"},
            project_bytes=5 * GIB,
            intermediate_bytes=25 * GIB,
            duplicate_keys=0,
            raw_hash_mismatches=0,
        )


@pytest.mark.parametrize(
    ("project_bytes", "filesystem_free_bytes", "message"),
    (
        (120 * GIB, 31 * GIB, "120 GB project hard stop"),
        (5 * GIB, 29 * GIB, "30 GB filesystem reserve"),
    ),
)
def test_checkpoint_enforces_soft_stop_and_filesystem_reserve_boundaries(
    project_bytes: int, filesystem_free_bytes: int, message: str
) -> None:
    with pytest.raises(CheckpointFailure, match=message):
        evaluate_checkpoint(
            number=2,
            required_manifests=(),
            present_manifests=set(),
            project_bytes=project_bytes,
            intermediate_bytes=1 * GIB,
            duplicate_keys=0,
            raw_hash_mismatches=0,
            filesystem_free_bytes=filesystem_free_bytes,
        )


def test_checkpoint2_requires_scaled_and_constructed_gad_authorities() -> None:
    assert set(_CHECKPOINT_REQUIREMENTS[2]) == {"gad_scaled_components", "gad_country_year"}


def test_approval_gate_lineage_scope_excludes_unapproved_downstream_layers(
    tmp_path: Path,
) -> None:
    class Manifest:
        def __init__(self, table_id: str, destination: Path) -> None:
            self.table_id = table_id
            self.destination = str(destination)

    normalized = Manifest(
        "wdi_country_year", tmp_path / "normalized/wdi.parquet"
    )
    provisional = Manifest(
        "provisional_sample", tmp_path / "harmonized/provisional.parquet"
    )
    gad = Manifest("gad_country_year", tmp_path / "measures/gad.parquet")
    scaled = Manifest(
        "gad_scaled_components", tmp_path / "measures/scaled.parquet"
    )
    downstream = Manifest("model_panel", tmp_path / "analysis/panel.parquet")

    assert _approval_gate_manifest_scope(
        1, (normalized, provisional, gad, scaled, downstream)
    ) == (normalized, provisional)
    assert _approval_gate_manifest_scope(
        2, (normalized, provisional, gad, scaled, downstream)
    ) == (provisional, gad, scaled)


def test_checkpoint_fails_closed_on_missing_or_invalid_evidence() -> None:
    with pytest.raises(CheckpointFailure, match="missing manifests: trade"):
        evaluate_checkpoint(
            number=1,
            required_manifests=("taxonomy", "trade"),
            present_manifests={"taxonomy"},
            project_bytes=5 * GIB,
            intermediate_bytes=1 * GIB,
            duplicate_keys=0,
            raw_hash_mismatches=0,
        )
    with pytest.raises(CheckpointFailure, match="duplicate keys"):
        evaluate_checkpoint(
            number=1,
            required_manifests=(),
            present_manifests=set(),
            project_bytes=5 * GIB,
            intermediate_bytes=1 * GIB,
            duplicate_keys=1,
            raw_hash_mismatches=0,
        )
    with pytest.raises(CheckpointFailure, match="raw hash mismatches"):
        evaluate_checkpoint(
            number=1,
            required_manifests=(),
            present_manifests=set(),
            project_bytes=5 * GIB,
            intermediate_bytes=1 * GIB,
            duplicate_keys=0,
            raw_hash_mismatches=1,
        )


def test_checkpoint_receipt_records_a_pass() -> None:
    receipt = evaluate_checkpoint(
        number=1,
        required_manifests=("taxonomy",),
        present_manifests={"taxonomy"},
        project_bytes=5 * GIB,
        intermediate_bytes=1 * GIB,
        duplicate_keys=0,
        raw_hash_mismatches=0,
        test_command="pytest -q",
        test_exit_code=0,
    )

    assert receipt.number == 1
    assert receipt.passed is True
    assert receipt.test_command == "pytest -q"
    assert receipt.test_exit_code == 0


def test_checkpoint_fails_when_manifest_input_hash_is_stale(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("frozen\n", encoding="utf-8")
    artifact = InputArtifact.from_path(source)
    source.write_text("changed\n", encoding="utf-8")

    lineage = audit_input_artifacts((artifact,))

    assert lineage.checked_artifacts == 1
    assert lineage.stale_count == 1
    with pytest.raises(CheckpointFailure, match="stale input artifacts"):
        evaluate_checkpoint(
            number=1,
            required_manifests=(),
            present_manifests=set(),
            project_bytes=5 * GIB,
            intermediate_bytes=1 * GIB,
            duplicate_keys=0,
            raw_hash_mismatches=0,
            stale_input_artifacts=lineage.stale_count,
        )


def test_checkpoint_discovers_partition_sidecars_in_authoritative_layers(
    tmp_path: Path,
) -> None:
    paths = resolve_project_paths(tmp_path / "code", tmp_path / "data")
    normalized = (
        paths.normalized
        / "baci/green_bilateral/year=1996/taxonomy_version=main_hs96.parquet.manifest.json"
    )
    harmonized = (
        paths.harmonized / "sample/provisional_sample.parquet.manifest.json"
    )
    central = paths.manifests / "colliding.manifest.json"
    for path in (normalized, harmonized, central):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    discovered = discover_authoritative_manifest_paths(paths)

    assert discovered == tuple(sorted((normalized, harmonized)))


def test_layer_usage_keeps_all_capacity_buckets_separate(tmp_path: Path) -> None:
    sizes = {
        "04_原始数据/raw.bin": 2,
        "05_中间数据/normalized/a.bin": 3,
        "05_中间数据/harmonized/a.bin": 5,
        "05_中间数据/measures/a.bin": 7,
        "05_中间数据/analysis/a.bin": 11,
        "05_中间数据/_tmp/a.bin": 13,
    }
    for relative, size in sizes.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
    audits = tmp_path / "code" / "06_结果"
    audits.mkdir(parents=True)
    (audits / "audit.json").write_bytes(b"x" * 17)

    usage = measure_layer_usage(tmp_path, audits_root=audits)

    assert usage.raw_bytes == 2
    assert usage.normalized_bytes == 3
    assert usage.harmonized_bytes == 5
    assert usage.measures_bytes == 7
    assert usage.analysis_bytes == 11
    assert usage.scratch_bytes == 13
    assert usage.audit_bytes == 17
    assert usage.intermediate_bytes == 3 + 5 + 7 + 11


def test_raw_snapshot_verifier_reports_missing_and_mismatch(tmp_path: Path) -> None:
    raw = tmp_path / "04_原始数据"
    raw.mkdir()
    (raw / "good.bin").write_bytes(b"good")
    (raw / "changed.bin").write_bytes(b"changed")
    snapshot = raw / "snapshot.txt"
    snapshot.write_text(
        "770e607624d689265ca6c44884d0807d9b054d23c473c106c72be9de08b7376c  ./good.bin\n"
        + "0" * 64
        + "  ./changed.bin\n"
        + "1" * 64
        + "  ./missing.bin\n",
        encoding="utf-8",
    )

    report = verify_raw_hash_snapshot(snapshot, raw)

    assert report.checked_files == 3
    assert report.mismatch_count == 2
    assert report.missing_files == ("missing.bin",)
    assert report.hash_mismatches == ("changed.bin",)


@pytest.mark.parametrize(
    ("argv", "command"),
    [
        (["raw-hash-verify", "--snapshot", "snapshot.txt"], "raw-hash-verify"),
        (["capacity-report"], "capacity-report"),
        (["verify-artifacts", "--layer", "normalized"], "verify-artifacts"),
        (["checkpoint", "--number", "1"], "checkpoint"),
    ],
)
def test_construction_audit_commands_share_data_root(
    argv: list[str], command: str, tmp_path: Path
) -> None:
    args = build_parser().parse_args([*argv, "--data-root", str(tmp_path)])

    assert args.command == command
    assert args.data_root == tmp_path


def test_verify_artifacts_uses_layer_sidecars_not_collision_prone_central_copies(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "05_中间数据/normalized/source/table.parquet"
    contract = TableContract(
        table_id="table",
        schema_version="1.0.0",
        primary_key=("id",),
        columns={"id": "String"},
        units={},
    )
    write_authoritative_table(
        pl.DataFrame({"id": ["A"]}),
        contract,
        destination,
        (),
        BuildIdentity(command="fixture", code_commit="a" * 40),
    )
    central = tmp_path / "05_中间数据/manifests/table.manifest.json"
    stale = json.loads(central.read_text(encoding="utf-8"))
    stale["destination"] = str(
        tmp_path
        / "05_中间数据/_tmp/build.table.deleted/data/05_中间数据/normalized/source/table.parquet"
    )
    central.write_text(json.dumps(stale), encoding="utf-8")

    assert main(
        [
            "verify-artifacts",
            "--layer",
            "normalized",
            "--data-root",
            str(tmp_path),
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "layer": "normalized",
        "manifest_count": 1,
        "status": "valid",
        "table_ids": ["table"],
    }


def test_checkpoint_cli_limits_review_gate_to_its_dag_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def run_fixture(
        number: int,
        paths: object,
        *,
        write_evidence: bool,
        approval_gate_verification: bool = False,
    ) -> CheckpointReceipt:
        observed.update(
            {
                "number": number,
                "write_evidence": write_evidence,
                "approval_gate_verification": approval_gate_verification,
            }
        )
        return CheckpointReceipt(
            number=number,
            checked_at_utc="fixture",
            git_commit="a" * 40,
            test_command="pytest -q",
            test_exit_code=0,
            raw_hash_mismatches=0,
            manifest_count=1,
            duplicate_keys=0,
            stale_input_artifacts=0,
            taxonomy_count_mismatches=0,
            unresolved_mappings=0,
            source_audit_failures=0,
            sample_audit_failures=0,
            layer_byte_counts={},
            project_bytes=0,
            intermediate_bytes=0,
            passed=True,
        )

    monkeypatch.setattr("green_debt.cli.run_checkpoint", run_fixture)

    assert main(
        ["checkpoint", "--number", "1", "--data-root", str(tmp_path)]
    ) == 0
    capsys.readouterr()
    assert observed == {
        "number": 1,
        "write_evidence": True,
        "approval_gate_verification": True,
    }


def test_checkpoint3_capacity_is_measured_after_mutable_audits_settle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code_root = tmp_path / "code"
    data_root = tmp_path / "data"
    code_root.mkdir()
    paths = resolve_project_paths(code_root, data_root)
    paths.raw.mkdir(parents=True)
    (paths.raw / "SHA256SUMS_fixture.txt").write_text("fixture\n", encoding="utf-8")

    state = {"audits_settled": False}

    def usage_fixture(*args: object, **kwargs: object) -> LayerUsage:
        return LayerUsage(
            raw_bytes=1,
            normalized_bytes=2,
            harmonized_bytes=3,
            measures_bytes=4,
            analysis_bytes=5,
            scratch_bytes=0,
            metadata_bytes=6,
            audit_bytes=90 if state["audits_settled"] else 97,
            filesystem_free_bytes=31 * GIB,
        )

    def checkpoint3_reports_fixture(*args: object, **kwargs: object) -> tuple[object, ...]:
        state["audits_settled"] = True
        return (
            {},
            {},
            {},
            pl.DataFrame({"section": [], "metric": []}, schema={"section": pl.String, "metric": pl.String}),
            pl.DataFrame({"criterion": []}, schema={"criterion": pl.String}),
            "delivery\n",
            ("frozen",),
        )

    manifest = SimpleNamespace(
        table_id="fixture",
        duplicate_primary_keys=0,
        input_artifacts=(),
    )
    monkeypatch.setattr(
        "green_debt.checkpoints.discover_authoritative_manifest_paths",
        lambda paths: (Path("fixture.manifest.json"),),
    )
    monkeypatch.setattr("green_debt.checkpoints.verify_manifest", lambda path: manifest)
    monkeypatch.setattr(
        "green_debt.checkpoints.audit_input_artifacts",
        lambda artifacts: SimpleNamespace(stale_count=0),
    )
    monkeypatch.setattr(
        "green_debt.checkpoints.verify_raw_hash_snapshot",
        lambda snapshot, raw: SimpleNamespace(mismatch_count=0),
    )
    monkeypatch.setattr("green_debt.checkpoints.measure_layer_usage", usage_fixture)
    monkeypatch.setattr(
        "green_debt.checkpoints._checkpoint1_source_coverage",
        lambda *args: (pl.DataFrame(), 0, 0, 0, 0),
    )
    monkeypatch.setattr(
        "green_debt.checkpoints._checkpoint2_gad_audit",
        lambda *args: (pl.DataFrame(), pl.DataFrame(), {}),
    )
    monkeypatch.setattr(
        "green_debt.checkpoints._checkpoint3_reports",
        checkpoint3_reports_fixture,
    )
    monkeypatch.setattr(
        "green_debt.checkpoints.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="1 passed in 0.01s\n"),
    )
    monkeypatch.setattr(
        "green_debt.checkpoints.evaluate_checkpoint",
        lambda **kwargs: CheckpointReceipt(
            number=3,
            checked_at_utc="fixture",
            git_commit="a" * 40,
            test_command="pytest -q",
            test_exit_code=0,
            raw_hash_mismatches=0,
            manifest_count=1,
            duplicate_keys=0,
            stale_input_artifacts=0,
            taxonomy_count_mismatches=0,
            unresolved_mappings=0,
            source_audit_failures=0,
            sample_audit_failures=0,
            layer_byte_counts={},
            project_bytes=0,
            intermediate_bytes=0,
            passed=True,
        ),
    )
    monkeypatch.setattr("green_debt.checkpoints._git_commit", lambda root: "a" * 40)
    monkeypatch.setattr("green_debt.checkpoints.checkpoint3_facts_from_reports", lambda **kwargs: {})
    monkeypatch.setattr("green_debt.checkpoints.validate_checkpoint3_facts", lambda facts: None)

    def validate_support_fixture(
        code_root: Path,
        *,
        results_and_iv: pl.DataFrame,
        leakage: pl.DataFrame,
        capacity_projection: dict[str, object],
        delivery_note: str,
    ) -> None:
        assert capacity_projection["usage_bytes"]["audits"] == 90

    monkeypatch.setattr(
        "green_debt.checkpoints.validate_checkpoint3_supporting_evidence",
        validate_support_fixture,
    )
    monkeypatch.setattr(
        "green_debt.checkpoints.validate_checkpoint3_bundle",
        lambda *args, **kwargs: {"test_counts": {"summary": "1 passed"}},
    )

    run_checkpoint(3, paths, write_evidence=False)
