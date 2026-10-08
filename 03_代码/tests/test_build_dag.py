import json
from pathlib import Path
import subprocess

import polars as pl
import pytest

from green_debt.build import (
    ApprovalGate,
    ApprovalRequired,
    BuildGraph,
    BuildNode,
    BuildController,
    BuildPublicationError,
    BuildStage,
    manifest_is_current,
    publish_version,
    registered_build_graph,
    validate_built_manifest_identity,
)
from green_debt.cli import _code_version, build_parser
from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    verify_manifest,
    write_authoritative_table,
)


def test_graph_orders_dependencies_and_skips_verified_outputs() -> None:
    calls: list[str] = []
    graph = BuildGraph(
        [
            BuildNode("normalized", (), lambda: calls.append("normalized")),
            BuildNode("measures", ("normalized",), lambda: calls.append("measures")),
            BuildNode("analysis", ("measures",), lambda: calls.append("analysis")),
        ]
    )

    assert graph.order("analysis") == ("normalized", "measures", "analysis")
    report = graph.run("analysis", manifest_is_current=lambda node: node == "normalized")

    assert calls == ["measures", "analysis"]
    assert [(item.name, item.status) for item in report] == [
        ("normalized", "current"),
        ("measures", "built"),
        ("analysis", "built"),
    ]


def test_graph_rejects_unknown_dependencies_and_cycles() -> None:
    with pytest.raises(ValueError, match="unknown dependency"):
        BuildGraph([BuildNode("analysis", ("missing",), lambda: None)])
    with pytest.raises(ValueError, match="cycle"):
        BuildGraph(
            [
                BuildNode("one", ("two",), lambda: None),
                BuildNode("two", ("one",), lambda: None),
            ]
        )


def test_stale_parent_forces_descendant_rebuild_even_if_descendant_claims_current() -> None:
    calls: list[str] = []
    graph = BuildGraph(
        [
            BuildNode("parent", (), lambda: calls.append("parent")),
            BuildNode("child", ("parent",), lambda: calls.append("child")),
        ]
    )

    report = graph.run("child", manifest_is_current=lambda node: node == "child")

    assert calls == ["parent", "child"]
    assert report[-1].reason == "upstream_rebuilt"


@pytest.mark.parametrize(
    ("field", "current"),
    (
        ("input_hashes", ("new-raw",)),
        ("parent_manifest_hashes", ("new-parent",)),
        ("config_hashes", ("new-config",)),
        ("contract_hashes", ("new-contract",)),
        ("taxonomy_hashes", ("new-taxonomy",)),
        ("scaler_hashes", ("new-scaler",)),
    ),
)
def test_manifest_hash_changes_invalidate_the_node(field: str, current: tuple[str, ...]) -> None:
    manifest = {
        "input_hashes": ["old-raw"],
        "parent_manifest_hashes": ["old-parent"],
        "config_hashes": ["old-config"],
        "contract_hashes": ["old-contract"],
        "taxonomy_hashes": ["old-taxonomy"],
        "scaler_hashes": ["old-scaler"],
        "executable_input_hashes": {"builder.py": "old-code"},
    }

    assert manifest_is_current(manifest, **{f"current_{field}": current}) is False


def test_node_code_changes_invalidate_but_readme_and_audits_do_not() -> None:
    manifest = {
        "input_hashes": [],
        "executable_input_hashes": {
            "03_code/src/builder.py": "builder-hash",
            "config/frozen.yaml": "config-hash",
        },
    }

    assert manifest_is_current(
        manifest,
        current_executable_input_hashes={
            "03_code/src/builder.py": "builder-hash",
            "config/frozen.yaml": "config-hash",
        },
    )
    assert not manifest_is_current(
        manifest,
        current_executable_input_hashes={
            "03_code/src/builder.py": "changed",
            "config/frozen.yaml": "config-hash",
        },
    )
    # README and compact audit files are deliberately absent from the node's
    # executable-input map, so changing them cannot make a data node stale.
    assert manifest_is_current(
        manifest,
        current_executable_input_hashes={
            "03_code/src/builder.py": "builder-hash",
            "config/frozen.yaml": "config-hash",
        },
        changed_non_executable_paths=("README.md", "06_results/checkpoint.csv"),
    )


def test_manifest_verifier_checks_output_and_schema_before_allowing_skip(tmp_path: Path) -> None:
    output = tmp_path / "table.parquet"
    schema = tmp_path / "table.schema.json"
    output.write_bytes(b"table")
    schema.write_bytes(b"schema")
    manifest = {
        "input_hashes": [],
        "outputs": [
            {
                "path": str(output),
                "sha256": "0d4fc4a78d3706edccafb665a8b2fdd9309e82c78625bb0f2b8e7bb9e1c4d21c",
                "schema_path": str(schema),
                "schema_sha256": "df0ad6e43880f09c90ebf95f19110178aba6890df0010ebda7485029e2b543b4",
            }
        ],
    }

    assert manifest_is_current(manifest)
    output.write_bytes(b"tampered")
    assert not manifest_is_current(manifest)


def test_atomic_publication_updates_current_only_after_success(tmp_path: Path) -> None:
    registry = tmp_path / "registry"

    first = publish_version(
        registry,
        "analysis",
        "build-one",
        lambda version: (version / "complete.txt").write_text("one", encoding="utf-8"),
    )
    assert first.build_id == "build-one"
    assert json.loads((registry / "analysis/CURRENT.json").read_text())["build_id"] == "build-one"

    def fail_after_partial(version: Path) -> None:
        (version / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("stage failed")

    with pytest.raises(BuildPublicationError, match="stage failed"):
        publish_version(registry, "analysis", "build-two", fail_after_partial)

    assert json.loads((registry / "analysis/CURRENT.json").read_text())["build_id"] == "build-one"
    assert (first.version_path / "complete.txt").read_text(encoding="utf-8") == "one"
    assert not (registry / "analysis/versions/build-two").exists()


def test_build_status_explains_stale_and_blocked_descendants() -> None:
    graph = BuildGraph(
        [
            BuildNode("raw_adapter", (), lambda: None),
            BuildNode("normalized", ("raw_adapter",), lambda: None),
            BuildNode("analysis", ("normalized",), lambda: None),
        ]
    )

    status = graph.status(
        "analysis",
        manifest_status=lambda name: (name != "normalized", "raw_hash_changed"),
    )

    assert [(item.name, item.status, item.reason) for item in status] == [
        ("raw_adapter", "current", "raw_hash_changed"),
        ("normalized", "stale", "raw_hash_changed"),
        ("analysis", "blocked", "dependency_stale:normalized"),
    ]


def test_registered_graph_covers_every_direct_l1_l4_authority_without_acquisition(
    tmp_path: Path,
) -> None:
    commands: list[tuple[str, ...]] = []
    graph = registered_build_graph(
        code_root=tmp_path / "code",
        data_root=tmp_path / "data",
        command_runner=lambda command: commands.append(command),
    )

    assert graph.order("checkpoint3") == (
        "taxonomy",
        "economies",
        "baci_hs96",
        "baci_hs07",
        "wdi",
        "irena",
        "openalex",
        "ilostat",
        "policy",
        "tiva",
        "provisional_sample",
        "checkpoint1",
        "complexity",
        "trade_components",
        "gsci",
        "supplier_capability",
        "tiva_measures",
        "gad_scaler",
        "gad",
        "checkpoint2",
        "outcomes",
        "instruments",
        "final_sample",
        "analysis_panels",
        "checkpoint3",
    )
    output_ids = {
        output_id
        for node in graph.nodes
        for output_id in node.output_ids
    }
    assert {
        "wdi_country_year",
        "irena_country_year",
        "openalex_country_year",
        "ilostat_skill",
        "policy_country_year",
        "tiva_activity_year",
        "tiva_activity_weights",
        "provisional_sample",
        "gpci_product_year",
        "trade_components_raw",
        "gsci_raw",
        "supplier_raw",
        "tiva_measures",
        "tiva_bounded_robustness",
        "gad_scaled_components",
        "gad_country_year",
        "outcomes_country_year",
        "outcomes_product_year",
        "iv_baseline_shares",
        "iv_partner_shocks",
        "iv_country_year",
        "final_sample",
        "regression_bounds",
        "giu_outcome_scalers",
        "model_panel",
    } <= output_ids
    for node in graph.nodes:
        node.build()
    assert commands
    assert not any("acquire" in argument for command in commands for argument in command)


@pytest.mark.parametrize(
    ("argv", "command"),
    (
        (["build", "--target", "checkpoint3", "--data-root", "/tmp/data"], "build"),
        (["build-status", "--data-root", "/tmp/data"], "build-status"),
        (
            [
                "reproduce-check",
                "--data-root",
                "/tmp/data",
                "--scratch-root",
                "/tmp/data/05_中间数据/_tmp/reproduce.ABC123",
            ],
            "reproduce-check",
        ),
        (
            [
                "cleanup-reproduction",
                "--receipt",
                "/tmp/data/05_中间数据/_tmp/reproduce.ABC123/reproduction_receipt.json",
            ],
            "cleanup-reproduction",
        ),
    ),
)
def test_orchestration_commands_have_the_frozen_cli_shape(
    argv: list[str], command: str
) -> None:
    parsed = build_parser().parse_args(argv)

    assert parsed.command == command


def test_controller_adopts_reviewed_authorities_then_skips_and_propagates_staleness(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    code_root = tmp_path / "code"
    parent_output = data_root / "parent.bin"
    child_output = data_root / "child.bin"
    parent_code = code_root / "parent.py"
    child_code = code_root / "child.py"
    for path, value in (
        (parent_output, "reviewed-parent"),
        (child_output, "reviewed-child"),
        (parent_code, "parent-code"),
        (child_code, "child-code"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    calls: list[str] = []

    def rebuild_parent(stage: BuildStage) -> None:
        calls.append("parent")
        staged = stage.path_for(parent_output)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text("rebuilt-parent", encoding="utf-8")

    def rebuild_child(stage: BuildStage) -> None:
        calls.append("child")
        staged = stage.path_for(child_output)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text("rebuilt-child", encoding="utf-8")

    controller = BuildController(
        BuildGraph(
            [
                BuildNode(
                    "parent",
                    (),
                    lambda: None,
                    authority_files=(parent_output,),
                    executable_inputs=(parent_code,),
                    staging_builder=rebuild_parent,
                ),
                BuildNode(
                    "child",
                    ("parent",),
                    lambda: None,
                    authority_files=(child_output,),
                    executable_inputs=(child_code,),
                    staging_builder=rebuild_child,
                ),
            ]
        ),
        code_root=code_root,
        data_root=data_root,
        implementation_commit="a" * 40,
    )

    adopted = controller.build("child")
    skipped = controller.build("child")
    assert calls == []
    assert [item.status for item in adopted] == ["adopted", "adopted"]
    assert [item.status for item in skipped] == ["current", "current"]

    # The fixed path is only a compatibility mirror after adoption. A real
    # executable input change invalidates the version authority and child.
    parent_code.write_text("parent-code-v2", encoding="utf-8")
    rebuilt = controller.build("child")
    assert calls == ["parent", "child"]
    assert [item.status for item in rebuilt] == ["built", "built"]
    assert rebuilt[-1].reason == "upstream_rebuilt"


def test_legacy_child_rebuilds_when_reviewed_input_lineage_is_stale(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    code_root = tmp_path / "code"
    parent_output = data_root / "05_中间数据/normalized/parent.parquet"
    child_output = data_root / "05_中间数据/measures/child.parquet"
    parent_code = code_root / "parent.py"
    child_code = code_root / "child.py"
    for path in (parent_code, child_code):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("v1", encoding="utf-8")
    parent_contract = TableContract(
        table_id="parent",
        schema_version="1.0.0",
        primary_key=("id",),
        columns={"id": "String", "value": "String"},
        units={},
    )
    child_contract = TableContract(
        table_id="child",
        schema_version="1.0.0",
        primary_key=("id",),
        columns={"id": "String", "parent_value": "String"},
        units={},
    )
    write_authoritative_table(
        pl.DataFrame({"id": ["row"], "value": ["v1"]}),
        parent_contract,
        parent_output,
        (),
        BuildIdentity(command="reviewed-parent", code_commit="0" * 40),
    )
    write_authoritative_table(
        pl.DataFrame({"id": ["row"], "parent_value": ["v1"]}),
        child_contract,
        child_output,
        (InputArtifact.from_path(parent_output),),
        BuildIdentity(command="reviewed-child", code_commit="0" * 40),
    )
    child_calls: list[str] = []

    def rebuild_parent(stage: BuildStage) -> None:
        write_authoritative_table(
            pl.DataFrame({"id": ["row"], "value": ["v2"]}),
            parent_contract,
            stage.path_for(parent_output),
            (),
            BuildIdentity(command="rebuilt-parent", code_commit="a" * 40),
        )

    def rebuild_child(stage: BuildStage) -> None:
        staged_parent = stage.path_for(parent_output)
        parent_value = pl.read_parquet(staged_parent)["value"].item()
        child_calls.append(parent_value)
        write_authoritative_table(
            pl.DataFrame({"id": ["row"], "parent_value": [parent_value]}),
            child_contract,
            stage.path_for(child_output),
            (InputArtifact.from_path(staged_parent),),
            BuildIdentity(command="rebuilt-child", code_commit="a" * 40),
        )

    graph = BuildGraph(
        (
            BuildNode(
                "parent",
                (),
                lambda: None,
                output_ids=("parent",),
                manifest_paths=(
                    parent_output.with_name(f"{parent_output.name}.manifest.json"),
                ),
                executable_inputs=(parent_code,),
                staging_builder=rebuild_parent,
            ),
            BuildNode(
                "child",
                ("parent",),
                lambda: None,
                output_ids=("child",),
                manifest_paths=(
                    child_output.with_name(f"{child_output.name}.manifest.json"),
                ),
                executable_inputs=(child_code,),
                staging_builder=rebuild_child,
            ),
        )
    )
    controller = BuildController(
        graph,
        code_root=code_root,
        data_root=data_root,
        implementation_commit="a" * 40,
    )
    assert controller.build("parent")[0].status == "adopted"
    parent_code.write_text("v2", encoding="utf-8")
    assert controller.build("parent")[0].status == "built"

    result = controller.build("child")

    assert [item.status for item in result] == ["current", "built"]
    assert child_calls == ["v2"]
    assert pl.read_parquet(child_output)["parent_value"].item() == "v2"


def test_controller_ignores_unregistered_readme_but_invalidates_node_code(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    code_root = tmp_path / "code"
    output = data_root / "authority.bin"
    executable = code_root / "builder.py"
    readme = code_root / "README.md"
    for path, value in ((output, "authority"), (executable, "v1"), (readme, "one")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    calls: list[str] = []
    def staged_build(stage: BuildStage) -> None:
        calls.append("built")
        staged = stage.path_for(output)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text("authority", encoding="utf-8")

    node = BuildNode(
        "authority",
        (),
        lambda: None,
        authority_files=(output,),
        executable_inputs=(executable,),
        staging_builder=staged_build,
    )
    controller = BuildController(
        BuildGraph((node,)),
        code_root=code_root,
        data_root=data_root,
        implementation_commit="b" * 40,
    )
    controller.build("authority")
    readme.write_text("two", encoding="utf-8")
    assert controller.status("authority")[0].status == "current"
    evidence_head_controller = BuildController(
        BuildGraph((node,)),
        code_root=code_root,
        data_root=data_root,
        implementation_commit="c" * 40,
    )
    assert evidence_head_controller.status("authority")[0].status == "current"

    executable.write_text("v2", encoding="utf-8")
    assert controller.status("authority")[0].status == "stale"
    controller.build("authority")
    assert calls == ["built"]


def test_controller_stages_multi_output_and_failed_second_output_preserves_authority(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    code_root = tmp_path / "code"
    executable = code_root / "builder.py"
    first = data_root / "05_中间数据/analysis/first.bin"
    second = data_root / "05_中间数据/analysis/second.bin"
    for path, value in ((executable, "v1"), (first, "old-first"), (second, "old-second")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def fail_after_first(stage: BuildStage) -> None:
        staged_first = stage.path_for(first)
        staged_first.parent.mkdir(parents=True, exist_ok=True)
        staged_first.write_text("new-first", encoding="utf-8")
        raise RuntimeError("second output failed")

    node = BuildNode(
        "analysis",
        (),
        lambda: None,
        authority_files=(first, second),
        executable_inputs=(executable,),
        staging_builder=fail_after_first,
    )
    controller = BuildController(
        BuildGraph((node,)),
        code_root=code_root,
        data_root=data_root,
        implementation_commit="a" * 40,
    )
    controller.build("analysis")
    old_current = (controller.registry_root / "analysis/CURRENT.json").read_bytes()
    old_state = controller._current_state_path("analysis")
    old_bundle = old_state.parent / "bundle/data/05_中间数据/analysis"
    assert (old_bundle / "first.bin").read_text() == "old-first"
    assert (old_bundle / "second.bin").read_text() == "old-second"

    executable.write_text("v2", encoding="utf-8")
    with pytest.raises(BuildPublicationError, match="second output failed"):
        controller.build("analysis")

    assert (controller.registry_root / "analysis/CURRENT.json").read_bytes() == old_current
    assert first.read_text() == "old-first"
    assert second.read_text() == "old-second"
    assert (old_bundle / "first.bin").read_text() == "old-first"
    assert (old_bundle / "second.bin").read_text() == "old-second"


def test_failed_staged_table_build_never_overwrites_fixed_metadata_mirrors(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    code_root = tmp_path / "code"
    executable = code_root / "builder.py"
    output = data_root / "05_中间数据/analysis/table.parquet"
    executable.parent.mkdir(parents=True)
    executable.write_text("v1", encoding="utf-8")
    contract = TableContract(
        table_id="table",
        schema_version="1.0.0",
        primary_key=("id",),
        columns={"id": "String"},
        units={},
    )
    write_authoritative_table(
        pl.DataFrame({"id": ["old"]}),
        contract,
        output,
        (),
        BuildIdentity(command="reviewed", code_commit="0" * 40),
    )
    central_manifest = data_root / "05_中间数据/manifests/table.manifest.json"
    reviewed_manifest = central_manifest.read_bytes()

    def fail_after_table_write(stage: BuildStage) -> None:
        write_authoritative_table(
            pl.DataFrame({"id": ["new"]}),
            contract,
            stage.path_for(output),
            (),
            BuildIdentity(command="staged", code_commit="a" * 40),
        )
        raise RuntimeError("stage failed after table write")

    node = BuildNode(
        "table",
        (),
        lambda: None,
        output_ids=("table",),
        manifest_paths=(output.with_name(f"{output.name}.manifest.json"),),
        executable_inputs=(executable,),
        staging_builder=fail_after_table_write,
    )
    controller = BuildController(
        BuildGraph((node,)),
        code_root=code_root,
        data_root=data_root,
        implementation_commit="a" * 40,
    )
    controller.build("table")
    executable.write_text("v2", encoding="utf-8")

    with pytest.raises(BuildPublicationError, match="stage failed after table write"):
        controller.build("table")

    assert central_manifest.read_bytes() == reviewed_manifest
    assert verify_manifest(central_manifest).destination == str(output.resolve())


def test_successful_staged_table_build_refreshes_fixed_metadata_mirrors(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    code_root = tmp_path / "code"
    executable = code_root / "builder.py"
    output = data_root / "05_中间数据/analysis/table.parquet"
    executable.parent.mkdir(parents=True)
    executable.write_text("v1", encoding="utf-8")
    contract = TableContract(
        table_id="table",
        schema_version="1.0.0",
        primary_key=("id",),
        columns={"id": "String"},
        units={},
    )
    write_authoritative_table(
        pl.DataFrame({"id": ["old"]}),
        contract,
        output,
        (),
        BuildIdentity(command="reviewed", code_commit="0" * 40),
    )

    def rebuild(stage: BuildStage) -> None:
        write_authoritative_table(
            pl.DataFrame({"id": ["new"]}),
            contract,
            stage.path_for(output),
            (),
            BuildIdentity(command="staged", code_commit="a" * 40),
        )

    node = BuildNode(
        "table",
        (),
        lambda: None,
        output_ids=("table",),
        manifest_paths=(output.with_name(f"{output.name}.manifest.json"),),
        executable_inputs=(executable,),
        staging_builder=rebuild,
    )
    controller = BuildController(
        BuildGraph((node,)),
        code_root=code_root,
        data_root=data_root,
        implementation_commit="a" * 40,
    )
    controller.build("table")
    executable.write_text("v2", encoding="utf-8")

    controller.build("table")

    sidecar = output.with_name(f"{output.name}.manifest.json")
    central_manifest = data_root / "05_中间数据/manifests/table.manifest.json"
    central_schema = data_root / "05_中间数据/schemas/table.schema.json"
    verified = verify_manifest(central_manifest)
    assert central_manifest.read_bytes() == sidecar.read_bytes()
    assert central_schema.read_bytes() == Path(verified.schema_path).read_bytes()
    assert "/build.table." not in verified.destination


def test_same_node_manifest_rebinding_refreshes_parent_hash_index(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    code_root = tmp_path / "code"
    executable = code_root / "builder.py"
    parent = data_root / "05_中间数据/analysis/parent.parquet"
    child = data_root / "05_中间数据/analysis/child.parquet"
    executable.parent.mkdir(parents=True)
    executable.write_text("v1", encoding="utf-8")
    parent_contract = TableContract(
        table_id="parent",
        schema_version="1.0.0",
        primary_key=("id",),
        columns={"id": "String"},
        units={},
    )
    child_contract = TableContract(
        table_id="child",
        schema_version="1.0.0",
        primary_key=("id",),
        columns={"id": "String"},
        units={},
    )

    def rebuild(stage: BuildStage) -> None:
        staged_parent = stage.path_for(parent)
        write_authoritative_table(
            pl.DataFrame({"id": ["parent"]}),
            parent_contract,
            staged_parent,
            (),
            BuildIdentity(command="stage", code_commit="a" * 40),
        )
        write_authoritative_table(
            pl.DataFrame({"id": ["child"]}),
            child_contract,
            stage.path_for(child),
            (InputArtifact.from_path(staged_parent),),
            BuildIdentity(command="stage", code_commit="a" * 40),
        )

    node = BuildNode(
        "tables",
        (),
        lambda: None,
        output_ids=("parent", "child"),
        manifest_paths=(
            parent.with_name(f"{parent.name}.manifest.json"),
            child.with_name(f"{child.name}.manifest.json"),
        ),
        executable_inputs=(executable,),
        staging_builder=rebuild,
    )
    controller = BuildController(
        BuildGraph((node,)),
        code_root=code_root,
        data_root=data_root,
        implementation_commit="a" * 40,
    )

    assert controller.build("tables")[0].status == "built"

    state_path = controller._current_state_path("tables")
    manifest = verify_manifest(
        state_path.parent
        / "bundle/data/05_中间数据/analysis/child.parquet.manifest.json"
    )
    bound_parent_hashes = tuple(
        sorted(
            artifact.parent_manifest_sha256
            for artifact in manifest.input_artifacts
            if artifact.parent_manifest_sha256 is not None
        )
    )
    assert manifest.parent_manifest_hashes == bound_parent_hashes


def test_successful_stage_keeps_fixed_compatibility_copy_separate_from_current_bundle(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    code_root = tmp_path / "code"
    output = data_root / "05_中间数据/analysis/table.bin"
    executable = code_root / "builder.py"
    for path, value in ((output, "old"), (executable, "v1")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def rebuild(stage: BuildStage) -> None:
        staged = stage.path_for(output)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text("new", encoding="utf-8")

    node = BuildNode(
        "analysis", (), lambda: None, authority_files=(output,),
        executable_inputs=(executable,), staging_builder=rebuild,
    )
    controller = BuildController(
        BuildGraph((node,)), code_root=code_root, data_root=data_root,
        implementation_commit="a" * 40,
    )
    controller.build("analysis")
    executable.write_text("v2", encoding="utf-8")

    controller.build("analysis")

    state = json.loads(controller._current_state_path("analysis").read_text())
    authoritative = Path(state["outputs"][0]["path"])
    assert authoritative.is_file() and not authoritative.is_symlink()
    assert output.is_file() and not output.is_symlink()
    assert output.resolve() != authoritative
    assert output.read_bytes() == authoritative.read_bytes()
    assert output.read_text() == "new"


def test_successful_stage_never_changes_tracked_code_compatibility_authority(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "project"
    code_root = data_root / ".worktrees/task"
    output = code_root / "02_数据字典/registry.csv"
    executable = code_root / "builder.py"
    for path, value in ((output, "same-bytes\n"), (executable, "v1")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def rebuild(stage: BuildStage) -> None:
        staged = stage.path_for(output)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text("new-bundle-bytes\n", encoding="utf-8")

    node = BuildNode(
        "registry", (), lambda: None, authority_files=(output,),
        executable_inputs=(executable,), staging_builder=rebuild,
    )
    controller = BuildController(
        BuildGraph((node,)), code_root=code_root, data_root=data_root,
        implementation_commit="a" * 40,
    )
    controller.build("registry")
    executable.write_text("v2", encoding="utf-8")
    controller.build("registry")

    assert output.is_file()
    assert not output.is_symlink()
    assert output.read_bytes() == b"same-bytes\n"
    state = json.loads(controller._current_state_path("registry").read_text())
    assert Path(state["outputs"][0]["path"]).read_bytes() == b"new-bundle-bytes\n"


def test_publication_preserves_every_registered_tracked_code_authority(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    data_root = tmp_path / "data"
    graph = registered_build_graph(code_root=code_root, data_root=data_root)
    registered = tuple(
        sorted(
            path.relative_to(code_root)
            for node in graph.nodes
            for path in node.authority_files
            if path.is_relative_to(code_root)
        )
    )
    assert registered == tuple(
        sorted(
            Path(path)
            for path in (
                "02_数据字典/economy_crosswalk_v1.csv",
                "02_数据字典/product_registry_hs07_v1.csv",
                "02_数据字典/product_registry_hs96_v1.parquet",
                "06_结果/GAD固定缩放器_v1.json",
                "06_结果/GAD固定缩放器_v1.manifest.json",
            )
        )
    )
    for relative in registered:
        destination = code_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"HEAD:{relative}\n".encode())
    executable = code_root / "builder.py"
    executable.write_text("v1\n", encoding="utf-8")

    def rebuild(stage: BuildStage) -> None:
        for relative in registered:
            staged = stage.code_root / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(f"BUILT:{relative}\n".encode())

    node = BuildNode(
        "all_code_authorities",
        (),
        lambda: None,
        authority_files=tuple(code_root / relative for relative in registered),
        executable_inputs=(executable,),
        staging_builder=rebuild,
    )
    controller = BuildController(
        BuildGraph((node,)),
        code_root=code_root,
        data_root=data_root,
        implementation_commit="a" * 40,
    )
    controller.build("all_code_authorities")
    executable.write_text("v2\n", encoding="utf-8")
    controller.build("all_code_authorities")
    for relative in registered:
        assert (code_root / relative).read_bytes() == f"HEAD:{relative}\n".encode()
    state = json.loads(controller._current_state_path("all_code_authorities").read_text())
    assert all(Path(output["path"]).read_bytes().startswith(b"BUILT:") for output in state["outputs"])

def test_raw_path_identity_and_hash_change_stales_root_and_blocks_descendant(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    code_root = tmp_path / "code"
    raw = data_root / "04_原始数据/source.csv"
    output = data_root / "05_中间数据/normalized/table.bin"
    code = code_root / "builder.py"
    for path, value in ((raw, "raw-v1"), (output, "table"), (code, "code")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    graph = BuildGraph(
        (
            BuildNode(
                "root",
                (),
                lambda: None,
                authority_files=(output,),
                raw_inputs=(raw,),
                executable_inputs=(code,),
            ),
            BuildNode("child", ("root",), lambda: None),
        )
    )
    controller = BuildController(
        graph,
        code_root=code_root,
        data_root=data_root,
        implementation_commit="b" * 40,
    )
    controller.build("child")
    state = json.loads(controller._current_state_path("root").read_text())
    assert state["raw_input_hashes"] == {
        "04_原始数据/source.csv": state["raw_input_hashes"]["04_原始数据/source.csv"]
    }

    raw.write_text("raw-v2", encoding="utf-8")
    status = controller.status("child")
    assert [(item.status, item.reason) for item in status] == [
        ("stale", "raw_input_hashes_changed"),
        ("blocked", "dependency_stale:root"),
    ]


def test_download_sidecar_is_not_misclassified_as_direct_table_manifest(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "source.bin"
    raw.write_bytes(b"source")
    raw.with_name("source.bin.manifest.json").write_text(
        json.dumps({"source_id": "download", "sha256": "x"}), encoding="utf-8"
    )

    assert BuildController._manifest_table_id(raw) is None


def test_code_input_identity_prefers_nested_worktree_root_over_shared_data_root(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "project"
    code_root = data_root / ".worktrees/task"
    code_file = code_root / "06_结果/support.csv"
    code_file.parent.mkdir(parents=True)
    code_file.write_text("support\n", encoding="utf-8")
    controller = BuildController(
        BuildGraph((BuildNode("node", (), lambda: None),)),
        code_root=code_root,
        data_root=data_root,
        implementation_commit="a" * 40,
    )

    assert controller._relative_identity(code_file) == "06_结果/support.csv"
    stage = BuildStage(
        data_root,
        code_root,
        tmp_path / "stage/data",
        tmp_path / "stage/code",
    )
    assert stage.path_for(code_file) == tmp_path / "stage/code/06_结果/support.csv"


def test_registered_nodes_bind_common_executable_closure_raw_roots_and_direct_ids(
    tmp_path: Path,
) -> None:
    graph = registered_build_graph(
        code_root=tmp_path / "code",
        data_root=tmp_path / "data",
        command_runner=lambda command: None,
    )
    common = {"build.py", "cli.py", "artifacts.py", "storage.py"}
    for name in ("taxonomy", "wdi", "analysis_panels"):
        assert common <= {path.name for path in graph.node(name).executable_inputs}
        # cli.py imports the package builders at process start, so the stable
        # executable closure is the complete tracked Python package.
        assert {"checkpoints.py", "instruments.py", "outcomes.py"} <= {
            path.name for path in graph.node(name).executable_inputs
        }
        assert graph.node(name).command_args
    assert len(graph.node("taxonomy").raw_inputs) == 6
    assert graph.node("economies").raw_globs
    assert graph.node("analysis_panels").direct_input_ids == (
        "final_sample",
        "gad_country_year",
        "iv_country_year",
        "outcomes_country_year",
        "outcomes_product_year",
        "provisional_sample",
        "wdi_country_year",
    )
    assert "provisional_sample" in graph.node("gad").dependencies
    assert graph.node("gad_scaler").command_args == ("apply-gad-scaler",)


def test_new_manifest_rejects_wrong_command_or_code_commit() -> None:
    manifest = {"command": "python -m green_debt.cli build-one --data-root /stage", "code_commit": "a" * 40}
    validate_built_manifest_identity(
        manifest,
        implementation_commit="a" * 40,
        exact_command="python -m green_debt.cli build-one --data-root /stage",
    )
    with pytest.raises(ValueError, match="command"):
        validate_built_manifest_identity(
            manifest,
            implementation_commit="a" * 40,
            exact_command="python -m green_debt.cli build-two --data-root /stage",
        )
    with pytest.raises(ValueError, match="code"):
        validate_built_manifest_identity(
            manifest,
            implementation_commit="b" * 40,
            exact_command=manifest["command"],
        )


def test_isolated_stage_can_bind_the_exact_committed_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GREEN_DEBT_IMPLEMENTATION_COMMIT", "d" * 40)

    assert _code_version() == "d" * 40


def test_approved_checkpoint_gate_blocks_after_parent_state_changes(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    code_root = tmp_path / "code"
    raw = data_root / "04_原始数据/source.bin"
    parent_output = data_root / "05_中间数据/harmonized/parent.bin"
    executable = code_root / "parent.py"
    receipt = code_root / "06_结果/检查点1_验收回执_v1.json"
    for path, value in ((raw, "raw"), (parent_output, "parent"), (executable, "code")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    receipt.parent.mkdir(parents=True)
    support = receipt.parent / "support.csv"
    support.write_text("status\nvalid\n", encoding="utf-8")
    receipt.write_text(json.dumps({"number": 1, "passed": True, "git_commit": "a" * 40}), encoding="utf-8")
    verified: list[str] = []

    def verify_gate_semantics() -> None:
        # This is a real file-level semantic check: a synchronized receipt edit
        # cannot substitute for the expected approved support content.
        assert support.read_text(encoding="utf-8") == "status\nvalid\n"
        verified.append("verified")
    calls: list[str] = []
    def rebuild_parent(stage: BuildStage) -> None:
        staged = stage.path_for(parent_output)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text("parent", encoding="utf-8")

    graph = BuildGraph(
        (
            BuildNode(
                "parent", (), lambda: None, authority_files=(parent_output,),
                raw_inputs=(raw,), executable_inputs=(executable,),
                staging_builder=rebuild_parent,
            ),
            BuildNode(
                "checkpoint1", ("parent",), lambda: calls.append("gate-ran"),
                approval_gate=ApprovalGate(
                    number=1,
                    receipt_path=receipt,
                    support_paths=(support,),
                    semantic_verifier=verify_gate_semantics,
                ),
            ),
            BuildNode("downstream", ("checkpoint1",), lambda: calls.append("downstream")),
        )
    )
    controller = BuildController(
        graph,
        code_root=code_root,
        data_root=data_root,
        implementation_commit="a" * 40,
    )
    first = controller.build("downstream")
    assert first[1].status == "adopted"
    assert calls == []
    assert verified == ["verified"]

    raw.write_text("changed", encoding="utf-8")
    with pytest.raises(ApprovalRequired, match="checkpoint1.*approval required"):
        controller.build("downstream")
    assert calls == []


def test_missing_gate_can_adopt_verified_receipt_after_same_run_parent_rebuild(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    code_root = tmp_path / "code"
    output = data_root / "05_中间数据/parent.bin"
    executable = code_root / "parent.py"
    receipt = code_root / "06_结果/receipt.json"
    for path, value in ((output, "same"), (executable, "v1")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps({"number": 1, "passed": True, "git_commit": "a" * 40}),
        encoding="utf-8",
    )
    verified: list[str] = []

    def rebuild(stage: BuildStage) -> None:
        staged = stage.path_for(output)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text("same", encoding="utf-8")

    parent = BuildNode(
        "parent", (), lambda: None, authority_files=(output,),
        executable_inputs=(executable,), staging_builder=rebuild,
    )
    parent_only = BuildController(
        BuildGraph((parent,)), code_root=code_root, data_root=data_root,
        implementation_commit="a" * 40,
    )
    parent_only.build("parent")
    executable.write_text("v2", encoding="utf-8")
    gate = BuildNode(
        "checkpoint1", ("parent",), lambda: None,
        approval_gate=ApprovalGate(
            1, receipt,
            semantic_verifier=lambda: verified.append("verified"),
        ),
    )
    controller = BuildController(
        BuildGraph((parent, gate)), code_root=code_root, data_root=data_root,
        implementation_commit="a" * 40,
    )

    result = controller.build("checkpoint1")

    assert [item.status for item in result] == ["built", "adopted"]
    assert verified == ["verified"]


def test_approval_gate_rejects_tampered_support_and_receipt_commit(tmp_path: Path) -> None:
    code_root = tmp_path / "repo"
    data_root = tmp_path / "data"
    code_root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=code_root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=code_root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=code_root, check=True)
    receipt = code_root / "06_结果/检查点1_验收回执_v1.json"
    support = code_root / "06_结果/support.csv"
    receipt.parent.mkdir(parents=True)
    support.write_text("status\nvalid\n", encoding="utf-8")
    receipt.write_text(json.dumps({"number": 1, "passed": True, "git_commit": "0" * 40}), encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=code_root, check=True)
    subprocess.run(["git", "commit", "-qm", "approval"], cwd=code_root, check=True)
    approved = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_root, check=True, capture_output=True, text=True).stdout.strip()
    receipt.write_text(json.dumps({"number": 1, "passed": True, "git_commit": approved}), encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=code_root, check=True)
    subprocess.run(["git", "commit", "-qm", "bind approval"], cwd=code_root, check=True)
    implementation = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_root, check=True, capture_output=True, text=True).stdout.strip()
    node = BuildNode(
        "checkpoint1", (), lambda: None,
        approval_gate=ApprovalGate(
            1, receipt, (support,),
            semantic_verifier=lambda: support.read_text(encoding="utf-8") == "status\nvalid\n",
        ),
    )
    controller = BuildController(
        BuildGraph((node,)), code_root=code_root, data_root=data_root,
        implementation_commit=implementation,
    )
    controller.build("checkpoint1")
    support.write_text("status\nforged\n", encoding="utf-8")
    assert controller.status("checkpoint1")[0].status == "stale"

    support.write_text("status\nvalid\n", encoding="utf-8")
    receipt.write_text(json.dumps({"number": 1, "passed": True, "git_commit": "f" * 40}), encoding="utf-8")
    assert controller.status("checkpoint1")[0].status == "stale"


def test_published_manifest_rebinds_staged_inputs_to_stable_authorities(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    code_root = tmp_path / "code"
    raw = data_root / "04_原始数据/source.csv"
    output = data_root / "05_中间数据/analysis/table.parquet"
    executable = code_root / "builder.py"
    config = code_root / "config/frozen.yaml"
    for path, value in ((raw, "raw"), (executable, "v1"), (config, "frozen: true\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    contract = TableContract(
        table_id="table", schema_version="1.0.0", primary_key=("id",),
        columns={"id": "String"}, units={},
    )

    def write(path: Path, command: str, commit: str, artifact: Path) -> None:
        write_authoritative_table(
            pl.DataFrame({"id": ["A"]}), contract, path,
            (InputArtifact.from_path(artifact),),
            BuildIdentity(command=command, code_commit=commit),
        )

    write(output, "old", "0" * 40, raw)
    node: BuildNode

    def rebuild(stage: BuildStage) -> None:
        staged_config = stage.path_for(config)
        staged_output = stage.path_for(output)
        command = " ".join(
            ("python", "-m", "green_debt.cli", "fixture", "--data-root", str(stage.data_root))
        )
        write(staged_output, command, "a" * 40, staged_config)

    node = BuildNode(
        "table", (), lambda: None, output_ids=("table",),
        manifest_paths=(output.with_name(f"{output.name}.manifest.json"),),
        executable_inputs=(executable,), raw_inputs=(raw,), config_inputs=(config,),
        command_args=("fixture",), staging_builder=rebuild,
    )
    controller = BuildController(
        BuildGraph((node,)), code_root=code_root, data_root=data_root,
        implementation_commit="a" * 40,
    )
    controller.build("table")
    executable.write_text("v2", encoding="utf-8")
    controller.build("table")

    state_path = controller._current_state_path("table")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    manifest_path = next(iter(state["direct_manifest_hashes"]))
    manifest = verify_manifest(Path(manifest_path))
    assert all("/build.table." not in artifact.path for artifact in manifest.input_artifacts)
    assert manifest.input_artifacts[0].path == str(config.resolve())
    assert controller.status("table")[0].status == "current"


def test_exact_manifest_command_matches_cli_builder_unicode_path_format(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "共享数据"
    code_root = tmp_path / "code"
    node = BuildNode(
        "baci", (), lambda: None,
        command_args=("normalize-baci", "--revision", "HS96"),
    )
    controller = BuildController(
        BuildGraph((node,)), code_root=code_root, data_root=data_root,
        implementation_commit="a" * 40,
    )
    stage = BuildStage(data_root, code_root, data_root / "stage", code_root / "stage")

    assert controller._exact_manifest_command(node, stage) == (
        "python -m green_debt.cli normalize-baci --revision HS96 --data-root "
        + str(stage.data_root)
    )
