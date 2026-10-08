"""Deterministic build orchestration and reproducibility safety primitives.

The construction commands continue to own formulas and authoritative Parquet
publication.  This module adds a small, versioned orchestration registry around
those commands.  It deliberately does not treat README or compact audit changes
as executable inputs to a data node.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any

import duckdb
import polars as pl

from green_debt.storage import (
    enforce_construction_capacity,
    measure_layer_usage,
    sha256_file,
)
from green_debt.artifacts import verify_manifest


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class BuildStage:
    """Isolated roots supplied to a node builder before publication."""

    source_data_root: Path
    source_code_root: Path
    data_root: Path
    code_root: Path

    def path_for(self, authority: Path) -> Path:
        resolved = authority.absolute()
        for source, staged in (
            (self.source_code_root.absolute(), self.code_root),
            (self.source_data_root.absolute(), self.data_root),
        ):
            try:
                return staged / resolved.relative_to(source)
            except ValueError:
                continue
        raise ValueError(f"authority is outside the staged roots: {authority}")


@dataclass(frozen=True)
class ApprovalGate:
    """A reviewed checkpoint receipt; orchestration may verify but never create it."""

    number: int
    receipt_path: Path
    support_paths: tuple[Path, ...] = ()
    semantic_verifier: Callable[[], object] | None = None


class ApprovalRequired(RuntimeError):
    """Raised when a reviewed gate no longer binds the current parent state."""


@dataclass(frozen=True)
class BuildNode:
    """One executable DAG node and the artifacts that define its contract."""

    name: str
    dependencies: tuple[str, ...]
    build: Callable[[], object]
    output_ids: tuple[str, ...] = ()
    manifest_paths: tuple[Path, ...] = ()
    manifest_globs: tuple[str, ...] = ()
    authority_files: tuple[Path, ...] = ()
    executable_inputs: tuple[Path, ...] = ()
    raw_inputs: tuple[Path, ...] = ()
    config_inputs: tuple[Path, ...] = ()
    contract_inputs: tuple[Path, ...] = ()
    taxonomy_inputs: tuple[Path, ...] = ()
    scaler_inputs: tuple[Path, ...] = ()
    raw_globs: tuple[str, ...] = ()
    direct_input_ids: tuple[str, ...] = ()
    command_args: tuple[str, ...] = ()
    staging_builder: Callable[[BuildStage], object] | None = None
    approval_gate: ApprovalGate | None = None

    def __post_init__(self) -> None:
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("build node name must be a non-empty token")
        if self.name in self.dependencies:
            raise ValueError(f"build graph cycle at {self.name}")
        inferred_args = getattr(self.build, "command_args", ())
        inferred_stage = getattr(self.build, "staging_builder", None)
        if not self.command_args and inferred_args:
            object.__setattr__(self, "command_args", tuple(inferred_args))
        if self.staging_builder is None and inferred_stage is not None:
            object.__setattr__(self, "staging_builder", inferred_stage)


@dataclass(frozen=True)
class BuildRunItem:
    name: str
    status: str
    reason: str


@dataclass(frozen=True)
class BuildStatusItem:
    name: str
    status: str
    reason: str


class BuildGraph:
    """Validated deterministic DAG with resumable, descendant-safe execution."""

    def __init__(self, nodes: Iterable[BuildNode]) -> None:
        materialized = tuple(nodes)
        names = [node.name for node in materialized]
        if len(names) != len(set(names)):
            raise ValueError("duplicate build node name")
        self._nodes = {node.name: node for node in materialized}
        for node in materialized:
            for dependency in node.dependencies:
                if dependency not in self._nodes:
                    raise ValueError(
                        f"unknown dependency {dependency!r} for node {node.name!r}"
                    )
        # Validate the whole graph immediately, including disconnected cycles.
        for name in names:
            self.order(name)

    @property
    def nodes(self) -> tuple[BuildNode, ...]:
        return tuple(self._nodes.values())

    def order(self, target: str) -> tuple[str, ...]:
        if target not in self._nodes:
            raise ValueError(f"unknown build target: {target}")
        ordered: list[str] = []
        permanent: set[str] = set()
        visiting: list[str] = []

        def visit(name: str) -> None:
            if name in permanent:
                return
            if name in visiting:
                cycle = " -> ".join((*visiting[visiting.index(name) :], name))
                raise ValueError(f"build graph cycle: {cycle}")
            visiting.append(name)
            for dependency in self._nodes[name].dependencies:
                visit(dependency)
            visiting.pop()
            permanent.add(name)
            ordered.append(name)

        visit(target)
        return tuple(ordered)

    def run(
        self,
        target: str,
        *,
        manifest_is_current: Callable[[str], bool],
    ) -> tuple[BuildRunItem, ...]:
        """Run stale nodes in order; a rebuilt parent always rebuilds descendants."""

        results: list[BuildRunItem] = []
        rebuilt: set[str] = set()
        for name in self.order(target):
            node = self._nodes[name]
            upstream_rebuilt = any(dependency in rebuilt for dependency in node.dependencies)
            if not upstream_rebuilt and manifest_is_current(name):
                results.append(BuildRunItem(name, "current", "verified_current"))
                continue
            reason = "upstream_rebuilt" if upstream_rebuilt else "manifest_stale"
            node.build()
            rebuilt.add(name)
            results.append(BuildRunItem(name, "built", reason))
        return tuple(results)

    def node(self, name: str) -> BuildNode:
        try:
            return self._nodes[name]
        except KeyError as exc:
            raise ValueError(f"unknown build target: {name}") from exc

    def status(
        self,
        target: str,
        *,
        manifest_status: Callable[[str], tuple[bool, str]],
    ) -> tuple[BuildStatusItem, ...]:
        """Report current/stale/blocked with the first causal reason per node."""

        results: list[BuildStatusItem] = []
        unavailable: set[str] = set()
        for name in self.order(target):
            node = self._nodes[name]
            blocking = next(
                (dependency for dependency in node.dependencies if dependency in unavailable),
                None,
            )
            if blocking is not None:
                unavailable.add(name)
                results.append(
                    BuildStatusItem(name, "blocked", f"dependency_stale:{blocking}")
                )
                continue
            current, reason = manifest_status(name)
            if current:
                results.append(BuildStatusItem(name, "current", reason))
            else:
                unavailable.add(name)
                results.append(BuildStatusItem(name, "stale", reason))
        return tuple(results)


_HASH_FIELDS = (
    "input_hashes",
    "parent_manifest_hashes",
    "config_hashes",
    "contract_hashes",
    "taxonomy_hashes",
    "scaler_hashes",
)


def _stored_hashes(manifest: Mapping[str, object], field: str) -> tuple[str, ...] | None:
    value = manifest.get(field)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return tuple(sorted(value))


def _verify_declared_outputs(manifest: Mapping[str, object]) -> bool:
    declared = manifest.get("outputs")
    if declared is None:
        destination = manifest.get("destination")
        output_hash = manifest.get("output_sha256")
        schema_path = manifest.get("schema_path")
        schema_hash = manifest.get("schema_sha256")
        if destination is None and output_hash is None:
            return True
        declared = [
            {
                "path": destination,
                "sha256": output_hash,
                "schema_path": schema_path,
                "schema_sha256": schema_hash,
            }
        ]
    if not isinstance(declared, list) or not declared:
        return False
    for item in declared:
        if not isinstance(item, Mapping):
            return False
        path_value = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path_value, str) or not isinstance(digest, str):
            return False
        path = Path(path_value)
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            return False
        schema_value = item.get("schema_path")
        schema_digest = item.get("schema_sha256")
        if schema_value is None and schema_digest is None:
            continue
        if not isinstance(schema_value, str) or not isinstance(schema_digest, str):
            return False
        schema = Path(schema_value)
        if schema.is_symlink() or not schema.is_file() or sha256_file(schema) != schema_digest:
            return False
    return True


def manifest_is_current(
    manifest: Mapping[str, object] | Path,
    current_input_hashes: tuple[str, ...] | None = None,
    *,
    current_parent_manifest_hashes: tuple[str, ...] | None = None,
    current_config_hashes: tuple[str, ...] | None = None,
    current_contract_hashes: tuple[str, ...] | None = None,
    current_taxonomy_hashes: tuple[str, ...] | None = None,
    current_scaler_hashes: tuple[str, ...] | None = None,
    current_executable_input_hashes: Mapping[str, str] | None = None,
    changed_non_executable_paths: tuple[str, ...] = (),
) -> bool:
    """Verify outputs and every node-specific semantic/executable input binding.

    ``changed_non_executable_paths`` is accepted for status reporting callers,
    but intentionally has no bearing on currency.  Only the executable-input
    map recorded for this node participates in code invalidation.
    """

    del changed_non_executable_paths
    if isinstance(manifest, Path):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
    else:
        payload = manifest
    if not _verify_declared_outputs(payload):
        return False
    current_by_field = {
        "input_hashes": current_input_hashes,
        "parent_manifest_hashes": current_parent_manifest_hashes,
        "config_hashes": current_config_hashes,
        "contract_hashes": current_contract_hashes,
        "taxonomy_hashes": current_taxonomy_hashes,
        "scaler_hashes": current_scaler_hashes,
    }
    for field in _HASH_FIELDS:
        current = current_by_field[field]
        if current is None:
            continue
        stored = _stored_hashes(payload, field)
        if stored is None or stored != tuple(sorted(current)):
            return False
    if current_executable_input_hashes is not None:
        stored_executable = payload.get("executable_input_hashes")
        if not isinstance(stored_executable, Mapping):
            return False
        normalized = {
            str(key): str(value) for key, value in stored_executable.items()
        }
        if normalized != {
            str(key): str(value)
            for key, value in current_executable_input_hashes.items()
        }:
            return False
    return True


def validate_built_manifest_identity(
    manifest: Mapping[str, object],
    *,
    implementation_commit: str,
    exact_command: str,
) -> None:
    """Reject a newly built manifest that does not name its exact code/command."""

    if manifest.get("command") != exact_command:
        raise ValueError("built manifest command differs from exact stage command")
    if manifest.get("code_commit") != implementation_commit:
        raise ValueError("built manifest code commit differs from implementation")


class BuildPublicationError(RuntimeError):
    """Raised after a version writer fails without changing CURRENT."""


@dataclass(frozen=True)
class PublishedVersion:
    node: str
    build_id: str
    version_path: Path
    current_path: Path


def _safe_token(value: str, label: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"unsafe {label}")
    return value


def publish_version(
    registry_root: Path,
    node: str,
    build_id: str,
    writer: Callable[[Path], object],
    *,
    validator: Callable[[Path], object] | None = None,
) -> PublishedVersion:
    """Publish a complete immutable node version, then atomically move CURRENT."""

    node = _safe_token(node, "node")
    build_id = _safe_token(build_id, "build_id")
    node_root = registry_root.resolve() / node
    versions = node_root / "versions"
    destination = versions / build_id
    partial = versions / f".{build_id}.partial"
    if destination.exists() or partial.exists():
        raise ValueError("build version already exists")
    partial.mkdir(parents=True)
    try:
        result = writer(partial)
        version_payload = {
            "schema_version": "1.0.0",
            "node": node,
            "build_id": build_id,
            "created_at_utc": _utc_now(),
            "writer_result": result if isinstance(result, (dict, list, str, int, float, bool, type(None))) else repr(result),
        }
        _write_bytes_atomic(partial / "VERSION.json", _canonical_json_bytes(version_payload))
        os.replace(partial, destination)
        if validator is not None:
            validator(destination)
        descriptor = os.open(versions, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        current_path = node_root / "CURRENT.json"
        _write_bytes_atomic(
            current_path,
            _canonical_json_bytes(
                {
                    "schema_version": "1.0.0",
                    "node": node,
                    "build_id": build_id,
                    "version_path": str(destination),
                }
            ),
        )
    except Exception as exc:
        if partial.is_dir() and not partial.is_symlink():
            shutil.rmtree(partial)
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        raise BuildPublicationError(f"version publication failed: {exc}") from exc
    return PublishedVersion(node, build_id, destination, node_root / "CURRENT.json")


class BuildController:
    """Publish complete output bundles; fixed paths are compatibility mirrors only."""

    def __init__(
        self,
        graph: BuildGraph,
        *,
        code_root: Path,
        data_root: Path,
        implementation_commit: str,
    ) -> None:
        if len(implementation_commit) != 40:
            raise ValueError("implementation_commit must be a full Git hash")
        self.graph = graph
        self.code_root = code_root.resolve()
        self.data_root = data_root.resolve()
        self.implementation_commit = implementation_commit
        self.registry_root = self.data_root / "05_中间数据/manifests/_build_registry"
        self._file_hash_cache: dict[tuple[str, int, int], str] = {}

    def _sha256(self, path: Path) -> str:
        resolved = path.resolve()
        metadata = resolved.stat()
        key = (str(resolved), metadata.st_size, metadata.st_mtime_ns)
        digest = self._file_hash_cache.get(key)
        if digest is None:
            digest = sha256_file(resolved)
            self._file_hash_cache[key] = digest
        return digest

    def _translate(self, path: Path, stage: BuildStage | None) -> Path:
        return path if stage is None else stage.path_for(path)

    def _compatibility_path(self, path: Path, stage: BuildStage | None) -> Path:
        if stage is None:
            return path.resolve()
        resolved = path.resolve()
        for staged, source in (
            (stage.code_root.resolve(), self.code_root),
            (stage.data_root.resolve(), self.data_root),
        ):
            try:
                return source / resolved.relative_to(staged)
            except ValueError:
                continue
        return resolved

    def _manifest_paths(
        self, node: BuildNode, *, stage: BuildStage | None = None
    ) -> tuple[Path, ...]:
        paths = {self._translate(path, stage) for path in node.manifest_paths}
        intermediate = (
            self.data_root / "05_中间数据"
            if stage is None
            else stage.data_root / "05_中间数据"
        )
        for pattern in node.manifest_globs:
            paths.update(intermediate.glob(pattern))
        return tuple(sorted(path.absolute() for path in paths))

    def _relative_identity(self, path: Path) -> str:
        resolved = path.resolve()
        # A Git worktree may live below the shared project/data root.  Prefer
        # the more specific code root so Git blob identities stay repo-relative.
        for root in (self.code_root, self.data_root):
            try:
                return resolved.relative_to(root).as_posix()
            except ValueError:
                continue
        raise ValueError(f"tracked build input escaped code/data roots: {path}")

    def _raw_paths(self, node: BuildNode) -> tuple[Path, ...]:
        paths = set(node.raw_inputs)
        raw = self.data_root / "04_原始数据"
        for pattern in node.raw_globs:
            paths.update(path for path in raw.glob(pattern) if path.is_file())
        if node.raw_globs and not paths:
            raise ValueError("declared raw inputs are missing")
        return tuple(sorted(paths))

    def _hash_files(
        self, paths: Sequence[Path], label: str, *, relative: bool = False
    ) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for path in paths:
            resolved = path.resolve()
            if path.is_symlink() or not resolved.is_file():
                raise ValueError(f"{label} is missing or unsafe: {path}")
            key = self._relative_identity(path) if relative else str(resolved)
            hashes[key] = self._sha256(resolved)
        return hashes

    @staticmethod
    def _manifest_table_id(path: Path) -> str | None:
        sidecar = path.with_name(f"{path.name}.manifest.json")
        if not sidecar.is_file():
            return None
        try:
            return verify_manifest(sidecar).table_id
        except ValueError:
            return None

    def _authority_snapshot(
        self,
        node: BuildNode,
        *,
        stage: BuildStage | None = None,
        require_built_identity: bool = False,
        expected_manifest_command: str | None = None,
        expected_implementation_commit: str | None = None,
    ) -> dict[str, object]:
        manifest_paths = self._manifest_paths(node, stage=stage)
        if (node.manifest_paths or node.manifest_globs) and not manifest_paths:
            raise ValueError("authoritative manifests are missing")
        outputs: list[dict[str, object]] = []
        direct_manifests: dict[str, str] = {}
        manifest_compatibility_paths: dict[str, str] = {}
        input_artifacts: dict[str, dict[str, object]] = {}
        direct_inputs: dict[str, list[dict[str, object]]] = {}
        actual_ids: set[str] = set()
        exact_command = expected_manifest_command or (
            self._exact_manifest_command(node, stage) if stage is not None else None
        )
        for manifest_path in manifest_paths:
            manifest = verify_manifest(manifest_path)
            if require_built_identity:
                if exact_command is None:
                    raise ValueError("built manifest identity lacks an exact command")
                validate_built_manifest_identity(
                    {"command": manifest.command, "code_commit": manifest.code_commit},
                    implementation_commit=(
                        expected_implementation_commit or self.implementation_commit
                    ),
                    exact_command=exact_command,
                )
            direct_manifests[str(manifest_path.resolve())] = self._sha256(manifest_path)
            manifest_compatibility_paths[str(manifest_path.resolve())] = str(
                self._compatibility_path(manifest_path, stage)
            )
            actual_ids.add(manifest.table_id)
            outputs.append(
                {
                    "path": manifest.destination,
                    "sha256": manifest.output_sha256,
                    "schema_path": manifest.schema_path,
                    "schema_sha256": manifest.schema_sha256,
                    "table_id": manifest.table_id,
                    "rows": manifest.rows,
                    "bytes": manifest.bytes,
                    "compatibility_path": str(
                        self._compatibility_path(Path(manifest.destination), stage)
                    ),
                    "compatibility_schema_path": str(
                        self._compatibility_path(Path(manifest.schema_path), stage)
                    ),
                }
            )
            for artifact in manifest.input_artifacts:
                path = Path(artifact.path)
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"manifest input is missing or unsafe: {path}")
                resolved_input = str(path.resolve())
                record = {
                    "sha256": artifact.sha256,
                    "bytes": path.stat().st_size,
                    "parent_manifest_sha256": artifact.parent_manifest_sha256,
                }
                if self._sha256(path) != artifact.sha256 or path.stat().st_size != artifact.bytes:
                    raise ValueError(f"manifest input hash changed: {path}")
                if artifact.parent_manifest_sha256 is not None:
                    parent = path.with_name(f"{path.name}.manifest.json")
                    if not parent.is_file() or self._sha256(parent) != artifact.parent_manifest_sha256:
                        raise ValueError(f"manifest parent hash changed: {path}")
                existing = input_artifacts.get(resolved_input)
                if existing is not None and existing != record:
                    raise ValueError(f"inconsistent input artifact binding: {path}")
                input_artifacts[resolved_input] = record
                table_id = self._manifest_table_id(path)
                if table_id is not None and table_id not in set(node.output_ids):
                    direct_inputs.setdefault(table_id, []).append(
                        {
                            "path": resolved_input,
                            "sha256": artifact.sha256,
                            "manifest_sha256": artifact.parent_manifest_sha256,
                        }
                    )
        if manifest_paths and actual_ids != set(node.output_ids):
            missing = sorted(set(node.output_ids) - actual_ids)
            extra = sorted(actual_ids - set(node.output_ids))
            raise ValueError(f"direct output ID mismatch: missing={missing}, extra={extra}")
        authority_paths = tuple(self._translate(path, stage) for path in node.authority_files)
        authority_files = self._hash_files(authority_paths, "authority file")
        for path, digest in authority_files.items():
            candidate = Path(path)
            outputs.append(
                {
                    "path": path,
                    "sha256": digest,
                    "bytes": candidate.stat().st_size,
                    "compatibility_path": str(
                        self._compatibility_path(candidate, stage)
                    ),
                }
            )
        if node.output_ids and not outputs:
            raise ValueError("node declares output IDs without verified authorities")
        if node.direct_input_ids and set(direct_inputs) != set(node.direct_input_ids):
            raise ValueError(
                "direct input ID mismatch: "
                f"expected={sorted(node.direct_input_ids)}, actual={sorted(direct_inputs)}"
            )
        return {
            "outputs": outputs,
            "direct_manifest_hashes": direct_manifests,
            "manifest_compatibility_paths": manifest_compatibility_paths,
            "input_artifacts": input_artifacts,
            "direct_input_bindings": direct_inputs,
            "executable_input_hashes": self._hash_files(
                node.executable_inputs, "executable input", relative=True
            ),
            "raw_input_hashes": self._hash_files(
                self._raw_paths(node), "raw input", relative=True
            ),
            "config_hashes": self._hash_files(node.config_inputs, "config input", relative=True),
            "contract_hashes": self._hash_files(node.contract_inputs, "contract input", relative=True),
            "taxonomy_hashes": self._hash_files(node.taxonomy_inputs, "taxonomy input", relative=True),
            "scaler_hashes": self._hash_files(node.scaler_inputs, "scaler input", relative=True),
        }

    def _current_state_path(self, name: str) -> Path:
        current = self.registry_root / name / "CURRENT.json"
        if current.is_symlink() or not current.is_file():
            raise ValueError("missing_current_pointer")
        try:
            payload = json.loads(current.read_text(encoding="utf-8"))
            build_id = _safe_token(str(payload["build_id"]), "build_id")
            version_value = str(payload["version_path"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid_current_pointer") from exc
        expected = (self.registry_root / name / "versions" / build_id).resolve()
        if Path(version_value).resolve() != expected:
            raise ValueError("mismatched_current_pointer")
        state = expected / "STATE.json"
        if expected.is_symlink() or state.is_symlink() or not state.is_file():
            raise ValueError("missing_current_state")
        return state

    def _parent_hashes(self, node: BuildNode) -> dict[str, str]:
        return {
            dependency: sha256_file(self._current_state_path(dependency))
            for dependency in node.dependencies
        }

    def _exact_argv(self, node: BuildNode, stage: BuildStage) -> tuple[str, ...]:
        if not node.command_args:
            return ()
        return (
            sys.executable,
            "-m",
            "green_debt.cli",
            *node.command_args,
            "--data-root",
            str(stage.data_root),
        )

    def _exact_manifest_command(self, node: BuildNode, stage: BuildStage | None) -> str | None:
        if not node.command_args or stage is None:
            return None
        return " ".join(
            (
                "python",
                "-m",
                "green_debt.cli",
                *node.command_args,
                "--data-root",
                str(stage.data_root),
            )
        )

    def _stage_for_bundle(self, bundle: Path) -> BuildStage:
        return BuildStage(self.data_root, self.code_root, bundle / "data", bundle / "code")

    def _copy_code_tree(self, destination: Path) -> None:
        ignored = shutil.ignore_patterns(".git", ".venv", ".superpowers", "__pycache__", "*.pyc")
        shutil.copytree(self.code_root, destination, ignore=ignored, dirs_exist_ok=True)

    def _link_parent_bundle(self, stage: BuildStage, dependency: str) -> None:
        state_path = self._current_state_path(dependency)
        bundle = state_path.parent / "bundle"
        if not bundle.is_dir() or bundle.is_symlink():
            return
        for prefix, destination_root in (("data", stage.data_root), ("code", stage.code_root)):
            source_root = bundle / prefix
            if not source_root.is_dir():
                continue
            for source in sorted(source_root.rglob("*")):
                if not source.is_file() or source.is_symlink():
                    continue
                target = destination_root / source.relative_to(source_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    if target.is_dir() and not target.is_symlink():
                        continue
                    target.unlink()
                target.symlink_to(source)

    def _prepare_stage(self, node: BuildNode, root: Path) -> BuildStage:
        stage = BuildStage(self.data_root, self.code_root, root / "data", root / "code")
        stage.data_root.mkdir(parents=True)
        if node.command_args:
            self._copy_code_tree(stage.code_root)
        else:
            stage.code_root.mkdir(parents=True)
        raw_target = stage.data_root / "04_原始数据"
        raw_source = self.data_root / "04_原始数据"
        if raw_source.is_dir() and not raw_target.exists():
            raw_target.symlink_to(raw_source)
        for dependency in node.dependencies:
            self._link_parent_bundle(stage, dependency)
        return stage

    def _source_to_bundle(self, source: Path, stage: BuildStage, bundle: Path) -> Path:
        for root, prefix in ((stage.code_root, "code"), (stage.data_root, "data")):
            try:
                return bundle / prefix / source.absolute().relative_to(root.absolute())
            except ValueError:
                continue
        raise ValueError(f"stage output escaped isolated roots: {source}")

    def _copy_output_bundle(
        self,
        node: BuildNode,
        stage: BuildStage,
        bundle: Path,
        *,
        logical_bundle: Path | None = None,
    ) -> None:
        logical = bundle if logical_bundle is None else logical_bundle
        manifest_paths = self._manifest_paths(node, stage=stage)
        files: set[Path] = set(self._translate(path, stage) for path in node.authority_files)
        manifest_payloads: list[tuple[Path, dict[str, object]]] = []
        for manifest_path in manifest_paths:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            files.update((manifest_path, Path(str(payload["destination"])), Path(str(payload["schema_path"]))))
            manifest_payloads.append((manifest_path, payload))
        for source in files:
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"staged output bundle member is missing or unsafe: {source}")
            destination = self._source_to_bundle(source, stage, bundle)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        output_mapping = {
            source.absolute(): self._source_to_bundle(source, stage, logical)
            for source in files
        }
        physical_manifest_by_logical: dict[Path, Path] = {}
        for manifest_path, payload in manifest_payloads:
            copied_manifest = self._source_to_bundle(manifest_path, stage, bundle)
            logical_manifest = self._source_to_bundle(manifest_path, stage, logical)
            physical_manifest_by_logical[logical_manifest] = copied_manifest
            payload["destination"] = str(
                self._source_to_bundle(Path(str(payload["destination"])), stage, logical)
            )
            payload["schema_path"] = str(
                self._source_to_bundle(Path(str(payload["schema_path"])), stage, logical)
            )
            artifacts = payload.get("input_artifacts")
            if not isinstance(artifacts, list):
                raise ValueError("staged manifest input_artifacts are invalid")
            for artifact in artifacts:
                if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                    raise ValueError("staged manifest input artifact is invalid")
                artifact_path = Path(str(artifact["path"]))
                stable_path: Path | None = None
                for staged_root, source_root in (
                    (stage.data_root.absolute(), self.data_root),
                    (stage.code_root.absolute(), self.code_root),
                ):
                    try:
                        relative = artifact_path.absolute().relative_to(staged_root)
                    except ValueError:
                        continue
                    stable_path = output_mapping.get(artifact_path.absolute())
                    if stable_path is None:
                        candidate = source_root / relative
                        if (
                            candidate.is_file()
                            and self._sha256(candidate) == str(artifact.get("sha256"))
                        ):
                            stable_path = candidate.resolve()
                    if stable_path is None:
                        raise ValueError(
                            f"staged manifest input has no stable authority: {artifact_path}"
                        )
                    break
                if stable_path is not None:
                    artifact["path"] = str(stable_path)
            _write_bytes_atomic(copied_manifest, _canonical_json_bytes(payload))
        # Rebind parent-manifest hashes after destinations and stable input paths
        # have been rewritten. Repeating propagates same-node dependencies.
        for _ in range(len(manifest_payloads) + 1):
            changed = False
            for manifest_path, payload in manifest_payloads:
                copied_manifest = self._source_to_bundle(manifest_path, stage, bundle)
                for artifact in payload["input_artifacts"]:
                    if artifact.get("parent_manifest_sha256") is None:
                        continue
                    stable = Path(str(artifact["path"]))
                    sidecar = stable.with_name(f"{stable.name}.manifest.json")
                    physical_sidecar = physical_manifest_by_logical.get(sidecar, sidecar)
                    if not physical_sidecar.is_file():
                        raise ValueError(f"stable parent manifest is missing: {sidecar}")
                    digest = sha256_file(physical_sidecar)
                    if artifact.get("parent_manifest_sha256") != digest:
                        artifact["parent_manifest_sha256"] = digest
                        changed = True
                parent_hashes = sorted(
                    str(artifact["parent_manifest_sha256"])
                    for artifact in payload["input_artifacts"]
                    if artifact.get("parent_manifest_sha256") is not None
                )
                if payload.get("parent_manifest_hashes") != parent_hashes:
                    payload["parent_manifest_hashes"] = parent_hashes
                    changed = True
                _write_bytes_atomic(copied_manifest, _canonical_json_bytes(payload))
            if not changed:
                break
        else:
            raise ValueError("same-node manifest lineage did not stabilize")

    def _publish_bundle(self, node: BuildNode, stage: BuildStage, *, action: str) -> PublishedVersion:
        nonce = hashlib.sha256(f"{node.name}:{action}:{_utc_now()}".encode()).hexdigest()[:16]
        build_id = f"{action}-{self.implementation_commit[:12]}-{nonce}"
        final_version = self.registry_root / node.name / "versions" / build_id

        def writer(version: Path) -> dict[str, object]:
            bundle = version / "bundle"
            self._copy_output_bundle(
                node,
                stage,
                bundle,
                logical_bundle=final_version / "bundle",
            )
            return {"bundle": "bundle"}

        def validate_and_write_state(version: Path) -> None:
            bundle_stage = self._stage_for_bundle(version / "bundle")
            snapshot = self._authority_snapshot(
                node,
                stage=bundle_stage,
                require_built_identity=action == "built" and bool(node.command_args),
                expected_manifest_command=self._exact_manifest_command(node, stage),
            )
            payload = {
                "schema_version": "2.0.0",
                "node": node.name,
                "action": action,
                "implementation_commit": self.implementation_commit,
                "output_ids": list(node.output_ids),
                "parent_state_hashes": self._parent_hashes(node),
                "command_args": list(node.command_args),
                "executed_argv": (
                    list(self._exact_argv(node, stage)) if action == "built" else []
                ),
                "build_data_root": str(stage.data_root) if action == "built" else None,
                "manifest_command": (
                    self._exact_manifest_command(node, stage) if action == "built" else None
                ),
                **snapshot,
            }
            state_bytes = _canonical_json_bytes(payload)
            _write_bytes_atomic(version / "STATE.json", state_bytes)

        return publish_version(
            self.registry_root,
            node.name,
            build_id,
            writer,
            validator=validate_and_write_state,
        )

    def _publish_gate(self, node: BuildNode) -> PublishedVersion:
        gate = node.approval_gate
        if gate is None:
            raise ValueError("not an approval gate")
        approval = self._approval_snapshot(gate, run_semantic=True)
        nonce = hashlib.sha256(
            (sha256_file(gate.receipt_path) + json.dumps(self._parent_hashes(node), sort_keys=True)).encode()
        ).hexdigest()[:16]
        build_id = f"approved-{self.implementation_commit[:12]}-{nonce}"

        def writer(version: Path) -> dict[str, object]:
            evidence = version / "bundle/code/06_结果"
            evidence.mkdir(parents=True)
            copied: dict[str, str] = {}
            for source in (gate.receipt_path, *gate.support_paths):
                destination = evidence / source.name
                shutil.copy2(source, destination)
                copied[source.name] = sha256_file(destination)
            payload = {
                "schema_version": "2.0.0",
                "node": node.name,
                "action": "approved_gate",
                "implementation_commit": self.implementation_commit,
                "output_ids": [],
                "parent_state_hashes": self._parent_hashes(node),
                "approval_number": gate.number,
                **approval,
                "command_args": [],
                "executed_argv": [],
                "manifest_command": None,
                "outputs": [],
                "direct_manifest_hashes": {},
                "manifest_compatibility_paths": {},
                "input_artifacts": {},
                "direct_input_bindings": {},
                "executable_input_hashes": self._hash_files(node.executable_inputs, "executable input", relative=True),
                "raw_input_hashes": {},
                "config_hashes": {},
                "contract_hashes": {},
                "taxonomy_hashes": {},
                "scaler_hashes": {},
            }
            _write_bytes_atomic(version / "STATE.json", _canonical_json_bytes(payload))
            return {"approval_receipt_sha256": sha256_file(gate.receipt_path)}

        return publish_version(self.registry_root, node.name, build_id, writer)

    def _approval_snapshot(
        self, gate: ApprovalGate, *, run_semantic: bool
    ) -> dict[str, object]:
        """Verify immutable receipt/support blobs and their approved Git lineage."""

        try:
            receipt = json.loads(gate.receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ApprovalRequired(
                f"checkpoint{gate.number} approval required: receipt unreadable"
            ) from exc
        passed = receipt.get("passed") is True or (
            isinstance(receipt.get("receipt"), Mapping)
            and receipt["receipt"].get("passed") is True
        )
        if int(receipt.get("number", -1)) != gate.number or not passed:
            raise ApprovalRequired(
                f"checkpoint{gate.number} approval required: receipt invalid"
            )
        support_hashes: dict[str, str] = {}
        for path in gate.support_paths:
            if path.is_symlink() or not path.is_file():
                raise ApprovalRequired(
                    f"checkpoint{gate.number} approval required: support missing"
                )
            support_hashes[self._relative_identity(path)] = self._sha256(path)
        receipt_hash = self._sha256(gate.receipt_path)
        commit_value = receipt.get("implementation_commit") or receipt.get("git_commit")
        if (
            not isinstance(commit_value, str)
            or len(commit_value) != 40
            or any(character not in "0123456789abcdef" for character in commit_value)
        ):
            raise ApprovalRequired(
                f"checkpoint{gate.number} approval required: receipt commit invalid"
            )
        git_checked = False
        git_probe = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=self.code_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if git_probe.returncode == 0 and git_probe.stdout.strip() == "true":
            git_checked = True
            for commit in (commit_value, self.implementation_commit):
                exists = subprocess.run(
                    ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                    cwd=self.code_root,
                    check=False,
                )
                if exists.returncode != 0:
                    raise ApprovalRequired(
                        f"checkpoint{gate.number} approval required: commit is unavailable"
                    )
            ancestor = subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit_value, self.implementation_commit],
                cwd=self.code_root,
                check=False,
            )
            if ancestor.returncode != 0:
                raise ApprovalRequired(
                    f"checkpoint{gate.number} approval required: lineage is invalid"
                )
            for path in (gate.receipt_path, *gate.support_paths):
                relative = self._relative_identity(path)
                blob = subprocess.run(
                    ["git", "show", f"{self.implementation_commit}:{relative}"],
                    cwd=self.code_root,
                    check=False,
                    capture_output=True,
                )
                if blob.returncode != 0 or blob.stdout != path.read_bytes():
                    raise ApprovalRequired(
                        f"checkpoint{gate.number} approval required: evidence blob is untracked or changed"
                    )
        elif commit_value != self.implementation_commit:
            raise ApprovalRequired(
                f"checkpoint{gate.number} approval required: lineage cannot be verified"
            )
        if gate.number == 2:
            checks = receipt.get("checks")
            if not isinstance(checks, list):
                raise ApprovalRequired("checkpoint2 approval required: checks missing")
            by_evidence = {
                str(item.get("evidence")): item
                for item in checks
                if isinstance(item, Mapping)
            }
            for path in gate.support_paths:
                relative = self._relative_identity(path)
                item = by_evidence.get(relative)
                if (
                    not isinstance(item, Mapping)
                    or item.get("status") != "pass"
                    or item.get("hash_algorithm") != "sha256"
                    or item.get("hash_kind") != "file_sha256"
                    or item.get("hash") != support_hashes[relative]
                ):
                    raise ApprovalRequired(
                        "checkpoint2 approval required: support hash binding differs"
                    )
        if run_semantic:
            if gate.semantic_verifier is None:
                raise ApprovalRequired(
                    f"checkpoint{gate.number} approval required: semantic verifier missing"
                )
            try:
                verified = gate.semantic_verifier()
            except Exception as exc:
                raise ApprovalRequired(
                    f"checkpoint{gate.number} approval required: verify-only failed"
                ) from exc
            if verified is False:
                raise ApprovalRequired(
                    f"checkpoint{gate.number} approval required: verify-only failed"
                )
        return {
            "approval_receipt_sha256": receipt_hash,
            "approval_support_hashes": support_hashes,
            "approval_receipt_commit": commit_value,
            "approval_git_lineage_verified": git_checked,
        }

    @staticmethod
    def _replace_compatibility_mirror(
        target: Path, authority: Path, *, use_symlink: bool
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.mirror.{os.getpid()}")
        temporary.unlink(missing_ok=True)
        if use_symlink:
            temporary.symlink_to(authority)
        else:
            shutil.copy2(authority, temporary)
        os.replace(temporary, target)

    def _update_compatibility_mirrors(self, published: PublishedVersion) -> None:
        state = json.loads((published.version_path / "STATE.json").read_text(encoding="utf-8"))
        for output in state["outputs"]:
            compatibility = Path(str(output["compatibility_path"]))
            if compatibility.is_relative_to(self.code_root):
                # Tracked dictionaries and frozen approval artifacts are not
                # runtime mirrors. Descendants consume the CURRENT bundle
                # injected into their isolated code copy instead.
                continue
            self._replace_compatibility_mirror(
                compatibility,
                Path(str(output["path"])),
                use_symlink=False,
            )
            if output.get("schema_path") is not None:
                compatibility_schema = Path(
                    str(output["compatibility_schema_path"])
                )
                self._replace_compatibility_mirror(
                    compatibility_schema,
                    Path(str(output["schema_path"])),
                    use_symlink=False,
                )
        for authority, compatibility in state["manifest_compatibility_paths"].items():
            compatibility_path = Path(str(compatibility))
            if compatibility_path.is_relative_to(self.code_root):
                continue
            authority_path = Path(authority)
            self._replace_compatibility_mirror(
                compatibility_path,
                authority_path,
                use_symlink=False,
            )
            try:
                manifest_payload = json.loads(
                    authority_path.read_text(encoding="utf-8")
                )
                table_id = _safe_token(
                    str(manifest_payload["table_id"]), "table_id"
                )
                schema_path = Path(str(manifest_payload["schema_path"]))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("published manifest metadata is invalid") from exc
            self._replace_compatibility_mirror(
                self.data_root
                / "05_中间数据/manifests"
                / f"{table_id}.manifest.json",
                authority_path,
                use_symlink=False,
            )
            self._replace_compatibility_mirror(
                self.data_root
                / "05_中间数据/schemas"
                / f"{table_id}.schema.json",
                schema_path,
                use_symlink=False,
            )

    def _manifest_status(self, name: str) -> tuple[bool, str]:
        node = self.graph.node(name)
        try:
            state_path = self._current_state_path(name)
        except ValueError as exc:
            return False, str(exc)
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or state.get("node") != name:
                return False, "invalid_current_state"
            if state.get("schema_version") != "2.0.0" or not (state_path.parent / "bundle").is_dir():
                return False, "legacy_state_requires_bundle_migration"
            if not self._implementation_binding_is_current(state):
                return False, "implementation_commit_lineage_changed"
            if node.approval_gate is not None:
                if state.get("action") != "approved_gate":
                    return False, "approval_receipt_not_verified"
                if state.get("parent_state_hashes") != self._parent_hashes(node):
                    return False, "approval_required_parent_state_changed"
                approval = self._approval_snapshot(
                    node.approval_gate, run_semantic=False
                )
                if any(state.get(key) != value for key, value in approval.items()):
                    return False, "approval_evidence_changed"
                return True, "verified_approved_gate"
            bundle_stage = self._stage_for_bundle(state_path.parent / "bundle")
            snapshot = self._authority_snapshot(
                node,
                stage=bundle_stage,
                require_built_identity=state.get("action") == "built" and bool(node.command_args),
                expected_manifest_command=(
                    str(state.get("manifest_command"))
                    if state.get("manifest_command") is not None
                    else None
                ),
                expected_implementation_commit=str(state.get("implementation_commit")),
            )
            if state.get("output_ids") != list(node.output_ids):
                return False, "output_ids_changed"
            for field in (
                "outputs",
                "direct_manifest_hashes",
                "manifest_compatibility_paths",
                "input_artifacts",
                "direct_input_bindings",
                "executable_input_hashes",
                "raw_input_hashes",
                "config_hashes",
                "contract_hashes",
                "taxonomy_hashes",
                "scaler_hashes",
            ):
                if state.get(field) != snapshot[field]:
                    return False, f"{field}_changed"
            if state.get("parent_state_hashes") != self._parent_hashes(node):
                return False, "parent_state_hashes_changed"
            if state.get("command_args") != list(node.command_args):
                return False, "command_args_changed"
            if state.get("action") == "built":
                recorded_root = state.get("build_data_root")
                if not isinstance(recorded_root, str):
                    return False, "build_data_root_missing"
                recorded_stage = BuildStage(
                    self.data_root,
                    self.code_root,
                    Path(recorded_root),
                    self.code_root,
                )
                if state.get("executed_argv") != list(self._exact_argv(node, recorded_stage)):
                    return False, "executed_argv_changed"
                if state.get("manifest_command") != self._exact_manifest_command(node, recorded_stage):
                    return False, "manifest_command_changed"
        except (OSError, ValueError, ApprovalRequired, json.JSONDecodeError):
            return False, "authority_verification_failed"
        return True, "verified_current"

    def _implementation_binding_is_current(
        self, state: Mapping[str, object]
    ) -> bool:
        recorded = state.get("implementation_commit")
        if (
            not isinstance(recorded, str)
            or len(recorded) != 40
            or any(character not in "0123456789abcdef" for character in recorded)
        ):
            return False
        if recorded == self.implementation_commit:
            return True
        probe = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=self.code_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0 or probe.stdout.strip() != "true":
            # Small embedding repositories may supply synthetic commit IDs;
            # byte-for-byte executable/input bindings remain authoritative.
            return True
        return subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                recorded,
                self.implementation_commit,
            ],
            cwd=self.code_root,
            check=False,
        ).returncode == 0

    def status(self, target: str = "checkpoint3") -> tuple[BuildStatusItem, ...]:
        return self.graph.status(target, manifest_status=self._manifest_status)

    def build(self, target: str) -> tuple[BuildRunItem, ...]:
        results: list[BuildRunItem] = []
        rebuilt: set[str] = set()
        for name in self.graph.order(target):
            node = self.graph.node(name)
            upstream_rebuilt = any(
                dependency in rebuilt for dependency in node.dependencies
            )
            current, reason = self._manifest_status(name)
            if current and not upstream_rebuilt:
                results.append(BuildRunItem(name, "current", reason))
                continue
            if node.approval_gate is not None:
                if reason in {
                    "missing_current_pointer",
                    "legacy_state_requires_bundle_migration",
                    "approval_receipt_not_verified",
                }:
                    self._publish_gate(node)
                    results.append(BuildRunItem(name, "adopted", "approved_receipt_verified"))
                    continue
                raise ApprovalRequired(f"{name} approval required after parent authority change")
            if not upstream_rebuilt and reason in {
                "missing_current_pointer",
                "legacy_state_requires_bundle_migration",
            }:
                # One-time migration verifies fixed reviewed bytes, copies the
                # complete bundle, and never treats the fixed mirror as CURRENT.
                try:
                    self._authority_snapshot(node)
                except ValueError:
                    # A reviewed legacy output whose recorded inputs no longer
                    # match current authorities is stale, not adoptable.  Let
                    # the normal isolated-build path reconstruct it below.
                    pass
                else:
                    stage = BuildStage(
                        self.data_root, self.code_root, self.data_root, self.code_root
                    )
                    self._publish_bundle(node, stage, action="adopted")
                    results.append(
                        BuildRunItem(name, "adopted", "reviewed_authority_verified")
                    )
                    continue
            tmp_parent = self.data_root / "05_中间数据/_tmp"
            tmp_parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f"build.{node.name}.", dir=tmp_parent))
            try:
                stage = self._prepare_stage(node, temporary)
                if node.staging_builder is None:
                    if node.output_ids or node.authority_files:
                        raise BuildPublicationError(f"{name} lacks an isolated staging builder")
                else:
                    node.staging_builder(stage)
                self._authority_snapshot(
                    node,
                    stage=stage,
                    require_built_identity=bool(node.command_args),
                )
                published = self._publish_bundle(node, stage, action="built")
                self._update_compatibility_mirrors(published)
            except BuildPublicationError:
                raise
            except Exception as exc:
                raise BuildPublicationError(f"isolated stage failed: {exc}") from exc
            finally:
                if temporary.is_dir() and not temporary.is_symlink():
                    shutil.rmtree(temporary)
            rebuilt.add(name)
            results.append(
                BuildRunItem(
                    name,
                    "built",
                    "upstream_rebuilt" if upstream_rebuilt else reason,
                )
            )
        return tuple(results)


class ReproductionMismatch(RuntimeError):
    """Raised when a rebuilt table differs from its reviewed authority."""


@dataclass(frozen=True)
class ReproductionFingerprint:
    rows: int
    content_sha256: str
    schema_sha256: str
    partition_order: tuple[str, ...]
    partition_rows: tuple[int, ...]


_REPRODUCTION_NUMERIC_TOLERANCE = 1e-12
_REPRODUCTION_DERIVED_HASH_COLUMNS = frozenset(
    {"canonical_hash", "giu_scaler_hash", "regression_bounds_hash"}
)


@dataclass(frozen=True)
class ReproductionContentComparison:
    """Bounded logical-row comparison after the fast canonical hash path."""

    matched: bool
    mismatch_rows: int
    max_scaled_float_error: float
    numeric_tolerance: float = _REPRODUCTION_NUMERIC_TOLERANCE


def _canonical_scalar(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical fingerprint rejects non-finite values")
        return {"float_hex": value.hex()}
    if isinstance(value, (date, datetime)):
        return {"iso8601": value.isoformat()}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, list):
        return [_canonical_scalar(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical_scalar(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return {"repr": repr(value)}


def _parquet_paths(root: Path) -> tuple[tuple[Path, str], ...]:
    resolved = root.resolve()
    if resolved.is_file():
        if resolved.suffix != ".parquet":
            raise ValueError("fingerprint target must be Parquet")
        return ((resolved, resolved.name),)
    if not resolved.is_dir():
        raise FileNotFoundError(root)
    paths = tuple(
        (path, path.relative_to(resolved).as_posix())
        for path in sorted(resolved.rglob("*.parquet"))
        if path.is_file() and not path.is_symlink()
    )
    if not paths:
        raise ValueError("fingerprint target contains no Parquet files")
    return paths


def canonical_parquet_fingerprint(
    root: Path,
    *,
    primary_key: tuple[str, ...],
    temporary_root: Path | None = None,
) -> ReproductionFingerprint:
    """Boundedly hash logical rows with external deterministic key ordering."""

    files = _parquet_paths(root)
    digest = hashlib.sha256()
    schema_digest = hashlib.sha256()
    schemas: list[dict[str, str]] = []
    partition_rows: list[int] = []
    total = 0
    for path, relative in files:
        parquet_schema = pl.read_parquet_schema(path)
        columns = tuple(parquet_schema)
        missing = set(primary_key) - set(columns)
        if missing:
            raise ValueError(f"primary key missing from Parquet: {sorted(missing)}")
        schema = {name: str(dtype) for name, dtype in parquet_schema.items()}
        schemas.append(schema)
        schema_digest.update(_canonical_json_bytes(schema))
        digest.update(_canonical_json_bytes({"partition": relative, "schema": schema}))
        connection = duckdb.connect()
        try:
            connection.execute("SET memory_limit='256MB'")
            connection.execute("SET threads=1")
            if temporary_root is not None:
                spill = temporary_root / hashlib.sha256(relative.encode()).hexdigest()[:16]
                spill.mkdir(parents=True, exist_ok=True)
                quoted_spill = str(spill).replace("'", "''")
                connection.execute(f"SET temp_directory='{quoted_spill}'")
            order = ""
            if primary_key:
                identifiers = ", ".join(
                    f'"{name.replace(chr(34), chr(34) * 2)}" NULLS LAST'
                    for name in primary_key
                )
                order = f" ORDER BY {identifiers}"
            reader = connection.execute(
                f"SELECT * FROM read_parquet(?){order}", [str(path)]
            ).to_arrow_reader(65_536)
            rows = 0
            for batch in reader:
                for row in batch.to_pylist():
                    digest.update(
                        _canonical_json_bytes(
                            {
                                name: _canonical_scalar(row[name])
                                for name in columns
                                if name not in _REPRODUCTION_DERIVED_HASH_COLUMNS
                            }
                        )
                    )
                rows += batch.num_rows
        finally:
            connection.close()
        partition_rows.append(rows)
        total += rows
    # Repeated schemas are retained intentionally: partition/schema alignment
    # is part of the deterministic reproduction contract.
    schema_digest.update(_canonical_json_bytes(schemas))
    return ReproductionFingerprint(
        rows=total,
        content_sha256=digest.hexdigest(),
        schema_sha256=schema_digest.hexdigest(),
        partition_order=tuple(relative for _, relative in files),
        partition_rows=tuple(partition_rows),
    )


def compare_reproduction_fingerprints(
    expected: ReproductionFingerprint,
    actual: ReproductionFingerprint,
) -> None:
    if expected.rows != actual.rows:
        raise ReproductionMismatch("row count mismatch")
    if expected.partition_order != actual.partition_order:
        raise ReproductionMismatch("partition ordering mismatch")
    if expected.partition_rows != actual.partition_rows:
        raise ReproductionMismatch("partition row-count mismatch")
    if expected.schema_sha256 != actual.schema_sha256:
        raise ReproductionMismatch("schema hash mismatch")
    if expected.content_sha256 != actual.content_sha256:
        raise ReproductionMismatch("content hash mismatch")


def _duckdb_identifier(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


def compare_reproduction_parquet_content(
    authority_root: Path,
    reproduced_root: Path,
    *,
    primary_key: tuple[str, ...],
    temporary_root: Path | None = None,
) -> ReproductionContentComparison:
    """Compare sorted logical rows, tolerating only machine-scale float noise.

    Non-floating values and null placement remain exact.  Float differences are
    accepted only when ``abs(a-b) / max(abs(a), abs(b), 1) <= 1e-12``.  Derived
    self-hash columns are excluded because they necessarily change when their
    machine-level float serialization changes; their source values are compared.
    """

    if not primary_key:
        raise ValueError("bounded reproduction comparison requires a primary key")
    authority_files = _parquet_paths(authority_root)
    reproduced_files = _parquet_paths(reproduced_root)
    authority_by_partition = {relative: path for path, relative in authority_files}
    reproduced_by_partition = {relative: path for path, relative in reproduced_files}
    if tuple(authority_by_partition) != tuple(reproduced_by_partition):
        raise ReproductionMismatch("partition ordering mismatch")

    mismatch_rows = 0
    max_scaled_float_error = 0.0
    row_number_name = "__green_debt_reproduction_row_number__"
    row_number = _duckdb_identifier(row_number_name)
    tolerance_sql = format(_REPRODUCTION_NUMERIC_TOLERANCE, ".17g")

    for relative, authority_path in authority_by_partition.items():
        reproduced_path = reproduced_by_partition[relative]
        authority_schema = pl.read_parquet_schema(authority_path)
        reproduced_schema = pl.read_parquet_schema(reproduced_path)
        if list(authority_schema.items()) != list(reproduced_schema.items()):
            raise ReproductionMismatch("schema mismatch")
        if row_number_name in authority_schema:
            raise ValueError(f"reserved reproduction column exists: {row_number_name}")
        missing = set(primary_key) - set(authority_schema)
        if missing:
            raise ValueError(f"primary key missing from Parquet: {sorted(missing)}")

        columns = tuple(
            name
            for name in authority_schema
            if name not in _REPRODUCTION_DERIVED_HASH_COLUMNS
        )
        float_columns = tuple(
            name
            for name in columns
            if str(authority_schema[name]) in {"Float32", "Float64"}
        )
        exact_columns = tuple(name for name in columns if name not in float_columns)
        order_sql = ", ".join(
            f"{_duckdb_identifier(name)} NULLS LAST" for name in primary_key
        )

        float_mismatches: list[str] = []
        scaled_errors: list[str] = []
        for name in float_columns:
            identifier = _duckdb_identifier(name)
            left = f"a.{identifier}"
            right = f"b.{identifier}"
            scale = f"greatest(abs({left}), abs({right}), 1.0)"
            float_mismatches.append(
                "("
                f"({left} IS NULL) <> ({right} IS NULL) OR "
                f"({left} IS NOT NULL AND {right} IS NOT NULL AND "
                f"(NOT isfinite({left}) OR NOT isfinite({right}) OR "
                f"abs({left} - {right}) > {tolerance_sql} * {scale}))"
                ")"
            )
            scaled_errors.append(
                "CASE WHEN "
                f"{left} IS NOT NULL AND {right} IS NOT NULL "
                f"AND isfinite({left}) AND isfinite({right}) "
                f"THEN abs({left} - {right}) / {scale} ELSE NULL END"
            )
        exact_mismatches = [
            f"a.{_duckdb_identifier(name)} IS DISTINCT FROM "
            f"b.{_duckdb_identifier(name)}"
            for name in exact_columns
        ]
        mismatch_terms = [
            f"a.{row_number} IS NULL",
            f"b.{row_number} IS NULL",
            *exact_mismatches,
            *float_mismatches,
        ]
        mismatch_sql = " OR ".join(f"({term})" for term in mismatch_terms)
        if not scaled_errors:
            max_error_sql = "0.0"
        elif len(scaled_errors) == 1:
            max_error_sql = scaled_errors[0]
        else:
            max_error_sql = f"greatest({', '.join(scaled_errors)})"

        connection = duckdb.connect()
        try:
            connection.execute("SET memory_limit='256MB'")
            connection.execute("SET threads=1")
            if temporary_root is not None:
                spill_key = hashlib.sha256(
                    f"{authority_root.resolve()}|{relative}".encode()
                ).hexdigest()[:16]
                spill = temporary_root / spill_key
                spill.mkdir(parents=True, exist_ok=True)
                quoted_spill = str(spill).replace("'", "''")
                connection.execute(f"SET temp_directory='{quoted_spill}'")
            query = f"""
                WITH authority AS (
                    SELECT row_number() OVER (ORDER BY {order_sql}) AS {row_number}, *
                    FROM read_parquet(?, hive_partitioning=false)
                ), reproduced AS (
                    SELECT row_number() OVER (ORDER BY {order_sql}) AS {row_number}, *
                    FROM read_parquet(?, hive_partitioning=false)
                )
                SELECT
                    coalesce(sum(CASE WHEN {mismatch_sql} THEN 1 ELSE 0 END), 0),
                    coalesce(max({max_error_sql}), 0.0)
                FROM authority AS a
                FULL OUTER JOIN reproduced AS b
                  ON a.{row_number} = b.{row_number}
            """
            partition_mismatches, partition_max_error = connection.execute(
                query, [str(authority_path), str(reproduced_path)]
            ).fetchone()
        finally:
            connection.close()
        mismatch_rows += int(partition_mismatches)
        max_scaled_float_error = max(
            max_scaled_float_error, float(partition_max_error)
        )

    return ReproductionContentComparison(
        matched=mismatch_rows == 0,
        mismatch_rows=mismatch_rows,
        max_scaled_float_error=max_scaled_float_error,
    )


def _receipt_binding(payload: Mapping[str, object]) -> str:
    stable = {key: value for key, value in payload.items() if key != "binding_sha256"}
    return hashlib.sha256(_canonical_json_bytes(stable)).hexdigest()


def _validate_scratch_target(
    data_root: Path,
    scratch_root: Path,
    *,
    workspace_root: Path | None,
) -> tuple[Path, Path]:
    lexical_data = data_root.absolute()
    lexical_intermediate = lexical_data / "05_中间数据"
    lexical_tmp = lexical_intermediate / "_tmp"
    lexical_scratch = scratch_root.absolute()
    for label, path in (
        ("data root", lexical_data),
        ("intermediate root", lexical_intermediate),
        ("scratch parent", lexical_tmp),
        ("reproduction scratch target", lexical_scratch),
    ):
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            raise
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} cannot be a symlink")
    data = lexical_data.resolve(strict=True)
    if not data.is_dir():
        raise ValueError("data root must be a real directory")
    try:
        scratch = lexical_scratch.resolve(strict=True)
    except FileNotFoundError:
        raise
    expected_parent = (data / "05_中间数据" / "_tmp").resolve(strict=True)
    if scratch.parent != expected_parent:
        raise ValueError("reproduction scratch target has the wrong parent")
    if not fnmatch.fnmatchcase(scratch.name, "reproduce.*") or scratch.name == "reproduce.":
        raise ValueError("reproduction scratch target has the wrong name")
    if not scratch.is_dir() or scratch == data or scratch == expected_parent:
        raise ValueError("reproduction scratch target is broader than one child")
    if workspace_root is not None:
        workspace = workspace_root.resolve(strict=True)
        if scratch == workspace or workspace.is_relative_to(scratch):
            raise ValueError("reproduction target cannot contain the workspace root")
    return data, scratch


def create_reproduction_receipt(
    *,
    data_root: Path,
    scratch_root: Path,
    artifacts: Sequence[Mapping[str, object] | ReproductionFingerprint],
    workspace_root: Path | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Write the cleanup authority inside the exact validated scratch child."""

    data, scratch = _validate_scratch_target(
        data_root, scratch_root, workspace_root=workspace_root
    )
    normalized_artifacts: list[object] = []
    for artifact in artifacts:
        normalized_artifacts.append(
            asdict(artifact)
            if isinstance(artifact, ReproductionFingerprint)
            else dict(artifact)
        )
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "data_root": str(data),
        "scratch_root": str(scratch),
        "workspace_root": (
            str(workspace_root.resolve(strict=True))
            if workspace_root is not None
            else None
        ),
        "artifacts": normalized_artifacts,
    }
    if metadata is not None:
        overlap = set(payload) & set(metadata)
        if overlap or "binding_sha256" in metadata:
            raise ValueError(f"reproduction metadata collides with receipt fields: {sorted(overlap)}")
        payload.update(dict(metadata))
    payload["binding_sha256"] = _receipt_binding(payload)
    receipt = scratch / "reproduction_receipt.json"
    _write_bytes_atomic(receipt, _canonical_json_bytes(payload))
    return receipt


def _validated_cleanup_paths(receipt_path: Path) -> tuple[Path, Path]:
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise FileNotFoundError(f"reproduction receipt is missing or unsafe: {receipt_path}")
    if receipt_path.name != "reproduction_receipt.json":
        raise ValueError("reproduction receipt must use the exact receipt name")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("reproduction receipt is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("reproduction receipt root is invalid")
    binding = payload.get("binding_sha256")
    if not isinstance(binding, str) or binding != _receipt_binding(payload):
        raise ValueError("reproduction receipt binding mismatch")
    data_value = payload.get("data_root")
    scratch_value = payload.get("scratch_root")
    workspace_value = payload.get("workspace_root")
    if not isinstance(data_value, str) or not isinstance(scratch_value, str):
        raise ValueError("reproduction receipt paths are invalid")
    if workspace_value is not None and not isinstance(workspace_value, str):
        raise ValueError("reproduction receipt workspace path is invalid")
    data, scratch = _validate_scratch_target(
        Path(data_value),
        Path(scratch_value),
        workspace_root=Path(workspace_value) if workspace_value is not None else None,
    )
    if receipt_path.resolve().parent != scratch:
        raise ValueError("reproduction receipt does not reside in the recorded scratch")
    return data, scratch


def _remove_directory_contents_fd(directory_fd: int) -> None:
    """Recursively unlink without ever resolving a child outside the opened fd."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    for name in os.listdir(directory_fd):
        child = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(child.st_mode):
            child_fd = os.open(
                name,
                os.O_RDONLY | directory_flag | nofollow,
                dir_fd=directory_fd,
            )
            try:
                _remove_directory_contents_fd(child_fd)
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)


@dataclass
class PreparedCleanup:
    """An fd-bound cleanup authority resistant to path substitution."""

    scratch: Path
    data_parent_fd: int
    data_fd: int
    intermediate_fd: int
    parent_fd: int
    scratch_fd: int
    data_name: str
    data_identity: tuple[int, int]
    intermediate_identity: tuple[int, int]
    parent_identity: tuple[int, int]
    device: int
    inode: int
    _closed: bool = False

    def close(self) -> None:
        if not self._closed:
            for descriptor in (
                self.scratch_fd,
                self.parent_fd,
                self.intermediate_fd,
                self.data_fd,
                self.data_parent_fd,
            ):
                os.close(descriptor)
            self._closed = True

    def remove(self) -> Path:
        if self._closed:
            raise ValueError("prepared cleanup is already closed")
        try:
            data_entry = os.stat(
                self.data_name,
                dir_fd=self.data_parent_fd,
                follow_symlinks=False,
            )
            intermediate_entry = os.stat(
                "05_中间数据", dir_fd=self.data_fd, follow_symlinks=False
            )
            parent_entry = os.stat(
                "_tmp", dir_fd=self.intermediate_fd, follow_symlinks=False
            )
            if (
                (data_entry.st_dev, data_entry.st_ino) != self.data_identity
                or (intermediate_entry.st_dev, intermediate_entry.st_ino)
                != self.intermediate_identity
                or (parent_entry.st_dev, parent_entry.st_ino)
                != self.parent_identity
                or (os.fstat(self.parent_fd).st_dev, os.fstat(self.parent_fd).st_ino)
                != self.parent_identity
            ):
                raise ValueError("reproduction scratch parent chain changed before deletion")
            current = os.stat(
                self.scratch.name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
            opened = os.fstat(self.scratch_fd)
            identity = (self.device, self.inode)
            if (
                stat.S_ISLNK(current.st_mode)
                or (current.st_dev, current.st_ino) != identity
                or (opened.st_dev, opened.st_ino) != identity
            ):
                raise ValueError("reproduction scratch object changed before deletion")
            _remove_directory_contents_fd(self.scratch_fd)
            final = os.stat(
                self.scratch.name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
            if (final.st_dev, final.st_ino) != identity:
                raise ValueError("reproduction scratch object changed before deletion")
            os.rmdir(self.scratch.name, dir_fd=self.parent_fd)
            return self.scratch
        finally:
            self.close()


def prepare_reproduction_cleanup(receipt_path: Path) -> PreparedCleanup:
    """Validate a receipt and bind open fds to the exact directory object."""

    data, scratch = _validated_cleanup_paths(receipt_path)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    flags = os.O_RDONLY | directory_flag | nofollow
    data_parent_fd = os.open(data.parent, flags)
    try:
        data_fd = os.open(data.name, flags, dir_fd=data_parent_fd)
    except Exception:
        os.close(data_parent_fd)
        raise
    try:
        intermediate_fd = os.open("05_中间数据", flags, dir_fd=data_fd)
    except Exception:
        os.close(data_fd)
        os.close(data_parent_fd)
        raise
    try:
        parent_fd = os.open("_tmp", flags, dir_fd=intermediate_fd)
    except Exception:
        os.close(intermediate_fd)
        os.close(data_fd)
        os.close(data_parent_fd)
        raise
    try:
        scratch_fd = os.open(
            scratch.name,
            flags,
            dir_fd=parent_fd,
        )
    except Exception:
        os.close(parent_fd)
        os.close(intermediate_fd)
        os.close(data_fd)
        os.close(data_parent_fd)
        raise
    data_opened = os.fstat(data_fd)
    data_entry = os.stat(data.name, dir_fd=data_parent_fd, follow_symlinks=False)
    intermediate_opened = os.fstat(intermediate_fd)
    intermediate_entry = os.stat(
        "05_中间数据", dir_fd=data_fd, follow_symlinks=False
    )
    parent_opened = os.fstat(parent_fd)
    parent_entry = os.stat("_tmp", dir_fd=intermediate_fd, follow_symlinks=False)
    opened = os.fstat(scratch_fd)
    entry = os.stat(scratch.name, dir_fd=parent_fd, follow_symlinks=False)
    chain_mismatch = any(
        stat.S_ISLNK(candidate.st_mode)
        or (candidate.st_dev, candidate.st_ino) != (actual.st_dev, actual.st_ino)
        for candidate, actual in (
            (data_entry, data_opened),
            (intermediate_entry, intermediate_opened),
            (parent_entry, parent_opened),
            (entry, opened),
        )
    )
    if chain_mismatch:
        os.close(scratch_fd)
        os.close(parent_fd)
        os.close(intermediate_fd)
        os.close(data_fd)
        os.close(data_parent_fd)
        raise ValueError("reproduction scratch object changed before deletion")
    return PreparedCleanup(
        scratch=scratch,
        data_parent_fd=data_parent_fd,
        data_fd=data_fd,
        intermediate_fd=intermediate_fd,
        parent_fd=parent_fd,
        scratch_fd=scratch_fd,
        data_name=data.name,
        data_identity=(data_opened.st_dev, data_opened.st_ino),
        intermediate_identity=(intermediate_opened.st_dev, intermediate_opened.st_ino),
        parent_identity=(parent_opened.st_dev, parent_opened.st_ino),
        device=opened.st_dev,
        inode=opened.st_ino,
    )


def cleanup_reproduction(receipt_path: Path) -> Path:
    """Remove only the fd/receipt-bound ``reproduce.*`` directory object."""

    return prepare_reproduction_cleanup(receipt_path).remove()


def _tree_sha256(root: Path) -> tuple[str, int, int]:
    """Hash relative names, sizes, and bytes without following symlinks."""

    resolved = root.resolve(strict=True)
    digest = hashlib.sha256()
    files = 0
    total = 0
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"reproduction input tree contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(resolved).as_posix()
        size = path.stat().st_size
        file_hash = sha256_file(path)
        digest.update(
            _canonical_json_bytes(
                {"path": relative, "bytes": size, "sha256": file_hash}
            )
        )
        files += 1
        total += size
    return digest.hexdigest(), files, total


def _reproduction_manifest_paths(data_root: Path) -> tuple[Path, ...]:
    intermediate = data_root / "05_中间数据"
    roots = tuple(intermediate / layer for layer in ("normalized", "harmonized", "measures", "analysis"))
    return tuple(
        sorted(
            path
            for root in roots
            if root.is_dir()
            for path in root.rglob("*.parquet.manifest.json")
            if path.is_file() and not path.is_symlink()
        )
    )


def run_reproduction_check(
    *,
    code_root: Path,
    data_root: Path,
    scratch_root: Path,
    graph: BuildGraph | None = None,
    implementation_commit: str | None = None,
) -> Path:
    """Independently rebuild L1-L4 below scratch and compare each authority once."""

    data, scratch = _validate_scratch_target(
        data_root, scratch_root, workspace_root=code_root if code_root.exists() else None
    )
    existing = tuple(scratch.iterdir())
    if existing:
        raise ValueError("reproduction scratch must be an empty mktemp child")
    raw = data / "04_原始数据"
    if not raw.is_dir() or raw.is_symlink():
        raise ValueError("immutable L0 raw root is missing or unsafe")
    raw_before, raw_files, raw_bytes = _tree_sha256(raw)
    authority_manifests = _reproduction_manifest_paths(data)
    if not authority_manifests:
        raise ValueError("reproduction found no authoritative manifests")
    if implementation_commit is None:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=code_root, check=False,
            capture_output=True, text=True,
        )
        implementation_commit = completed.stdout.strip()
    if len(implementation_commit) != 40:
        raise ValueError("reproduction requires an exact implementation commit")
    rebuild_root = scratch / "rebuild"
    stage = BuildStage(data, code_root.resolve(), rebuild_root / "data", rebuild_root / "code")
    stage.data_root.mkdir(parents=True)
    controller_graph = graph or registered_build_graph(code_root=code_root, data_root=data)
    controller = BuildController(
        controller_graph,
        code_root=code_root,
        data_root=data,
        implementation_commit=implementation_commit,
    )
    if any(node.command_args for node in controller_graph.nodes):
        controller._copy_code_tree(stage.code_root)
    else:
        stage.code_root.mkdir(parents=True)
    (stage.data_root / "04_原始数据").symlink_to(raw)
    rebuild_commands: list[dict[str, object]] = []
    for name in controller_graph.order("checkpoint3") if graph is None else tuple(
        node.name for node in controller_graph.nodes
    ):
        node = controller_graph.node(name)
        if node.approval_gate is not None or name.startswith("checkpoint"):
            continue
        if node.staging_builder is None:
            raise ReproductionMismatch(f"node lacks independent staging builder: {name}")
        node.staging_builder(stage)
        controller._authority_snapshot(
            node,
            stage=stage,
            require_built_identity=bool(node.command_args),
        )
        command_record: dict[str, object] = {
            "node": name,
            "kind": "command" if node.command_args else "staging_builder",
        }
        if node.command_args:
            command_record["argv"] = list(controller._exact_argv(node, stage))
        rebuild_commands.append(command_record)
    reproduced_manifests = _reproduction_manifest_paths(stage.data_root)
    def manifest_identity(path: Path, root: Path) -> tuple[str, str]:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        return relative, verify_manifest(path).table_id

    authority_by_identity = {
        manifest_identity(path, data): path for path in authority_manifests
    }
    reproduced_by_identity = {
        manifest_identity(path, stage.data_root): path for path in reproduced_manifests
    }
    if len(authority_by_identity) != len(authority_manifests) or len(
        reproduced_by_identity
    ) != len(reproduced_manifests):
        raise ReproductionMismatch("reproduction manifest path/table identity is duplicated")
    if set(authority_by_identity) != set(reproduced_by_identity):
        raise ReproductionMismatch(
            "reproduced manifest identities differ from authority: "
            f"missing={sorted(set(authority_by_identity) - set(reproduced_by_identity))}, "
            f"extra={sorted(set(reproduced_by_identity) - set(authority_by_identity))}"
        )
    artifacts: list[dict[str, object]] = []
    fingerprint_tmp = scratch / "fingerprint_tmp"
    fingerprint_tmp.mkdir()
    for identity in sorted(authority_by_identity):
        manifest_relative, table_id = identity
        authority_manifest_path = authority_by_identity[identity]
        reproduced_manifest_path = reproduced_by_identity[identity]
        authority_manifest = verify_manifest(authority_manifest_path)
        reproduced_manifest = verify_manifest(reproduced_manifest_path)
        authority_output = Path(authority_manifest.destination)
        reproduced_output = Path(reproduced_manifest.destination)
        if authority_output.resolve() == reproduced_output.resolve():
            raise ReproductionMismatch("reproduction compared the same authority path")
        authority_fp = canonical_parquet_fingerprint(
            authority_output,
            primary_key=authority_manifest.primary_key,
            temporary_root=fingerprint_tmp / "authority",
        )
        reproduced_fp = canonical_parquet_fingerprint(
            reproduced_output,
            primary_key=reproduced_manifest.primary_key,
            temporary_root=fingerprint_tmp / "reproduced",
        )
        try:
            compare_reproduction_fingerprints(authority_fp, reproduced_fp)
        except ReproductionMismatch as exc:
            if str(exc) != "content hash mismatch":
                raise
            content_comparison = compare_reproduction_parquet_content(
                authority_output,
                reproduced_output,
                primary_key=authority_manifest.primary_key,
                temporary_root=fingerprint_tmp / "paired",
            )
            if not content_comparison.matched:
                raise ReproductionMismatch(
                    "content mismatch after bounded numeric comparison: "
                    f"mismatch_rows={content_comparison.mismatch_rows}, "
                    "max_scaled_float_error="
                    f"{content_comparison.max_scaled_float_error:.17g}"
                ) from exc
        else:
            content_comparison = ReproductionContentComparison(
                matched=True,
                mismatch_rows=0,
                max_scaled_float_error=0.0,
            )
        if authority_fp.rows != authority_manifest.rows or reproduced_fp.rows != reproduced_manifest.rows:
            raise ReproductionMismatch("manifest and canonical row count differ")
        artifacts.append(
            {
                "table_id": table_id,
                "manifest_identity": manifest_relative,
                "authority_manifest_path": str(authority_manifest_path.resolve()),
                "authority_manifest_sha256": sha256_file(authority_manifest_path),
                "reproduced_manifest_path": str(reproduced_manifest_path.resolve()),
                "reproduced_manifest_sha256": sha256_file(reproduced_manifest_path),
                "authority_path": str(authority_output.resolve()),
                "reproduced_path": str(reproduced_output.resolve()),
                "authority_bytes": authority_output.stat().st_size,
                "reproduced_bytes": reproduced_output.stat().st_size,
                "rows": authority_fp.rows,
                "primary_key": list(authority_manifest.primary_key),
                "partition_order": list(authority_fp.partition_order),
                "partition_rows": list(authority_fp.partition_rows),
                "schema_sha256": authority_fp.schema_sha256,
                "authority_content_sha256": authority_fp.content_sha256,
                "reproduced_content_sha256": reproduced_fp.content_sha256,
                "content_hash_kind": (
                    "logical_rows_sorted_by_primary_key_with_1e-12_float_tolerance"
                ),
                "content_hashes_equal": (
                    authority_fp.content_sha256 == reproduced_fp.content_sha256
                ),
                "numeric_tolerance": content_comparison.numeric_tolerance,
                "max_scaled_float_error": (
                    content_comparison.max_scaled_float_error
                ),
                "mismatch_rows": content_comparison.mismatch_rows,
                "matched": content_comparison.matched,
            }
        )
    raw_after, after_files, after_bytes = _tree_sha256(raw)
    if (raw_before, raw_files, raw_bytes) != (raw_after, after_files, after_bytes):
        raise ReproductionMismatch("immutable L0 raw tree changed during reproduction")
    usage = measure_layer_usage(data, audits_root=code_root / "06_结果")
    enforce_construction_capacity(usage)
    return create_reproduction_receipt(
        data_root=data,
        scratch_root=scratch,
        workspace_root=code_root if code_root.exists() else None,
        artifacts=artifacts,
        metadata={
            "created_at_utc": _utc_now(),
            "raw_tree_sha256_before": raw_before,
            "raw_tree_sha256_after": raw_after,
            "raw_files": raw_files,
            "raw_bytes": raw_bytes,
            "artifact_count": len(artifacts),
            "all_artifacts_matched": True,
            "implementation_commit": implementation_commit,
            "rebuild_commands": rebuild_commands,
            "scratch_bytes_before_receipt": sum(
                path.stat().st_size
                for path in scratch.rglob("*")
                if path.is_file() and not path.is_symlink()
            ),
            "intermediate_bytes": usage.intermediate_bytes,
            "intermediate_plus_scratch_bytes": usage.intermediate_bytes
            + usage.scratch_bytes,
            "project_bytes": usage.project_bytes,
            "publishes_current": False,
        },
    )


CHECKPOINT3_CRITERIA = (
    "raw_hashes",
    "schemas_and_manifests",
    "taxonomy_counts",
    "duplicate_authoritative_keys",
    "source_missing_zero_rules",
    "single_frozen_gad_scaler",
    "timing_overlap_violations",
    "leakage_violations",
    "fresh_full_test_exit_code",
    "capacity",
)


def validate_checkpoint3_facts(facts: Mapping[str, object]) -> None:
    """Fail closed unless the exact ten frozen Checkpoint-3 criteria pass."""

    if set(facts) != set(CHECKPOINT3_CRITERIA):
        missing = sorted(set(CHECKPOINT3_CRITERIA) - set(facts))
        extra = sorted(set(facts) - set(CHECKPOINT3_CRITERIA))
        raise ValueError(f"checkpoint3 criterion set mismatch: missing={missing}, extra={extra}")
    for name in (
        "raw_hashes",
        "schemas_and_manifests",
        "source_missing_zero_rules",
        "single_frozen_gad_scaler",
    ):
        if facts[name] is not True:
            raise ValueError(f"checkpoint3 criterion failed: {name}")
    taxonomy = facts["taxonomy_counts"]
    if taxonomy != {"cleg": 248, "apec": 126, "overlap": 54}:
        raise ValueError("checkpoint3 criterion failed: taxonomy_counts")
    for name in (
        "duplicate_authoritative_keys",
        "timing_overlap_violations",
        "leakage_violations",
        "fresh_full_test_exit_code",
    ):
        if facts[name] != 0:
            raise ValueError(f"checkpoint3 criterion failed: {name}")
    capacity = facts["capacity"]
    if not isinstance(capacity, Mapping):
        raise ValueError("checkpoint3 criterion failed: capacity")
    try:
        limits = capacity["limits"]
        passed = (
            int(capacity["intermediate_plus_scratch_bytes"]) < int(limits["intermediate"])
            and int(capacity["project_bytes"]) < int(limits["project"])
            and int(capacity["absolute_project_bytes"]) < int(limits["absolute"])
            and int(capacity["filesystem_reserve_bytes"]) >= int(limits["filesystem_reserve"])
        )
    except (KeyError, TypeError, ValueError):
        passed = False
    if not passed:
        raise ValueError("checkpoint3 criterion failed: capacity")


def _relative_support_path(path: Path, code_root: Path) -> str:
    try:
        return path.resolve().relative_to(code_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("checkpoint3 support file escaped code root") from exc


def create_checkpoint3_receipt_payload(
    *,
    implementation_commit: str,
    facts: Mapping[str, object],
    support_files: Mapping[str, Path],
    code_root: Path,
) -> dict[str, object]:
    validate_checkpoint3_facts(facts)
    if len(implementation_commit) != 40:
        raise ValueError("checkpoint3 implementation commit must be a full Git hash")
    support: list[dict[str, object]] = []
    for name, path in support_files.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"checkpoint3 support file is missing: {name}")
        support.append(
            {
                "name": name,
                "path": _relative_support_path(path, code_root),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": "1.0.0",
        "implementation_commit": implementation_commit,
        "facts": dict(facts),
        "support_files": support,
        "evidence_commit_resolution": "evidence_only_descendant_verified_by_verify_only",
    }


def validate_checkpoint3_lineage(
    implementation_commit: str,
    head_commit: str,
    changed_files: tuple[str, ...],
    *,
    is_ancestor: bool,
) -> None:
    """Allow only compact result/docs descendants after implementation."""

    if not is_ancestor:
        raise ValueError("checkpoint3 implementation commit is not an ancestor of HEAD")
    if implementation_commit == head_commit:
        if changed_files:
            raise ValueError("checkpoint3 implementation HEAD has descendant changes")
        return
    allowed_prefixes = ("06_结果/", "06_results/")
    allowed_exact = {"README.md"}
    for path in changed_files:
        if path in allowed_exact:
            continue
        if not path.startswith(allowed_prefixes):
            raise ValueError("checkpoint3 descendant is not evidence-only")
        if Path(path).suffix.lower() not in {".csv", ".json", ".md"}:
            raise ValueError("checkpoint3 descendant is not compact evidence")


def _git_output(code_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=code_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"git inspection failed: {' '.join(arguments)}")
    return completed.stdout.strip()


def validate_checkpoint3_bundle(
    receipt_path: Path,
    code_root: Path,
    *,
    recomputed_facts: Mapping[str, object],
    verify_git_state: bool = True,
) -> dict[str, object]:
    """Verify support blobs, semantic facts, tracked bytes, and evidence lineage."""

    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint3 receipt cannot be read") from exc
    if not isinstance(payload, dict):
        raise ValueError("checkpoint3 receipt root is invalid")
    recorded_facts = payload.get("facts")
    if not isinstance(recorded_facts, Mapping):
        raise ValueError("checkpoint3 receipt facts are invalid")
    try:
        validate_checkpoint3_facts(recorded_facts)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint3 receipt facts are invalid") from exc
    validate_checkpoint3_facts(recomputed_facts)
    recorded_stable = dict(recorded_facts)
    current_stable = dict(recomputed_facts)
    recorded_stable["capacity"] = dict(recorded_stable["capacity"])
    current_stable["capacity"] = dict(current_stable["capacity"])
    recorded_stable["capacity"].pop("filesystem_reserve_bytes", None)
    current_stable["capacity"].pop("filesystem_reserve_bytes", None)
    if recorded_stable != current_stable:
        raise ValueError("checkpoint3 receipt facts differ from recomputation")
    support = payload.get("support_files")
    if not isinstance(support, list) or not support:
        raise ValueError("checkpoint3 receipt support files are missing")
    support_paths: list[str] = []
    for item in support:
        if not isinstance(item, dict):
            raise ValueError("checkpoint3 support metadata is invalid")
        try:
            relative = str(item["path"])
            path = (code_root / relative).resolve()
            if not path.is_relative_to(code_root.resolve()):
                raise ValueError("checkpoint3 support path escaped code root")
            if path.is_symlink() or not path.is_file():
                raise ValueError("checkpoint3 support file is missing")
            if sha256_file(path) != item["sha256"] or path.stat().st_size != int(item["bytes"]):
                raise ValueError("checkpoint3 support hash or size mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and "support" in str(exc):
                raise
            raise ValueError("checkpoint3 support metadata is invalid") from exc
        support_paths.append(relative)
    implementation = payload.get("implementation_commit")
    if not isinstance(implementation, str) or len(implementation) != 40:
        raise ValueError("checkpoint3 implementation commit is invalid")
    if verify_git_state:
        status = _git_output(code_root, "status", "--porcelain", "--untracked-files=no")
        if status:
            raise ValueError("checkpoint3 verify-only requires a clean tracked worktree")
        relative_receipt = receipt_path.resolve().relative_to(code_root.resolve()).as_posix()
        head_bytes = subprocess.run(
            ["git", "show", f"HEAD:{relative_receipt}"],
            cwd=code_root,
            check=False,
            capture_output=True,
        )
        if head_bytes.returncode != 0 or head_bytes.stdout != receipt_path.read_bytes():
            raise ValueError("checkpoint3 receipt differs from the tracked HEAD blob")
        head = _git_output(code_root, "rev-parse", "HEAD")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", implementation, head],
            cwd=code_root,
            check=False,
        ).returncode == 0
        changed_output = _git_output(
            code_root,
            "-c",
            "core.quotePath=false",
            "diff",
            "--name-only",
            f"{implementation}..{head}",
        )
        changed = tuple(line for line in changed_output.splitlines() if line)
        validate_checkpoint3_lineage(
            implementation, head, changed, is_ancestor=ancestor
        )
        allowed = set(support_paths) | {relative_receipt}
        changed_set = set(changed)
        if relative_receipt not in changed_set or not changed_set.issubset(allowed):
            raise ValueError("checkpoint3 evidence lineage does not match receipt support files")
    return payload


# A convenient namespaced constructor keeps the verification API compact while
# remaining a normal callable for existing users.
validate_checkpoint3_bundle.create_receipt_payload = create_checkpoint3_receipt_payload  # type: ignore[attr-defined]


_DELIVERY_FIELDS = (
    "table_id",
    "path",
    "primary_key",
    "period",
    "sample_versions",
    "gad_versions",
    "outcome_families",
    "units",
    "rows",
    "bytes",
    "manifest_path",
)


def render_delivery_note(panels: Sequence[Mapping[str, object]]) -> str:
    """Render a complete human-readable L4 inventory without model results."""

    if not panels:
        raise ValueError("delivery inventory must contain at least one L4 artifact")
    lines = [
        "# Data cleaning and variable construction delivery",
        "",
        "This delivery contains constructed data only; no causal estimate was run.",
        "",
    ]
    for panel in panels:
        missing = [field for field in _DELIVERY_FIELDS if field not in panel]
        if missing:
            raise ValueError(f"delivery inventory is incomplete: {missing}")
        period = panel["period"]
        period_text = (
            "not applicable"
            if period is None
            else f"{period[0]}\u2013{period[1]}"  # type: ignore[index]
        )
        lines.extend(
            [
                f"## {panel['table_id']}",
                "",
                f"- Path: {panel['path']}",
                f"- Primary key: {', '.join(str(item) for item in panel['primary_key'])}",
                f"- Period: {period_text}",
                f"- Sample versions: {', '.join(str(item) for item in panel['sample_versions'])}",
                f"- GAD versions: {', '.join(str(item) for item in panel['gad_versions'])}",
                f"- Outcome families: {', '.join(str(item) for item in panel['outcome_families'])}",
                f"- Units: {json.dumps(panel['units'], ensure_ascii=False, sort_keys=True)}",
                f"- Rows: {int(panel['rows']):,}",
                f"- Bytes: {int(panel['bytes']):,}",
                f"- Manifest: {panel['manifest_path']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _default_command_runner(command: tuple[str, ...]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"direct stage command failed with exit code {completed.returncode}: "
            + " ".join(command)
        )


def _baci_output_ids(revision: str) -> tuple[str, ...]:
    if revision == "HS96":
        years = range(1996, 2025)
        names = (
            "baci_economy_product__all_hs96__{year}",
            "baci_economy_product__main_hs96__{year}",
            "baci_economy_product__broad_hs96__{year}",
            "baci_economy_product__apec_hs96__{year}",
            "baci_economy_year_totals__all_hs96__{year}",
            "baci_green_bilateral__main_hs96__{year}",
            "baci_green_bilateral__broad_hs96__{year}",
            "baci_green_bilateral__apec_hs96__{year}",
        )
    elif revision == "HS07":
        years = range(2007, 2025)
        names = (
            "baci_economy_product__all_hs07_native__{year}",
            "baci_economy_product__hs07_native__{year}",
            "baci_economy_year_totals__all_hs07_native__{year}",
            "baci_green_bilateral__hs07_native__{year}",
        )
    else:
        raise ValueError(f"unsupported BACI revision: {revision}")
    return tuple(template.format(year=year) for year in years for template in names)


def registered_build_graph(
    *,
    code_root: Path,
    data_root: Path,
    command_runner: Callable[[tuple[str, ...]], object] | None = None,
) -> BuildGraph:
    """Register the reviewed L1-L4 direct builders and virtual checkpoints."""

    code = code_root.resolve()
    data = data_root.resolve()
    intermediate = data / "05_中间数据"
    normalized = intermediate / "normalized"
    harmonized = intermediate / "harmonized"
    measures = intermediate / "measures"
    analysis = intermediate / "analysis"
    runner = _default_command_runner if command_runner is None else command_runner

    def command(*arguments: str) -> Callable[[], object]:
        direct = (
            sys.executable,
            "-m",
            "green_debt.cli",
            *arguments,
            "--data-root",
            str(data),
        )
        def run_direct() -> object:
            return runner(direct)

        def run_staged(stage: BuildStage) -> object:
            staged = (
                sys.executable,
                "-m",
                "green_debt.cli",
                *arguments,
                "--data-root",
                str(stage.data_root),
            )
            if command_runner is not None:
                return runner(staged)
            environment = os.environ.copy()
            python_path = str(stage.code_root / "03_代码/src")
            if environment.get("PYTHONPATH"):
                python_path += os.pathsep + environment["PYTHONPATH"]
            environment["PYTHONPATH"] = python_path
            environment["GREEN_DEBT_IMPLEMENTATION_COMMIT"] = _git_output(
                code, "rev-parse", "HEAD"
            )
            completed = subprocess.run(
                staged,
                cwd=stage.code_root,
                env=environment,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"direct stage command failed with exit code {completed.returncode}: "
                    + " ".join(staged)
                )
            return None

        run_direct.command_args = tuple(arguments)  # type: ignore[attr-defined]
        run_direct.staging_builder = run_staged  # type: ignore[attr-defined]
        return run_direct

    def approved_checkpoint(number: int) -> Callable[[], object]:
        def verify() -> object:
            from green_debt.checkpoints import verify_checkpoint_approval_gate
            from green_debt.paths import resolve_project_paths

            return verify_checkpoint_approval_gate(
                number, resolve_project_paths(code, data)
            )

        return verify

    source = code / "03_代码/src/green_debt"
    contracts = code / "03_代码/contracts"
    configuration = code / "config"
    dictionaries = code / "02_数据字典"

    def sidecar(path: Path) -> Path:
        return path.with_name(f"{path.name}.manifest.json")

    raw = data / "04_原始数据"

    def annual(template: str, years: range = range(1996, 2025)) -> tuple[str, ...]:
        return tuple(template.format(year=year) for year in years)

    baci_all = annual("baci_economy_product__all_hs96__{year}")
    baci_main = annual("baci_economy_product__main_hs96__{year}")
    baci_totals = annual("baci_economy_year_totals__all_hs96__{year}")
    baci_bilateral = annual("baci_green_bilateral__main_hs96__{year}")

    nodes = (
        BuildNode(
            "taxonomy",
            (),
            command("build-taxonomy"),
            output_ids=("product_registry_hs07", "product_registry_hs96"),
            authority_files=(
                dictionaries / "product_registry_hs07_v1.csv",
                dictionaries / "product_registry_hs96_v1.parquet",
            ),
            executable_inputs=(source / "taxonomy.py",),
            config_inputs=(configuration / "construction.yaml",),
            contract_inputs=(
                contracts / "product_registry_hs07.json",
                contracts / "product_registry_hs96.json",
            ),
            raw_inputs=(
                raw / "taxonomy/oecd/oecd_cleg_sauvage_2014.pdf",
                raw / "classifications/apec/apec_environmental_goods_54_2012.html",
                raw / "classifications/wits/Concordance_H3_to_H1.zip",
                raw / "classifications/wits/Concordance_H1_to_BE.zip",
                raw / "classifications/unsd/HS1996_to_BEC.xls",
                raw / "baci/202601/BACI_HS07_V202601.zip",
            ),
        ),
        BuildNode(
            "economies",
            (),
            command("build-economy-crosswalk"),
            output_ids=("economy_crosswalk",),
            authority_files=(dictionaries / "economy_crosswalk_v1.csv",),
            executable_inputs=(source / "economies.py",),
            config_inputs=(configuration / "construction.yaml",),
            contract_inputs=(contracts / "economy_crosswalk.json",),
            raw_globs=(
                "baci/202601/BACI_HS*.zip",
                "wdi/20260821/**/*.json",
                "openalex/2026082*/works_aggregate/*.csv",
                "irena/20260821/**/*",
                "ilostat/20260821/**/*",
                "oecd_tiva/20260823/data/*",
                "oecd_eps/20260821/**/*",
                "oecd_ifcma/202604/*.csv",
            ),
        ),
        BuildNode(
            "baci_hs96",
            ("taxonomy", "economies"),
            command("normalize-baci", "--revision", "HS96"),
            output_ids=_baci_output_ids("HS96"),
            manifest_globs=(
                "normalized/baci/**/year=19*/taxonomy_version=*hs96.parquet.manifest.json",
                "normalized/baci/**/year=20*/taxonomy_version=*hs96.parquet.manifest.json",
            ),
            executable_inputs=(source / "sources/baci.py",),
            contract_inputs=(
                contracts / "baci_economy_product.json",
                contracts / "baci_economy_year_totals.json",
                contracts / "baci_green_bilateral.json",
            ),
            taxonomy_inputs=(dictionaries / "product_registry_hs96_v1.parquet",),
            raw_inputs=(raw / "baci/202601/BACI_HS96_V202601.zip",),
        ),
        BuildNode(
            "baci_hs07",
            ("taxonomy", "economies"),
            command("normalize-baci", "--revision", "HS07"),
            output_ids=_baci_output_ids("HS07"),
            manifest_globs=(
                "normalized/baci/**/taxonomy_version=*hs07_native.parquet.manifest.json",
            ),
            executable_inputs=(source / "sources/baci.py",),
            contract_inputs=(
                contracts / "baci_economy_product.json",
                contracts / "baci_economy_year_totals.json",
                contracts / "baci_green_bilateral.json",
            ),
            taxonomy_inputs=(dictionaries / "product_registry_hs07_v1.csv",),
            raw_inputs=(raw / "baci/202601/BACI_HS07_V202601.zip",),
        ),
        BuildNode(
            "wdi",
            ("economies",),
            command("normalize-wdi"),
            output_ids=("wdi_country_year",),
            manifest_paths=(sidecar(normalized / "wdi/wdi_country_year.parquet"),),
            executable_inputs=(source / "sources/wdi.py",),
            contract_inputs=(contracts / "wdi_country_year.json",),
            raw_globs=("wdi/20260821/**/*.json",),
        ),
        BuildNode(
            "irena",
            ("economies",),
            command("normalize-irena"),
            output_ids=("irena_country_year",),
            manifest_paths=(sidecar(normalized / "irena/irena_country_year.parquet"),),
            executable_inputs=(source / "sources/irena.py",),
            contract_inputs=(contracts / "irena_country_year.json",),
            raw_globs=("irena/20260821/**/*",),
        ),
        BuildNode(
            "openalex",
            ("economies",),
            command("normalize-openalex"),
            output_ids=("openalex_country_year",),
            manifest_paths=(sidecar(normalized / "openalex/openalex_country_year.parquet"),),
            executable_inputs=(source / "sources/openalex.py", source / "science.py"),
            contract_inputs=(contracts / "openalex_country_year.json",),
            raw_inputs=(
                raw / "openalex/20260823/works_aggregate/openalex_country_year_counts_1992_1995.csv",
                raw / "openalex/20260822/works_aggregate/openalex_country_year_counts_1996_2024.csv",
            ),
        ),
        BuildNode(
            "ilostat",
            ("economies",),
            command("normalize-ilostat"),
            output_ids=("ilostat_skill",),
            manifest_paths=(sidecar(normalized / "ilostat/ilostat_skill.parquet"),),
            executable_inputs=(source / "sources/ilostat.py",),
            contract_inputs=(contracts / "ilostat_skill.json",),
            raw_globs=("ilostat/20260821/**/*",),
        ),
        BuildNode(
            "policy",
            ("economies",),
            command("normalize-policy"),
            output_ids=("policy_country_year",),
            manifest_paths=(sidecar(normalized / "policy/policy_country_year.parquet"),),
            executable_inputs=(source / "sources/policy.py",),
            contract_inputs=(contracts / "policy_country_year.json",),
            raw_globs=("oecd_eps/20260821/**/*", "oecd_ifcma/202604/*.csv"),
        ),
        BuildNode(
            "tiva",
            ("economies",),
            command("normalize-tiva"),
            output_ids=("tiva_activity_year", "tiva_activity_weights"),
            manifest_paths=(
                sidecar(normalized / "tiva/tiva_activity_year.parquet"),
                sidecar(harmonized / "tiva/tiva_activity_weights.parquet"),
            ),
            executable_inputs=(source / "sources/tiva.py",),
            contract_inputs=(
                contracts / "tiva_activity_year.json",
                contracts / "tiva_activity_weights.json",
            ),
            raw_globs=("oecd_tiva/20260823/data/*",),
        ),
        BuildNode(
            "provisional_sample",
            ("baci_hs96", "baci_hs07", "wdi", "irena", "openalex", "ilostat", "policy", "tiva"),
            command("build-provisional-sample"),
            output_ids=("provisional_sample",),
            manifest_paths=(sidecar(harmonized / "sample/provisional_sample.parquet"),),
            executable_inputs=(source / "sample.py",),
            config_inputs=(configuration / "construction.yaml",),
            contract_inputs=(contracts / "provisional_sample.json",),
            direct_input_ids=(
                *annual("baci_economy_product__main_hs96__{year}", range(1996, 2000)),
                *annual("baci_economy_year_totals__all_hs96__{year}", range(1996, 2000)),
                "openalex_country_year",
                "tiva_activity_year",
                "wdi_country_year",
            ),
        ),
        BuildNode(
            "checkpoint1",
            ("provisional_sample",),
            lambda: None,
            approval_gate=ApprovalGate(
                1,
                code / "06_结果/检查点1_验收回执_v1.json",
                (code / "06_结果/检查点1_来源覆盖审计_v1.csv",),
                semantic_verifier=approved_checkpoint(1),
            ),
        ),
        BuildNode(
            "complexity",
            ("baci_hs96", "taxonomy"),
            command("build-complexity", "--taxonomy", "main_hs96"),
            output_ids=("gpci_product_year",),
            manifest_paths=(sidecar(measures / "trade/gpci_product_year.parquet"),),
            executable_inputs=(source / "trade.py",),
            contract_inputs=(contracts / "gpci_product_year.json",),
            direct_input_ids=baci_all,
        ),
        BuildNode(
            "trade_components",
            ("complexity", "baci_hs96", "wdi"),
            command("build-trade-components", "--taxonomy", "main_hs96"),
            output_ids=("trade_components_raw",),
            manifest_paths=(sidecar(measures / "trade/trade_components_raw.parquet"),),
            executable_inputs=(source / "trade.py",),
            contract_inputs=(contracts / "trade_components_raw.json",),
            direct_input_ids=(*baci_main, *baci_totals, "gpci_product_year", "wdi_country_year"),
        ),
        BuildNode(
            "gsci",
            ("openalex", "wdi"),
            command("build-gsci"),
            output_ids=("gsci_raw",),
            manifest_paths=(sidecar(measures / "science/gsci_raw.parquet"),),
            executable_inputs=(source / "science.py",),
            contract_inputs=(contracts / "gsci_raw.json",),
            direct_input_ids=("openalex_country_year", "wdi_country_year"),
        ),
        BuildNode(
            "supplier_capability",
            ("complexity", "baci_hs96"),
            command("build-supplier-raw", "--taxonomy", "main_hs96"),
            output_ids=("supplier_raw",),
            manifest_paths=(sidecar(measures / "trade/supplier_raw.parquet"),),
            executable_inputs=(source / "trade.py",),
            contract_inputs=(contracts / "supplier_raw.json",),
            direct_input_ids=(*baci_all, *baci_totals),
        ),
        BuildNode(
            "tiva_measures",
            ("tiva",),
            command("build-tiva-measures"),
            output_ids=("tiva_measures", "tiva_bounded_robustness"),
            manifest_paths=(
                sidecar(measures / "tiva/tiva_measures.parquet"),
                sidecar(measures / "tiva/tiva_bounded_robustness.parquet"),
            ),
            executable_inputs=(source / "tiva.py",),
            contract_inputs=(
                contracts / "tiva_measures.json",
                contracts / "tiva_bounded_robustness.json",
            ),
            direct_input_ids=("tiva_activity_year",),
        ),
        BuildNode(
            "gad_scaler",
            ("provisional_sample", "trade_components", "gsci", "supplier_capability", "tiva_measures"),
            command("apply-gad-scaler"),
            output_ids=("gad_scaled_components",),
            manifest_paths=(sidecar(measures / "gad/gad_scaled_components.parquet"),),
            authority_files=(
                code / "06_结果/GAD固定缩放器_v1.json",
                code / "06_结果/GAD固定缩放器_v1.manifest.json",
            ),
            executable_inputs=(source / "scaling.py",),
            contract_inputs=(
                contracts / "gad_scaled_components.json",
                contracts / "gad_scaler_sample_semantics.json",
            ),
            direct_input_ids=(
                "gsci_raw",
                "provisional_sample",
                "supplier_raw",
                "tiva_measures",
                "trade_components_raw",
            ),
        ),
        BuildNode(
            "gad",
            ("gad_scaler", "provisional_sample"),
            command("build-gad", "--all-registered-variants"),
            output_ids=("gad_country_year",),
            manifest_paths=(sidecar(measures / "gad/gad_country_year.parquet"),),
            executable_inputs=(source / "gad.py",),
            config_inputs=(configuration / "construction.yaml",),
            contract_inputs=(contracts / "gad_country_year.json",),
            scaler_inputs=(code / "06_结果/GAD固定缩放器_v1.json",),
            direct_input_ids=("gad_scaled_components", "provisional_sample"),
        ),
        BuildNode(
            "checkpoint2",
            ("checkpoint1", "gad"),
            lambda: None,
            approval_gate=ApprovalGate(
                2,
                code / "06_结果/检查点2_验收回执_v1.json",
                (
                    code / "06_结果/检查点2_GAD构造审计_v1.csv",
                    code / "06_结果/检查点2_初始化覆盖_v1.csv",
                    code / "06_结果/检查点2_容量报告_v1.json",
                    code / "06_结果/GAD固定缩放器_v1.manifest.json",
                ),
                semantic_verifier=approved_checkpoint(2),
            ),
        ),
        BuildNode(
            "outcomes",
            ("trade_components", "complexity", "tiva_measures", "wdi", "irena", "baci_hs96"),
            command("build-outcomes"),
            output_ids=("outcomes_country_year", "outcomes_product_year"),
            manifest_paths=(
                sidecar(measures / "outcomes/outcomes_country_year.parquet"),
                sidecar(measures / "outcomes/outcomes_product_year.parquet"),
            ),
            executable_inputs=(source / "outcomes.py",),
            config_inputs=(configuration / "outcome_gad_map.yaml",),
            contract_inputs=(
                contracts / "outcomes_country_year.json",
                contracts / "outcomes_product_year.json",
            ),
            direct_input_ids=(
                *baci_all,
                *baci_main,
                *baci_totals,
                "gpci_product_year",
                "irena_country_year",
                "tiva_measures",
                "trade_components_raw",
                "wdi_country_year",
            ),
        ),
        BuildNode(
            "instruments",
            ("gad", "baci_hs96", "taxonomy", "provisional_sample"),
            command("build-instruments", "--taxonomy", "main_hs96"),
            output_ids=("iv_baseline_shares", "iv_partner_shocks", "iv_country_year"),
            manifest_paths=(
                sidecar(measures / "instruments/iv_baseline_shares.parquet"),
                sidecar(measures / "instruments/iv_partner_shocks.parquet"),
                sidecar(measures / "instruments/iv_country_year.parquet"),
            ),
            executable_inputs=(source / "instruments.py",),
            contract_inputs=(
                contracts / "iv_baseline_shares.json",
                contracts / "iv_partner_shocks.json",
                contracts / "iv_country_year.json",
            ),
            direct_input_ids=(*baci_main, *baci_bilateral, "gad_country_year", "provisional_sample"),
        ),
        BuildNode(
            "final_sample",
            ("gad", "instruments", "provisional_sample"),
            command("freeze-samples"),
            output_ids=("final_sample",),
            manifest_paths=(sidecar(harmonized / "sample/final_sample.parquet"),),
            executable_inputs=(source / "sample.py",),
            contract_inputs=(contracts / "final_sample.json",),
            direct_input_ids=("gad_country_year", "iv_baseline_shares", "provisional_sample"),
        ),
        BuildNode(
            "analysis_panels",
            ("outcomes", "instruments", "final_sample", "gad", "provisional_sample", "wdi"),
            command("build-analysis-panels"),
            output_ids=("regression_bounds", "giu_outcome_scalers", "model_panel"),
            manifest_paths=(
                sidecar(analysis / "regression_bounds.parquet"),
                sidecar(analysis / "giu_outcome_scalers.parquet"),
                sidecar(analysis / "lp_panel.parquet"),
            ),
            executable_inputs=(source / "sample.py",),
            config_inputs=(configuration / "outcome_gad_map.yaml",),
            contract_inputs=(
                contracts / "regression_bounds.json",
                contracts / "giu_outcome_scalers.json",
                contracts / "model_panel.json",
            ),
            direct_input_ids=(
                "final_sample",
                "gad_country_year",
                "iv_country_year",
                "outcomes_country_year",
                "outcomes_product_year",
                "provisional_sample",
                "wdi_country_year",
            ),
        ),
        BuildNode(
            "checkpoint3",
            ("checkpoint2", "analysis_panels"),
            lambda: None,
            executable_inputs=(source / "build.py", source / "checkpoints.py", source / "cli.py"),
        ),
    )
    runtime_package = Path(__file__).resolve().parent
    # The CLI imports every builder module at process start.  Consequently the
    # honest executable closure is the complete tracked Python package, mapped
    # into the caller's code root (not this worktree's absolute package path).
    common_executable = tuple(
        source / path.relative_to(runtime_package)
        for path in sorted(runtime_package.rglob("*.py"))
    )
    closed_nodes = tuple(
        replace(
            node,
            executable_inputs=tuple(
                dict.fromkeys((*node.executable_inputs, *common_executable))
            ),
        )
        for node in nodes
    )
    return BuildGraph(closed_nodes)
