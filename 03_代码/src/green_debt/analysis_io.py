"""Fail-closed authority and output primitives for empirical analysis."""

from __future__ import annotations

from collections.abc import Callable
import ctypes
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import struct
import sys
import threading
import time
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from green_debt.analysis_spec import AnalysisSpec, load_analysis_spec
from green_debt.artifacts import (
    BuildIdentity,
    InputArtifact,
    TableContract,
    verify_manifest,
    write_authoritative_table,
)
from green_debt.config import load_project_config
from green_debt.storage import (
    GIB,
    directory_usage_bytes,
    project_usage_bytes,
    sha256_file,
)


FROZEN_MODEL_PANEL_ROWS = 55_514
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_TERMS = ("gimc_a", "gimc_gad_a")
MODEL_CELL_FIELDS = (
    "analysis_family",
    "outcome_id",
    "horizon",
    "gad_version",
    "sample_version",
)
MODEL_PROVENANCE_FIELDS = (
    "run_id",
    "spec_id",
    "input_authority_hash",
    "git_commit",
    "renv_lock_sha256",
    "evidence_policy_sha256",
    "created_at_utc",
)
EVIDENCE_POLICY_RELATIVE_PATH = Path("config/evidence_policy.json")
_CLEANUP_INVENTORY_MAX_DEPTH = 64
_CLEANUP_INVENTORY_MAX_ENTRIES = 4_096
_CLEANUP_INVENTORY_MAX_CANONICAL_BYTES = 1_048_576
_CLEANUP_BATCH_MAX_ENTRIES = 32
_CLEANUP_BATCH_MAX_CANONICAL_BYTES = 262_144
_CLEANUP_LEGACY_BATCH_MAX_ANCESTORS = 4_096


def planned_analysis_stages() -> tuple[str, ...]:
    """Return the single frozen order for a complete offline analysis."""

    return (
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


@dataclass(frozen=True)
class AnalysisCommand:
    stage: str | None
    command: tuple[str, ...]


def planned_analysis_commands(
    *, code_root: Path, data_root: Path, output_root: Path
) -> tuple[AnalysisCommand, ...]:
    """Build the single command table shared by production and reproduction."""

    python = str(code_root / ".venv/bin/python")
    rscript = str(code_root / "03_代码/R/run_analysis.R")
    common = ("--data-root", str(data_root), "--output-root", str(output_root))

    def cli(stage: str | None, command: str, *extra: str) -> AnalysisCommand:
        return AnalysisCommand(
            stage,
            (python, "-m", "green_debt.cli", command, *extra, *common),
        )

    def r(stage: str, command: str) -> AnalysisCommand:
        return AnalysisCommand(
            stage,
            ("Rscript", "--vanilla", rscript, command, *common),
        )

    return (
        AnalysisCommand(None, (python, "-m", "green_debt.cli", "analysis-deps", "--verify")),
        cli("preflight", "analysis-preflight"),
        cli("diagnostics", "analysis-diagnostics"),
        r("lp", "lp"),
        cli("lp_ingest", "analysis-ingest-models", "--kind", "lp"),
        r("threshold", "threshold"),
        r("weak_iv", "weak-iv"),
        r("shift_share", "shift-share"),
        cli(
            "threshold_and_iv_audit_ingest",
            "analysis-ingest-models",
            "--kind",
            "threshold-and-iv-audit",
        ),
        r("report", "report"),
        cli("audit", "analysis-output-audit"),
    )


@dataclass(frozen=True)
class InputAuthority:
    table_id: str
    rows: int
    output_sha256: str
    schema_sha256: str
    manifest_sha256: str
    contract_sha256: str
    project_config_sha256: str
    outcome_map_sha256: str
    analysis_config_sha256: str
    table_hashes: tuple[tuple[str, str], ...]
    input_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RunContext:
    run_id: str
    spec_id: str
    input_authority_hash: str
    git_commit: str
    renv_lock_sha256: str
    evidence_policy_sha256: str
    seed: int
    created_at_utc: str


@dataclass(frozen=True)
class AnalysisPaths:
    code_root: Path
    data_root: Path
    output_root: Path


@dataclass(frozen=True)
class DiagnosticBundle:
    table_paths: tuple[Path, ...]
    gate_path: Path
    cell_count: int


@dataclass(frozen=True)
class AnalysisAuditReport:
    status: str
    missing_confirmatory_cells: tuple[tuple[str, int], ...]
    failed_contracts: tuple[str, ...]
    output_bytes: int


@dataclass(frozen=True)
class AnalysisPreflight:
    status: str
    table_id: str
    rows: int
    authority_tables: int
    confirmatory_cells: int
    registered_cells: int
    write_count: int
    input_authority_hash: str
    evidence_policy_sha256: str
    output_bytes: int
    project_bytes: int


class AnalysisReproductionMismatch(ValueError):
    """A clean-room output differs from the formal logical result."""


class AnalysisReproductionInterrupted(KeyboardInterrupt):
    """A catchable process signal interrupted a journaled reproduction."""


class _CleanupInventoryLimitExceeded(ValueError):
    """A bounded inventory cannot be persisted as one receipt field."""

    def __init__(self, kind: str, observed: int, limit: int) -> None:
        self.kind = kind
        self.observed = observed
        self.limit = limit
        super().__init__(
            f"reproduction cleanup inventory {kind} limit exceeded: "
            f"{observed} > {limit}"
        )

    def payload(self) -> dict[str, int | str]:
        return {
            "kind": self.kind,
            "observed": self.observed,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class LogicalAnalysisComparison:
    matched: bool
    rows: int
    max_scaled_float_error: float


@dataclass(frozen=True)
class AnalysisTreeSnapshot:
    sha256: str
    files: int
    bytes: int


@dataclass(frozen=True)
class AnalysisOutputComparison:
    compared_parquet: int
    compared_json: int
    compared_csv: int
    compared_figures: int
    max_scaled_float_error: float
    threshold_registry_hash: str


@dataclass(frozen=True)
class AnalysisReproductionCapability:
    data_root: Path
    formal_output_root: Path
    scratch_root: Path
    marker_path: Path
    execution_id: str

    def open_stage_authority(
        self, stage: str
    ) -> "AnalysisReproductionStageAuthority":
        if stage not in _REPRODUCTION_BROKER_STAGES:
            raise ValueError("analysis reproduction stage is invalid")
        marker = _json_object(
            self.marker_path, "analysis reproduction capability marker"
        )
        binding = {
            "execution_id": self.execution_id,
            "data_root": str(self.data_root),
            "formal_output_root": str(self.formal_output_root),
            "scratch_root": str(self.scratch_root),
            "git_commit": marker["git_commit"],
            "marker_instance_id": marker["instance_id"],
            "marker_sha256": sha256_file(self.marker_path),
            "scratch_device": marker["scratch_device"],
            "scratch_inode": marker["scratch_inode"],
            "orchestrator_pid": marker["orchestrator_pid"],
        }
        return AnalysisReproductionStageAuthority(
            capability=self,
            stage=stage,
            broker=AnalysisReproductionBroker(binding, stage),
        )


def _capability_marker_environment(
    capability: AnalysisReproductionCapability,
) -> dict[str, str]:
    return {
        "GREEN_DEBT_ANALYSIS_REPRO_MARKER": str(capability.marker_path),
        "GREEN_DEBT_ANALYSIS_REPRO_SCRATCH": str(capability.scratch_root),
        "GREEN_DEBT_ANALYSIS_REPRO_EXECUTION_ID": capability.execution_id,
    }


_REPRODUCTION_BROKER_STAGES = (
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


class AnalysisReproductionBroker:
    """One-stage parent broker with an in-memory HMAC key and inherited FDs."""

    def __init__(self, binding: dict[str, Any], stage: str) -> None:
        server, client = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        secret_server, secret_client = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_STREAM
        )
        secret = secrets.token_bytes(32)
        self._server = server
        self._client = client
        self._secret_server = secret_server
        self._secret_client = secret_client
        self._secret = secret
        self._binding = dict(binding)
        self._stage = stage
        self._stage_root_pid: int | None = None
        self._stage_pid_ready = threading.Event()
        self._issued_nonces: set[str] = set()
        self._used_nonces: set[str] = set()
        self._closed = False
        self._thread = threading.Thread(
            target=self._serve,
            name=f"analysis-reproduction-{binding['execution_id']}",
            daemon=True,
        )
        self._secret_thread = threading.Thread(
            target=self._serve_secret,
            name=f"analysis-reproduction-secret-{binding['execution_id']}",
            daemon=True,
        )
        self._thread.start()
        self._secret_thread.start()

    @property
    def client_fd(self) -> int:
        if self._closed:
            raise ValueError("analysis reproduction broker is closed")
        return self._client.fileno()

    @property
    def secret_fd(self) -> int:
        if self._closed:
            raise ValueError("analysis reproduction broker is closed")
        return self._secret_client.fileno()

    def set_stage_root_pid(self, pid: int) -> None:
        if pid <= 0 or self._stage_root_pid is not None:
            raise ValueError("analysis reproduction stage PID is invalid")
        self._stage_root_pid = pid
        self._stage_pid_ready.set()

    def _serve(self) -> None:
        stream = self._server.makefile("rwb", buffering=0)
        try:
            while True:
                line = stream.readline()
                if not line:
                    return
                try:
                    request = json.loads(line)
                    response = self._authorize(request)
                except Exception as exc:
                    response = {"status": "denied", "error": str(exc)}
                stream.write(_canonical_json_bytes(response) + b"\n")
        finally:
            stream.close()

    def _serve_secret(self) -> None:
        stream = self._secret_server.makefile("rwb", buffering=0)
        try:
            while True:
                line = stream.readline()
                if not line:
                    return
                try:
                    request = json.loads(line)
                    response = self._issue_secret(request)
                except Exception as exc:
                    response = {"status": "denied", "error": str(exc)}
                stream.write(_canonical_json_bytes(response) + b"\n")
        finally:
            stream.close()

    def _validate_request_identity(self, request: dict[str, Any]) -> tuple[str, int]:
        stage = request.get("stage")
        nonce = request.get("nonce")
        child_pid = request.get("child_pid")
        if stage != self._stage:
            raise ValueError("broker stage mismatch")
        if (
            not isinstance(nonce, str)
            or not re.fullmatch(r"[0-9a-f]{64}", nonce)
            or not isinstance(child_pid, int)
            or child_pid <= 0
        ):
            raise ValueError("broker request authentication is invalid")
        for field, expected in self._binding.items():
            if request.get(field) != expected:
                raise ValueError(f"broker binding mismatch: {field}")
        if not self._stage_pid_ready.wait(timeout=5) or self._stage_root_pid is None:
            raise ValueError("broker stage process identity is unavailable")
        if not _is_process_ancestor(self._stage_root_pid, child_pid):
            raise ValueError("broker requester is outside the stage process group")
        return nonce, child_pid

    def _issue_secret(self, request: object) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("broker secret request must be an object")
        nonce, child_pid = self._validate_request_identity(request)
        if nonce in self._issued_nonces or nonce in self._used_nonces:
            raise ValueError("broker secret nonce replay")
        self._issued_nonces.add(nonce)
        return {
            "status": "issued",
            "nonce": nonce,
            "stage": self._stage,
            "child_pid": child_pid,
            "secret": self._secret.hex(),
        }

    def _authorize(self, request: object) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("broker request must be an object")
        stage = request.get("stage")
        nonce = request.get("nonce")
        request_mac = request.get("request_mac")
        child_pid = request.get("child_pid")
        if stage != self._stage:
            raise ValueError("broker stage mismatch")
        if (
            not isinstance(nonce, str)
            or not re.fullmatch(r"[0-9a-f]{64}", nonce)
            or not isinstance(child_pid, int)
            or child_pid <= 0
            or not isinstance(request_mac, str)
            or not re.fullmatch(r"[0-9a-f]{64}", request_mac)
        ):
            raise ValueError("broker request authentication is invalid")
        nonce, child_pid = self._validate_request_identity(request)
        if nonce not in self._issued_nonces or nonce in self._used_nonces:
            raise ValueError("broker request replay or unissued nonce")
        unsigned = {key: value for key, value in request.items() if key != "request_mac"}
        expected_mac = hmac.new(
            self._secret, _canonical_json_bytes(unsigned), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(request_mac, expected_mac):
            raise ValueError("broker request authentication failed")
        self._used_nonces.add(nonce)
        response = {
            "status": "authorized",
            "nonce": nonce,
            "stage": stage,
            "execution_id": self._binding["execution_id"],
            "marker_instance_id": self._binding["marker_instance_id"],
            "child_pid": child_pid,
        }
        response["response_mac"] = hmac.new(
            self._secret, _canonical_json_bytes(response), hashlib.sha256
        ).hexdigest()
        return response

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._client.close()
        self._secret_client.close()
        for server in (self._server, self._secret_server):
            try:
                server.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            server.close()
        self._thread.join(timeout=2)
        self._secret_thread.join(timeout=2)


@dataclass
class AnalysisReproductionStageAuthority:
    capability: AnalysisReproductionCapability
    stage: str
    broker: AnalysisReproductionBroker = field(repr=False)

    @property
    def environment(self) -> dict[str, str]:
        return {
            "GREEN_DEBT_ANALYSIS_REPRO_BROKER_FD": str(self.broker.client_fd),
            "GREEN_DEBT_ANALYSIS_REPRO_SECRET_FD": str(self.broker.secret_fd),
            "GREEN_DEBT_ANALYSIS_REPRO_MARKER": str(self.capability.marker_path),
            "GREEN_DEBT_ANALYSIS_REPRO_SCRATCH": str(self.capability.scratch_root),
            "GREEN_DEBT_ANALYSIS_REPRO_EXECUTION_ID": self.capability.execution_id,
            "GREEN_DEBT_ANALYSIS_REPRO_STAGE": self.stage,
        }

    @property
    def pass_fds(self) -> tuple[int, int]:
        return self.broker.client_fd, self.broker.secret_fd

    def set_stage_root_pid(self, pid: int) -> None:
        self.broker.set_stage_root_pid(pid)

    def close(self) -> None:
        self.broker.close()


_REPRODUCTION_MARKER_NAME = ".analysis-reproduction-capability.json"
_REPRODUCTION_EXECUTION_RE = re.compile(
    r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8,32}$"
)


def _process_identity(pid: int) -> tuple[str, str]:
    """Return the kernel-visible command and start identity for one Darwin PID."""

    if pid <= 0:
        raise ValueError("orchestrator peer PID is invalid")
    command = subprocess.run(
        ["/bin/ps", "-ww", "-p", str(pid), "-o", "command="],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    ).stdout.strip()
    started = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "lstart="],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    ).stdout.strip()
    if not command or not started:
        raise ValueError("orchestrator process identity is unavailable")
    return command, started


def _darwin_process_arguments(pid: int) -> tuple[str, tuple[str, ...]]:
    """Read one process executable/argv from Darwin KERN_PROCARGS2."""

    if pid <= 0:
        raise ValueError("orchestrator peer PID is invalid")
    libc = ctypes.CDLL(None, use_errno=True)
    mib = (ctypes.c_int * 3)(1, 49, pid)  # CTL_KERN, KERN_PROCARGS2
    for _attempt in range(3):
        size = ctypes.c_size_t()
        if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
            raise ValueError("orchestrator kernel argv size is unavailable")
        buffer = ctypes.create_string_buffer(size.value)
        if libc.sysctl(
            mib, 3, buffer, ctypes.byref(size), None, 0
        ) == 0:
            raw = buffer.raw[: size.value]
            break
        if ctypes.get_errno() != 12:  # ENOMEM: argv grew between calls
            raise ValueError("orchestrator kernel argv is unavailable")
    else:
        raise ValueError("orchestrator kernel argv changed during capture")
    if len(raw) < struct.calcsize("=i"):
        raise ValueError("orchestrator kernel argv is truncated")
    argc = struct.unpack_from("=i", raw)[0]
    if argc <= 0 or argc > 4096:
        raise ValueError("orchestrator kernel argc is invalid")
    offset = struct.calcsize("=i")
    executable_end = raw.find(b"\0", offset)
    if executable_end < 0:
        raise ValueError("orchestrator kernel executable is truncated")
    executable = raw[offset:executable_end].decode(
        "utf-8", errors="surrogateescape"
    )
    offset = executable_end + 1
    while offset < len(raw) and raw[offset] == 0:
        offset += 1
    arguments: list[str] = []
    for _index in range(argc):
        end = raw.find(b"\0", offset)
        if end < 0:
            raise ValueError("orchestrator kernel argv is truncated")
        arguments.append(
            raw[offset:end].decode("utf-8", errors="surrogateescape")
        )
        offset = end + 1
    if not executable or len(arguments) != argc or not arguments[0]:
        raise ValueError("orchestrator kernel argv is invalid")
    return executable, tuple(arguments)


def _validate_reproduction_cli_arguments(
    executable: str, arguments: tuple[str, ...], marker: dict[str, Any]
) -> None:
    """Require the peer to be exactly the frozen Python module CLI invocation."""

    expected_python = os.path.realpath(sys.executable)
    executable_realpath = os.path.realpath(executable)
    argv0_realpath = os.path.realpath(arguments[0])
    if (
        executable_realpath != expected_python
        or argv0_realpath != expected_python
        or marker.get("orchestrator_executable_realpath") != expected_python
    ):
        raise ValueError("orchestrator Python executable identity mismatch")
    if arguments[1:4] != (
        "-m",
        "green_debt.cli",
        "analysis-reproduce-check",
    ):
        raise ValueError("broker peer argv is not analysis-reproduce-check")
    parsed: dict[str, str] = {}
    index = 4
    while index < len(arguments):
        token = arguments[index]
        if token in {"--data-root", "--output-root"}:
            if token in parsed or index + 1 >= len(arguments):
                raise ValueError("orchestrator CLI arguments are invalid")
            parsed[token] = arguments[index + 1]
            index += 2
            continue
        matched = next(
            (
                option
                for option in ("--data-root", "--output-root")
                if token.startswith(option + "=")
            ),
            None,
        )
        if matched is None or matched in parsed:
            raise ValueError("orchestrator CLI arguments contain an extra token")
        parsed[matched] = token.split("=", 1)[1]
        index += 1
    if set(parsed) != {"--data-root", "--output-root"}:
        raise ValueError("orchestrator CLI arguments are incomplete")
    expected_roots = {
        "--data-root": marker.get("data_root"),
        "--output-root": marker.get("formal_output_root"),
    }
    for option, expected in expected_roots.items():
        try:
            observed = str(Path(parsed[option]).resolve(strict=True))
        except OSError as exc:
            raise ValueError("orchestrator CLI root is unavailable") from exc
        if observed != expected:
            raise ValueError(f"orchestrator CLI {option} binding mismatch")


def _process_parent_pid(pid: int) -> int:
    output = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "ppid="],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    ).stdout.strip()
    if not output.isdecimal():
        raise ValueError("process ancestry is unavailable")
    return int(output)


def _is_process_ancestor(ancestor_pid: int, child_pid: int) -> bool:
    observed: set[int] = set()
    current = child_pid
    while current > 0 and current not in observed:
        if current == ancestor_pid:
            return True
        observed.add(current)
        current = _process_parent_pid(current)
    return False


def _validate_orchestrator_peer(
    broker_fd: int, marker: dict[str, Any]
) -> int:
    """Anchor a broker endpoint to the live analysis-reproduce-check ancestor."""

    try:
        with socket.socket(fileno=os.dup(broker_fd)) as peer_socket:
            peer_raw = peer_socket.getsockopt(0, 2, 4)
        peer_pid = int.from_bytes(peer_raw, byteorder=sys.byteorder, signed=True)
    except OSError as exc:
        raise ValueError("orchestrator broker peer PID is unavailable") from exc
    if peer_pid != marker.get("orchestrator_pid"):
        raise ValueError("orchestrator broker peer PID mismatch")
    if not _is_process_ancestor(peer_pid, os.getpid()):
        raise ValueError("orchestrator broker peer is not an ancestor")
    executable, arguments = _darwin_process_arguments(peer_pid)
    _command, started = _process_identity(peer_pid)
    argv_sha256 = hashlib.sha256(
        _canonical_json_bytes(list(arguments))
    ).hexdigest()
    start_sha256 = hashlib.sha256(started.encode("utf-8")).hexdigest()
    if (
        argv_sha256 != marker.get("orchestrator_argv_sha256")
        or start_sha256 != marker.get("orchestrator_start_sha256")
    ):
        raise ValueError("orchestrator process identity mismatch")
    _validate_reproduction_cli_arguments(executable, arguments, marker)
    return peer_pid
_CANONICAL_JSON_EXCLUDED_FIELDS = frozenset(
    {"created_at_utc", "execution_id"}
)


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _without_reproduction_ephemera(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _without_reproduction_ephemera(item)
            for key, item in value.items()
            if key not in _CANONICAL_JSON_EXCLUDED_FIELDS
        }
    if isinstance(value, list):
        return [_without_reproduction_ephemera(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical reproduction JSON contains a nonfinite number")
    return value


def canonical_json_for_analysis_reproduction(payload: Any) -> bytes:
    """Canonicalize JSON while excluding exactly two execution-only fields."""

    return _canonical_json_bytes(_without_reproduction_ephemera(payload))


def _logical_value_equal(left: Any, right: Any) -> tuple[bool, float]:
    if left is None or right is None:
        return left is None and right is None, 0.0
    left_float = float(left)
    right_float = float(right)
    if math.isnan(left_float) or math.isnan(right_float):
        return math.isnan(left_float) and math.isnan(right_float), 0.0
    if math.isinf(left_float) or math.isinf(right_float):
        return left_float == right_float, 0.0
    scaled = abs(left_float - right_float) / max(
        1.0, abs(left_float), abs(right_float)
    )
    return scaled <= 1e-12, scaled


def compare_logical_frames(
    formal: pl.DataFrame,
    reproduced: pl.DataFrame,
    *,
    primary_key: tuple[str, ...],
) -> LogicalAnalysisComparison:
    """Compare exact keys/nonfloats and bounded floats independent of row order."""

    if formal.schema != reproduced.schema:
        raise AnalysisReproductionMismatch("logical table schema mismatch")
    missing_keys = sorted(set(primary_key) - set(formal.columns))
    if missing_keys:
        raise ValueError(
            "logical comparison primary key is missing: " + ", ".join(missing_keys)
        )
    if formal.height != reproduced.height:
        raise AnalysisReproductionMismatch("logical table row keys differ")
    if primary_key:
        for label, frame in (("formal", formal), ("reproduced", reproduced)):
            duplicates = frame.group_by(*primary_key).len().filter(pl.col("len") > 1)
            if duplicates.height:
                raise AnalysisReproductionMismatch(
                    f"{label} logical table contains duplicate row keys"
                )
        formal = formal.sort(*primary_key)
        reproduced = reproduced.sort(*primary_key)
        if not formal.select(*primary_key).equals(reproduced.select(*primary_key)):
            raise AnalysisReproductionMismatch("logical table row keys differ")
    nonfloat = [
        name for name, dtype in formal.schema.items() if not dtype.is_float()
    ]
    if nonfloat and not formal.select(*nonfloat).equals(
        reproduced.select(*nonfloat)
    ):
        raise AnalysisReproductionMismatch("logical table non-float values differ")
    maximum = 0.0
    mismatch = False
    for name, dtype in formal.schema.items():
        if not dtype.is_float():
            continue
        for left, right in zip(
            formal.get_column(name).to_list(),
            reproduced.get_column(name).to_list(),
            strict=True,
        ):
            equal, scaled = _logical_value_equal(left, right)
            maximum = max(maximum, scaled)
            mismatch = mismatch or not equal
    if mismatch:
        raise AnalysisReproductionMismatch(
            "logical table float tolerance exceeded; "
            f"max_scaled_float_error={maximum:.17g}"
        )
    return LogicalAnalysisComparison(
        matched=True,
        rows=formal.height,
        max_scaled_float_error=maximum,
    )


def compare_logical_parquet(
    formal_path: Path,
    reproduced_path: Path,
    *,
    primary_key: tuple[str, ...],
) -> LogicalAnalysisComparison:
    """Read and logically compare two Parquet tables."""

    return compare_logical_frames(
        pl.read_parquet(formal_path),
        pl.read_parquet(reproduced_path),
        primary_key=primary_key,
    )


def _figure_source_payload(path: Path) -> tuple[str, str, tuple[str, ...]]:
    payload = _json_object(path, "figure provenance")
    figure_id = payload.get("figure_id")
    plotted = payload.get("plotted_source_data_hash")
    sources = payload.get("source_table_hashes")
    if (
        not isinstance(figure_id, str)
        or not isinstance(plotted, str)
        or not re.fullmatch(r"[0-9a-f]{64}", plotted)
        or not isinstance(sources, dict)
        or not sources
        or any(
            not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in sources.values()
        )
    ):
        raise ValueError(f"invalid figure source-data provenance: {path.name}")
    return figure_id, plotted, tuple(sorted(str(key) for key in sources))


def compare_analysis_figure_sources(
    formal_output_root: Path, reproduced_output_root: Path
) -> tuple[tuple[str, str], ...]:
    """Compare figure source-data identities, deliberately excluding PDFs."""

    def load(root: Path) -> dict[str, tuple[str, tuple[str, ...]]]:
        paths = sorted((root / "figures").glob("figure_*.provenance.json"))
        result: dict[str, tuple[str, bytes]] = {}
        for path in paths:
            figure_id, plotted, sources = _figure_source_payload(path)
            if figure_id in result:
                raise AnalysisReproductionMismatch("duplicate figure source identity")
            result[figure_id] = (plotted, sources)
        return result

    formal = load(formal_output_root)
    reproduced = load(reproduced_output_root)
    if not formal or formal != reproduced:
        raise AnalysisReproductionMismatch("figure source-data hashes differ")
    return tuple((key, formal[key][0]) for key in sorted(formal))


def analysis_tree_snapshot(root: Path) -> AnalysisTreeSnapshot:
    """Hash names, sizes, and bytes of a read-only tree without following links."""

    resolved = _resolved_real_directory(root, "analysis snapshot root")
    digest = hashlib.sha256()
    files = 0
    total = 0
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"analysis snapshot contains a symbolic link: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(resolved).as_posix()
        size = path.stat().st_size
        digest.update(
            _canonical_json_bytes(
                {"path": relative, "bytes": size, "sha256": sha256_file(path)}
            )
        )
        files += 1
        total += size
    return AnalysisTreeSnapshot(digest.hexdigest(), files, total)


def _logical_primary_key(path: Path, frame: pl.DataFrame) -> tuple[str, ...]:
    manifest_path = path.with_name(f"{path.name}.manifest.json")
    if manifest_path.is_file() and not manifest_path.is_symlink():
        payload = _json_object(manifest_path, "logical table manifest")
        primary_key = payload.get("primary_key")
        if (
            not isinstance(primary_key, list)
            or not primary_key
            or any(not isinstance(value, str) for value in primary_key)
        ):
            raise ValueError("logical table manifest primary key is invalid")
        return tuple(primary_key)
    candidates = tuple(
        name for name, dtype in frame.schema.items() if not dtype.is_float()
    )
    if candidates and not frame.group_by(*candidates).len().filter(
        pl.col("len") > 1
    ).height:
        return candidates
    return ()


def _relative_files(root: Path, pattern: str) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.glob(pattern))
        if path.is_file()
        and not path.is_symlink()
        and "_tmp" not in path.relative_to(root).parts
    }


def compare_analysis_output_trees(
    formal_output_root: Path, reproduced_output_root: Path
) -> AnalysisOutputComparison:
    """Compare every logical table plus the single registry and figure sources."""

    formal = _resolved_real_directory(formal_output_root, "formal output root")
    reproduced = _resolved_real_directory(
        reproduced_output_root, "reproduced output root"
    )
    formal_parquet = _relative_files(formal, "**/*.parquet")
    reproduced_parquet = _relative_files(reproduced, "**/*.parquet")
    if set(formal_parquet) != set(reproduced_parquet):
        raise AnalysisReproductionMismatch("logical Parquet output set differs")
    maximum = 0.0
    for relative in sorted(formal_parquet):
        left = pl.read_parquet(formal_parquet[relative])
        right = pl.read_parquet(reproduced_parquet[relative])
        result = compare_logical_frames(
            left,
            right,
            primary_key=_logical_primary_key(formal_parquet[relative], left),
        )
        maximum = max(maximum, result.max_scaled_float_error)

    formal_csv = _relative_files(formal, "tables/*.csv")
    reproduced_csv = _relative_files(reproduced, "tables/*.csv")
    if set(formal_csv) != set(reproduced_csv):
        raise AnalysisReproductionMismatch("logical CSV output set differs")
    for relative in sorted(formal_csv):
        left = pl.read_csv(formal_csv[relative], infer_schema_length=10_000)
        right = pl.read_csv(reproduced_csv[relative], infer_schema_length=10_000)
        result = compare_logical_frames(
            left,
            right,
            primary_key=_logical_primary_key(formal_csv[relative], left),
        )
        maximum = max(maximum, result.max_scaled_float_error)

    formal_registry_path = formal / "registries/threshold_registry_v1.json"
    reproduced_registry_path = reproduced / "registries/threshold_registry_v1.json"
    formal_registry = _json_object(formal_registry_path, "formal threshold registry")
    reproduced_registry = _json_object(
        reproduced_registry_path, "reproduced threshold registry"
    )
    if canonical_json_for_analysis_reproduction(
        formal_registry
    ) != canonical_json_for_analysis_reproduction(reproduced_registry):
        raise AnalysisReproductionMismatch("canonical threshold registry differs")
    threshold_hash = formal_registry.get("registry_hash")
    if (
        not isinstance(threshold_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", threshold_hash)
        or reproduced_registry.get("registry_hash") != threshold_hash
    ):
        raise AnalysisReproductionMismatch("threshold registry hash differs")
    figures = compare_analysis_figure_sources(formal, reproduced)
    formal_summary = formal / "analysis_summary.md"
    reproduced_summary = reproduced / "analysis_summary.md"
    if formal_summary.exists() != reproduced_summary.exists() or (
        formal_summary.exists()
        and formal_summary.read_bytes() != reproduced_summary.read_bytes()
    ):
        raise AnalysisReproductionMismatch("analysis summary differs")
    return AnalysisOutputComparison(
        compared_parquet=len(formal_parquet),
        compared_json=1,
        compared_csv=len(formal_csv),
        compared_figures=len(figures),
        max_scaled_float_error=maximum,
        threshold_registry_hash=threshold_hash,
    )


def _atomic_write_json_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("xb") as handle:
            handle.write(_json_document_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _json_document_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )


def _write_analysis_reproduction_receipt(
    path: Path, payload: dict[str, Any]
) -> None:
    """Project one exact replacement document before any receipt state write."""

    document_bytes = len(_json_document_bytes(payload))
    formal = _resolved_real_directory(
        Path(payload["formal_output_root"]), "formal output root"
    )
    data = _resolved_real_directory(
        Path(payload["data_root"]), "analysis data root"
    )
    spec = load_analysis_spec(PROJECT_ROOT / "config/analysis.yaml")
    project = load_project_config(PROJECT_ROOT / "config/project.yaml")
    output_limit = spec.outputs.quota_gb * GIB
    project_limit_gb = project.storage.absolute_limit_gb
    if project_limit_gb is None:
        raise ValueError("absolute project limit is not configured")
    receipt = Path(os.path.abspath(os.fspath(path)))
    execution_id = payload.get("execution_id")
    if (
        not isinstance(execution_id, str)
        or receipt != _analysis_reproduction_journal_path(formal, execution_id)
    ):
        raise ValueError("analysis reproduction receipt path is not formally bound")
    try:
        existing = receipt.lstat()
    except FileNotFoundError:
        existing_bytes = 0
    else:
        if stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode):
            raise ValueError("analysis reproduction receipt is unsafe")
        existing_bytes = existing.st_size
    partial = receipt.with_name(f"{receipt.name}.partial")
    try:
        stale_partial = partial.lstat()
    except FileNotFoundError:
        stale_partial_bytes = 0
    else:
        if stat.S_ISLNK(stale_partial.st_mode) or not stat.S_ISREG(
            stale_partial.st_mode
        ):
            raise ValueError("analysis reproduction receipt partial is unsafe")
        stale_partial_bytes = stale_partial.st_size
    output_before = directory_usage_bytes(formal)
    project_before = combined_project_usage_bytes(data, formal)
    replaced_bytes = existing_bytes + stale_partial_bytes
    if output_before < replaced_bytes or project_before < replaced_bytes:
        raise ValueError("analysis reproduction receipt usage accounting is invalid")
    projected_output = output_before - replaced_bytes + document_bytes
    projected_project = project_before - replaced_bytes + document_bytes
    if projected_output > output_limit:
        raise RuntimeError(
            "10 GB cumulative analysis output quota blocks reproduction journal write"
        )
    if projected_project >= project_limit_gb * GIB:
        raise RuntimeError(
            "150 GB absolute project limit blocks reproduction journal write"
        )
    _atomic_write_json_object(path, payload)


def _write_cleanup_reproduction_receipt(
    path: Path, payload: dict[str, Any]
) -> None:
    """Keep the cleanup journal interception point while sharing quota accounting."""

    _write_analysis_reproduction_receipt(path, payload)


def _resolved_real_directory(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    if _existing_path_has_symlink(lexical):
        raise ValueError(f"{label} must not use a symbolic link")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is missing or unsafe") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a real directory")
    return resolved


def mint_analysis_reproduction_capability(
    *,
    data_root: Path,
    formal_output_root: Path,
    execution_id: str,
    git_commit: str,
    formal_run_id: str,
    formal_created_at_utc: str,
    formal_audited_at_utc: str,
) -> AnalysisReproductionCapability:
    """Mint one unguessable execution-bound capability and real scratch child."""

    if not _REPRODUCTION_EXECUTION_RE.fullmatch(execution_id):
        raise ValueError("invalid analysis reproduction execution id")
    if not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise ValueError("invalid analysis reproduction Git commit")
    if not re.fullmatch(r"[0-9a-f]{16}", formal_run_id):
        raise ValueError("invalid formal analysis run id")
    data = _resolved_real_directory(data_root, "analysis data root")
    formal = _resolved_real_directory(formal_output_root, "formal output root")
    temporary = formal / "_tmp"
    if temporary.exists():
        temporary = _resolved_real_directory(temporary, "reproduction temporary root")
    else:
        temporary.mkdir(mode=0o700)
    scratch = temporary / f"reproduce.{execution_id}"
    scratch.mkdir(mode=0o700)
    scratch = _resolved_real_directory(scratch, "reproduction scratch root")
    stat_result = scratch.stat()
    orchestrator_executable, orchestrator_arguments = (
        _darwin_process_arguments(os.getpid())
    )
    _orchestrator_command, orchestrator_started = _process_identity(os.getpid())
    core: dict[str, Any] = {
        "schema_version": 4,
        "execution_id": execution_id,
        "instance_id": secrets.token_hex(16),
        "data_root": str(data),
        "formal_output_root": str(formal),
        "scratch_root": str(scratch),
        "git_commit": git_commit,
        "formal_run_id": formal_run_id,
        "formal_created_at_utc": formal_created_at_utc,
        "formal_audited_at_utc": formal_audited_at_utc,
        "scratch_device": stat_result.st_dev,
        "scratch_inode": stat_result.st_ino,
        "orchestrator_pid": os.getpid(),
        "orchestrator_executable_realpath": os.path.realpath(
            orchestrator_executable
        ),
        "orchestrator_argv_sha256": hashlib.sha256(
            _canonical_json_bytes(list(orchestrator_arguments))
        ).hexdigest(),
        "orchestrator_start_sha256": hashlib.sha256(
            orchestrator_started.encode("utf-8")
        ).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
    }
    marker = scratch / _REPRODUCTION_MARKER_NAME
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(core, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        marker.chmod(0o600)
    except Exception:
        marker.unlink(missing_ok=True)
        scratch.rmdir()
        raise
    return AnalysisReproductionCapability(
        data_root=data,
        formal_output_root=formal,
        scratch_root=scratch,
        marker_path=marker,
        execution_id=execution_id,
    )


def validate_analysis_reproduction_capability(
    *,
    data_root: Path,
    formal_output_root: Path,
    scratch_root: Path,
    expected_git_commit: str,
    environment: dict[str, str] | os._Environ[str] | None = None,
    require_live_broker: bool = True,
) -> dict[str, Any]:
    """Validate marker/path binding and obtain one live broker authorization."""

    env = os.environ if environment is None else environment
    data = _resolved_real_directory(data_root, "analysis data root")
    formal = _resolved_real_directory(formal_output_root, "formal output root")
    scratch = _resolved_real_directory(scratch_root, "reproduction scratch root")
    temporary = _resolved_real_directory(
        formal / "_tmp", "reproduction temporary root"
    )
    if scratch.parent != temporary or not scratch.name.startswith("reproduce."):
        raise ValueError("reproduction scratch has the wrong canonical parent or name")
    execution_id = scratch.name.removeprefix("reproduce.")
    if not _REPRODUCTION_EXECUTION_RE.fullmatch(execution_id):
        raise ValueError("reproduction scratch has an invalid execution id")
    marker_path = scratch / _REPRODUCTION_MARKER_NAME
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError("reproduction capability marker is missing or unsafe")
    expected_environment = {
        "GREEN_DEBT_ANALYSIS_REPRO_MARKER": str(marker_path),
        "GREEN_DEBT_ANALYSIS_REPRO_SCRATCH": str(scratch),
        "GREEN_DEBT_ANALYSIS_REPRO_EXECUTION_ID": execution_id,
    }
    for field, expected in expected_environment.items():
        if env.get(field) != expected:
            raise ValueError(f"analysis reproduction capability {field} mismatch")
    marker = _json_object(marker_path, "analysis reproduction capability marker")
    required = {
        "schema_version",
        "execution_id",
        "instance_id",
        "data_root",
        "formal_output_root",
        "scratch_root",
        "git_commit",
        "formal_run_id",
        "formal_created_at_utc",
        "formal_audited_at_utc",
        "scratch_device",
        "scratch_inode",
        "created_at_utc",
        "orchestrator_pid",
        "orchestrator_executable_realpath",
        "orchestrator_argv_sha256",
        "orchestrator_start_sha256",
    }
    if set(marker) != required or marker["schema_version"] != 4:
        raise ValueError("analysis reproduction capability marker schema mismatch")
    identities = {
        "execution_id": execution_id,
        "data_root": str(data),
        "formal_output_root": str(formal),
        "scratch_root": str(scratch),
        "git_commit": expected_git_commit,
    }
    for field, expected in identities.items():
        if marker.get(field) != expected:
            label = "Git commit" if field == "git_commit" else field.replace("_", " ")
            raise ValueError(f"analysis reproduction capability {label} mismatch")
    observed = scratch.stat()
    if (
        marker["scratch_device"] != observed.st_dev
        or marker["scratch_inode"] != observed.st_ino
    ):
        raise ValueError("analysis reproduction scratch identity changed")
    if require_live_broker:
        _authorize_analysis_reproduction_stage(marker, marker_path, env)
    return marker


def _authorize_analysis_reproduction_stage(
    marker: dict[str, Any],
    marker_path: Path,
    environment: dict[str, str] | os._Environ[str],
) -> None:
    fd_text = environment.get("GREEN_DEBT_ANALYSIS_REPRO_BROKER_FD", "")
    secret_fd_text = environment.get("GREEN_DEBT_ANALYSIS_REPRO_SECRET_FD", "")
    stage = environment.get("GREEN_DEBT_ANALYSIS_REPRO_STAGE", "")
    if (
        not fd_text.isdecimal()
        or not secret_fd_text.isdecimal()
        or stage not in _REPRODUCTION_BROKER_STAGES
    ):
        raise ValueError("live reproduction orchestrator broker is required")
    fd = int(fd_text)
    secret_fd = int(secret_fd_text)
    _validate_orchestrator_peer(fd, marker)
    _validate_orchestrator_peer(secret_fd, marker)
    request = {
        "execution_id": marker["execution_id"],
        "data_root": marker["data_root"],
        "formal_output_root": marker["formal_output_root"],
        "scratch_root": marker["scratch_root"],
        "git_commit": marker["git_commit"],
        "marker_instance_id": marker["instance_id"],
        "marker_sha256": sha256_file(marker_path),
        "scratch_device": marker["scratch_device"],
        "scratch_inode": marker["scratch_inode"],
        "orchestrator_pid": marker["orchestrator_pid"],
        "stage": stage,
        "nonce": secrets.token_hex(32),
        "child_pid": os.getpid(),
    }
    try:
        secret_channel = socket.socket(fileno=os.dup(secret_fd))
        secret_channel.settimeout(5)
        secret_stream = secret_channel.makefile("rwb", buffering=0)
        secret_stream.write(_canonical_json_bytes(request) + b"\n")
        secret_line = secret_stream.readline()
        secret_stream.close()
        secret_channel.close()
        secret_response = json.loads(secret_line)
        if (
            not isinstance(secret_response, dict)
            or secret_response.get("status") != "issued"
            or secret_response.get("nonce") != request["nonce"]
            or secret_response.get("stage") != stage
            or secret_response.get("child_pid") != os.getpid()
            or not isinstance(secret_response.get("secret"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", secret_response["secret"])
        ):
            raise ValueError("live reproduction broker secret response is invalid")
        secret = bytes.fromhex(secret_response["secret"])
    except (OSError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("live reproduction broker secret is unavailable") from exc
    request["request_mac"] = hmac.new(
        secret, _canonical_json_bytes(request), hashlib.sha256
    ).hexdigest()
    try:
        channel = socket.socket(fileno=os.dup(fd))
        channel.settimeout(5)
        stream = channel.makefile("rwb", buffering=0)
        stream.write(_canonical_json_bytes(request) + b"\n")
        response_line = stream.readline()
        stream.close()
        channel.close()
    except (OSError, TimeoutError) as exc:
        raise ValueError("live reproduction orchestrator broker is unavailable") from exc
    try:
        response = json.loads(response_line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("live reproduction broker response is invalid") from exc
    response_mac = response.pop("response_mac", None) if isinstance(response, dict) else None
    expected_response = {
        "status": "authorized",
        "nonce": request["nonce"],
        "stage": stage,
        "execution_id": marker["execution_id"],
        "marker_instance_id": marker["instance_id"],
        "child_pid": os.getpid(),
    }
    expected_mac = hmac.new(
        secret, _canonical_json_bytes(expected_response), hashlib.sha256
    ).hexdigest()
    if response != expected_response or not isinstance(response_mac, str) or not hmac.compare_digest(
        response_mac, expected_mac
    ):
        raise ValueError("live reproduction broker denied stage authorization")


def _analysis_receipt_hash(payload: dict[str, Any]) -> str:
    bound = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return hashlib.sha256(_canonical_json_bytes(bound)).hexdigest()


def _analysis_reproduction_journal_path(
    formal_output_root: Path, execution_id: str
) -> Path:
    return (
        formal_output_root
        / "reproduction_receipts"
        / f"reproduction.{execution_id}.json"
    )


def _pending_reproduction_payload(
    capability: AnalysisReproductionCapability, formal_manifest_path: Path
) -> dict[str, Any]:
    marker = validate_analysis_reproduction_capability(
        data_root=capability.data_root,
        formal_output_root=capability.formal_output_root,
        scratch_root=capability.scratch_root,
        expected_git_commit=_json_object(
            capability.marker_path, "analysis reproduction capability marker"
        )["git_commit"],
        environment=_capability_marker_environment(capability),
        require_live_broker=False,
    )
    manifest_path = formal_manifest_path.resolve(strict=True)
    manifest = _json_object(manifest_path, "formal run manifest")
    if (
        manifest_path != capability.formal_output_root / "run_manifest.json"
        or manifest.get("status") != "success"
        or manifest.get("run_id") != marker["formal_run_id"]
        or manifest.get("git_commit") != marker["git_commit"]
    ):
        raise ValueError("formal run manifest identity mismatch")
    scratch_stat = capability.scratch_root.stat()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": 1,
        "receipt_kind": "analysis-reproduction",
        "status": "pending",
        "cleanup_status": "pending",
        "cleaned_at_utc": None,
        "failure": None,
        "execution_id": capability.execution_id,
        "created_at_utc": now,
        "state_updated_at_utc": now,
        "data_root": str(capability.data_root),
        "formal_output_root": str(capability.formal_output_root),
        "scratch_root": str(capability.scratch_root),
        "scratch_device": scratch_stat.st_dev,
        "scratch_inode": scratch_stat.st_ino,
        "marker_path": str(capability.marker_path),
        "marker_sha256": sha256_file(capability.marker_path),
        "marker_instance_id": marker["instance_id"],
        "marker_schema_version": marker["schema_version"],
        "formal_run_manifest": str(manifest_path),
        "formal_run_manifest_sha256": sha256_file(manifest_path),
        "formal_run_id": manifest["run_id"],
        "spec_id": manifest.get("spec_id"),
        "input_authority_hash": manifest.get("input_authority_hash"),
        "git_commit": manifest["git_commit"],
        "renv_lock_sha256": manifest.get("renv_lock_sha256"),
        "evidence_policy_sha256": manifest.get("evidence_policy_sha256"),
        "threshold_registry_hash": manifest.get("threshold_registry_hash"),
        "comparison_scope": None,
        "comparison_results": None,
    }


def create_pending_analysis_reproduction_receipt(
    *,
    capability: AnalysisReproductionCapability,
    formal_manifest_path: Path,
) -> Path:
    """Atomically journal a new scratch before any child stage can run."""

    receipt = _analysis_reproduction_journal_path(
        capability.formal_output_root, capability.execution_id
    )
    receipt.parent.mkdir(mode=0o700, exist_ok=True)
    if receipt.exists():
        raise ValueError("analysis reproduction journal already exists")
    payload = _pending_reproduction_payload(capability, formal_manifest_path)
    payload["receipt_sha256"] = _analysis_receipt_hash(payload)
    _write_analysis_reproduction_receipt(receipt, payload)
    return receipt


def create_analysis_reproduction_receipt(
    *,
    capability: AnalysisReproductionCapability,
    formal_manifest_path: Path,
    data_snapshot_before: str,
    data_snapshot_after: str,
    manifest_snapshot_before: str,
    manifest_snapshot_after: str,
    compared_parquet: int,
    compared_json: int,
    compared_csv: int,
    compared_figures: int,
    max_scaled_float_error: float,
    threshold_registry_hash: str,
) -> Path:
    """Write the formal-root receipt that binds one successful comparison."""

    marker = validate_analysis_reproduction_capability(
        data_root=capability.data_root,
        formal_output_root=capability.formal_output_root,
        scratch_root=capability.scratch_root,
        expected_git_commit=_json_object(
            capability.marker_path, "analysis reproduction capability marker"
        )["git_commit"],
        environment=_capability_marker_environment(capability),
        require_live_broker=False,
    )
    manifest_path = formal_manifest_path.resolve(strict=True)
    if manifest_path != capability.formal_output_root / "run_manifest.json":
        raise ValueError("formal run manifest path mismatch")
    manifest = _json_object(manifest_path, "formal run manifest")
    if manifest.get("status") != "success" or manifest.get("run_id") != marker["formal_run_id"]:
        raise ValueError("formal run manifest identity mismatch")
    if manifest.get("git_commit") != marker["git_commit"]:
        raise ValueError("formal run manifest Git identity mismatch")
    for value, label in (
        (data_snapshot_before, "data snapshot before"),
        (data_snapshot_after, "data snapshot after"),
        (manifest_snapshot_before, "manifest snapshot before"),
        (manifest_snapshot_after, "manifest snapshot after"),
        (threshold_registry_hash, "threshold registry hash"),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"invalid {label}")
    if data_snapshot_before != data_snapshot_after:
        raise AnalysisReproductionMismatch("05_\u4e2d\u95f4\u6570\u636e snapshot changed")
    if manifest_snapshot_before != manifest_snapshot_after:
        raise AnalysisReproductionMismatch("authority manifest snapshot changed")
    if threshold_registry_hash != manifest.get("threshold_registry_hash"):
        raise AnalysisReproductionMismatch("threshold registry hash changed")
    if (
        any(value < 0 for value in (
            compared_parquet,
            compared_json,
            compared_csv,
            compared_figures,
        ))
        or not math.isfinite(max_scaled_float_error)
        or not 0 <= max_scaled_float_error <= 1e-12
    ):
        raise ValueError("invalid analysis reproduction comparison result")
    receipt = _analysis_reproduction_journal_path(
        capability.formal_output_root, capability.execution_id
    )
    if receipt.exists():
        pending = _json_object(receipt, "analysis reproduction journal")
        if (
            pending.get("receipt_sha256") != _analysis_receipt_hash(pending)
            or pending.get("status") != "pending"
            or pending.get("cleanup_status") != "pending"
            or pending.get("scratch_root") != str(capability.scratch_root)
        ):
            raise ValueError("analysis reproduction journal state is invalid")
    else:
        receipt = create_pending_analysis_reproduction_receipt(
            capability=capability,
            formal_manifest_path=manifest_path,
        )
        pending = _json_object(receipt, "analysis reproduction journal")
    scratch_stat = capability.scratch_root.stat()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": "analysis-reproduction",
        "status": "matched",
        "cleanup_status": "pending",
        "cleaned_at_utc": None,
        "failure": None,
        "execution_id": capability.execution_id,
        "created_at_utc": pending["created_at_utc"],
        "state_updated_at_utc": now,
        "data_root": str(capability.data_root),
        "formal_output_root": str(capability.formal_output_root),
        "scratch_root": str(capability.scratch_root),
        "scratch_device": scratch_stat.st_dev,
        "scratch_inode": scratch_stat.st_ino,
        "marker_path": str(capability.marker_path),
        "marker_sha256": sha256_file(capability.marker_path),
        "marker_instance_id": marker["instance_id"],
        "marker_schema_version": marker["schema_version"],
        "formal_run_manifest": str(manifest_path),
        "formal_run_manifest_sha256": sha256_file(manifest_path),
        "formal_run_id": manifest["run_id"],
        "spec_id": manifest.get("spec_id"),
        "input_authority_hash": manifest.get("input_authority_hash"),
        "git_commit": manifest["git_commit"],
        "renv_lock_sha256": manifest.get("renv_lock_sha256"),
        "evidence_policy_sha256": manifest.get("evidence_policy_sha256"),
        "threshold_registry_hash": threshold_registry_hash,
        "data_snapshot_before": data_snapshot_before,
        "data_snapshot_after": data_snapshot_after,
        "manifest_snapshot_before": manifest_snapshot_before,
        "manifest_snapshot_after": manifest_snapshot_after,
        "comparison_scope": {
            "parquet": "keys_and_nonfloats_exact_floats_scaled_1e-12",
            "json": "canonical_excluding_only_created_at_utc_and_execution_id",
            "figures": (
                "plotted_source_hash_and_source_labels; "
                "underlying_source_tables_logically_compared_separately"
            ),
            "threshold": "single_formal_registry_hash",
            "run_manifest": "identity_and_result_fields_bound_by_receipt",
        },
        "comparison_results": {
            "parquet": compared_parquet,
            "json": compared_json,
            "csv": compared_csv,
            "figures": compared_figures,
            "max_scaled_float_error": max_scaled_float_error,
            "all_matched": True,
        },
    }
    payload["receipt_sha256"] = _analysis_receipt_hash(payload)
    _write_analysis_reproduction_receipt(receipt, payload)
    return receipt


def create_failed_analysis_reproduction_receipt(
    *, formal_output_root: Path, scratch_root: Path, failure: str
) -> Path:
    """Bind one failed historical attempt so only its exact scratch is cleanable."""

    formal = _resolved_real_directory(formal_output_root, "formal output root")
    temporary = _resolved_real_directory(
        formal / "_tmp", "reproduction temporary root"
    )
    scratch = _resolved_real_directory(scratch_root, "reproduction scratch root")
    execution_id = scratch.name.removeprefix("reproduce.")
    if (
        scratch.parent != temporary
        or scratch.name != f"reproduce.{execution_id}"
        or not _REPRODUCTION_EXECUTION_RE.fullmatch(execution_id)
    ):
        raise ValueError("failed reproduction cleanup target is too broad")
    marker_path = scratch / _REPRODUCTION_MARKER_NAME
    scratch_stat = scratch.stat()
    receipt = _analysis_reproduction_journal_path(formal, execution_id)
    pending: dict[str, Any] | None = None
    if receipt.exists():
        pending = _json_object(receipt, "analysis reproduction journal")
        pending_bindings = {
            "status": "pending",
            "cleanup_status": "pending",
            "execution_id": execution_id,
            "formal_output_root": str(formal),
            "scratch_root": str(scratch),
            "scratch_device": scratch_stat.st_dev,
            "scratch_inode": scratch_stat.st_ino,
            "marker_path": str(marker_path),
        }
        if pending.get("receipt_sha256") != _analysis_receipt_hash(pending):
            raise ValueError("analysis reproduction journal hash mismatch")
        for field, expected in pending_bindings.items():
            if pending.get(field) != expected:
                raise ValueError("analysis reproduction journal state is invalid")
        authority = {
            "data_root": pending.get("data_root"),
            "git_commit": pending.get("git_commit"),
            "formal_run_id": pending.get("formal_run_id"),
            "instance_id": pending.get("marker_instance_id"),
            "schema_version": pending.get("marker_schema_version"),
            "marker_sha256": pending.get("marker_sha256"),
        }
    else:
        if marker_path.is_symlink() or not marker_path.is_file():
            raise ValueError("failed reproduction marker is missing or unsafe")
        marker = _json_object(
            marker_path, "analysis reproduction capability marker"
        )
        marker_bindings = {
            "schema_version": 4,
            "execution_id": execution_id,
            "formal_output_root": str(formal),
            "scratch_root": str(scratch),
            "scratch_device": scratch_stat.st_dev,
            "scratch_inode": scratch_stat.st_ino,
        }
        for field, expected in marker_bindings.items():
            if marker.get(field) != expected:
                raise ValueError("failed reproduction marker binding mismatch")
        authority = {
            "data_root": marker.get("data_root"),
            "git_commit": marker.get("git_commit"),
            "formal_run_id": marker.get("formal_run_id"),
            "instance_id": marker.get("instance_id"),
            "schema_version": marker.get("schema_version"),
            "marker_sha256": sha256_file(marker_path),
        }
    for field, pattern in (
        ("git_commit", r"[0-9a-f]{40}"),
        ("formal_run_id", r"[0-9a-f]{16}"),
        ("instance_id", r"[0-9a-f]{32}"),
    ):
        if not isinstance(authority.get(field), str) or not re.fullmatch(
            pattern, authority[field]
        ):
            raise ValueError("failed reproduction authority identity is invalid")
    if authority.get("schema_version") != 4 or not isinstance(
        authority.get("marker_sha256"), str
    ):
        raise ValueError("failed reproduction authority marker is invalid")
    if not isinstance(failure, str) or not failure.strip():
        raise ValueError("failed reproduction reason is required")
    scratch_manifest = scratch / "run_manifest.json"
    observation: dict[str, Any] = {"state": "missing", "sha256": None}
    scratch_manifest_sha256: str | None = None
    try:
        manifest_stat = scratch_manifest.lstat()
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISLNK(manifest_stat.st_mode):
            observation["state"] = "symlink"
        elif not stat.S_ISREG(manifest_stat.st_mode):
            observation["state"] = "special"
        else:
            try:
                raw = scratch_manifest.read_bytes()
            except PermissionError:
                observation["state"] = "permission_error"
            except OSError:
                observation["state"] = "read_error"
            else:
                scratch_manifest_sha256 = hashlib.sha256(raw).hexdigest()
                observation["sha256"] = scratch_manifest_sha256
                try:
                    manifest = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    observation["state"] = "invalid_json"
                else:
                    if not isinstance(manifest, dict):
                        observation["state"] = "invalid_json"
                    elif (
                        manifest.get("run_id") != authority["formal_run_id"]
                        or manifest.get("git_commit") != authority["git_commit"]
                    ):
                        observation["state"] = "mismatch"
                    else:
                        observation["state"] = "regular_hash"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": "analysis-reproduction",
        "status": "failed",
        "cleanup_status": "pending",
        "cleaned_at_utc": None,
        "failure": failure.strip(),
        "execution_id": execution_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "data_root": authority.get("data_root"),
        "formal_output_root": str(formal),
        "scratch_root": str(scratch),
        "scratch_device": scratch_stat.st_dev,
        "scratch_inode": scratch_stat.st_ino,
        "marker_path": str(marker_path),
        "marker_sha256": authority["marker_sha256"],
        "marker_instance_id": authority["instance_id"],
        "marker_schema_version": authority["schema_version"],
        "formal_run_id": authority["formal_run_id"],
        "git_commit": authority["git_commit"],
        "scratch_run_manifest": (
            str(scratch_manifest) if scratch_manifest_sha256 is not None else None
        ),
        "scratch_run_manifest_sha256": scratch_manifest_sha256,
        "scratch_run_manifest_observation": observation,
    }
    if pending is not None:
        pending.update(
            {
                "status": "failed",
                "failure": failure.strip(),
                "state_updated_at_utc": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "scratch_run_manifest": payload["scratch_run_manifest"],
                "scratch_run_manifest_sha256": payload[
                    "scratch_run_manifest_sha256"
                ],
                "scratch_run_manifest_observation": observation,
            }
        )
        payload = pending
    else:
        receipt.parent.mkdir(mode=0o700, exist_ok=True)
        payload["state_updated_at_utc"] = payload["created_at_utc"]
    payload["receipt_sha256"] = _analysis_receipt_hash(payload)
    _write_analysis_reproduction_receipt(receipt, payload)
    return receipt


def _analysis_manifest_snapshot(data_root: Path) -> AnalysisTreeSnapshot:
    intermediate = _resolved_real_directory(
        data_root / "05_中间数据", "read-only intermediate root"
    )
    paths = tuple(sorted(intermediate.rglob("*.manifest.json")))
    if not paths:
        raise ValueError("analysis reproduction found no authority manifests")
    digest = hashlib.sha256()
    total = 0
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("analysis authority manifest is unsafe")
        size = path.stat().st_size
        digest.update(
            _canonical_json_bytes(
                {
                    "path": path.relative_to(intermediate).as_posix(),
                    "bytes": size,
                    "sha256": sha256_file(path),
                }
            )
        )
        total += size
    return AnalysisTreeSnapshot(digest.hexdigest(), len(paths), total)


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_analysis_process_group(
    process: subprocess.Popen[Any], *, timeout: float = 5.0
) -> None:
    """TERM, then KILL, one stage group and wait until no member remains."""

    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=min(1.0, timeout))
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic() + timeout
    while _process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.02)
    if _process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.wait(timeout=max(0.1, timeout))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=max(0.1, timeout))
    deadline = time.monotonic() + timeout
    while _process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.02)
    if _process_group_exists(pgid):
        raise RuntimeError("analysis stage process group did not terminate")


_ANALYSIS_STAGE_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def _install_grouped_stage_signal_handlers(
    handler: Callable[[int, Any], None],
    race_hook: Callable[[str], None],
) -> dict[int, Any]:
    """Install both deferred handlers while both signals are blocked."""

    blocked = set(_ANALYSIS_STAGE_SIGNALS)
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    installed: list[int] = []
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in _ANALYSIS_STAGE_SIGNALS
    }
    try:
        for signum in _ANALYSIS_STAGE_SIGNALS:
            signal.signal(signum, handler)
            installed.append(signum)
            race_hook(
                "install_after_sigint"
                if signum == signal.SIGINT
                else "install_after_sigterm"
            )
        race_hook("install_before_unmask")
    except BaseException:
        for signum in installed:
            signal.signal(signum, previous_handlers[signum])
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        raise
    try:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    except BaseException:
        # Delivery can raise at the exact unmask boundary.  Re-block the pair
        # before rolling back so callers never observe a half-installed set.
        signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        restoration_error: BaseException | None = None
        for signum in _ANALYSIS_STAGE_SIGNALS:
            try:
                signal.signal(signum, previous_handlers[signum])
            except BaseException as exc:
                if restoration_error is None:
                    restoration_error = exc
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        except BaseException as exc:
            if restoration_error is None:
                restoration_error = exc
        if restoration_error is not None:
            raise restoration_error
        raise
    return previous_handlers


def _restore_grouped_stage_signal_handlers(
    previous_handlers: dict[int, Any],
    deferred_signals: list[tuple[int, Any, bool]],
    race_hook: Callable[[str], None],
) -> None:
    """Restore both handlers before unblocking and record pending delivery."""

    if set(previous_handlers) != set(_ANALYSIS_STAGE_SIGNALS):
        raise RuntimeError("analysis stage signal handlers are incomplete")
    blocked = set(_ANALYSIS_STAGE_SIGNALS)
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    restoration_error: BaseException | None = None
    try:
        for signum in _ANALYSIS_STAGE_SIGNALS:
            try:
                signal.signal(signum, previous_handlers[signum])
                race_hook(
                    "restore_after_sigint"
                    if signum == signal.SIGINT
                    else "restore_after_sigterm"
                )
            except BaseException as exc:
                if restoration_error is None:
                    restoration_error = exc
        race_hook("restore_before_unmask")
        for signum in _ANALYSIS_STAGE_SIGNALS:
            if signum in signal.sigpending():
                deferred_signals.append((signum, None, True))
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    if restoration_error is not None:
        raise restoration_error


def _raise_deferred_stage_signal(
    deferred_signals: list[tuple[int, Any, bool]],
    previous_handlers: dict[int, Any],
) -> None:
    """Dispatch the old handler, then fail even when it returns or ignores."""

    if not deferred_signals:
        return
    signum, frame, already_dispatched = deferred_signals[0]
    handler = previous_handlers[signum]
    if not already_dispatched and callable(handler):
        handler(signum, frame)
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise AnalysisReproductionInterrupted(signal.Signals(signum).name)


def _run_analysis_stage_process(
    command: tuple[str, ...],
    environment: dict[str, str],
    pass_fds: tuple[int, ...],
    on_started: Callable[[int], None] | None,
    *,
    cwd: Path = PROJECT_ROOT,
    _signal_race_hook: Callable[[str], None] | None = None,
) -> None:
    """Run and reap exactly one isolated analysis stage process group."""

    process: subprocess.Popen[Any] | None = None
    deferred_signals: list[tuple[int, Any, bool]] = []

    def defer_signal(signum: int, frame: Any) -> None:
        deferred_signals.append((signum, frame, False))

    signal_race_hook = _signal_race_hook or (lambda _stage: None)
    previous_handlers = _install_grouped_stage_signal_handlers(
        defer_signal, signal_race_hook
    )
    handlers_active = True
    try:
        if deferred_signals:
            _restore_grouped_stage_signal_handlers(
                previous_handlers, deferred_signals, signal_race_hook
            )
            handlers_active = False
            _raise_deferred_stage_signal(deferred_signals, previous_handlers)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            pass_fds=pass_fds,
            start_new_session=True,
        )
        if deferred_signals:
            _restore_grouped_stage_signal_handlers(
                previous_handlers, deferred_signals, signal_race_hook
            )
            handlers_active = False
            _raise_deferred_stage_signal(deferred_signals, previous_handlers)
        if on_started is not None:
            on_started(process.pid)
        _restore_grouped_stage_signal_handlers(
            previous_handlers, deferred_signals, signal_race_hook
        )
        handlers_active = False
        _raise_deferred_stage_signal(deferred_signals, previous_handlers)
        returncode = process.wait()
    except BaseException as exc:
        if process is not None:
            _terminate_analysis_process_group(process)
        if handlers_active:
            try:
                _restore_grouped_stage_signal_handlers(
                    previous_handlers, deferred_signals, signal_race_hook
                )
                handlers_active = False
            except BaseException as restore_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "failed to restore analysis stage signal state: "
                        f"{type(restore_error).__name__}: {restore_error}"
                    )
        raise
    if returncode != 0:
        _terminate_analysis_process_group(process)
        raise subprocess.CalledProcessError(returncode, command)
    deadline = time.monotonic() + 5.0
    while _process_group_exists(process.pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    if _process_group_exists(process.pid):
        _terminate_analysis_process_group(process)
        raise RuntimeError("analysis stage left a live descendant")


def run_analysis_command_plan(
    *,
    code_root: Path,
    data_root: Path,
    output_root: Path,
    command_runner: Callable[[tuple[str, ...], dict[str, str]], None] | None = None,
) -> None:
    """Execute the frozen production plan offline and fail at the first stage."""

    code = Path(os.path.abspath(os.fspath(code_root)))
    base_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GREEN_DEBT_ANALYSIS_REPRO_")
    }
    for planned in planned_analysis_commands(
        code_root=code,
        data_root=data_root,
        output_root=output_root,
    ):
        environment = dict(base_environment)
        if planned.stage is not None:
            environment["GREEN_DEBT_ANALYSIS_REPRO_STAGE"] = planned.stage
        if command_runner is None:
            _run_analysis_stage_process(
                planned.command,
                environment,
                (),
                None,
                cwd=code,
            )
        else:
            command_runner(planned.command, environment)


def _remove_unjournaled_scratch(scratch: Path) -> None:
    """Rollback a just-minted scratch when its first journal write failed."""

    temporary = scratch.parent
    temporary_fd = os.open(
        temporary, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    scratch_fd = -1
    try:
        scratch_fd = os.open(
            scratch.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=temporary_fd,
        )
        held = os.fstat(scratch_fd)
        if held.st_dev != os.fstat(temporary_fd).st_dev:
            raise ValueError("unjournaled scratch crosses a mount boundary")
        inventory = _inventory_tree_at(scratch_fd, held.st_dev)
        _remove_tree_at(scratch_fd, held.st_dev, inventory)
        _verified_entry_stat(temporary_fd, scratch.name, held)
        os.rmdir(scratch.name, dir_fd=temporary_fd)
    finally:
        if scratch_fd >= 0:
            os.close(scratch_fd)
        os.close(temporary_fd)


def _enforce_reproduction_tree_resource_limits(scratch: Path) -> None:
    """Bound scratch metadata after every writer stage, before comparison."""

    scratch_fd = os.open(
        scratch, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        _inventory_tree_at(scratch_fd, os.fstat(scratch_fd).st_dev)
    finally:
        os.close(scratch_fd)


def _run_analysis_reproduction_check_locked(
    *,
    code_root: Path,
    data_root: Path,
    formal_output_root: Path,
    expected_git_commit: str | None = None,
    command_runner: Callable[[tuple[str, ...], dict[str, str]], None] | None = None,
    enforce_production_counts: bool = True,
) -> Path:
    """Run the full offline command once in a minted scratch and receipt results."""

    code = code_root.resolve()
    data = _resolved_real_directory(data_root, "analysis data root")
    formal = resolve_frozen_analysis_output(code, formal_output_root)
    formal = _resolved_real_directory(formal, "formal output root")
    manifest_path = formal / "run_manifest.json"
    manifest = _json_object(manifest_path, "formal run manifest")
    commit = expected_git_commit or current_analysis_git_commit(code)
    required_manifest = {
        "status": "success",
        "git_commit": commit,
    }
    for field, expected in required_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(f"formal run manifest {field} mismatch")
    for field, length in (
        ("run_id", 16),
        ("input_authority_hash", 64),
        ("renv_lock_sha256", 64),
        ("evidence_policy_sha256", 64),
        ("threshold_registry_hash", 64),
    ):
        value = manifest.get(field)
        if not isinstance(value, str) or not re.fullmatch(
            rf"[0-9a-f]{{{length}}}", value
        ):
            raise ValueError(f"formal run manifest {field} is invalid")
    spec_id = manifest.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id:
        raise ValueError("formal run manifest spec_id is invalid")
    run_material = json.dumps(
        [
            spec_id,
            manifest["input_authority_hash"],
            commit,
            manifest["renv_lock_sha256"],
            manifest["evidence_policy_sha256"],
        ],
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(run_material).hexdigest()[:16] != manifest["run_id"]:
        raise ValueError("formal run id is not the deterministic frozen identity")
    for field in ("created_at_utc", "audited_at_utc"):
        if not isinstance(manifest.get(field), str) or not manifest[field].endswith("Z"):
            raise ValueError(f"formal run manifest {field} is invalid")
    formal_manifest_sha = sha256_file(manifest_path)
    registry_path = formal / "registries/threshold_registry_v1.json"
    registry = _json_object(registry_path, "formal threshold registry")
    if registry.get("registry_hash") != manifest["threshold_registry_hash"]:
        raise ValueError("formal threshold registry differs from run manifest")
    formal_registry_sha = sha256_file(registry_path)
    intermediate_before = analysis_tree_snapshot(data / "05_中间数据")
    manifests_before = _analysis_manifest_snapshot(data)
    execution_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + secrets.token_hex(4)
    )
    capability = mint_analysis_reproduction_capability(
        data_root=data,
        formal_output_root=formal,
        execution_id=execution_id,
        git_commit=commit,
        formal_run_id=manifest["run_id"],
        formal_created_at_utc=manifest["created_at_utc"],
        formal_audited_at_utc=manifest["audited_at_utc"],
    )
    try:
        create_pending_analysis_reproduction_receipt(
            capability=capability,
            formal_manifest_path=manifest_path,
        )
    except BaseException:
        _remove_unjournaled_scratch(capability.scratch_root)
        raise
    try:
        scratch_registry = (
            capability.scratch_root / "registries/threshold_registry_v1.json"
        )
        scratch_registry.parent.mkdir()
        scratch_registry.write_bytes(registry_path.read_bytes())
        if sha256_file(scratch_registry) != formal_registry_sha:
            raise RuntimeError("copied threshold registry hash mismatch")
        _enforce_reproduction_tree_resource_limits(capability.scratch_root)
        base_environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GREEN_DEBT_ANALYSIS_REPRO_")
        }
        for planned in planned_analysis_commands(
            code_root=code,
            data_root=data,
            output_root=capability.scratch_root,
        ):
            if planned.stage is None:
                if command_runner is None:
                    _run_analysis_stage_process(
                        planned.command,
                        base_environment,
                        (),
                        None,
                        cwd=code,
                    )
                else:
                    command_runner(planned.command, dict(base_environment))
                continue
            authority = capability.open_stage_authority(planned.stage)
            try:
                environment = {**base_environment, **authority.environment}
                if command_runner is None:
                    _run_analysis_stage_process(
                        planned.command,
                        environment,
                        authority.pass_fds,
                        authority.set_stage_root_pid,
                        cwd=code,
                    )
                else:
                    authority.set_stage_root_pid(os.getpid())
                    command_runner(planned.command, environment)
            finally:
                authority.close()
            _enforce_reproduction_tree_resource_limits(
                capability.scratch_root
            )
        scratch_manifest = _json_object(
            capability.scratch_root / "run_manifest.json",
            "reproduced run manifest",
        )
        bound_fields = (
            "status", "run_id", "spec_id", "input_authority_hash",
            "git_commit", "renv_lock_sha256", "evidence_policy_sha256",
            "threshold_registry_hash",
            "created_at_utc", "audited_at_utc",
        )
        for field in bound_fields:
            if scratch_manifest.get(field) != manifest.get(field):
                raise AnalysisReproductionMismatch(
                    f"reproduced run manifest {field} differs"
                )
        if sha256_file(manifest_path) != formal_manifest_sha:
            raise AnalysisReproductionMismatch("formal run manifest changed")
        if sha256_file(registry_path) != formal_registry_sha:
            raise AnalysisReproductionMismatch("formal threshold registry changed")
        comparison = compare_analysis_output_trees(formal, capability.scratch_root)
        if enforce_production_counts and (
            comparison.compared_parquet,
            comparison.compared_json,
            comparison.compared_csv,
            comparison.compared_figures,
        ) != (23, 1, 8, 7):
            raise AnalysisReproductionMismatch(
                "analysis reproduction did not compare the complete frozen output set"
            )
        intermediate_after = analysis_tree_snapshot(data / "05_中间数据")
        manifests_after = _analysis_manifest_snapshot(data)
        if intermediate_before != intermediate_after:
            raise AnalysisReproductionMismatch(
                "05_中间数据 changed during reproduction"
            )
        if manifests_before != manifests_after:
            raise AnalysisReproductionMismatch(
                "authority manifests changed during reproduction"
            )
        spec = load_analysis_spec(code / "config/analysis.yaml")
        project = load_project_config(code / "config/project.yaml")
        if directory_usage_bytes(formal) > spec.outputs.quota_gb * GIB:
            raise RuntimeError("10 GB cumulative analysis output quota exceeded")
        project_bytes = combined_project_usage_bytes(data, formal)
        if (
            project.storage.absolute_limit_gb is None
            or project_bytes >= project.storage.absolute_limit_gb * GIB
        ):
            raise RuntimeError("150 GB absolute project limit reached")
        return create_analysis_reproduction_receipt(
            capability=capability,
            formal_manifest_path=manifest_path,
            data_snapshot_before=intermediate_before.sha256,
            data_snapshot_after=intermediate_after.sha256,
            manifest_snapshot_before=manifests_before.sha256,
            manifest_snapshot_after=manifests_after.sha256,
            compared_parquet=comparison.compared_parquet,
            compared_json=comparison.compared_json,
            compared_csv=comparison.compared_csv,
            compared_figures=comparison.compared_figures,
            max_scaled_float_error=comparison.max_scaled_float_error,
            threshold_registry_hash=comparison.threshold_registry_hash,
        )
    except BaseException as exc:
        try:
            create_failed_analysis_reproduction_receipt(
                formal_output_root=formal,
                scratch_root=capability.scratch_root,
                failure=f"{type(exc).__name__}: {exc}",
            )
        except Exception as journal_error:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "failed to update reproduction journal: "
                    f"{type(journal_error).__name__}: {journal_error}"
                )
        raise


def _reject_unclean_reproduction(formal: Path) -> None:
    receipts = formal / "reproduction_receipts"
    if receipts.exists():
        if receipts.is_symlink() or not receipts.is_dir():
            raise ValueError("analysis reproduction journal root is unsafe")
        for receipt in sorted(receipts.glob("reproduction.*.json")):
            payload = _json_object(receipt, "analysis reproduction journal")
            if payload.get("receipt_sha256") != _analysis_receipt_hash(payload):
                raise ValueError("analysis reproduction journal hash mismatch")
            if payload.get("cleanup_status") != "cleaned":
                raise ValueError("an unclean analysis reproduction is active")
    temporary = formal / "_tmp"
    if temporary.exists():
        temporary = _resolved_real_directory(
            temporary, "reproduction temporary root"
        )
        if any(temporary.glob("reproduce.*")):
            raise ValueError("an orphan or unclean reproduction scratch exists")


def run_analysis_reproduction_check(
    *,
    code_root: Path,
    data_root: Path,
    formal_output_root: Path,
    expected_git_commit: str | None = None,
    command_runner: Callable[[tuple[str, ...], dict[str, str]], None] | None = None,
    enforce_production_counts: bool = True,
    _signal_race_hook: Callable[[str], None] | None = None,
) -> Path:
    """Serialize one journaled reproduction and reject any unclean predecessor."""

    code = code_root.resolve()
    formal = _resolved_real_directory(
        resolve_frozen_analysis_output(code, formal_output_root),
        "formal output root",
    )
    temporary = formal / "_tmp"
    temporary.mkdir(mode=0o700, exist_ok=True)
    lock_path = temporary / ".analysis-reproduction.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    previous_handlers: dict[int, Any] | None = None
    deferred_signals: list[tuple[int, Any, bool]] = []
    handlers_active = False
    body_error: BaseException | None = None
    result: Path | None = None

    def interrupt(signum: int, _frame: Any) -> None:
        raise AnalysisReproductionInterrupted(signal.Signals(signum).name)

    race_hook = _signal_race_hook or (lambda _stage: None)
    try:
        if threading.current_thread() is threading.main_thread():
            previous_handlers = _install_grouped_stage_signal_handlers(
                interrupt, race_hook
            )
            handlers_active = True
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("an analysis reproduction is already active") from exc
        _reject_unclean_reproduction(formal)
        result = _run_analysis_reproduction_check_locked(
            code_root=code,
            data_root=data_root,
            formal_output_root=formal,
            expected_git_commit=expected_git_commit,
            command_runner=command_runner,
            enforce_production_counts=enforce_production_counts,
        )
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        restore_error: BaseException | None = None
        if previous_handlers is not None and handlers_active:
            try:
                _restore_grouped_stage_signal_handlers(
                    previous_handlers, deferred_signals, race_hook
                )
                handlers_active = False
            except BaseException as exc:
                restore_error = exc
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
        if restore_error is not None:
            if body_error is not None and hasattr(body_error, "add_note"):
                body_error.add_note(
                    "failed to restore reproduction lock signal state: "
                    f"{type(restore_error).__name__}: {restore_error}"
                )
            else:
                raise restore_error
    if previous_handlers is not None:
        _raise_deferred_stage_signal(deferred_signals, previous_handlers)
    if result is None:
        raise RuntimeError("analysis reproduction returned no receipt")
    return result


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _verified_entry_stat(parent_fd: int, name: str, expected: os.stat_result) -> None:
    observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not _same_file_identity(observed, expected) or stat.S_IFMT(
        observed.st_mode
    ) != stat.S_IFMT(expected.st_mode):
        raise ValueError("reproduction cleanup entry changed during exchange")


def _cleanup_entry_type(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    raise ValueError("reproduction cleanup found an unsupported special file")


def _open_cleanup_directory_at(
    parent_fd: int,
    name: str,
    observed: os.stat_result,
    expected: dict[str, Any] | None = None,
) -> int:
    """Open one known directory without following a replacement symlink."""

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        child_fd = os.open(name, flags, dir_fd=parent_fd)
    except PermissionError:
        os.chmod(
            name,
            stat.S_IMODE(observed.st_mode) | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        child_fd = os.open(name, flags, dir_fd=parent_fd)
    opened = os.fstat(child_fd)
    if not _same_file_identity(opened, observed) or (
        expected is not None
        and (
            opened.st_dev != expected["device"]
            or opened.st_ino != expected["inode"]
            or _cleanup_entry_type(opened.st_mode) != expected["type"]
        )
    ):
        os.close(child_fd)
        raise ValueError("reproduction cleanup child identity changed")
    return child_fd


def _cleanup_inventory_entry(
    parent_fd: int,
    name: str,
    relative: str,
    observed: os.stat_result,
) -> dict[str, Any]:
    entry_type = _cleanup_entry_type(observed.st_mode)
    entry: dict[str, Any] = {
        "path": relative,
        "type": entry_type,
        "device": observed.st_dev,
        "inode": observed.st_ino,
    }
    if entry_type == "symlink":
        entry["target"] = os.readlink(name, dir_fd=parent_fd)
    return entry


def _inventory_tree_at(
    directory_fd: int, root_device: int, prefix: str = ""
) -> list[dict[str, Any]]:
    """Capture a bounded inventory with iterative descriptor traversal."""

    if os.fstat(directory_fd).st_dev != root_device:
        raise ValueError("reproduction cleanup refuses a mount boundary")
    root_fd = os.dup(directory_fd)
    stack: list[tuple[int, Any, str, int]] = []
    inventory: list[dict[str, Any]] = []
    prefix_depth = 0 if not prefix else len(prefix.split("/"))
    try:
        stack.append((root_fd, os.scandir(root_fd), prefix, prefix_depth))
        while stack:
            parent_fd, iterator, parent_prefix, parent_depth = stack[-1]
            try:
                directory_entry = next(iterator)
            except StopIteration:
                iterator.close()
                os.close(parent_fd)
                stack.pop()
                continue
            name = directory_entry.name
            if not name or "/" in name or name in {".", ".."}:
                raise ValueError("reproduction cleanup found an unsafe entry label")
            relative = f"{parent_prefix}/{name}" if parent_prefix else name
            depth = parent_depth + 1
            if depth > _CLEANUP_INVENTORY_MAX_DEPTH:
                raise _CleanupInventoryLimitExceeded(
                    "depth", depth, _CLEANUP_INVENTORY_MAX_DEPTH
                )
            observed = os.stat(
                name, dir_fd=parent_fd, follow_symlinks=False
            )
            if observed.st_dev != root_device:
                raise ValueError("reproduction cleanup refuses a mount boundary")
            inventory.append(
                _cleanup_inventory_entry(
                    parent_fd, name, relative, observed
                )
            )
            if len(inventory) > _CLEANUP_INVENTORY_MAX_ENTRIES:
                raise _CleanupInventoryLimitExceeded(
                    "entries",
                    len(inventory),
                    _CLEANUP_INVENTORY_MAX_ENTRIES,
                )
            if stat.S_ISDIR(observed.st_mode):
                child_fd = _open_cleanup_directory_at(
                    parent_fd, name, observed
                )
                stack.append(
                    (child_fd, os.scandir(child_fd), relative, depth)
                )
        inventory.sort(key=lambda entry: entry["path"])
        canonical_bytes = len(_canonical_json_bytes(inventory))
        if canonical_bytes > _CLEANUP_INVENTORY_MAX_CANONICAL_BYTES:
            raise _CleanupInventoryLimitExceeded(
                "canonical_bytes",
                canonical_bytes,
                _CLEANUP_INVENTORY_MAX_CANONICAL_BYTES,
            )
        return inventory
    finally:
        for opened_fd, iterator, _parent_prefix, _depth in stack:
            try:
                iterator.close()
            finally:
                try:
                    os.close(opened_fd)
                except OSError:
                    pass


def _cleanup_inventory_map(
    inventory: Any,
    *,
    enforce_limits: bool = True,
) -> dict[str, dict[str, Any]]:
    if not isinstance(inventory, list):
        raise ValueError("analysis reproduction cleaning inventory is invalid")
    if enforce_limits:
        if len(inventory) > _CLEANUP_INVENTORY_MAX_ENTRIES:
            raise _CleanupInventoryLimitExceeded(
                "entries",
                len(inventory),
                _CLEANUP_INVENTORY_MAX_ENTRIES,
            )
        canonical_bytes = len(_canonical_json_bytes(inventory))
        if canonical_bytes > _CLEANUP_INVENTORY_MAX_CANONICAL_BYTES:
            raise _CleanupInventoryLimitExceeded(
                "canonical_bytes",
                canonical_bytes,
                _CLEANUP_INVENTORY_MAX_CANONICAL_BYTES,
            )
    result: dict[str, dict[str, Any]] = {}
    for entry in inventory:
        if not isinstance(entry, dict) or set(entry) not in (
            {"path", "type", "device", "inode"},
            {"path", "type", "device", "inode", "target"},
        ):
            raise ValueError("analysis reproduction cleaning inventory is invalid")
        path = entry.get("path")
        entry_type = entry.get("type")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or "//" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or path in result
            or (
                enforce_limits
                and len(path.split("/")) > _CLEANUP_INVENTORY_MAX_DEPTH
            )
            or entry_type not in {"directory", "regular", "symlink", "fifo", "socket"}
            or not isinstance(entry.get("device"), int)
            or not isinstance(entry.get("inode"), int)
            or (entry_type == "symlink") != ("target" in entry)
            or (
                entry_type == "symlink"
                and not isinstance(entry.get("target"), str)
            )
        ):
            raise ValueError("analysis reproduction cleaning inventory is invalid")
        result[path] = entry
    for path in result:
        if "/" not in path:
            continue
        parent = path.rsplit("/", 1)[0]
        if parent not in result or result[parent]["type"] != "directory":
            raise ValueError(
                "analysis reproduction cleaning inventory parent is invalid"
            )
    return result


def _validate_inventory_subset_at(
    directory_fd: int,
    root_device: int,
    inventory: dict[str, dict[str, Any]],
    prefix: str = "",
) -> None:
    """Require every remaining entry to be an unchanged member of the inventory."""

    if os.fstat(directory_fd).st_dev != root_device:
        raise ValueError("reproduction cleanup refuses a mount boundary")
    for name in sorted(os.listdir(directory_fd)):
        relative = f"{prefix}/{name}" if prefix else name
        expected = inventory.get(relative)
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            expected is None
            or observed.st_dev != root_device
            or observed.st_dev != expected["device"]
            or observed.st_ino != expected["inode"]
            or _cleanup_entry_type(observed.st_mode) != expected["type"]
        ):
            raise ValueError("reproduction cleanup inventory entry changed")
        if expected["type"] == "symlink":
            if os.readlink(name, dir_fd=directory_fd) != expected["target"]:
                raise ValueError("reproduction cleanup symbolic link changed")
        elif expected["type"] == "directory":
            child_fd = _open_cleanup_directory_at(directory_fd, name, observed)
            try:
                _validate_inventory_subset_at(
                    child_fd, root_device, inventory, relative
                )
            finally:
                os.close(child_fd)


def _expected_cleanup_entry_at(
    parent_fd: int,
    name: str,
    relative: str,
    root_device: int,
    inventory: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], os.stat_result]:
    """Revalidate one persisted identity immediately before an operation."""

    if os.fstat(parent_fd).st_dev != root_device:
        raise ValueError("reproduction cleanup refuses a mount boundary")
    expected = inventory.get(relative)
    if expected is None or expected.get("path") != relative:
        raise ValueError("reproduction cleanup inventory entry is missing")
    observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        expected["device"] != root_device
        or observed.st_dev != root_device
        or observed.st_dev != expected["device"]
        or observed.st_ino != expected["inode"]
        or _cleanup_entry_type(observed.st_mode) != expected["type"]
    ):
        raise ValueError("reproduction cleanup inventory entry changed")
    if expected["type"] == "symlink" and os.readlink(
        name, dir_fd=parent_fd
    ) != expected["target"]:
        raise ValueError("reproduction cleanup symbolic link changed")
    return expected, observed


def _renamed_cleanup_entry_at(
    parent_fd: int,
    name: str,
    root_device: int,
    expected: dict[str, Any],
) -> os.stat_result:
    """Revalidate an expected entry under its unpredictable quarantine label."""

    if os.fstat(parent_fd).st_dev != root_device:
        raise ValueError("reproduction cleanup refuses a mount boundary")
    observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        expected["device"] != root_device
        or observed.st_dev != expected["device"]
        or observed.st_ino != expected["inode"]
        or _cleanup_entry_type(observed.st_mode) != expected["type"]
    ):
        raise ValueError("reproduction cleanup quarantined entry changed")
    if expected["type"] == "symlink" and os.readlink(
        name, dir_fd=parent_fd
    ) != expected["target"]:
        raise ValueError("reproduction cleanup symbolic link changed")
    return observed


def _remove_tree_at(
    directory_fd: int,
    root_device: int,
    inventory: list[dict[str, Any]],
    prefix: str = "",
    on_removed: Callable[[], None] | None = None,
    entry_race_hook: Callable[[str, str, str | None], None] | None = None,
) -> None:
    """Remove only inventoried children relative to one held directory FD."""

    inventory_map = _cleanup_inventory_map(inventory)
    _validate_inventory_subset_at(
        directory_fd, root_device, inventory_map, prefix
    )
    race_hook = entry_race_hook or (
        lambda _stage, _relative, _renamed_leaf: None
    )
    for name in sorted(os.listdir(directory_fd)):
        relative = f"{prefix}/{name}" if prefix else name
        expected, observed = _expected_cleanup_entry_at(
            directory_fd, name, relative, root_device, inventory_map
        )
        if expected["type"] == "directory":
            race_hook("entry_stat_to_open", relative, None)
            expected, observed = _expected_cleanup_entry_at(
                directory_fd, name, relative, root_device, inventory_map
            )
            child_fd = _open_cleanup_directory_at(
                directory_fd, name, observed, expected
            )
            try:
                _remove_tree_at(
                    child_fd,
                    root_device,
                    inventory,
                    relative,
                    on_removed,
                    race_hook,
                )
                race_hook("entry_walk_to_rmdir", relative, None)
                expected, current = _expected_cleanup_entry_at(
                    directory_fd, name, relative, root_device, inventory_map
                )
                opened = os.fstat(child_fd)
                if (
                    not _same_file_identity(opened, current)
                    or opened.st_dev != expected["device"]
                    or opened.st_ino != expected["inode"]
                    or expected["type"] != "directory"
                ):
                    raise ValueError(
                        "reproduction cleanup directory identity changed"
                    )
                os.rmdir(name, dir_fd=directory_fd)
            finally:
                os.close(child_fd)
        else:
            race_hook("entry_stat_to_rename", relative, None)
            expected, _observed = _expected_cleanup_entry_at(
                directory_fd, name, relative, root_device, inventory_map
            )
            quarantine = f".cleanup-file.{secrets.token_hex(16)}"
            os.rename(
                name,
                quarantine,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            race_hook("entry_rename_to_unlink", relative, quarantine)
            _renamed_cleanup_entry_at(
                directory_fd, quarantine, root_device, expected
            )
            os.unlink(quarantine, dir_fd=directory_fd)
        if on_removed is not None:
            on_removed()


def _select_bounded_cleanup_batch(
    root_fd: int, root_device: int
) -> dict[str, Any] | None:
    """Select one constant-depth promotion/removal without retaining ancestors."""

    if os.fstat(root_fd).st_dev != root_device:
        raise ValueError("reproduction cleanup refuses a mount boundary")
    with os.scandir(root_fd) as iterator:
        directory_entry = next(iterator, None)
    if directory_entry is None:
        return None
    name = directory_entry.name
    if not name or "/" in name or name in {".", ".."}:
        raise ValueError("reproduction cleanup found an unsafe entry label")
    observed = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    if observed.st_dev != root_device:
        raise ValueError("reproduction cleanup refuses a mount boundary")
    entry = _cleanup_inventory_entry(root_fd, name, name, observed)
    destination = f".cleanup-node.{secrets.token_hex(16)}"
    try:
        os.stat(destination, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise ValueError("reproduction cleanup destination unexpectedly exists")
    if entry["type"] != "directory":
        batch: dict[str, Any] = {
            "schema_version": 2,
            "operation": "remove",
            "destination": destination,
            "entries": [entry],
        }
        _validate_bounded_cleanup_batch(batch)
        return batch

    child_fd = _open_cleanup_directory_at(root_fd, name, observed, entry)
    try:
        with os.scandir(child_fd) as iterator:
            child_directory_entry = next(iterator, None)
        if child_directory_entry is None:
            batch = {
                "schema_version": 2,
                "operation": "remove",
                "destination": destination,
                "entries": [entry],
            }
        else:
            child_name = child_directory_entry.name
            if not child_name or "/" in child_name or child_name in {".", ".."}:
                raise ValueError("reproduction cleanup found an unsafe entry label")
            child_observed = os.stat(
                child_name, dir_fd=child_fd, follow_symlinks=False
            )
            if child_observed.st_dev != root_device:
                raise ValueError("reproduction cleanup refuses a mount boundary")
            child_entry = _cleanup_inventory_entry(
                child_fd,
                child_name,
                f"{name}/{child_name}",
                child_observed,
            )
            batch = {
                "schema_version": 2,
                "operation": "promote",
                "destination": destination,
                "entries": [entry, child_entry],
            }
    finally:
        os.close(child_fd)
    _validate_bounded_cleanup_batch(batch)
    return batch


def _validate_bounded_cleanup_batch(
    batch: Any,
) -> tuple[str, list[dict[str, Any]], str | None]:
    """Validate both constant-depth batches and bounded pre-upgrade journals."""

    if not isinstance(batch, dict):
        raise ValueError("analysis reproduction cleanup batch is invalid")
    if batch.get("schema_version") == 2:
        if set(batch) != {
            "schema_version", "operation", "destination", "entries"
        }:
            raise ValueError("analysis reproduction cleanup batch is invalid")
        operation = batch.get("operation")
        destination = batch.get("destination")
        entries = batch.get("entries")
        if (
            operation not in {"remove", "promote"}
            or not isinstance(destination, str)
            or not re.fullmatch(r"\.cleanup-node\.[0-9a-f]{32}", destination)
            or not isinstance(entries, list)
            or len(entries) != (1 if operation == "remove" else 2)
            or len(entries) > _CLEANUP_BATCH_MAX_ENTRIES
        ):
            raise ValueError("analysis reproduction cleanup batch is invalid")
        if len(_canonical_json_bytes(batch)) > _CLEANUP_BATCH_MAX_CANONICAL_BYTES:
            raise ValueError(
                "reproduction cleanup batch canonical_bytes limit exceeded"
            )
        inventory = _cleanup_inventory_map(entries, enforce_limits=False)
        parent = entries[0]
        if "/" in parent["path"]:
            raise ValueError("analysis reproduction cleanup batch is invalid")
        if operation == "promote":
            child = entries[1]
            if (
                parent["type"] != "directory"
                or child["path"].count("/") != 1
                or child["path"].rsplit("/", 1)[0] != parent["path"]
            ):
                raise ValueError("analysis reproduction cleanup batch is invalid")
        if len(inventory) != len(entries):
            raise ValueError("analysis reproduction cleanup batch is invalid")
        return operation, entries, destination

    ancestors = batch.get("ancestors")
    entries = batch.get("entries")
    if (
        set(batch) != {"ancestors", "entries"}
        or not isinstance(ancestors, list)
        or not isinstance(entries, list)
        or not entries
        or len(ancestors) > _CLEANUP_LEGACY_BATCH_MAX_ANCESTORS
        or len(entries) > _CLEANUP_BATCH_MAX_ENTRIES
    ):
        raise ValueError("analysis reproduction cleanup batch is invalid")
    if len(_canonical_json_bytes(batch)) > _CLEANUP_BATCH_MAX_CANONICAL_BYTES:
        raise ValueError(
            "reproduction cleanup batch canonical_bytes limit exceeded"
        )
    _cleanup_inventory_map([*ancestors, *entries], enforce_limits=False)
    return "legacy", entries, None


def _open_bounded_batch_parent(
    root_fd: int,
    root_device: int,
    ancestors: list[dict[str, Any]],
) -> int:
    current_fd = os.dup(root_fd)
    inventory = _cleanup_inventory_map(ancestors, enforce_limits=False)
    try:
        for expected in ancestors:
            relative = expected["path"]
            name = relative.rsplit("/", 1)[-1]
            expected, observed = _expected_cleanup_entry_at(
                current_fd, name, relative, root_device, inventory
            )
            child_fd = _open_cleanup_directory_at(
                current_fd, name, observed, expected
            )
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _remove_legacy_bounded_cleanup_batch(
    root_fd: int,
    root_device: int,
    batch: dict[str, Any],
) -> None:
    ancestors = batch.get("ancestors")
    entries = batch.get("entries")
    if not isinstance(ancestors, list) or not isinstance(entries, list) or not entries:
        raise ValueError("analysis reproduction cleanup batch is invalid")
    inventory = _cleanup_inventory_map(
        [*ancestors, *entries], enforce_limits=False
    )
    parent_fd = _open_bounded_batch_parent(
        root_fd, root_device, ancestors
    )
    try:
        for expected in entries:
            relative = expected["path"]
            name = relative.rsplit("/", 1)[-1]
            try:
                expected, observed = _expected_cleanup_entry_at(
                    parent_fd, name, relative, root_device, inventory
                )
            except FileNotFoundError:
                continue
            if expected["type"] == "directory":
                child_fd = _open_cleanup_directory_at(
                    parent_fd, name, observed, expected
                )
                try:
                    with os.scandir(child_fd) as iterator:
                        if next(iterator, None) is not None:
                            raise ValueError(
                                "reproduction cleanup batch directory is not empty"
                            )
                    expected, current = _expected_cleanup_entry_at(
                        parent_fd, name, relative, root_device, inventory
                    )
                    if not _same_file_identity(os.fstat(child_fd), current):
                        raise ValueError(
                            "reproduction cleanup batch directory changed"
                        )
                    os.rmdir(name, dir_fd=parent_fd)
                finally:
                    os.close(child_fd)
            else:
                expected, _observed = _expected_cleanup_entry_at(
                    parent_fd, name, relative, root_device, inventory
                )
                quarantine = f".cleanup-file.{secrets.token_hex(16)}"
                os.rename(
                    name,
                    quarantine,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                _renamed_cleanup_entry_at(
                    parent_fd, quarantine, root_device, expected
                )
                os.unlink(quarantine, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


def _bounded_batch_entry_presence(
    parent_fd: int, name: str
) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _remove_frontier_cleanup_batch(
    root_fd: int,
    root_device: int,
    batch: dict[str, Any],
    entry_race_hook: Callable[[str, str, str | None], None] | None = None,
) -> None:
    operation, entries, destination = _validate_bounded_cleanup_batch(batch)
    if operation == "legacy" or destination is None:
        raise ValueError("analysis reproduction cleanup batch is invalid")
    inventory = _cleanup_inventory_map(entries, enforce_limits=False)
    race_hook = entry_race_hook or (
        lambda _stage, _relative, _renamed_leaf: None
    )
    if operation == "promote":
        parent_expected = entries[0]
        child_expected = entries[1]
        parent_name = parent_expected["path"]
        child_relative = child_expected["path"]
        child_name = child_relative.rsplit("/", 1)[-1]
        parent_expected, parent_observed = _expected_cleanup_entry_at(
            root_fd, parent_name, parent_name, root_device, inventory
        )
        parent_fd = _open_cleanup_directory_at(
            root_fd, parent_name, parent_observed, parent_expected
        )
        try:
            source = _bounded_batch_entry_presence(parent_fd, child_name)
            promoted = _bounded_batch_entry_presence(root_fd, destination)
            if source is not None and promoted is not None:
                raise ValueError(
                    "reproduction cleanup promotion found both bound entries"
                )
            if source is not None:
                child_expected, _child_observed = _expected_cleanup_entry_at(
                    parent_fd,
                    child_name,
                    child_relative,
                    root_device,
                    inventory,
                )
                race_hook(
                    "bounded_entry_stat_to_rename",
                    child_relative,
                    destination,
                )
                parent_expected, current_parent = _expected_cleanup_entry_at(
                    root_fd,
                    parent_name,
                    parent_name,
                    root_device,
                    inventory,
                )
                if not _same_file_identity(os.fstat(parent_fd), current_parent):
                    raise ValueError(
                        "reproduction cleanup batch directory changed"
                    )
                child_expected, _child_observed = _expected_cleanup_entry_at(
                    parent_fd,
                    child_name,
                    child_relative,
                    root_device,
                    inventory,
                )
                if _bounded_batch_entry_presence(root_fd, destination) is not None:
                    raise ValueError(
                        "reproduction cleanup destination unexpectedly exists"
                    )
                os.rename(
                    child_name,
                    destination,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=root_fd,
                )
                _renamed_cleanup_entry_at(
                    root_fd, destination, root_device, child_expected
                )
            elif promoted is not None:
                _renamed_cleanup_entry_at(
                    root_fd, destination, root_device, child_expected
                )
            else:
                raise ValueError(
                    "reproduction cleanup promotion lost both bound entries"
                )
        finally:
            os.close(parent_fd)
        race_hook(
            "bounded_rename_to_journal_clear",
            child_relative,
            destination,
        )
        return

    expected = entries[0]
    relative = expected["path"]
    source = _bounded_batch_entry_presence(root_fd, relative)
    quarantined = _bounded_batch_entry_presence(root_fd, destination)
    if source is not None and quarantined is not None:
        raise ValueError("reproduction cleanup removal found both bound entries")
    if source is not None:
        expected, observed = _expected_cleanup_entry_at(
            root_fd, relative, relative, root_device, inventory
        )
        opened_fd = -1
        if expected["type"] == "directory":
            opened_fd = _open_cleanup_directory_at(
                root_fd, relative, observed, expected
            )
            with os.scandir(opened_fd) as iterator:
                if next(iterator, None) is not None:
                    os.close(opened_fd)
                    raise ValueError(
                        "reproduction cleanup batch directory is not empty"
                    )
        try:
            race_hook("bounded_entry_stat_to_rename", relative, destination)
            expected, current = _expected_cleanup_entry_at(
                root_fd, relative, relative, root_device, inventory
            )
            if opened_fd >= 0 and not _same_file_identity(
                os.fstat(opened_fd), current
            ):
                raise ValueError("reproduction cleanup batch directory changed")
            if _bounded_batch_entry_presence(root_fd, destination) is not None:
                raise ValueError(
                    "reproduction cleanup destination unexpectedly exists"
                )
            os.rename(
                relative,
                destination,
                src_dir_fd=root_fd,
                dst_dir_fd=root_fd,
            )
        finally:
            if opened_fd >= 0:
                os.close(opened_fd)
        quarantined = _renamed_cleanup_entry_at(
            root_fd, destination, root_device, expected
        )
    elif quarantined is not None:
        _renamed_cleanup_entry_at(
            root_fd, destination, root_device, expected
        )
    else:
        return
    if expected["type"] == "directory":
        child_fd = _open_cleanup_directory_at(
            root_fd, destination, quarantined, expected
        )
        try:
            with os.scandir(child_fd) as iterator:
                if next(iterator, None) is not None:
                    raise ValueError(
                        "reproduction cleanup batch directory is not empty"
                    )
            _renamed_cleanup_entry_at(
                root_fd, destination, root_device, expected
            )
            os.rmdir(destination, dir_fd=root_fd)
        finally:
            os.close(child_fd)
    else:
        race_hook("bounded_rename_to_unlink", relative, destination)
        _renamed_cleanup_entry_at(
            root_fd, destination, root_device, expected
        )
        os.unlink(destination, dir_fd=root_fd)
    race_hook("bounded_rename_to_journal_clear", relative, destination)


def _remove_bounded_cleanup_batch(
    root_fd: int,
    root_device: int,
    batch: dict[str, Any],
    entry_race_hook: Callable[[str, str, str | None], None] | None = None,
) -> None:
    operation, _entries, _destination = _validate_bounded_cleanup_batch(batch)
    if operation == "legacy":
        _remove_legacy_bounded_cleanup_batch(root_fd, root_device, batch)
    else:
        _remove_frontier_cleanup_batch(
            root_fd, root_device, batch, entry_race_hook
        )


def _remove_tree_in_bounded_batches(
    root_fd: int,
    root_device: int,
    payload: dict[str, Any],
    receipt_path: Path,
    entry_race_hook: Callable[[str, str, str | None], None] | None = None,
) -> None:
    """Persist one expected batch at a time so an overlimit tree is recoverable."""

    while True:
        batch = payload.get("cleaning_batch")
        if batch is None:
            batch = _select_bounded_cleanup_batch(root_fd, root_device)
            if batch is None:
                return
            payload["cleaning_batch"] = batch
            payload["receipt_sha256"] = _analysis_receipt_hash(payload)
            _write_cleanup_reproduction_receipt(receipt_path, payload)
        _remove_bounded_cleanup_batch(
            root_fd, root_device, batch, entry_race_hook
        )
        payload["cleaning_batch"] = None
        payload["receipt_sha256"] = _analysis_receipt_hash(payload)
        _write_cleanup_reproduction_receipt(receipt_path, payload)


def _cleanup_analysis_reproduction_locked(
    receipt_path: Path,
    *,
    _race_hook: Callable[[str, Path, Path | None], None] | None = None,
    _entry_race_hook: Callable[[str, str, str | None], None] | None = None,
) -> Path:
    """Delete only the unchanged directory object bound by a valid receipt."""

    receipt_lexical = Path(os.path.abspath(os.fspath(receipt_path)))
    if receipt_lexical.is_symlink() or not receipt_lexical.is_file():
        raise ValueError("analysis reproduction receipt is missing or unsafe")
    payload = _json_object(receipt_lexical, "analysis reproduction receipt")
    if payload.get("receipt_sha256") != _analysis_receipt_hash(payload):
        raise ValueError("analysis reproduction receipt hash mismatch")
    required_identity = {
        "schema_version": 1,
        "receipt_kind": "analysis-reproduction",
    }
    for field, expected in required_identity.items():
        if payload.get(field) != expected:
            raise ValueError("analysis reproduction receipt state is invalid")
    formal = _resolved_real_directory(
        Path(payload["formal_output_root"]), "formal output root"
    )
    status = payload.get("status")
    cleanup_status = payload.get("cleanup_status")
    execution_id = payload.get("execution_id")
    expected_receipt = _analysis_reproduction_journal_path(formal, execution_id)
    if receipt_lexical.resolve() != expected_receipt or (
        status not in {"pending", "matched", "failed", "cleaning"}
    ):
        raise ValueError("analysis reproduction receipt has the wrong location")
    resuming = status == "cleaning"
    if resuming:
        if cleanup_status != "cleaning" or payload.get("pre_cleanup_status") not in {
            "pending", "matched", "failed"
        }:
            raise ValueError("analysis reproduction cleaning state is invalid")
    elif cleanup_status != "pending":
        raise ValueError("analysis reproduction receipt state is invalid")
    if status == "matched":
        formal_manifest = formal / "run_manifest.json"
        if (
            str(formal_manifest) != payload.get("formal_run_manifest")
            or not formal_manifest.is_file()
            or formal_manifest.is_symlink()
            or sha256_file(formal_manifest)
            != payload.get("formal_run_manifest_sha256")
        ):
            raise ValueError("analysis reproduction formal run manifest changed")
        formal_payload = _json_object(formal_manifest, "formal run manifest")
        for field in (
            "run_id",
            "spec_id",
            "input_authority_hash",
            "git_commit",
            "renv_lock_sha256",
            "evidence_policy_sha256",
            "threshold_registry_hash",
        ):
            receipt_field = "formal_run_id" if field == "run_id" else field
            if formal_payload.get(field) != payload.get(receipt_field):
                raise ValueError(
                    "analysis reproduction formal run manifest identity changed"
                )
    elif status == "failed":
        scratch_manifest_value = payload.get("scratch_run_manifest")
        scratch_manifest_hash = payload.get("scratch_run_manifest_sha256")
        if (scratch_manifest_value is None) != (scratch_manifest_hash is None):
            raise ValueError("failed reproduction manifest binding is invalid")
        if scratch_manifest_value is not None:
            scratch_manifest = Path(scratch_manifest_value)
            if (
                scratch_manifest.is_symlink()
                or not scratch_manifest.is_file()
                or sha256_file(scratch_manifest) != scratch_manifest_hash
            ):
                raise ValueError("failed reproduction manifest changed")
    temporary = _resolved_real_directory(
        formal / "_tmp", "reproduction temporary root"
    )
    scratch = Path(os.path.abspath(os.fspath(payload["scratch_root"])))
    if (
        not isinstance(execution_id, str)
        or scratch.parent != temporary
        or scratch.name != f"reproduce.{execution_id}"
        or not _REPRODUCTION_EXECUTION_RE.fullmatch(execution_id)
    ):
        raise ValueError("analysis reproduction cleanup target is too broad or unbound")
    if not resuming:
        scratch = _resolved_real_directory(scratch, "reproduction scratch root")
        scratch_stat = scratch.stat()
        if (
            scratch_stat.st_dev != payload.get("scratch_device")
            or scratch_stat.st_ino != payload.get("scratch_inode")
        ):
            raise ValueError("analysis reproduction scratch identity changed")
        marker = scratch / _REPRODUCTION_MARKER_NAME
        if str(marker) != payload.get("marker_path"):
            raise ValueError("analysis reproduction marker path changed")
        if status == "matched":
            if (
                marker.is_symlink()
                or not marker.is_file()
                or sha256_file(marker) != payload.get("marker_sha256")
            ):
                raise ValueError("analysis reproduction marker changed")
            marker_payload = _json_object(
                marker, "analysis reproduction capability marker"
            )
            marker_bindings = {
                "instance_id": payload.get("marker_instance_id"),
                "schema_version": payload.get("marker_schema_version"),
                "execution_id": execution_id,
                "scratch_root": str(scratch),
                "formal_output_root": str(formal),
                "data_root": payload.get("data_root"),
                "git_commit": payload.get("git_commit"),
                "formal_run_id": payload.get("formal_run_id"),
            }
            for field, expected in marker_bindings.items():
                if marker_payload.get(field) != expected:
                    raise ValueError(
                        "analysis reproduction marker binding mismatch"
                    )
        preflight_fd = os.open(
            scratch, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            try:
                inventory = _inventory_tree_at(
                    preflight_fd, scratch_stat.st_dev
                )
                cleaning_strategy = "persisted_inventory_v1"
                cleaning_limit: dict[str, int | str] | None = None
            except _CleanupInventoryLimitExceeded as limit_error:
                inventory = None
                cleaning_strategy = "bounded_batches_v1"
                cleaning_limit = limit_error.payload()
            if inventory is None:
                try:
                    marker_stat = os.stat(
                        _REPRODUCTION_MARKER_NAME,
                        dir_fd=preflight_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    marker_entry = None
                else:
                    marker_entry = _cleanup_inventory_entry(
                        preflight_fd,
                        _REPRODUCTION_MARKER_NAME,
                        _REPRODUCTION_MARKER_NAME,
                        marker_stat,
                    )
            else:
                marker_entry = _cleanup_inventory_map(inventory).get(
                    _REPRODUCTION_MARKER_NAME
                )
        finally:
            os.close(preflight_fd)
        marker_observation: dict[str, Any] = {
            "state": "missing" if marker_entry is None else marker_entry["type"],
            "initial_sha256": payload.get("marker_sha256"),
        }
        if marker_entry is not None:
            marker_observation.update(marker_entry)
        quarantine_name = f".cleanup-{execution_id}-{secrets.token_hex(16)}"
        payload["pre_cleanup_status"] = status
        payload["status"] = "cleaning"
        payload["cleanup_status"] = "cleaning"
        payload["cleaning_original_name"] = scratch.name
        payload["cleaning_quarantine_name"] = quarantine_name
        payload["cleaning_inventory_version"] = 2
        payload["cleaning_strategy"] = cleaning_strategy
        payload["cleaning_limit"] = cleaning_limit
        payload["cleaning_inventory"] = inventory
        payload["cleaning_inventory_bytes"] = (
            len(_canonical_json_bytes(inventory))
            if inventory is not None
            else None
        )
        payload["cleaning_batch"] = None
        payload["cleaning_marker_observation"] = marker_observation
        payload["state_updated_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        payload["receipt_sha256"] = _analysis_receipt_hash(payload)
        _write_cleanup_reproduction_receipt(receipt_lexical, payload)
    else:
        quarantine_name = payload.get("cleaning_quarantine_name")
        if (
            payload.get("cleaning_original_name") != scratch.name
            or not isinstance(quarantine_name, str)
            or not re.fullmatch(
                rf"\.cleanup-{re.escape(execution_id)}-[0-9a-f]{{32}}",
                quarantine_name,
            )
        ):
            raise ValueError("analysis reproduction quarantine binding is invalid")
        if payload.get("cleaning_inventory_version") not in {1, 2} or not isinstance(
            payload.get("cleaning_marker_observation"), dict
        ):
            raise ValueError("analysis reproduction cleaning inventory is invalid")
        inventory = payload.get("cleaning_inventory")
        cleaning_strategy = payload.get(
            "cleaning_strategy", "persisted_inventory_v1"
        )
        if cleaning_strategy == "persisted_inventory_v1":
            _cleanup_inventory_map(inventory)
        elif cleaning_strategy == "bounded_batches_v1":
            if inventory is not None or not isinstance(
                payload.get("cleaning_limit"), dict
            ):
                raise ValueError(
                    "analysis reproduction bounded cleanup state is invalid"
                )
            batch = payload.get("cleaning_batch")
            if batch is not None:
                _validate_bounded_cleanup_batch(batch)
        else:
            raise ValueError("analysis reproduction cleaning strategy is invalid")
    hook = _race_hook or (lambda _stage, _scratch, _quarantine: None)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    formal_fd = os.open(formal, directory_flags)
    temporary_fd = -1
    scratch_fd = -1
    quarantine_path = temporary / quarantine_name
    try:
        temporary_fd = os.open("_tmp", directory_flags, dir_fd=formal_fd)
        temporary_stat = os.fstat(temporary_fd)
        entries: dict[str, os.stat_result] = {}
        for name in (scratch.name, quarantine_name):
            try:
                entries[name] = os.stat(
                    name, dir_fd=temporary_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                pass
        if len(entries) > 1:
            raise ValueError("analysis reproduction cleanup found both bound entries")
        if not entries:
            if not resuming:
                raise ValueError("analysis reproduction scratch disappeared")
        else:
            active_name = next(iter(entries))
            scratch_fd = os.open(active_name, directory_flags, dir_fd=temporary_fd)
            held_stat = os.fstat(scratch_fd)
            if (
                held_stat.st_dev != payload.get("scratch_device")
                or held_stat.st_ino != payload.get("scratch_inode")
                or held_stat.st_dev != temporary_stat.st_dev
            ):
                raise ValueError("analysis reproduction scratch identity changed")
            inventory_map = (
                _cleanup_inventory_map(inventory)
                if cleaning_strategy == "persisted_inventory_v1"
                else None
            )
            if inventory_map is not None:
                _validate_inventory_subset_at(
                    scratch_fd, held_stat.st_dev, inventory_map
                )
            if active_name == scratch.name:
                hook("validation_to_rename", scratch, None)
                os.rename(
                    scratch.name,
                    quarantine_name,
                    src_dir_fd=temporary_fd,
                    dst_dir_fd=temporary_fd,
                )
                _verified_entry_stat(temporary_fd, quarantine_name, held_stat)
            hook("rename_to_walk", scratch, quarantine_path)
            _verified_entry_stat(temporary_fd, quarantine_name, held_stat)
            if inventory_map is not None:
                _validate_inventory_subset_at(
                    scratch_fd, held_stat.st_dev, inventory_map
                )
                _remove_tree_at(
                    scratch_fd,
                    held_stat.st_dev,
                    inventory,
                    on_removed=lambda: hook(
                        "walk_progress", scratch, quarantine_path
                    ),
                    entry_race_hook=_entry_race_hook,
                )
            else:
                _remove_tree_in_bounded_batches(
                    scratch_fd,
                    held_stat.st_dev,
                    payload,
                    receipt_lexical,
                    _entry_race_hook,
                )
            hook("walk_to_rmdir", scratch, quarantine_path)
            _verified_entry_stat(temporary_fd, quarantine_name, held_stat)
            os.rmdir(quarantine_name, dir_fd=temporary_fd)
    finally:
        if scratch_fd >= 0:
            os.close(scratch_fd)
        if temporary_fd >= 0:
            os.close(temporary_fd)
        os.close(formal_fd)
    payload["status"] = "cleaned"
    payload["cleanup_status"] = "cleaned"
    payload["cleaned_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    payload["state_updated_at_utc"] = payload["cleaned_at_utc"]
    payload["receipt_sha256"] = _analysis_receipt_hash(payload)
    _write_cleanup_reproduction_receipt(receipt_lexical, payload)
    return scratch


def cleanup_analysis_reproduction(
    receipt_path: Path,
    *,
    _race_hook: Callable[[str, Path, Path | None], None] | None = None,
    _entry_race_hook: Callable[[str, str, str | None], None] | None = None,
) -> Path:
    """Serialize cleanup against the writer, then remove only its bound scratch."""

    receipt = Path(os.path.abspath(os.fspath(receipt_path)))
    if receipt.is_symlink() or not receipt.is_file():
        raise ValueError("analysis reproduction receipt is missing or unsafe")
    payload = _json_object(receipt, "analysis reproduction receipt")
    if payload.get("receipt_sha256") != _analysis_receipt_hash(payload):
        raise ValueError("analysis reproduction receipt hash mismatch")
    formal = _resolved_real_directory(
        Path(payload["formal_output_root"]), "formal output root"
    )
    temporary = _resolved_real_directory(
        formal / "_tmp", "reproduction temporary root"
    )
    temporary_fd = os.open(
        temporary, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        lock_fd = os.open(
            ".analysis-reproduction.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
            dir_fd=temporary_fd,
        )
    finally:
        os.close(temporary_fd)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("an active analysis reproduction holds the cleanup lock") from exc
        return _cleanup_analysis_reproduction_locked(
            receipt,
            _race_hook=_race_hook,
            _entry_race_hook=_entry_race_hook,
        )
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def canonical_authority_hash(authority: InputAuthority) -> str:
    payload = json.dumps(
        sorted(authority.input_hashes),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_threshold_number(value: int | float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("threshold registry contains a nonfinite number")
    text = f"{number:.17f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _canonical_threshold_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        return _canonical_threshold_number(value)
    if isinstance(value, list):
        return [_canonical_threshold_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical_threshold_value(value[key])
            for key in sorted(value)
        }
    raise ValueError(
        f"threshold registry contains an unsupported value: {type(value).__name__}"
    )


def canonical_threshold_registry_hash(payload: dict[str, Any]) -> str:
    """Hash every registry field except its hash and creation timestamp.

    Numeric scalars use a fixed 17-decimal representation so the R producer and
    Python verifier calculate identical bytes from IEEE-754 values.
    """

    if not isinstance(payload, dict):
        raise ValueError("threshold registry must be a JSON object")
    bound = {
        key: value
        for key, value in payload.items()
        if key not in {"registry_hash", "created_at_utc"}
    }
    canonical = json.dumps(
        _canonical_threshold_value(bound),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _require_hex_hash(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"threshold registry {field} must be a lowercase SHA-256")
    return text


def _require_json_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"threshold registry {field} must be an integer")
    return value


def verify_threshold_registry_payload(
    payload: dict[str, Any], spec: AnalysisSpec, context: RunContext
) -> None:
    """Fail closed when a frozen threshold registry is retargeted or altered."""

    if not isinstance(payload, dict):
        raise ValueError("threshold registry must be a JSON object")
    required = {
        "registry_id",
        "selection_outcome",
        "horizon",
        "gad_version",
        "sample_version",
        "criterion",
        "quantile_type",
        "tie_break",
        "seed",
        "sample_hash",
        "input_authority_hash",
        "q",
        "percentile",
        "low_share",
        "high_share",
        "bootstrap_draws",
        "bootstrap_valid_draws",
        "bootstrap_failed_draws",
        "bootstrap_status",
        "percentile_conf_low",
        "percentile_conf_high",
        "q_conf_low",
        "q_conf_high",
        "candidates",
        "created_at_utc",
        "registry_hash",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            f"threshold registry is missing: {', '.join(missing)}"
        )
    observed_hash = _require_hex_hash(payload["registry_hash"], "registry hash")
    if observed_hash != canonical_threshold_registry_hash(payload):
        raise ValueError("threshold registry hash mismatch")

    cell = spec.threshold_cell()
    if payload["registry_id"] != "threshold_registry_v1":
        raise ValueError("threshold registry id mismatch")
    if payload["selection_outcome"] != cell.outcome_id:
        raise ValueError("threshold registry selection outcome mismatch")
    for field in ("horizon", "quantile_type", "seed"):
        _require_json_integer(payload[field], field)
    identities = {
        "horizon": cell.horizon,
        "gad_version": cell.gad_version,
        "sample_version": cell.sample_version,
        "criterion": spec.threshold.criterion,
        "quantile_type": spec.threshold.quantile_type,
        "tie_break": spec.threshold.tie_break,
        "seed": spec.seed,
        "input_authority_hash": context.input_authority_hash,
    }
    for field, expected in identities.items():
        if payload[field] != expected:
            raise ValueError(f"threshold registry {field} mismatch")
    _require_hex_hash(payload["sample_hash"], "sample hash")
    _require_hex_hash(payload["input_authority_hash"], "input authority hash")

    q = float(payload["q"])
    percentile = _require_json_integer(payload["percentile"], "percentile")
    low_share = float(payload["low_share"])
    high_share = float(payload["high_share"])
    if not math.isfinite(q):
        raise ValueError("threshold registry q must be finite")
    if percentile not in spec.threshold.percentiles:
        raise ValueError("threshold registry percentile is outside the frozen grid")
    if (
        not math.isfinite(low_share)
        or not math.isfinite(high_share)
        or low_share < spec.threshold.minimum_regime_share
        or high_share < spec.threshold.minimum_regime_share
        or not math.isclose(low_share + high_share, 1.0, abs_tol=1e-12)
    ):
        raise ValueError("threshold registry regime shares are invalid")

    draws = _require_json_integer(payload["bootstrap_draws"], "bootstrap_draws")
    valid = _require_json_integer(
        payload["bootstrap_valid_draws"], "bootstrap_valid_draws"
    )
    failed = _require_json_integer(
        payload["bootstrap_failed_draws"], "bootstrap_failed_draws"
    )
    if draws != 999 or valid < 0 or failed < 0 or valid + failed != draws:
        raise ValueError("threshold bootstrap draw accounting mismatch")
    status = str(payload["bootstrap_status"])
    bound_fields = (
        "percentile_conf_low",
        "percentile_conf_high",
        "q_conf_low",
        "q_conf_high",
    )
    if status == "available":
        if valid < 900 or any(payload[field] is None for field in bound_fields):
            raise ValueError("threshold bootstrap availability is inconsistent")
        bounds = [float(payload[field]) for field in bound_fields]
        if any(not math.isfinite(value) for value in bounds):
            raise ValueError("threshold bootstrap bounds must be finite")
        if bounds[0] > bounds[1] or bounds[2] > bounds[3]:
            raise ValueError("threshold bootstrap confidence bounds are reversed")
    elif status == "insufficient_valid_draws":
        if valid >= 900 or any(payload[field] is not None for field in bound_fields):
            raise ValueError("threshold failed-bootstrap bounds must be null")
    else:
        raise ValueError("threshold bootstrap status is invalid")

    candidates = payload["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("threshold registry candidate grid must be non-empty")
    candidate_percentiles: set[int] = set()
    candidate_q: set[float] = set()
    selected_found = False
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("threshold candidate must be a JSON object")
        required_candidate = {
            "percentile", "q", "low_share", "high_share", "eligible",
            "ssr", "n", "status",
        }
        missing_candidate = sorted(required_candidate - set(candidate))
        if missing_candidate:
            raise ValueError(
                "threshold candidate is missing: "
                + ", ".join(missing_candidate)
            )
        candidate_percentile = _require_json_integer(
            candidate["percentile"], "candidate percentile"
        )
        candidate_value = float(candidate["q"])
        candidate_low_share = float(candidate["low_share"])
        candidate_high_share = float(candidate["high_share"])
        candidate_eligible = candidate["eligible"]
        candidate_status = str(candidate["status"])
        if (
            candidate_percentile in candidate_percentiles
            or candidate_value in candidate_q
            or candidate_percentile not in spec.threshold.percentiles
            or not math.isfinite(candidate_value)
            or not math.isfinite(candidate_low_share)
            or not math.isfinite(candidate_high_share)
            or not 0 <= candidate_low_share <= 1
            or not 0 <= candidate_high_share <= 1
            or not math.isclose(
                candidate_low_share + candidate_high_share,
                1.0,
                abs_tol=1e-12,
            )
            or not isinstance(candidate_eligible, bool)
            or candidate_status
            not in {"estimated", "fit_failed", "regime_share_below_minimum"}
        ):
            raise ValueError("threshold registry candidate grid is invalid")
        if candidate_status == "estimated":
            candidate_n = _require_json_integer(candidate["n"], "candidate n")
            candidate_ssr = float(candidate["ssr"])
            if (
                not candidate_eligible
                or candidate_n <= 0
                or not math.isfinite(candidate_ssr)
                or candidate_ssr < 0
            ):
                raise ValueError("estimated threshold candidate is invalid")
        elif candidate["n"] is not None or candidate["ssr"] is not None:
            raise ValueError("failed threshold candidate estimates must be null")
        candidate_percentiles.add(candidate_percentile)
        candidate_q.add(candidate_value)
        if (
            candidate_percentile == percentile
            and candidate_value == q
            and candidate_status == "estimated"
        ):
            selected_found = True
    if not selected_found:
        raise ValueError("threshold registry selected candidate is absent")


def _git_output(code_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=code_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"cannot bind analysis Git identity: {detail}")
    return completed.stdout.strip()


def current_analysis_git_commit(code_root: Path) -> str:
    """Return the reporting/audit implementation commit at execution time."""

    commit = _git_output(code_root.resolve(), "rev-parse", "HEAD")
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise ValueError("analysis Git commit must be a full lowercase hash")
    return commit


def load_evidence_policy(path: Path) -> tuple[dict[str, Any], str]:
    """Load and strictly validate the tracked evidence/inference authority."""

    try:
        raw = path.read_bytes()
        policy = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evidence policy {path}: {exc}") from exc
    if not isinstance(policy, dict):
        raise ValueError("evidence policy must be a JSON object")
    concentration = policy.get("concentration")
    conventional = policy.get("conventional_inference")
    wild = policy.get("wild_cluster_bootstrap")
    software = policy.get("software")
    threshold = policy.get("threshold_binding")
    if not all(isinstance(value, dict) for value in (
        concentration, conventional, wild, software, threshold
    )):
        raise ValueError("evidence policy authority sections are incomplete")
    if (
        policy.get("schema_version") != "1.0.0"
        or policy.get("policy_id") != "green_absorptive_debt_evidence_policy_v1"
        or concentration.get("status") != "unresolved_no_preregistered_cutoff"
        or concentration.get("unresolved_policy") != "exploratory"
        or concentration.get("aggregation") != "diagnostic_disclosure_only"
        or concentration.get("metrics")
        != {
            "exposure": ["hhi", "top1_share"],
            "rotemberg": ["hhi_absolute", "top1_absolute_share"],
        }
        or concentration.get("cutoffs") != {"hhi": None, "top1_share": None}
        or conventional.get("confidence_level") != 0.95
        or conventional.get("reference_distribution") != "cluster_t"
        or conventional.get("reference_df_rule") != "clusters_minus_one"
        or conventional.get("ssc_config")
        != "fixest:K.adj=TRUE;K.fixef=nonnested;K.exact=FALSE;G.adj=TRUE;G.df=min;t.df=min"
        or wild.get("minimum_clusters") != 20
        or wild.get("maximum_clusters") != 29
        or wild.get("seed") != 20260820
        or wild.get("draws") != 9999
        or wild.get("unbounded_bounds") is not None
        or threshold.get("registry_hash")
        != "606c77365627c222cef81e1351a5fe24b38128d08d6f082045ebab1e85292d37"
        or float(threshold.get("q", math.nan)) != 1.4777372886633888
        or threshold.get("percentile") != 47
        or threshold.get("input_authority_hash")
        != "9a79ba80640376369a6f6a3be90918d9b9798e2f6775f67544a288ec16e5032a"
    ):
        raise ValueError("evidence policy differs from the frozen authority")
    versions = software.get("packages")
    if (
        software.get("r_version") != "4.6.1"
        or software.get("python_version") != "3.12.13"
        or software.get("julia_version") != "1.12.7"
        or not isinstance(versions, dict)
        or not versions
    ):
        raise ValueError("evidence policy software authority is incomplete")
    return policy, hashlib.sha256(raw).hexdigest()


def _tracked_clean_evidence_policy(code_root: Path) -> tuple[dict[str, Any], str]:
    path = code_root / EVIDENCE_POLICY_RELATIVE_PATH
    try:
        _git_output(code_root, "ls-files", "--error-unmatch", "--", str(EVIDENCE_POLICY_RELATIVE_PATH))
    except ValueError as exc:
        raise ValueError("evidence policy must be tracked by Git") from exc
    if _git_output(code_root, "status", "--porcelain", "--", str(EVIDENCE_POLICY_RELATIVE_PATH)):
        raise ValueError("evidence policy is dirty")
    return load_evidence_policy(path)


def build_run_context(
    spec: AnalysisSpec, authority: InputAuthority, code_root: Path
) -> RunContext:
    """Bind one logical run to immutable inputs, code, and a clean R lock."""

    root = code_root.resolve()
    lock_path = root / "renv.lock"
    if not lock_path.is_file():
        raise ValueError("renv.lock is missing")
    try:
        _git_output(root, "ls-files", "--error-unmatch", "--", "renv.lock")
    except ValueError as exc:
        raise ValueError("renv.lock must be tracked by Git") from exc
    if _git_output(root, "status", "--porcelain", "--", "renv.lock"):
        raise ValueError("renv.lock is dirty")

    git_commit = _git_output(root, "rev-parse", "HEAD")
    if len(git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in git_commit
    ):
        raise ValueError("analysis Git commit must be a full lowercase hash")
    authority_hash = canonical_authority_hash(authority)
    lock_hash = sha256_file(lock_path)
    _, evidence_policy_sha256 = _tracked_clean_evidence_policy(root)
    run_material = json.dumps(
        [
            spec.spec_id,
            authority_hash,
            git_commit,
            lock_hash,
            evidence_policy_sha256,
        ],
        separators=(",", ":"),
    ).encode("utf-8")
    run_id = hashlib.sha256(run_material).hexdigest()[:16]
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return RunContext(
        run_id=run_id,
        spec_id=spec.spec_id,
        input_authority_hash=authority_hash,
        git_commit=git_commit,
        renv_lock_sha256=lock_hash,
        evidence_policy_sha256=evidence_policy_sha256,
        seed=spec.seed,
        created_at_utc=created_at,
    )


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def _existing_path_has_symlink(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def resolve_frozen_analysis_output(
    code_root: Path, output_root: Path, spec: AnalysisSpec | None = None
) -> Path:
    """Resolve only the frozen output root, rejecting symlink aliases."""

    code = code_root.resolve()
    frozen_spec = spec or load_analysis_spec(code / "config/analysis.yaml")
    declared = Path(frozen_spec.outputs.root)
    if declared.is_absolute() or ".." in declared.parts:
        raise ValueError("frozen outputs.root must be a project-relative path")
    requested_lexical = Path(os.path.abspath(os.fspath(output_root)))
    if _existing_path_has_symlink(requested_lexical):
        raise ValueError("analysis output root must not use a symbolic link")
    expected = (code / declared).resolve()
    requested = requested_lexical.resolve()
    if requested != expected:
        raise ValueError(
            f"analysis output root must equal frozen outputs.root: {expected}"
        )
    return expected


def resolve_authorized_analysis_output(
    code_root: Path,
    data_root: Path,
    output_root: Path,
    spec: AnalysisSpec | None = None,
    *,
    expected_git_commit: str | None = None,
    environment: dict[str, str] | os._Environ[str] | None = None,
) -> Path:
    """Resolve the formal root or the one live reproduction capability target."""

    code = code_root.resolve()
    frozen_spec = spec or load_analysis_spec(code / "config/analysis.yaml")
    try:
        return resolve_frozen_analysis_output(code, output_root, frozen_spec)
    except ValueError as formal_error:
        env = os.environ if environment is None else environment
        capability_fields = (
            "GREEN_DEBT_ANALYSIS_REPRO_BROKER_FD",
            "GREEN_DEBT_ANALYSIS_REPRO_MARKER",
            "GREEN_DEBT_ANALYSIS_REPRO_SCRATCH",
            "GREEN_DEBT_ANALYSIS_REPRO_EXECUTION_ID",
            "GREEN_DEBT_ANALYSIS_REPRO_STAGE",
        )
        if any(not env.get(field) for field in capability_fields):
            raise formal_error
        declared = Path(frozen_spec.outputs.root)
        if declared.is_absolute() or ".." in declared.parts:
            raise ValueError("frozen outputs.root must be project-relative") from formal_error
        formal = (code / declared).resolve()
        commit = expected_git_commit or current_analysis_git_commit(code)
        try:
            validate_analysis_reproduction_capability(
                data_root=data_root,
                formal_output_root=formal,
                scratch_root=output_root,
                expected_git_commit=commit,
                environment=env,
            )
        except ValueError as capability_error:
            raise ValueError(
                "analysis output is neither the frozen root nor a valid "
                "execution-bound reproduction capability"
            ) from capability_error
        return Path(output_root).resolve(strict=True)


def _authorized_reproduction_marker(
    *,
    code_root: Path,
    data_root: Path,
    output_root: Path,
    expected_git_commit: str,
    environment: dict[str, str] | os._Environ[str] | None,
) -> dict[str, Any] | None:
    code = code_root.resolve()
    spec = load_analysis_spec(code / "config/analysis.yaml")
    try:
        resolve_frozen_analysis_output(code, output_root, spec)
    except ValueError:
        declared = Path(spec.outputs.root)
        formal = (code / declared).resolve()
        return validate_analysis_reproduction_capability(
            data_root=data_root,
            formal_output_root=formal,
            scratch_root=output_root,
            expected_git_commit=expected_git_commit,
            environment=environment,
        )
    return None


def bind_analysis_reproduction_context(
    *,
    code_root: Path,
    data_root: Path,
    output_root: Path,
    context: RunContext,
    environment: dict[str, str] | os._Environ[str] | None = None,
) -> RunContext:
    """Reuse the formal provenance timestamp for one validated clean-room run."""

    marker = _authorized_reproduction_marker(
        code_root=code_root,
        data_root=data_root,
        output_root=output_root,
        expected_git_commit=context.git_commit,
        environment=environment,
    )
    if marker is None:
        return context
    if marker["formal_run_id"] != context.run_id:
        raise ValueError("analysis reproduction formal run id mismatch")
    return replace(context, created_at_utc=str(marker["formal_created_at_utc"]))


def analysis_reproduction_audited_at(
    *,
    code_root: Path,
    data_root: Path,
    output_root: Path,
    default: str,
    expected_git_commit: str,
    environment: dict[str, str] | os._Environ[str] | None = None,
) -> str:
    """Return the formal audit timestamp only for a valid scratch capability."""

    marker = _authorized_reproduction_marker(
        code_root=code_root,
        data_root=data_root,
        output_root=output_root,
        expected_git_commit=expected_git_commit,
        environment=environment,
    )
    return default if marker is None else str(marker["formal_audited_at_utc"])


def combined_project_usage_bytes(data_root: Path, output_root: Path) -> int:
    """Count data plus any external output exactly once."""

    data = data_root.resolve()
    output = output_root.resolve()
    if output.is_relative_to(data):
        return project_usage_bytes(data)
    if data.is_relative_to(output):
        return directory_usage_bytes(output)
    return project_usage_bytes(data) + directory_usage_bytes(output)


def analysis_preflight(
    code_root: Path, data_root: Path, output_root: Path
) -> AnalysisPreflight:
    """Validate the frozen analysis authority and budgets without writing files."""

    code = code_root.resolve()
    data = data_root.resolve()
    output = output_root.resolve()
    intermediate = (data / "05_中间数据").resolve()
    if _paths_overlap(output, intermediate):
        raise ValueError("output root must not overlap read-only 05_中间数据")

    spec = load_analysis_spec(code / "config/analysis.yaml")
    _, evidence_policy_sha256 = load_evidence_policy(
        code / EVIDENCE_POLICY_RELATIVE_PATH
    )
    output = resolve_authorized_analysis_output(
        code, data, output_root, spec
    )
    authority = validate_analysis_authority(code, data)
    project = load_project_config(code / "config/project.yaml")
    output_quota_gb = project.storage.output_quota_gb
    absolute_limit_gb = project.storage.absolute_limit_gb
    if output_quota_gb is None or output_quota_gb != spec.outputs.quota_gb:
        raise ValueError("analysis output quota is not consistently frozen")
    if absolute_limit_gb is None:
        raise ValueError("absolute project limit is not configured")

    output_bytes = directory_usage_bytes(output)
    if output_bytes >= output_quota_gb * GIB:
        raise RuntimeError(f"{output_quota_gb} GB analysis output quota reached")
    project_bytes = combined_project_usage_bytes(data, output)
    if project_bytes >= absolute_limit_gb * GIB:
        raise RuntimeError(f"{absolute_limit_gb} GB absolute project limit reached")

    return AnalysisPreflight(
        status="ready",
        table_id=authority.table_id,
        rows=authority.rows,
        authority_tables=len(authority.table_hashes),
        confirmatory_cells=len(spec.confirmatory_cells()),
        registered_cells=len(spec.registered_cells()),
        write_count=0,
        input_authority_hash=canonical_authority_hash(authority),
        evidence_policy_sha256=evidence_policy_sha256,
        output_bytes=output_bytes,
        project_bytes=project_bytes,
    )


def load_table_contract(path: Path) -> TableContract:
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid table contract {path}: {exc}") from exc
    period = payload.get("period")
    return TableContract(
        table_id=str(payload["table_id"]),
        schema_version=str(payload["schema_version"]),
        primary_key=tuple(str(value) for value in payload["primary_key"]),
        columns={str(key): str(value) for key, value in payload["columns"].items()},
        units={str(key): str(value) for key, value in payload.get("units", {}).items()},
        period=(int(period[0]), int(period[1])) if period else None,
        zero_semantics={
            str(key): str(value)
            for key, value in payload.get("zero_semantics", {}).items()
        },
        null_semantics={
            str(key): str(value)
            for key, value in payload.get("null_semantics", {}).items()
        },
        transformations=tuple(
            str(value) for value in payload.get("transformations", ())
        ),
    )


def _authority_tables(
    code_root: Path, data_root: Path
) -> tuple[tuple[str, Path, Path], ...]:
    return (
        (
            "model_panel",
            data_root / "05_中间数据/analysis/lp_panel.parquet",
            code_root / "03_代码/contracts/model_panel.json",
        ),
        (
            "iv_baseline_shares",
            data_root
            / "05_中间数据/measures/instruments/iv_baseline_shares.parquet",
            code_root / "03_代码/contracts/iv_baseline_shares.json",
        ),
        (
            "iv_partner_shocks",
            data_root
            / "05_中间数据/measures/instruments/iv_partner_shocks.parquet",
            code_root / "03_代码/contracts/iv_partner_shocks.json",
        ),
    )


def _require_contract_matches_manifest(
    table_id: str, contract: TableContract, manifest: object
) -> None:
    for field in (
        "table_id",
        "schema_version",
        "primary_key",
        "columns",
        "units",
        "period",
        "zero_semantics",
        "null_semantics",
        "transformations",
    ):
        if getattr(contract, field) != getattr(manifest, field):
            raise ValueError(f"contract-manifest mismatch for {table_id}: {field}")


def validate_analysis_authority(code_root: Path, data_root: Path) -> InputAuthority:
    """Verify live tables plus their sidecars, contracts, and frozen configs."""

    verified: list[tuple[object, Path, Path]] = []
    bound: dict[str, str] = {}
    for expected_id, table_path, contract_path in _authority_tables(
        code_root.resolve(), data_root.resolve()
    ):
        manifest_path = table_path.with_name(f"{table_path.name}.manifest.json")
        schema_path = table_path.with_name(f"{table_path.name}.schema.json")
        manifest = verify_manifest(manifest_path)
        if manifest.table_id != expected_id:
            raise ValueError(f"unexpected table authority: {manifest.table_id}")
        if sha256_file(table_path) != manifest.output_sha256:
            raise ValueError(f"live output hash mismatch for {expected_id}")
        if sha256_file(schema_path) != manifest.schema_sha256:
            raise ValueError(f"live schema hash mismatch for {expected_id}")
        contract = load_table_contract(contract_path)
        _require_contract_matches_manifest(expected_id, contract, manifest)
        verified.append((manifest, manifest_path, contract_path))
        bound[f"{expected_id}.output"] = manifest.output_sha256
        bound[f"{expected_id}.schema"] = manifest.schema_sha256
        bound[f"{expected_id}.manifest"] = sha256_file(manifest_path)
        bound[f"{expected_id}.contract"] = sha256_file(contract_path)

    model_manifest, model_manifest_path, model_contract_path = verified[0]
    if model_manifest.rows != FROZEN_MODEL_PANEL_ROWS:
        raise ValueError(
            "unexpected frozen model-panel rows: "
            f"{model_manifest.rows} != {FROZEN_MODEL_PANEL_ROWS}"
        )

    project_path = code_root / "config/project.yaml"
    outcome_map_path = code_root / "config/outcome_gad_map.yaml"
    analysis_path = code_root / "config/analysis.yaml"
    bound["config.project"] = sha256_file(project_path)
    bound["config.outcome_gad_map"] = sha256_file(outcome_map_path)
    bound["config.analysis"] = sha256_file(analysis_path)

    return InputAuthority(
        table_id=model_manifest.table_id,
        rows=model_manifest.rows,
        output_sha256=model_manifest.output_sha256,
        schema_sha256=model_manifest.schema_sha256,
        manifest_sha256=sha256_file(model_manifest_path),
        contract_sha256=sha256_file(model_contract_path),
        project_config_sha256=bound["config.project"],
        outcome_map_sha256=bound["config.outcome_gad_map"],
        analysis_config_sha256=bound["config.analysis"],
        table_hashes=tuple(
            (manifest.table_id, manifest.output_sha256)
            for manifest, _, _ in verified
        ),
        input_hashes=tuple(sorted(bound.items())),
    )


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _cell_key(value: Any) -> tuple[str, str, int, str, str]:
    if isinstance(value, dict):
        getter = value.__getitem__
    else:
        getter = lambda field: getattr(value, field)
    return (
        str(getter("analysis_family")),
        str(getter("outcome_id")),
        int(getter("horizon")),
        str(getter("gad_version")),
        str(getter("sample_version")),
    )


def _require_context_identity(
    payload: dict[str, Any], context: RunContext, label: str
) -> None:
    for field, expected in asdict(context).items():
        if field not in payload:
            raise ValueError(f"{label} is missing {field}")
        if payload[field] != expected:
            raise ValueError(f"{label} {field} does not match the run context")


def _read_staging_csv(path: Path, contract: TableContract) -> pl.DataFrame:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"receipt-bound staging file is unavailable: {path.name}")
    try:
        string_overrides = {
            name: pl.String
            for name, dtype in contract.columns.items()
            if dtype == "String"
        }
        frame = pl.read_csv(
            path,
            null_values="",
            infer_schema_length=10_000,
            schema_overrides=string_overrides,
        )
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise ValueError(f"invalid staging CSV {path.name}: {exc}") from exc
    expected = tuple(contract.columns)
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"extra {', '.join(extra)}")
        raise ValueError(
            f"staging CSV {path.name} has invalid columns: {'; '.join(details)}"
        )
    expressions = [
        pl.col(name).cast(getattr(pl, dtype), strict=True).alias(name)
        for name, dtype in contract.columns.items()
    ]
    try:
        return frame.select(*expressions)
    except pl.exceptions.PolarsError as exc:
        raise ValueError(
            f"staging CSV {path.name} violates its contract: {exc}"
        ) from exc


def _require_no_duplicate_keys(
    frame: pl.DataFrame, contract: TableContract
) -> None:
    duplicate = (
        frame.group_by(*contract.primary_key)
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate:
        raise ValueError(
            f"duplicate primary key groups in {contract.table_id}: {duplicate}"
        )


def _require_provenance(frame: pl.DataFrame, context: RunContext) -> None:
    expected = asdict(context)
    expected.pop("seed")
    for field in MODEL_PROVENANCE_FIELDS:
        if field not in frame.columns:
            raise ValueError(f"model staging output is missing {field}")
        values = frame.get_column(field).unique().to_list()
        if values != [expected[field]]:
            raise ValueError(f"model staging {field} does not match run context")


def _require_result_metadata(
    frame: pl.DataFrame, context: RunContext, inference_kind: str
) -> None:
    policy, policy_sha256 = load_evidence_policy(
        PROJECT_ROOT / EVIDENCE_POLICY_RELATIVE_PATH
    )
    software = policy["software"]
    expected_strings = {
        "evidence_policy_sha256": policy_sha256,
        "r_version": software["r_version"],
        "python_version": software["python_version"],
        "julia_version": software["julia_version"],
        "package_versions_json": json.dumps(
            software["packages"], sort_keys=True, separators=(",", ":")
        ),
    }
    for field, expected in expected_strings.items():
        if frame.get_column(field).unique().to_list() != [expected]:
            raise ValueError(f"result metadata mismatch: {field}")
    if frame.get_column("random_seed").unique().to_list() != [context.seed]:
        raise ValueError("result metadata mismatch: random_seed")
    invalid_years = frame.filter(
        (pl.col("year_min") < 2000)
        | (pl.col("year_max") > 2022)
        | (pl.col("year_min") > pl.col("year_max"))
    )
    if invalid_years.height:
        raise ValueError("result metadata year range is invalid")
    if inference_kind == "cluster_t":
        conventional = policy["conventional_inference"]
        invalid = frame.filter(
            (pl.col("ssc_config") != conventional["ssc_config"])
            | (pl.col("reference_distribution") != "cluster_t")
            | (pl.col("reference_df") != pl.col("clusters").cast(pl.Float64) - 1.0)
        )
        if invalid.height:
            raise ValueError("result metadata cluster t/SSC authority mismatch")
    elif inference_kind == "cr2_htz_f":
        invalid = frame.filter(
            (pl.col("ssc_config") != "clubSandwich:CR2;Wald_test=HTZ")
            | (pl.col("reference_distribution") != "cr2_htz_f")
            | pl.col("reference_df").is_null()
            | (pl.col("reference_df") <= 0)
        )
        if invalid.height:
            raise ValueError("result metadata CR2/HTZ authority mismatch")
    elif inference_kind == "not_used":
        invalid = frame.filter(
            (
                pl.col("ssc_config")
                != "shock_cluster_sandwich:exporter_hs6;reference=not_used"
            )
            | (pl.col("reference_distribution") != "not_used")
            | pl.col("reference_df").is_null()
            | (pl.col("reference_df") <= 0)
        )
        if invalid.height:
            raise ValueError("result metadata shock reference authority mismatch")
    else:
        raise ValueError(f"unknown result inference metadata kind: {inference_kind}")


def _gate_inference_status(cell: dict[str, Any]) -> str:
    rank = int(cell["rank"])
    clusters = int(cell["clusters"])
    status = str(cell["gate_status"])
    if rank < 2:
        expected = "fail_rank_deficient"
        inference = "fail_rank_deficient"
    elif clusters < 20:
        expected = "exploratory_lt20_clusters"
        inference = expected
    elif clusters < 30:
        expected = "wild_bootstrap_required"
        inference = expected
    else:
        expected = "ready_cluster_robust"
        inference = "cluster_robust"
    if status != expected:
        raise ValueError(
            f"stage A gate status is inconsistent with rank/clusters: {status}"
        )
    return inference


def _require_confidence_arithmetic(frame: pl.DataFrame) -> None:
    estimate = frame.get_column("estimate").to_numpy()
    standard_error = frame.get_column("std_error").to_numpy()
    conf_low = frame.get_column("conf_low").to_numpy()
    conf_high = frame.get_column("conf_high").to_numpy()
    p_value = frame.get_column("p_value").to_numpy()
    reference_df = frame.get_column("reference_df").to_numpy()
    distribution = frame.get_column("reference_distribution").to_list()
    if np.any(~np.isfinite(standard_error)) or np.any(standard_error <= 0):
        raise ValueError("model standard errors must be finite and positive")
    if any(value != "cluster_t" for value in distribution):
        raise ValueError("conventional reference distribution must be cluster_t")
    if np.any(~np.isfinite(reference_df)) or np.any(reference_df <= 0):
        raise ValueError("conventional cluster t reference df is invalid")
    critical = stats.t.ppf(0.975, reference_df)
    if not np.allclose(
        conf_low, estimate - critical * standard_error, atol=1e-8, rtol=1e-8
    ):
        raise ValueError("conventional 95% confidence arithmetic mismatch")
    if not np.allclose(
        conf_high, estimate + critical * standard_error, atol=1e-8, rtol=1e-8
    ):
        raise ValueError("conventional 95% confidence arithmetic mismatch")
    expected_p = 2 * stats.t.sf(np.abs(estimate / standard_error), reference_df)
    if not np.allclose(p_value, expected_p, atol=1e-8, rtol=1e-8):
        raise ValueError("conventional p-value arithmetic mismatch")


def _require_wild_fields(
    row: dict[str, Any], expected_status: str, spec: AnalysisSpec
) -> None:
    fields = (
        "wild_conf_low",
        "wild_conf_high",
        "wild_p_value",
        "wild_draws",
        "wild_seed",
    )
    values = tuple(row[field] for field in fields)
    row_status = str(row["inference_status"])
    if expected_status == "wild_bootstrap_required":
        if any(row[field] is None for field in fields[2:]):
            raise ValueError(
                "wild-bootstrap fields are required for 20-29 clusters"
            )
        if int(row["wild_draws"]) != spec.inference.wild_bootstrap_draws:
            raise ValueError("wild-bootstrap draw count mismatch")
        if int(row["wild_seed"]) != spec.seed:
            raise ValueError("wild-bootstrap seed mismatch")
        if not 0 <= float(row["wild_p_value"]) <= 1:
            raise ValueError("wild-bootstrap p-value must be a probability")
        if row_status == "wild_bootstrap_required":
            if any(row[field] is None for field in fields[:2]):
                raise ValueError(
                    "finite wild-bootstrap interval bounds are required"
                )
            if float(row["wild_conf_low"]) > float(row["wild_conf_high"]):
                raise ValueError(
                    "wild-bootstrap confidence interval is reversed"
                )
        elif row_status == "wild_bootstrap_unbounded":
            if any(row[field] is not None for field in fields[:2]):
                raise ValueError(
                    "unbounded wild-bootstrap interval bounds must be null"
                )
        else:
            raise ValueError(
                "model inference status does not match the stage A gate"
            )
    else:
        if row_status != expected_status:
            raise ValueError(
                "model inference status does not match the stage A gate"
            )
        if any(value is not None for value in values):
            raise ValueError(
                "wild-bootstrap fields must be null outside 20-29 clusters"
            )


def _validate_estimate_table(
    frame: pl.DataFrame,
    estimator: str,
    expected_keys: set[tuple[str, str, int, str, str]],
    gates: dict[tuple[str, str, int, str, str], dict[str, Any]],
    spec: AnalysisSpec,
) -> None:
    _require_confidence_arithmetic(frame)
    if frame.get_column("estimator").unique().to_list() != [estimator]:
        raise ValueError(f"{estimator} staging table contains another estimator")
    grouped: dict[
        tuple[str, str, int, str, str], list[dict[str, Any]]
    ] = {}
    for row in frame.to_dicts():
        key = _cell_key(row)
        grouped.setdefault(key, []).append(row)
    if set(grouped) != expected_keys:
        raise ValueError(
            f"{estimator} does not represent the exact registered cells"
        )
    for key, rows in grouped.items():
        if (
            {str(row["term"]) for row in rows} != set(MODEL_TERMS)
            or len(rows) != 2
        ):
            raise ValueError(
                f"{estimator} must contain both registered terms once"
            )
        gate = gates[key]
        expected_inference = _gate_inference_status(gate)
        for row in rows:
            if int(row["horizon"]) == 0:
                raise ValueError(
                    "horizon zero cannot appear in model estimates"
                )
            if (
                int(row["n"]) != int(gate["n"])
                or int(row["economies"]) != int(gate["economies"])
                or int(row["clusters"]) != int(gate["clusters"])
            ):
                raise ValueError(
                    "model sample counts do not match the stage A gate"
                )
            if str(row["first_stage_status"]) != str(
                gate["first_stage_status"]
            ):
                raise ValueError(
                    "model first-stage status does not match the stage A gate"
                )
            _require_wild_fields(row, expected_inference, spec)


def _covariance_matrices(
    frame: pl.DataFrame,
    expected_pairs: set[
        tuple[str, tuple[str, str, int, str, str]]
    ],
) -> dict[
    tuple[str, tuple[str, str, int, str, str]], np.ndarray
]:
    grouped: dict[
        tuple[str, tuple[str, str, int, str, str]],
        list[dict[str, Any]],
    ] = {}
    for row in frame.to_dicts():
        pair = (str(row["estimator"]), _cell_key(row))
        grouped.setdefault(pair, []).append(row)
    if set(grouped) != expected_pairs:
        raise ValueError(
            "covariance table does not represent every estimated cell"
        )
    matrices: dict[
        tuple[str, tuple[str, str, int, str, str]], np.ndarray
    ] = {}
    term_index = {term: index for index, term in enumerate(MODEL_TERMS)}
    for pair, rows in grouped.items():
        ordered_pairs = {
            (row["term_i"], row["term_j"]) for row in rows
        }
        expected_ordered = {
            (left, right)
            for left in MODEL_TERMS
            for right in MODEL_TERMS
        }
        if len(rows) != 4 or ordered_pairs != expected_ordered:
            raise ValueError(
                "covariance requires every ordered term pair exactly once"
            )
        matrix = np.empty((2, 2), dtype=float)
        for row in rows:
            matrix[
                term_index[str(row["term_i"])],
                term_index[str(row["term_j"])],
            ] = float(row["covariance"])
        if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0):
            raise ValueError(
                "model covariance must be symmetric within 1e-10"
            )
        eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
        if float(eigenvalues.min()) < -1e-10:
            raise ValueError(
                "model covariance must be positive semidefinite"
            )
        matrices[pair] = matrix
    return matrices


def _validate_covariance_diagonals(
    estimate_frames: dict[str, pl.DataFrame],
    matrices: dict[
        tuple[str, tuple[str, str, int, str, str]], np.ndarray
    ],
) -> None:
    term_index = {term: index for index, term in enumerate(MODEL_TERMS)}
    for estimator, frame in estimate_frames.items():
        for row in frame.to_dicts():
            key = (estimator, _cell_key(row))
            index = term_index[str(row["term"])]
            variance = matrices[key][index, index]
            if not math.isclose(
                variance,
                float(row["std_error"]) ** 2,
                abs_tol=1e-10,
                rel_tol=1e-8,
            ):
                raise ValueError(
                    "model covariance diagonal does not match standard error"
                )


def _validate_marginal_effects(
    frame: pl.DataFrame,
    estimate_frames: dict[str, pl.DataFrame],
    matrices: dict[
        tuple[str, tuple[str, str, int, str, str]], np.ndarray
    ],
    expected_pairs: set[
        tuple[str, tuple[str, str, int, str, str]]
    ],
    gates: dict[tuple[str, str, int, str, str], dict[str, Any]],
    spec: AnalysisSpec,
) -> None:
    _require_confidence_arithmetic(frame)
    coefficients: dict[
        tuple[str, tuple[str, str, int, str, str]], dict[str, float]
    ] = {}
    for estimator, estimates in estimate_frames.items():
        for row in estimates.to_dicts():
            pair = (estimator, _cell_key(row))
            coefficients.setdefault(pair, {})[
                str(row["term"])
            ] = float(row["estimate"])
    grouped: dict[
        tuple[str, tuple[str, str, int, str, str]],
        list[dict[str, Any]],
    ] = {}
    for row in frame.to_dicts():
        pair = (str(row["estimator"]), _cell_key(row))
        grouped.setdefault(pair, []).append(row)
    if set(grouped) != expected_pairs:
        raise ValueError(
            "marginal effects do not represent every estimated cell"
        )
    expected_quantiles = tuple(spec.inference.marginal_gad_quantiles)
    for pair, rows in grouped.items():
        if len(rows) != len(expected_quantiles):
            raise ValueError(
                "marginal effects require every frozen GAD quantile"
            )
        observed_quantiles = sorted(
            float(row["gad_quantile"]) for row in rows
        )
        if not np.allclose(
            observed_quantiles,
            expected_quantiles,
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError(
                "marginal-effect GAD quantiles are not frozen"
            )
        _, key = pair
        gate = gates[key]
        expected_inference = _gate_inference_status(gate)
        beta = coefficients[pair]["gimc_a"]
        theta = coefficients[pair]["gimc_gad_a"]
        covariance = matrices[pair]
        for row in rows:
            q = float(row["gad_value"])
            expected_estimate = beta + q * theta
            expected_variance = (
                covariance[0, 0]
                + q * q * covariance[1, 1]
                + 2 * q * covariance[0, 1]
            )
            if expected_variance < -1e-10:
                raise ValueError(
                    "marginal-effect variance is negative"
                )
            expected_error = math.sqrt(max(0.0, expected_variance))
            if not math.isclose(
                float(row["estimate"]),
                expected_estimate,
                abs_tol=1e-8,
                rel_tol=1e-8,
            ):
                raise ValueError(
                    "marginal-effect estimate does not match model coefficients"
                )
            if not math.isclose(
                float(row["std_error"]),
                expected_error,
                abs_tol=1e-8,
                rel_tol=1e-8,
            ):
                raise ValueError(
                    "marginal-effect standard error omits model covariance"
                )
            if (
                int(row["n"]) != int(gate["n"])
                or int(row["economies"]) != int(gate["economies"])
                or int(row["clusters"]) != int(gate["clusters"])
            ):
                raise ValueError(
                    "marginal sample counts do not match the stage A gate"
                )
            if str(row["first_stage_status"]) != str(
                gate["first_stage_status"]
            ):
                raise ValueError(
                    "marginal first-stage status does not match the stage A gate"
                )
            _require_wild_fields(row, expected_inference, spec)


def _require_reference_row_arithmetic(
    row: dict[str, Any], label: str, *, prefix: str = ""
) -> None:
    fields = tuple(
        f"{prefix}{name}"
        for name in ("estimate", "std_error", "conf_low", "conf_high", "p_value")
    )
    if any(row[field] is None for field in fields):
        raise ValueError(f"{label} has incomplete conventional inference")
    estimate, standard_error, conf_low, conf_high, p_value = (
        float(row[field]) for field in fields
    )
    if (
        not all(
            math.isfinite(value)
            for value in (estimate, standard_error, conf_low, conf_high, p_value)
        )
        or standard_error <= 0
        or not 0 <= p_value <= 1
    ):
        raise ValueError(f"{label} has invalid conventional inference")
    if row.get("reference_distribution") != "cluster_t":
        raise ValueError(f"{label} reference distribution mismatch")
    reference_df = float(row.get("reference_df", math.nan))
    if not math.isfinite(reference_df) or reference_df <= 0:
        raise ValueError(f"{label} reference df mismatch")
    critical = float(stats.t.ppf(0.975, reference_df))
    expected_p = float(stats.t.sf(abs(estimate / standard_error), reference_df) * 2)
    if (
        not math.isclose(
            conf_low,
            estimate - critical * standard_error,
            abs_tol=1e-8,
            rel_tol=1e-8,
        )
        or not math.isclose(
            conf_high,
            estimate + critical * standard_error,
            abs_tol=1e-8,
            rel_tol=1e-8,
        )
        or not math.isclose(p_value, expected_p, abs_tol=1e-8, rel_tol=1e-8)
    ):
        raise ValueError(f"{label} conventional confidence arithmetic mismatch")


def _same_nullable(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isclose(float(left), float(right), abs_tol=1e-12, rel_tol=1e-12)
    return left == right


def _validate_threshold_wild(
    row: dict[str, Any], *, prefix: str, required: bool, label: str
) -> None:
    status_field = f"{prefix}wild_inference_status"
    value_fields = tuple(
        f"{prefix}wild_{name}" for name in ("estimate", "std_error", "p_value")
    )
    bound_fields = tuple(
        f"{prefix}wild_{name}" for name in ("conf_low", "conf_high")
    )
    draws_field = f"{prefix}wild_draws"
    seed_field = f"{prefix}wild_seed"
    status = str(row[status_field])
    if not required:
        if status != "not_required" or any(
            row[field] is not None
            for field in (*value_fields, *bound_fields, draws_field, seed_field)
        ):
            raise ValueError(f"{label} wild bootstrap must be absent outside 20-29")
        return
    if status not in {"wild_bootstrap_bounded", "wild_bootstrap_unbounded"}:
        raise ValueError(f"20-29 clusters require {label} wild bootstrap")
    if any(row[field] is None for field in (*value_fields, draws_field, seed_field)):
        raise ValueError(f"20-29 clusters require {label} wild bootstrap fields")
    estimate, standard_error, p_value = (
        float(row[field]) for field in value_fields
    )
    if (
        not all(math.isfinite(value) for value in (estimate, standard_error, p_value))
        or standard_error <= 0
        or not 0 <= p_value <= 1
        or int(row[draws_field]) != 9999
        or int(row[seed_field]) != 20260820
    ):
        raise ValueError(f"20-29 clusters have invalid {label} wild bootstrap")
    bounds = tuple(row[field] for field in bound_fields)
    if status == "wild_bootstrap_unbounded":
        if any(value is not None for value in bounds):
            raise ValueError(f"unbounded {label} wild bootstrap requires null bounds")
    elif any(value is None for value in bounds) or float(bounds[0]) > float(bounds[1]):
        raise ValueError(f"bounded {label} wild bootstrap requires finite ordered bounds")


def _required_gate_mapping(
    gate: dict[str, Any],
    expected: set[tuple[str, str, int, str, str]],
) -> dict[tuple[str, str, int, str, str], dict[str, Any]]:
    rows = gate.get("cells")
    if not isinstance(rows, list):
        raise ValueError("analysis gate cells must be a list")
    result: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("analysis gate contains a non-object cell")
        key = _cell_key(row)
        if key not in expected:
            continue
        if key in result:
            raise ValueError("analysis gate contains a duplicate audit cell")
        result[key] = row
    if set(result) != expected:
        raise ValueError("analysis gate lacks an exact threshold/audit cell")
    return result


def _validate_threshold_estimates(
    frame: pl.DataFrame,
    expected: set[tuple[str, str, int, str, str]],
    gates: dict[tuple[str, str, int, str, str], dict[str, Any]],
    registry: dict[str, Any],
) -> None:
    grouped: dict[
        tuple[str, str, int, str, str], list[dict[str, Any]]
    ] = {}
    for row in frame.to_dicts():
        grouped.setdefault(_cell_key(row), []).append(row)
    if set(grouped) != expected:
        raise ValueError("threshold estimates do not cover exactly 39 cells")
    for key, rows in grouped.items():
        if len(rows) != 2 or {str(row["regime"]) for row in rows} != {
            "low",
            "high",
        }:
            raise ValueError("threshold estimates require low and high regimes")
        gate = gates[key]
        by_regime = {str(row["regime"]): row for row in rows}
        low = by_regime["low"]
        high = by_regime["high"]
        paired_fields = (
            "low_high_covariance",
            "difference_estimate", "difference_std_error",
            "difference_conf_low", "difference_conf_high", "difference_p_value",
            "difference_wild_estimate", "difference_wild_std_error",
            "difference_wild_conf_low", "difference_wild_conf_high",
            "difference_wild_p_value", "difference_wild_inference_status",
            "difference_wild_draws", "difference_wild_seed",
        )
        if any(not _same_nullable(low[field], high[field]) for field in paired_fields):
            raise ValueError("paired high-minus-low contrast fields are inconsistent")
        if sum(int(row["regime_n"]) for row in rows) != int(gate["n"]):
            raise ValueError("threshold regime counts do not reconstruct model n")
        if not math.isclose(
            sum(float(row["regime_share"]) for row in rows),
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError("threshold regime shares do not sum to one")
        expected_inference = _gate_inference_status(gate)
        for row in rows:
            if (
                row["estimator"] != "threshold_iv"
                or row["analysis_family"] != "confirmatory"
                or row["selection_outcome"] != registry["selection_outcome"]
                or row["registry_hash"] != registry["registry_hash"]
                or row["registry_sample_hash"] != registry["sample_hash"]
                or not math.isclose(
                    float(row["q"]), float(registry["q"]), abs_tol=0, rel_tol=0
                )
            ):
                raise ValueError("threshold result differs from frozen registry")
            if (
                int(row["n"]) != int(gate["n"])
                or int(row["economies"]) != int(gate["economies"])
                or int(row["clusters"]) != int(gate["clusters"])
            ):
                raise ValueError("threshold sample counts differ from stage A")
            if row["inference_status"] == "fit_failed":
                raise ValueError(
                    "all 39 threshold cells require validated low/high estimates "
                    "and high-minus-low contrasts"
                )
            else:
                if row["inference_status"] != expected_inference:
                    raise ValueError("threshold cluster rule differs from stage A")
                if not math.isclose(
                    float(row["reference_df"]),
                    float(int(gate["clusters"]) - 1),
                    abs_tol=0,
                    rel_tol=0,
                ):
                    raise ValueError("threshold cluster t reference df mismatch")
                _require_reference_row_arithmetic(row, "threshold estimate")
                _validate_threshold_wild(
                    row,
                    prefix="",
                    required=expected_inference == "wild_bootstrap_required",
                    label=f"{row['regime']} regime",
                )
                if (
                    expected_inference == "wild_bootstrap_required"
                    and not math.isclose(
                        float(row["wild_estimate"]),
                        float(row["estimate"]),
                        abs_tol=1e-8,
                        rel_tol=1e-8,
                    )
                ):
                    raise ValueError(
                        "threshold wild-bootstrap point estimate differs from "
                        "the same-sample conventional estimate"
                    )
        if low["inference_status"] != "fit_failed":
            if not math.isclose(
                float(low["difference_estimate"]),
                float(high["estimate"]) - float(low["estimate"]),
                abs_tol=1e-10,
                rel_tol=1e-10,
            ):
                raise ValueError("threshold high-minus-low contrast estimate mismatch")
            expected_variance = (
                float(high["std_error"]) ** 2
                + float(low["std_error"]) ** 2
                - 2 * float(low["low_high_covariance"])
            )
            if expected_variance <= 0 or not math.isclose(
                float(low["difference_std_error"]) ** 2,
                expected_variance,
                abs_tol=1e-10,
                rel_tol=1e-8,
            ):
                raise ValueError("threshold contrast variance mismatch")
            _require_reference_row_arithmetic(
                low, "threshold contrast", prefix="difference_"
            )
            _validate_threshold_wild(
                low,
                prefix="difference_",
                required=expected_inference == "wild_bootstrap_required",
                label="difference",
            )
            if (
                expected_inference == "wild_bootstrap_required"
                and not math.isclose(
                    float(low["difference_wild_estimate"]),
                    float(low["difference_estimate"]),
                    abs_tol=1e-8,
                    rel_tol=1e-8,
                )
            ):
                raise ValueError(
                    "threshold contrast wild-bootstrap reparameterization "
                    "changed the same-sample point estimate"
                )


def _validate_weak_iv_sets(
    frame: pl.DataFrame,
    expected: set[tuple[str, str, int, str, str]],
    gates: dict[tuple[str, str, int, str, str], dict[str, Any]],
) -> None:
    rows = {_cell_key(row): row for row in frame.to_dicts()}
    if len(rows) != frame.height or set(rows) != expected:
        raise ValueError("weak-IV sets do not cover exactly 45 cells")
    allowed = {"bounded", "unbounded", "disjoint", "empty", "unavailable"}
    for key, row in rows.items():
        gate = gates[key]
        status = str(row["status"])
        if row["estimator"] != "lp_iv_ar" or status not in allowed:
            raise ValueError("weak-IV set has an invalid estimator or status")
        if (
            int(row["n"]) != int(gate["n"])
            or int(row["economies"]) != int(gate["economies"])
            or int(row["clusters"]) != int(gate["clusters"])
        ):
            raise ValueError("weak-IV sample counts differ from stage A")
        expansions = int(row["expansions"])
        accepted = int(row["accepted_points"])
        if not 0 <= expansions <= 4:
            raise ValueError("weak-IV expansion count is outside the frozen rule")
        _require_hex_hash(row["accepted_hash"], "accepted-grid hash")
        bounds = tuple(
            row[field]
            for field in ("beta_low", "beta_high", "theta_low", "theta_high")
        )
        if status in {"bounded", "disjoint"}:
            if accepted <= 0 or any(value is None for value in bounds):
                raise ValueError("bounded/disjoint AR sets require finite bounds")
            numeric = tuple(float(value) for value in bounds)
            if (
                any(not math.isfinite(value) for value in numeric)
                or numeric[0] > numeric[1]
                or numeric[2] > numeric[3]
            ):
                raise ValueError("weak-IV projected bounds are invalid")
        elif status == "unbounded":
            if accepted <= 0 or all(value is not None for value in bounds):
                raise ValueError(
                    "unbounded AR sets require accepted points and a null side"
                )
            numeric = tuple(
                None if value is None else float(value) for value in bounds
            )
            if any(
                value is not None and not math.isfinite(value)
                for value in numeric
            ):
                raise ValueError("weak-IV projected bounds are invalid")
            if (
                numeric[0] is not None
                and numeric[1] is not None
                and numeric[0] > numeric[1]
            ) or (
                numeric[2] is not None
                and numeric[3] is not None
                and numeric[2] > numeric[3]
            ):
                raise ValueError("weak-IV projected bounds are reversed")
        elif any(value is not None for value in bounds):
            raise ValueError("empty or unavailable AR bounds must be null")
        if status == "empty" and accepted != 0:
            raise ValueError("empty AR set must have zero accepted points")
        expected_inference = "unavailable" if status == "unavailable" else "available"
        if row["inference_status"] != expected_inference:
            raise ValueError("weak-IV inference status is inconsistent")


def _validate_shift_share_summaries(
    frame: pl.DataFrame,
    expected: set[tuple[str, str, int, str, str]],
    gates: dict[tuple[str, str, int, str, str], dict[str, Any]],
) -> dict[tuple[str, str, int, str, str], dict[str, Any]]:
    rows = {_cell_key(row): row for row in frame.to_dicts()}
    if len(rows) != frame.height or set(rows) != expected:
        raise ValueError("shift-share summaries do not cover exactly 45 cells")
    probability_fields = (
        "hhi_absolute",
        "top1_absolute_share",
        "top5_absolute_share",
        "negative_weight_share",
    )
    weight_fields = (
        "signed_weight_sum",
        "absolute_weight_sum",
        *probability_fields,
    )
    standard_error_fields = (
        "shock_std_error_gimc",
        "shock_std_error_interaction",
    )
    for key, row in rows.items():
        gate = gates[key]
        if row["estimator"] != "lp_iv_shift_share":
            raise ValueError("shift-share summary estimator mismatch")
        if (
            int(row["n"]) != int(gate["n"])
            or int(row["economies"]) != int(gate["economies"])
        ):
            raise ValueError("shift-share sample counts differ from stage A")
        observations = int(row["shock_observations"])
        clusters = int(row["shock_clusters"])
        if observations <= 0 or clusters <= 0 or clusters > observations:
            raise ValueError("shift-share shock observation/cluster counts are invalid")
        for field in ("z_reconstruction_error", "z_gad_reconstruction_error"):
            value = float(row[field])
            if not math.isfinite(value) or value < 0 or value > 1e-12:
                raise ValueError("shift-share instrument reconstruction exceeds 1e-12")
        status = str(row["shock_inference_status"])
        cross_moment_rank = int(row["cross_moment_rank"])
        if cross_moment_rank not in {0, 1, 2}:
            raise ValueError("shift-share cross-moment rank is invalid")
        if status == "available":
            if any(
                row[field] is None
                for field in (*weight_fields, *standard_error_fields)
            ):
                raise ValueError("available shock inference has null diagnostics")
            if (
                cross_moment_rank != 2
                or clusters < 2
                or not math.isfinite(float(row["shock_std_error_gimc"]))
                or not math.isfinite(
                    float(row["shock_std_error_interaction"])
                )
                or float(row["shock_std_error_gimc"]) < 0
                or float(row["shock_std_error_interaction"]) < 0
            ):
                raise ValueError("shift-share available inference is inconsistent")
        elif status == "unavailable":
            if any(row[field] is not None for field in standard_error_fields):
                raise ValueError("unavailable shock standard errors must be null")
            weight_presence = tuple(
                row[field] is not None for field in weight_fields
            )
            if any(weight_presence) != all(weight_presence):
                raise ValueError(
                    "unavailable shock weight diagnostics must be all present or null"
                )
            if all(weight_presence) and cross_moment_rank != 2:
                raise ValueError("rank-deficient shock audit cannot report weights")
        else:
            raise ValueError("shift-share shock inference status is invalid")
        if row["absolute_weight_sum"] is not None:
            if any(
                not math.isfinite(float(row[field])) for field in weight_fields
            ):
                raise ValueError("shift-share weight diagnostics must be finite")
            if not math.isclose(
                float(row["signed_weight_sum"]), 1.0, abs_tol=1e-10
            ):
                raise ValueError("generalized Rotemberg weights do not sum to one")
            if float(row["absolute_weight_sum"]) < 1.0 - 1e-10:
                raise ValueError("absolute generalized weights cannot sum below one")
            if any(
                not 0 <= float(row[field]) <= 1.0 + 1e-12
                for field in probability_fields
            ):
                raise ValueError("shift-share concentration share is invalid")
            if (
                float(row["top1_absolute_share"])
                > float(row["top5_absolute_share"]) + 1e-12
            ):
                raise ValueError("shift-share concentration ordering is invalid")
    return rows


def _validate_shift_share_weights(
    frame: pl.DataFrame,
    summaries: dict[tuple[str, str, int, str, str], dict[str, Any]],
) -> None:
    grouped: dict[
        tuple[str, str, int, str, str], list[dict[str, Any]]
    ] = {}
    for row in frame.to_dicts():
        grouped.setdefault(_cell_key(row), []).append(row)
    weighted = {
        key
        for key, row in summaries.items()
        if row["absolute_weight_sum"] is not None
    }
    if set(grouped) != weighted:
        raise ValueError("top shift-share weights do not cover weighted cells")
    for rows in grouped.values():
        if not 1 <= len(rows) <= 100:
            raise ValueError("shift-share top-weight count must be 1 through 100")
        ordered = sorted(rows, key=lambda row: int(row["absolute_rank"]))
        if [int(row["absolute_rank"]) for row in ordered] != list(
            range(1, len(rows) + 1)
        ):
            raise ValueError("shift-share absolute ranks are not contiguous")
        if len({str(row["shock_id"]) for row in rows}) != len(rows):
            raise ValueError("shift-share top weights contain duplicate shocks")
        for row in rows:
            expected_cluster = f"{row['exporter']}|{row['hs6']}"
            expected_shock = f"{expected_cluster}|{int(row['year'])}"
            signed = float(row["signed_weight"])
            absolute = float(row["absolute_weight"])
            if (
                row["shock_cluster_id"] != expected_cluster
                or row["shock_id"] != expected_shock
                or not math.isclose(absolute, abs(signed), abs_tol=1e-12)
            ):
                raise ValueError("shift-share shock identity or absolute weight mismatch")


def _ingest_threshold_iv_audit_bundle(
    staging: Path,
    output: Path,
    run_context: RunContext,
    spec: AnalysisSpec,
) -> tuple[Path, ...]:
    if (
        staging.name != "threshold-and-iv-audit"
        or staging.parent.name != run_context.run_id
        or staging.parent.parent.name != "_staging"
    ):
        raise ValueError("threshold/audit staging is not bound to the run receipt")
    receipt_path = staging / "receipt.json"
    receipt = _json_object(receipt_path, "threshold-and-IV-audit receipt")
    _require_context_identity(receipt, run_context, "threshold-and-IV-audit receipt")
    if receipt.get("kind") != "threshold-and-iv-audit" or set(
        receipt.get("completed_stages", ())
    ) != {"threshold", "weak-iv", "shift-share"}:
        raise ValueError("threshold/audit receipt does not contain all three stages")

    analysis_root = output.parent
    gate_path = analysis_root / "registries/analysis_gate_v1.json"
    registry_path = analysis_root / "registries/threshold_registry_v1.json"
    gate = _json_object(gate_path, "analysis gate")
    registry = _json_object(registry_path, "threshold registry")
    _require_context_identity(gate, run_context, "analysis gate")
    if gate.get("status") != "frozen":
        raise ValueError("analysis gate is not frozen")
    if receipt.get("gate_sha256") != sha256_file(gate_path):
        raise ValueError("threshold/audit gate receipt hash mismatch")
    verify_threshold_registry_payload(registry, spec, run_context)
    registry_binding = {
        "registry_sha256": sha256_file(registry_path),
        "registry_hash": registry["registry_hash"],
        "registry_sample_hash": registry["sample_hash"],
        "registry_q": registry["q"],
        "selection_outcome": registry["selection_outcome"],
    }
    for field, expected_value in registry_binding.items():
        observed = receipt.get(field)
        if field == "registry_q":
            matches = observed is not None and math.isclose(
                float(observed), float(expected_value), abs_tol=0, rel_tol=0
            )
        else:
            matches = observed == expected_value
        if not matches:
            raise ValueError(f"threshold/audit receipt {field} mismatch")

    confirmatory = {_cell_key(cell) for cell in spec.confirmatory_cells()}
    audit_cells = confirmatory | {
        _cell_key(cell) for cell in spec.vulnerability_cells()
    }
    gates = _required_gate_mapping(gate, audit_cells)
    contract_root = PROJECT_ROOT / "03_代码/contracts/analysis"
    contracts = {
        "threshold_estimates": load_table_contract(
            contract_root / "threshold_estimate.json"
        ),
        "weak_iv_sets": load_table_contract(contract_root / "weak_iv_set.json"),
        "shift_share_summary": load_table_contract(
            contract_root / "shift_share_summary.json"
        ),
        "shift_share_weights": load_table_contract(
            contract_root / "shift_share_weight.json"
        ),
    }
    files = receipt.get("files")
    if not isinstance(files, dict) or set(files) != set(contracts):
        raise ValueError("threshold/audit receipt must bind exactly four files")
    frames: dict[str, pl.DataFrame] = {}
    staging_paths: dict[str, Path] = {}
    for name, contract in contracts.items():
        metadata = files[name]
        expected_name = f"{name}.csv"
        if not isinstance(metadata, dict) or metadata.get("name") != expected_name:
            raise ValueError(f"threshold/audit receipt does not bind {expected_name}")
        path = staging / expected_name
        if metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"threshold/audit receipt hash mismatch for {expected_name}")
        frame = _read_staging_csv(path, contract)
        if metadata.get("rows") != frame.height:
            raise ValueError(f"threshold/audit row count mismatch for {expected_name}")
        _require_no_duplicate_keys(frame, contract)
        _require_provenance(frame, run_context)
        _require_result_metadata(
            frame,
            run_context,
            {
                "threshold_estimates": "cluster_t",
                "weak_iv_sets": "cr2_htz_f",
                "shift_share_summary": "not_used",
                "shift_share_weights": "not_used",
            }[name],
        )
        frames[name] = frame
        staging_paths[name] = path

    _validate_threshold_estimates(
        frames["threshold_estimates"], confirmatory, gates, registry
    )
    _validate_weak_iv_sets(frames["weak_iv_sets"], audit_cells, gates)
    summaries = _validate_shift_share_summaries(
        frames["shift_share_summary"], audit_cells, gates
    )
    _validate_shift_share_weights(frames["shift_share_weights"], summaries)

    destinations = {
        "threshold_estimates": output / "threshold_estimates.parquet",
        "weak_iv_sets": output / "weak_iv_sets.parquet",
        "shift_share_summary": analysis_root
        / "diagnostics/shift_share_summary.parquet",
        "shift_share_weights": analysis_root
        / "diagnostics/shift_share_weights.parquet",
    }
    build = BuildIdentity(
        command=(
            "python -m green_debt.cli analysis-ingest-models "
            "--kind threshold-and-iv-audit"
        ),
        code_commit=run_context.git_commit,
        created_at_utc=run_context.created_at_utc,
    )
    published: list[Path] = []
    for name, destination in destinations.items():
        contract = contracts[name]
        contract_path = contract_root / f"{contract.table_id}.json"
        write_authoritative_table(
            frames[name].sort(*contract.primary_key),
            contract,
            destination,
            (
                InputArtifact.from_path(staging_paths[name]),
                InputArtifact.from_path(receipt_path),
                InputArtifact.from_path(gate_path),
                InputArtifact.from_path(registry_path),
                InputArtifact.from_path(contract_path),
            ),
            build,
        )
        published.append(destination)
    shutil.rmtree(staging)
    return tuple(published)


def ingest_model_bundle(
    staging_dir: Path,
    output_dir: Path,
    run_context: RunContext,
    spec: AnalysisSpec,
) -> tuple[Path, ...]:
    """Validate one receipt-bound R model bundle and publish authoritative tables."""

    staging = staging_dir.resolve()
    output = output_dir.resolve()
    if staging_dir.is_symlink() or not staging.is_dir():
        raise ValueError("model staging directory must be a real directory")
    if staging.name == "threshold-and-iv-audit":
        return _ingest_threshold_iv_audit_bundle(
            staging, output, run_context, spec
        )
    if (
        staging.name != "lp"
        or staging.parent.name != run_context.run_id
        or staging.parent.parent.name != "_staging"
    ):
        raise ValueError(
            "LP staging directory is not bound to the run receipt"
        )
    receipt_path = staging / "receipt.json"
    receipt = _json_object(receipt_path, "LP staging receipt")
    _require_context_identity(
        receipt, run_context, "LP staging receipt"
    )
    if receipt.get("kind") != "lp":
        raise ValueError("LP staging receipt has the wrong kind")

    gate_path = output.parent / "registries" / "analysis_gate_v1.json"
    gate = _json_object(gate_path, "analysis gate")
    _require_context_identity(gate, run_context, "analysis gate")
    if gate.get("status") != "frozen":
        raise ValueError("analysis gate is not frozen")
    if receipt.get("gate_sha256") != sha256_file(gate_path):
        raise ValueError(
            "analysis gate hash does not match the LP receipt"
        )

    expected_cells = {
        _cell_key(cell)
        for cell in spec.registered_cells()
        if cell.analysis_family != "threshold_selection"
    }
    gate_rows = gate.get("cells")
    if not isinstance(gate_rows, list):
        raise ValueError("analysis gate cells must be a list")
    gates: dict[
        tuple[str, str, int, str, str], dict[str, Any]
    ] = {}
    for raw in gate_rows:
        if not isinstance(raw, dict):
            raise ValueError(
                "analysis gate contains a non-object cell"
            )
        key = _cell_key(raw)
        if key not in expected_cells:
            continue
        if key in gates:
            raise ValueError(
                "analysis gate contains duplicate continuous cells"
            )
        gates[key] = raw
    if set(gates) != expected_cells:
        raise ValueError(
            "analysis gate does not contain the exact 84 continuous cells"
        )

    failed_iv = {
        key
        for key, value in gates.items()
        if _gate_inference_status(value) == "fail_rank_deficient"
    }
    skipped = receipt.get("skipped_cells")
    if not isinstance(skipped, list):
        raise ValueError(
            "LP receipt skipped_cells must be a list"
        )
    skipped_iv: set[tuple[str, str, int, str, str]] = set()
    for item in skipped:
        if (
            not isinstance(item, dict)
            or item.get("estimator") != "lp_iv"
        ):
            raise ValueError(
                "LP receipt contains an invalid skipped cell"
            )
        if item.get("status") != "fail_rank_deficient":
            raise ValueError(
                "LP receipt skipped cell has an invalid status"
            )
        skipped_iv.add(_cell_key(item))
    if skipped_iv != failed_iv:
        raise ValueError(
            "LP-IV rank-deficient cells lack exact receipt gates"
        )

    contract_root = (
        PROJECT_ROOT / "03_代码/contracts/analysis"
    )
    estimate_contract = load_table_contract(
        contract_root / "model_estimate.json"
    )
    covariance_contract = load_table_contract(
        contract_root / "model_covariance.json"
    )
    marginal_contract = load_table_contract(
        contract_root / "marginal_effect.json"
    )
    contracts = {
        "lp_fe": estimate_contract,
        "lp_iv": estimate_contract,
        "model_covariance": covariance_contract,
        "marginal_effects": marginal_contract,
    }
    files = receipt.get("files")
    if not isinstance(files, dict) or set(files) != set(contracts):
        raise ValueError(
            "LP receipt must bind exactly four staging files"
        )
    frames: dict[str, pl.DataFrame] = {}
    staging_paths: dict[str, Path] = {}
    for name, contract in contracts.items():
        metadata = files[name]
        expected_name = f"{name}.csv"
        if (
            not isinstance(metadata, dict)
            or metadata.get("name") != expected_name
        ):
            raise ValueError(
                f"LP receipt does not bind {expected_name}"
            )
        path = staging / expected_name
        if metadata.get("sha256") != sha256_file(path):
            raise ValueError(
                f"LP receipt hash mismatch for {expected_name}"
            )
        frame = _read_staging_csv(path, contract)
        if metadata.get("rows") != frame.height:
            raise ValueError(
                f"LP receipt row count mismatch for {expected_name}"
            )
        _require_no_duplicate_keys(frame, contract)
        _require_provenance(frame, run_context)
        _require_result_metadata(frame, run_context, "cluster_t")
        frames[name] = frame
        staging_paths[name] = path

    fe_keys = expected_cells
    iv_keys = expected_cells - failed_iv
    _validate_estimate_table(
        frames["lp_fe"], "lp_fe", fe_keys, gates, spec
    )
    _validate_estimate_table(
        frames["lp_iv"], "lp_iv", iv_keys, gates, spec
    )
    expected_pairs = {
        *(("lp_fe", key) for key in fe_keys),
        *(("lp_iv", key) for key in iv_keys),
    }
    matrices = _covariance_matrices(
        frames["model_covariance"], expected_pairs
    )
    estimate_frames = {
        "lp_fe": frames["lp_fe"],
        "lp_iv": frames["lp_iv"],
    }
    _validate_covariance_diagonals(
        estimate_frames, matrices
    )
    _validate_marginal_effects(
        frames["marginal_effects"],
        estimate_frames,
        matrices,
        expected_pairs,
        gates,
        spec,
    )

    destinations = {
        "lp_fe": output / "lp_fe.parquet",
        "lp_iv": output / "lp_iv.parquet",
        "model_covariance": output / "model_covariance.parquet",
        "marginal_effects": output / "marginal_effects.parquet",
    }
    build = BuildIdentity(
        command=(
            "python -m green_debt.cli "
            "analysis-ingest-models --kind lp"
        ),
        code_commit=run_context.git_commit,
        created_at_utc=run_context.created_at_utc,
    )
    published: list[Path] = []
    for name, destination in destinations.items():
        contract = contracts[name]
        frame = frames[name].sort(*contract.primary_key)
        contract_path = (
            contract_root / f"{contract.table_id}.json"
        )
        write_authoritative_table(
            frame,
            contract,
            destination,
            (
                InputArtifact.from_path(staging_paths[name]),
                InputArtifact.from_path(receipt_path),
                InputArtifact.from_path(gate_path),
                InputArtifact.from_path(contract_path),
            ),
            build,
        )
        published.append(destination)

    shutil.rmtree(staging)
    return tuple(published)


def load_stage_a_run_context(output_root: Path) -> RunContext:
    """Load the exact run identity frozen by the stage-A diagnostics."""

    summary_path = output_root.resolve() / "diagnostics/stage_a_summary.json"
    payload = _json_object(summary_path, "stage A summary")
    if payload.get("status") != "valid":
        raise ValueError("stage A summary is not valid")
    required = {
        "run_id": str,
        "spec_id": str,
        "input_authority_hash": str,
        "git_commit": str,
        "renv_lock_sha256": str,
        "evidence_policy_sha256": str,
        "seed": int,
        "created_at_utc": str,
    }
    values: dict[str, Any] = {}
    for field, expected_type in required.items():
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, expected_type):
            raise ValueError(
                f"stage A summary has invalid run context field: {field}"
            )
        values[field] = value
    return RunContext(**values)
