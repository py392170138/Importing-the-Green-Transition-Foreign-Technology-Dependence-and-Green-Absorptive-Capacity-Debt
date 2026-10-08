"""Storage-budget, hashing, and filesystem evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil


GIB = 1024**3


def project_usage_bytes(project_root: Path) -> int:
    """Measure file bytes inside the project without following symlinks."""

    total = 0
    for path in project_root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def directory_usage_bytes(directory: Path) -> int:
    """Measure one directory tree, returning zero when it is absent."""

    if not directory.exists():
        return 0
    if not directory.is_dir():
        raise ValueError(f"usage target must be a directory: {directory}")
    return project_usage_bytes(directory)


def filesystem_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DiskSnapshot:
    current_project_bytes: int
    projected_additional_bytes: int
    projected_peak_bytes: int
    filesystem_free_bytes: int
    reserve_bytes: int
    checked_at_utc: str


@dataclass(frozen=True)
class LayerUsage:
    raw_bytes: int
    normalized_bytes: int
    harmonized_bytes: int
    measures_bytes: int
    analysis_bytes: int
    scratch_bytes: int
    metadata_bytes: int
    audit_bytes: int
    filesystem_free_bytes: int

    @property
    def intermediate_bytes(self) -> int:
        return (
            self.normalized_bytes
            + self.harmonized_bytes
            + self.measures_bytes
            + self.analysis_bytes
        )

    @property
    def project_bytes(self) -> int:
        return (
            self.raw_bytes
            + self.intermediate_bytes
            + self.scratch_bytes
            + self.metadata_bytes
            + self.audit_bytes
        )

    def byte_counts(self) -> dict[str, int]:
        return {
            "raw": self.raw_bytes,
            "normalized": self.normalized_bytes,
            "harmonized": self.harmonized_bytes,
            "measures": self.measures_bytes,
            "analysis": self.analysis_bytes,
            "scratch": self.scratch_bytes,
            "metadata": self.metadata_bytes,
            "audits": self.audit_bytes,
        }


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(f"no existing ancestor for {path}")
        candidate = parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate


def measure_layer_usage(
    data_root: Path,
    *,
    audits_root: Path | None = None,
    free_reader: Callable[[Path], int] = filesystem_free_bytes,
) -> LayerUsage:
    """Measure the bounded raw, L1-L4, runtime, scratch, and audit buckets."""

    data = data_root.resolve()
    intermediate = data / "05_中间数据"
    schemas = intermediate / "schemas"
    manifests = intermediate / "manifests"
    audit_path = (audits_root or (data / "06_结果")).resolve()
    free_path = _nearest_existing_directory(data)
    return LayerUsage(
        raw_bytes=directory_usage_bytes(data / "04_原始数据"),
        normalized_bytes=directory_usage_bytes(intermediate / "normalized"),
        harmonized_bytes=directory_usage_bytes(intermediate / "harmonized"),
        measures_bytes=directory_usage_bytes(intermediate / "measures"),
        analysis_bytes=directory_usage_bytes(intermediate / "analysis"),
        scratch_bytes=directory_usage_bytes(intermediate / "_tmp"),
        metadata_bytes=(
            directory_usage_bytes(schemas) + directory_usage_bytes(manifests)
        ),
        audit_bytes=directory_usage_bytes(audit_path),
        filesystem_free_bytes=free_reader(free_path),
    )


def enforce_construction_capacity(
    usage: LayerUsage,
    *,
    projected_additional_bytes: int = 0,
    intermediate_limit_bytes: int = 25 * GIB,
    hard_stop_bytes: int = 120 * GIB,
    reserve_bytes: int = 30 * GIB,
) -> None:
    """Fail closed at the construction quota, project stop, or disk reserve."""

    if projected_additional_bytes < 0:
        raise ValueError("projected additional bytes cannot be negative")
    if usage.intermediate_bytes + usage.scratch_bytes >= intermediate_limit_bytes:
        raise RuntimeError("25 GB intermediate quota reached")
    if usage.project_bytes + projected_additional_bytes >= hard_stop_bytes:
        raise RuntimeError("projected total reaches 120 GB hard stop")
    if usage.filesystem_free_bytes < projected_additional_bytes + reserve_bytes:
        raise RuntimeError("filesystem cannot preserve 30 GB reserve")


class DiskBudgetGuard:
    """Fail closed before projected project usage or free-space limits."""

    def __init__(
        self,
        *,
        project_root: Path,
        hard_stop_bytes: int,
        reserve_bytes: int,
        usage_reader: Callable[[Path], int] = project_usage_bytes,
        free_reader: Callable[[Path], int] = filesystem_free_bytes,
    ) -> None:
        self.project_root = project_root
        self.hard_stop_bytes = hard_stop_bytes
        self.reserve_bytes = reserve_bytes
        self._usage_reader = usage_reader
        self._free_reader = free_reader

    def check(self, projected_additional_bytes: int) -> DiskSnapshot:
        if projected_additional_bytes < 0:
            raise ValueError("projected additional bytes cannot be negative")

        current = self._usage_reader(self.project_root)
        free = self._free_reader(self.project_root)
        projected_peak = current + projected_additional_bytes
        if projected_peak >= self.hard_stop_bytes:
            stop_gb = self.hard_stop_bytes // GIB
            raise RuntimeError(
                f"projected peak reaches {stop_gb}GB hard stop"
            )
        if free < projected_additional_bytes + self.reserve_bytes:
            reserve_gb = self.reserve_bytes // GIB
            raise RuntimeError(
                f"filesystem cannot preserve {reserve_gb}GB reserve"
            )

        checked_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return DiskSnapshot(
            current_project_bytes=current,
            projected_additional_bytes=projected_additional_bytes,
            projected_peak_bytes=projected_peak,
            filesystem_free_bytes=free,
            reserve_bytes=self.reserve_bytes,
            checked_at_utc=checked_at,
        )
