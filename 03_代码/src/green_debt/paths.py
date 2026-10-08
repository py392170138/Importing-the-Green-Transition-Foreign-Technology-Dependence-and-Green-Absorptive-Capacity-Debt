"""Pure resolution of separate code and shared-data roots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    code_root: Path
    data_root: Path
    raw: Path
    normalized: Path
    harmonized: Path
    measures: Path
    analysis: Path
    schemas: Path
    manifests: Path
    scratch: Path
    audits: Path


def resolve_project_paths(code_root: Path, data_root: Path | None) -> ProjectPaths:
    """Derive project locations without creating any directories."""

    code = code_root.resolve()
    data = (data_root if data_root is not None else code_root).resolve()
    if data.exists() and not data.is_dir():
        raise ValueError(f"data_root must be a directory: {data}")

    intermediate = data / "05_中间数据"
    paths = ProjectPaths(
        code_root=code,
        data_root=data,
        raw=data / "04_原始数据",
        normalized=intermediate / "normalized",
        harmonized=intermediate / "harmonized",
        measures=intermediate / "measures",
        analysis=intermediate / "analysis",
        schemas=intermediate / "schemas",
        manifests=intermediate / "manifests",
        scratch=intermediate / "_tmp",
        audits=code / "06_结果",
    )
    data_paths = (
        paths.raw,
        paths.normalized,
        paths.harmonized,
        paths.measures,
        paths.analysis,
        paths.schemas,
        paths.manifests,
        paths.scratch,
    )
    if any(not path.is_relative_to(data) for path in data_paths):
        raise ValueError("derived data path escaped data_root")
    return paths
