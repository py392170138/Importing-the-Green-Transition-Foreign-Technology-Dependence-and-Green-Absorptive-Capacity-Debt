"""Validated source catalog and frozen download batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from green_debt.acquire import DownloadSpec


@dataclass(frozen=True)
class SourceEntry:
    source_id: str
    kind: str
    enabled: bool
    reason: str | None
    landing_url: str
    download_url: str | None
    api_base_url: str | None
    allowed_hosts: tuple[str, ...]
    version: str
    destination: str | None
    expected_max_bytes: int | None
    expected_sha256: str | None
    projected_working_bytes: int
    public_access: bool
    authentication: str


@dataclass(frozen=True)
class SourceCatalog:
    project_root: Path
    sources: Mapping[str, SourceEntry]
    batches: Mapping[str, tuple[str, ...]]

    def download_spec(self, source_id: str) -> DownloadSpec:
        try:
            entry = self.sources[source_id]
        except KeyError as exc:
            raise ValueError(f"unknown source_id: {source_id}") from exc
        if not entry.enabled:
            raise ValueError(
                f"source is disabled: {source_id} ({entry.reason or 'no reason'})"
            )
        if entry.kind != "file" or entry.download_url is None:
            raise ValueError(f"source has no frozen file download: {source_id}")
        if entry.destination is None or entry.expected_max_bytes is None:
            raise ValueError(f"source file limits are incomplete: {source_id}")
        destination = (self.project_root / entry.destination).resolve()
        if not destination.is_relative_to(self.project_root):
            raise ValueError(f"source destination escapes project: {source_id}")
        return DownloadSpec(
            source_id=entry.source_id,
            source_version=entry.version,
            url=entry.download_url,
            allowed_hosts=entry.allowed_hosts,
            destination=destination,
            expected_max_bytes=entry.expected_max_bytes,
            expected_sha256=entry.expected_sha256,
            projected_working_bytes=entry.projected_working_bytes,
        )

    def batch_specs(self, batch_name: str) -> tuple[DownloadSpec, ...]:
        try:
            source_ids = self.batches[batch_name]
        except KeyError as exc:
            raise ValueError(f"unknown batch: {batch_name}") from exc
        if not source_ids:
            raise ValueError(f"batch is empty: {batch_name}")
        return tuple(self.download_spec(source_id) for source_id in source_ids)


def _required_text(raw: Mapping[str, object], key: str, source_id: str) -> str:
    value = str(raw.get(key, "")).strip()
    if not value:
        raise ValueError(f"{source_id}.{key} is required")
    return value


def _optional_text(raw: Mapping[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _hosts(raw: object, source_id: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source_id}.allowed_hosts must be a non-empty list")
    normalized = tuple(
        sorted({str(value).strip().lower().rstrip(".") for value in raw})
    )
    if any(not host or "/" in host or ":" in host for host in normalized):
        raise ValueError(f"{source_id}.allowed_hosts contains an invalid host")
    return normalized


def _positive_int(
    raw: Mapping[str, object],
    key: str,
    source_id: str,
    *,
    required: bool,
) -> int | None:
    value = raw.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{source_id}.{key} must be a positive integer")
    return value


def load_source_catalog(path: Path, project_root: Path) -> SourceCatalog:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("sources catalog schema_version must be 1")
    raw_sources = raw.get("sources")
    raw_batches = raw.get("batches")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise ValueError("sources catalog must contain sources")
    if not isinstance(raw_batches, dict):
        raise ValueError("sources catalog must contain batches")

    entries: dict[str, SourceEntry] = {}
    for source_id, value in raw_sources.items():
        if not isinstance(source_id, str) or not isinstance(value, dict):
            raise ValueError("source records must be keyed mappings")
        enabled = value.get("enabled")
        public_access = value.get("public_access")
        if not isinstance(enabled, bool) or not isinstance(public_access, bool):
            raise ValueError(
                f"{source_id}.enabled and public_access must be booleans"
            )
        kind = _required_text(value, "kind", source_id)
        file_enabled = enabled and kind == "file"
        expected_max = _positive_int(
            value,
            "expected_max_bytes",
            source_id,
            required=file_enabled,
        )
        projected = _positive_int(
            value,
            "projected_working_bytes",
            source_id,
            required=True,
        )
        assert projected is not None
        if expected_max is not None and projected < expected_max:
            raise ValueError(
                f"{source_id}.projected_working_bytes is below file maximum"
            )
        entry = SourceEntry(
            source_id=source_id,
            kind=kind,
            enabled=enabled,
            reason=_optional_text(value, "reason"),
            landing_url=_required_text(value, "landing_url", source_id),
            download_url=_optional_text(value, "download_url"),
            api_base_url=_optional_text(value, "api_base_url"),
            allowed_hosts=_hosts(value.get("allowed_hosts"), source_id),
            version=_required_text(value, "version", source_id),
            destination=_optional_text(value, "destination"),
            expected_max_bytes=expected_max,
            expected_sha256=_optional_text(value, "expected_sha256"),
            projected_working_bytes=projected,
            public_access=public_access,
            authentication=_required_text(value, "authentication", source_id),
        )
        if entry.enabled and not entry.public_access:
            raise ValueError(f"enabled source is not public: {source_id}")
        allowed_authentication = {"none"}
        if entry.kind == "api":
            allowed_authentication.add("free_api_key_optional")
        if entry.enabled and entry.authentication not in allowed_authentication:
            raise ValueError(f"enabled source requires authentication: {source_id}")
        entries[source_id] = entry

    batches: dict[str, tuple[str, ...]] = {}
    for batch_name, values in raw_batches.items():
        if not isinstance(batch_name, str) or not isinstance(values, list):
            raise ValueError("batches must map names to source-id lists")
        source_ids = tuple(str(value) for value in values)
        unknown = [value for value in source_ids if value not in entries]
        if unknown:
            raise ValueError(
                f"batch {batch_name} contains unknown sources: {unknown}"
            )
        batches[batch_name] = source_ids

    return SourceCatalog(
        project_root=project_root.resolve(),
        sources=MappingProxyType(entries),
        batches=MappingProxyType(batches),
    )
