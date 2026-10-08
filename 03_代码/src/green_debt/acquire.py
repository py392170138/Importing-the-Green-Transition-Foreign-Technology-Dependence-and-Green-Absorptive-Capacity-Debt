"""Atomic, audited downloads for exact public-data files."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx

from green_debt.network import DirectHttpClient
from green_debt.storage import sha256_file


SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_url(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, "", ""))


def manifest_path_for(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.manifest.json")


@dataclass(frozen=True)
class DownloadSpec:
    source_id: str
    source_version: str
    url: str
    allowed_hosts: tuple[str, ...]
    destination: Path
    expected_max_bytes: int
    expected_sha256: str | None
    projected_working_bytes: int


@dataclass(frozen=True)
class AcquisitionRecord:
    source_id: str
    source_version: str
    url: str
    allowed_host: str
    started_at_utc: str
    completed_at_utc: str
    status: str
    bytes: int
    sha256: str | None
    etag: str | None
    last_modified: str | None
    connected_tunnel_count: int
    route_interface: str | None
    resolved_ip: str | None
    system_proxy_enabled: bool
    proxy_bypass_enforced: bool
    automatic_redirects: bool
    proxy_mode: str
    current_project_bytes: int
    projected_peak_bytes: int
    destination: str
    code_version: str
    error: str | None = None
    route_exception_authorization: str | None = None
    route_exception_used: bool = False


class GateEvidence(Protocol):
    connected_tunnel_count: int
    route_interface: str
    resolved_ip: str
    system_proxy_enabled: bool
    proxy_bypass_enforced: bool
    automatic_redirects: bool
    current_project_bytes: int
    projected_peak_bytes: int


class DirectGate(Protocol):
    def check(
        self,
        url: str,
        *,
        projected_additional_bytes: int,
        route_exception_authorization: str | None = None,
    ) -> GateEvidence: ...


@dataclass(frozen=True)
class _TestEvidence:
    connected_tunnel_count: int
    route_interface: str
    resolved_ip: str
    system_proxy_enabled: bool
    proxy_bypass_enforced: bool
    automatic_redirects: bool
    current_project_bytes: int
    projected_peak_bytes: int


class _TestGate:
    def check(
        self,
        url: str,
        *,
        projected_additional_bytes: int,
        route_exception_authorization: str | None = None,
    ) -> _TestEvidence:
        return _TestEvidence(
            connected_tunnel_count=0,
            route_interface="en0",
            resolved_ip="198.51.100.10",
            system_proxy_enabled=False,
            proxy_bypass_enforced=True,
            automatic_redirects=False,
            current_project_bytes=0,
            projected_peak_bytes=projected_additional_bytes,
        )


def append_acquisition_log(log_path: Path, record: object) -> None:
    """Append one JSON record and fsync it so route evidence is durable."""

    if is_dataclass(record) and not isinstance(record, type):
        payload = asdict(record)
    elif isinstance(record, dict):
        payload = dict(record)
    else:
        raise TypeError("acquisition log record must be a dataclass or mapping")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class AcquisitionRunner:
    def __init__(
        self,
        *,
        project_root: Path,
        gate: DirectGate,
        client: DirectHttpClient,
        log_path: Path,
        code_version: str = "unknown",
    ) -> None:
        self._project_root = project_root.resolve()
        self._gate = gate
        self._client = client
        self._log_path = log_path
        self._code_version = code_version

    @classmethod
    def for_test(
        cls,
        project_root: Path,
        transport: httpx.BaseTransport | None = None,
    ) -> AcquisitionRunner:
        return cls(
            project_root=project_root,
            gate=_TestGate(),
            client=DirectHttpClient(transport=transport),
            log_path=project_root / "下载日志.jsonl",
            code_version="test",
        )

    def close(self) -> None:
        self._client.close()

    def _validate(self, spec: DownloadSpec) -> str:
        if SOURCE_ID_PATTERN.fullmatch(spec.source_id) is None:
            raise ValueError("invalid source_id")
        if not spec.source_version.strip():
            raise ValueError("source_version is required")
        parsed = urlsplit(spec.url)
        if parsed.scheme.lower() != "https":
            raise ValueError("HTTPS is required")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("URL credentials are forbidden")
        host = (parsed.hostname or "").lower().rstrip(".")
        allowed = tuple(value.lower().rstrip(".") for value in spec.allowed_hosts)
        if not host or host not in allowed:
            raise ValueError(f"host is not allowlisted: {host}")
        if spec.expected_max_bytes <= 0:
            raise ValueError("expected_max_bytes must be positive")
        if spec.projected_working_bytes < spec.expected_max_bytes:
            raise ValueError(
                "projected_working_bytes must cover expected_max_bytes"
            )
        if (
            spec.expected_sha256 is not None
            and SHA256_PATTERN.fullmatch(spec.expected_sha256.lower()) is None
        ):
            raise ValueError("expected_sha256 must be 64 lowercase hex characters")
        destination = spec.destination.resolve()
        if not destination.is_relative_to(self._project_root):
            raise ValueError("destination must stay inside the project")
        return host

    def _append_record(self, record: AcquisitionRecord) -> None:
        append_acquisition_log(self._log_path, record)

    def _verified_existing(
        self,
        spec: DownloadSpec,
        *,
        host: str,
        started_at: str,
        route_exception_authorization: str | None,
        route_exception_used: bool,
    ) -> AcquisitionRecord | None:
        if not spec.destination.exists():
            return None
        manifest_path = manifest_path_for(spec.destination)
        if not manifest_path.is_file():
            raise RuntimeError(
                "destination exists without a verification manifest; refusing overwrite"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_bytes = spec.destination.stat().st_size
        actual_sha256 = sha256_file(spec.destination)
        if actual_bytes != manifest.get("bytes"):
            raise RuntimeError("existing destination size does not match manifest")
        if actual_sha256 != manifest.get("sha256"):
            raise RuntimeError("existing destination hash does not match manifest")
        if (
            spec.expected_sha256 is not None
            and actual_sha256 != spec.expected_sha256.lower()
        ):
            raise RuntimeError("existing destination hash does not match source spec")
        completed_at = _utc_now()
        record = AcquisitionRecord(
            source_id=spec.source_id,
            source_version=spec.source_version,
            url=_safe_url(spec.url),
            allowed_host=host,
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            status="verified_existing",
            bytes=actual_bytes,
            sha256=actual_sha256,
            etag=manifest.get("etag"),
            last_modified=manifest.get("last_modified"),
            connected_tunnel_count=0,
            route_interface=None,
            resolved_ip=None,
            system_proxy_enabled=False,
            proxy_bypass_enforced=True,
            automatic_redirects=False,
            proxy_mode="direct_only",
            current_project_bytes=0,
            projected_peak_bytes=0,
            destination=str(spec.destination.relative_to(self._project_root)),
            code_version=self._code_version,
            route_exception_authorization=(
                manifest.get("route_exception_authorization")
                if manifest.get("route_exception_authorization") is not None
                else route_exception_authorization
            ),
            route_exception_used=bool(
                manifest.get("route_exception_used", route_exception_used)
            ),
        )
        self._append_record(record)
        return record

    def acquire(
        self,
        spec: DownloadSpec,
        *,
        route_exception_authorization: str | None = None,
        route_exception_used: bool = False,
    ) -> AcquisitionRecord:
        started_at = _utc_now()
        host = self._validate(spec)
        authorization = (
            route_exception_authorization.strip()
            if route_exception_authorization is not None
            else None
        )
        if authorization == "":
            raise ValueError("route exception authorization must be nonempty")
        if route_exception_used and authorization is None:
            raise ValueError("a used route exception requires authorization")
        existing = self._verified_existing(
            spec,
            host=host,
            started_at=started_at,
            route_exception_authorization=authorization,
            route_exception_used=route_exception_used,
        )
        if existing is not None:
            return existing

        partial = spec.destination.with_name(f"{spec.destination.name}.partial")
        if partial.exists():
            partial.unlink()
        evidence: GateEvidence | None = None
        try:
            evidence = self._gate.check(
                spec.url,
                projected_additional_bytes=spec.projected_working_bytes,
                route_exception_authorization=authorization,
            )
            effective_authorization = getattr(
                evidence, "route_exception_authorization", authorization
            )
            effective_route_exception_used = bool(
                getattr(evidence, "route_exception_used", route_exception_used)
            )
            if effective_route_exception_used and not effective_authorization:
                raise RuntimeError("used route exception lacks authorization")
            spec.destination.parent.mkdir(parents=True, exist_ok=True)
            with self._client.stream("GET", spec.url) as response:
                if response.is_redirect:
                    raise RuntimeError("redirect refused; register the exact target host")
                if response.status_code != 200:
                    raise RuntimeError(
                        f"unexpected HTTP status {response.status_code}"
                    )
                raw_length = response.headers.get("Content-Length")
                declared_length: int | None = None
                if raw_length is not None:
                    try:
                        declared_length = int(raw_length)
                    except ValueError as exc:
                        raise RuntimeError("invalid content length") from exc
                    if declared_length > spec.expected_max_bytes:
                        raise RuntimeError(
                            "content length exceeds expected maximum"
                        )

                digest = hashlib.sha256()
                written = 0
                with partial.open("xb") as handle:
                    for chunk in response.iter_bytes():
                        written += len(chunk)
                        if written > spec.expected_max_bytes:
                            raise RuntimeError(
                                "response exceeds expected maximum"
                            )
                        handle.write(chunk)
                        digest.update(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())

                if declared_length is not None and written != declared_length:
                    raise RuntimeError("content length does not match response body")
                actual_sha256 = digest.hexdigest()
                if (
                    spec.expected_sha256 is not None
                    and actual_sha256 != spec.expected_sha256.lower()
                ):
                    raise RuntimeError("SHA-256 does not match expected value")

                etag = response.headers.get("ETag")
                last_modified = response.headers.get("Last-Modified")

            os.replace(partial, spec.destination)
            completed_at = _utc_now()
            record = AcquisitionRecord(
                source_id=spec.source_id,
                source_version=spec.source_version,
                url=_safe_url(spec.url),
                allowed_host=host,
                started_at_utc=started_at,
                completed_at_utc=completed_at,
                status="downloaded",
                bytes=written,
                sha256=actual_sha256,
                etag=etag,
                last_modified=last_modified,
                connected_tunnel_count=evidence.connected_tunnel_count,
                route_interface=evidence.route_interface,
                resolved_ip=evidence.resolved_ip,
                system_proxy_enabled=evidence.system_proxy_enabled,
                proxy_bypass_enforced=evidence.proxy_bypass_enforced,
                automatic_redirects=evidence.automatic_redirects,
                proxy_mode="direct_only",
                current_project_bytes=evidence.current_project_bytes,
                projected_peak_bytes=evidence.projected_peak_bytes,
                destination=str(spec.destination.relative_to(self._project_root)),
                code_version=self._code_version,
                route_exception_authorization=effective_authorization,
                route_exception_used=effective_route_exception_used,
            )
            manifest_path = manifest_path_for(spec.destination)
            manifest_partial = manifest_path.with_name(
                f"{manifest_path.name}.partial"
            )
            manifest_partial.write_text(
                json.dumps(asdict(record), ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            os.replace(manifest_partial, manifest_path)
            self._append_record(record)
            return record
        except Exception as exc:
            if partial.exists():
                partial.unlink()
            if spec.destination.exists() and not manifest_path_for(
                spec.destination
            ).exists():
                spec.destination.unlink()
            completed_at = _utc_now()
            message = str(exc).replace(spec.url, _safe_url(spec.url))
            failed = AcquisitionRecord(
                source_id=spec.source_id,
                source_version=spec.source_version,
                url=_safe_url(spec.url),
                allowed_host=host,
                started_at_utc=started_at,
                completed_at_utc=completed_at,
                status="failed",
                bytes=0,
                sha256=None,
                etag=None,
                last_modified=None,
                connected_tunnel_count=(
                    evidence.connected_tunnel_count if evidence else 0
                ),
                route_interface=evidence.route_interface if evidence else None,
                resolved_ip=evidence.resolved_ip if evidence else None,
                system_proxy_enabled=(
                    evidence.system_proxy_enabled if evidence else False
                ),
                proxy_bypass_enforced=True,
                automatic_redirects=False,
                proxy_mode="direct_only",
                current_project_bytes=(
                    evidence.current_project_bytes if evidence else 0
                ),
                projected_peak_bytes=(
                    evidence.projected_peak_bytes if evidence else 0
                ),
                destination=str(spec.destination.relative_to(self._project_root)),
                code_version=self._code_version,
                error=message,
                route_exception_authorization=(
                    getattr(evidence, "route_exception_authorization", authorization)
                    if evidence
                    else authorization
                ),
                route_exception_used=(
                    bool(getattr(evidence, "route_exception_used", route_exception_used))
                    if evidence
                    else route_exception_used
                ),
            )
            self._append_record(failed)
            raise
