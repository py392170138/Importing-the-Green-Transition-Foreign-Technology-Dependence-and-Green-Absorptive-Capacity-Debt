"""Fail-closed physical-direct gate for proxy-off download windows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlparse

import yaml

from green_debt.network import (
    DirectHttpClient,
    DirectRouteGuard,
    direct_environment,
)
from green_debt.storage import DiskBudgetGuard


@dataclass(frozen=True)
class DataHostPolicy:
    allowed_hosts: tuple[str, ...]


@dataclass(frozen=True)
class NightDirectEvidence:
    status: str
    host: str
    resolved_ip: str
    route_interface: str
    connected_tunnel_count: int
    proxy_bypass_enforced: bool
    automatic_redirects: bool
    system_proxy_enabled: bool
    bytes_downloaded: int
    current_project_bytes: int
    projected_peak_bytes: int
    route_exception_authorization: str | None
    route_exception_used: bool
    checked_at_utc: str


def _normalize_hosts(values: object) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError("allowed_hosts must be a non-empty list")
    hosts: set[str] = set()
    for value in values:
        host = str(value).strip().lower().rstrip(".")
        if not host or "/" in host or ":" in host or host.startswith("."):
            raise ValueError(f"invalid allowed host: {value}")
        hosts.add(host)
    return tuple(sorted(hosts))


def load_data_host_policy(path: Path) -> DataHostPolicy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("data-host policy must be a mapping")
    return DataHostPolicy(
        allowed_hosts=_normalize_hosts(raw.get("allowed_hosts")),
    )


def validate_data_url(url: str, policy: DataHostPolicy) -> str:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        raise ValueError("HTTPS is required")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")
    host = parsed.hostname.lower().rstrip(".")
    if host not in policy.allowed_hosts:
        raise ValueError(f"host is not allowlisted: {host}")
    return host


def parse_connected_tunnel_services(scutil_output: str) -> tuple[str, ...]:
    services: list[str] = []
    for line in scutil_output.splitlines():
        if "(Connected)" not in line:
            continue
        match = re.search(r'"([^"]+)"', line)
        if match:
            services.append(match.group(1))
    return tuple(services)


def read_connected_tunnel_services() -> tuple[str, ...]:
    completed = subprocess.run(
        ["scutil", "--nc", "list"],
        check=True,
        capture_output=True,
        text=True,
        env=direct_environment(),
    )
    return parse_connected_tunnel_services(completed.stdout)


class NightDirectGate:
    def __init__(
        self,
        *,
        policy: DataHostPolicy,
        route_guard: DirectRouteGuard,
        disk_guard: DiskBudgetGuard,
        tunnel_reader: Callable[[], tuple[str, ...]],
        allow_logged_user_route_exception: bool = False,
    ) -> None:
        self._policy = policy
        self._route_guard = route_guard
        self._disk_guard = disk_guard
        self._tunnel_reader = tunnel_reader
        self._allow_logged_user_route_exception = (
            allow_logged_user_route_exception
        )

    def check(
        self,
        url: str,
        *,
        projected_additional_bytes: int,
        route_exception_authorization: str | None = None,
    ) -> NightDirectEvidence:
        host = validate_data_url(url, self._policy)
        authorization = (
            route_exception_authorization.strip()
            if route_exception_authorization is not None
            else None
        )
        if authorization == "":
            raise ValueError("route exception authorization must be nonempty")
        exception_permitted = bool(
            self._allow_logged_user_route_exception and authorization
        )
        connected_tunnels = self._tunnel_reader()
        if connected_tunnels and not exception_permitted:
            raise RuntimeError(
                "connected tunnel detected; close Shadowrocket/VPN before download"
            )

        client = DirectHttpClient()
        try:
            proxy_bypass = (
                client.trust_env is False
                and client.explicit_proxy is None
                and client.follow_redirects is False
            )
            route = self._route_guard.check(
                host,
                proxy_bypass_enforced=proxy_bypass,
                allow_rejected_interface=exception_permitted,
            )
            disk = self._disk_guard.check(projected_additional_bytes)
        finally:
            client.close()

        route_exception_used = bool(connected_tunnels) or route.rejected_interface
        return NightDirectEvidence(
            status=(
                "authorized_route_exception_ready"
                if route_exception_used
                else "physical_direct_ready"
            ),
            host=host,
            resolved_ip=route.resolved_ip,
            route_interface=route.interface,
            connected_tunnel_count=len(connected_tunnels),
            proxy_bypass_enforced=proxy_bypass,
            automatic_redirects=client.follow_redirects,
            system_proxy_enabled=route.system_proxy_enabled,
            bytes_downloaded=0,
            current_project_bytes=disk.current_project_bytes,
            projected_peak_bytes=disk.projected_peak_bytes,
            route_exception_authorization=authorization,
            route_exception_used=route_exception_used,
            checked_at_utc=route.checked_at_utc,
        )


Executor = Callable[[str, list[str], dict[str, str]], None]


def run_project_python_after_check(
    *,
    gate: NightDirectGate,
    url: str,
    projected_additional_bytes: int,
    python_executable: Path,
    python_arguments: Sequence[str],
    executor: Executor = os.execvpe,
    base_environment: Mapping[str, str] | None = None,
    route_exception_authorization: str | None = None,
) -> NightDirectEvidence:
    if not python_arguments:
        raise ValueError("project Python arguments are required")
    if projected_additional_bytes <= 0:
        raise ValueError("projected bytes must be positive for a download run")
    evidence = gate.check(
        url,
        projected_additional_bytes=projected_additional_bytes,
        route_exception_authorization=route_exception_authorization,
    )
    executable = str(python_executable)
    arguments = [executable, *python_arguments]
    environment = direct_environment(base_environment)
    executor(executable, arguments, environment)
    return evidence
