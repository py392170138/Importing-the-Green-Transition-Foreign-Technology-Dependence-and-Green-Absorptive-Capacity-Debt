"""Direct-only network primitives with sanitized route evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import re
import socket
import subprocess
from typing import Any

import httpx


PROXY_ENVIRONMENT_KEYS = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    }
)
SYSTEM_PROXY_TYPES = {
    "HTTPEnable": "HTTP",
    "HTTPSEnable": "HTTPS",
    "SOCKSEnable": "SOCKS",
    "FTPEnable": "FTP",
    "RTSPEnable": "RTSP",
}


def direct_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a copy of an environment with proxy variables removed."""

    environment = dict(os.environ if base is None else base)
    for key in PROXY_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    environment["NO_PROXY"] = "*"
    environment["no_proxy"] = "*"
    return environment


def read_macos_system_proxy_flags() -> dict[str, str]:
    """Read only enable flags from macOS proxy configuration."""

    completed = subprocess.run(
        ["scutil", "--proxy"],
        check=True,
        capture_output=True,
        text=True,
        env=direct_environment(),
    )
    flags: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key in SYSTEM_PROXY_TYPES:
            flags[key] = value
    return flags


def resolve_host(host: str) -> str:
    """Resolve a host to the first IPv4 or IPv6 address."""

    addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    if not addresses:
        raise RuntimeError(f"host resolution returned no addresses: {host}")
    for family, _, _, _, sockaddr in addresses:
        if family in {socket.AF_INET, socket.AF_INET6}:
            return str(sockaddr[0])
    raise RuntimeError(f"host resolution returned no routable address: {host}")


def read_route_interface(host: str, ip: str) -> str:
    """Return the macOS route interface for a resolved target."""

    completed = subprocess.run(
        ["/sbin/route", "-n", "get", ip],
        check=True,
        capture_output=True,
        text=True,
        env=direct_environment(),
    )
    match = re.search(r"^\s*interface:\s*(\S+)\s*$", completed.stdout, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"could not determine route interface for host {host}")
    return match.group(1)


@dataclass(frozen=True)
class RouteEvidence:
    host: str
    resolved_ip: str
    interface: str
    system_proxy_enabled: bool
    enabled_system_proxy_types: tuple[str, ...]
    proxy_bypass_enforced: bool
    rejected_interface: bool
    checked_at_utc: str


class DirectRouteGuard:
    """Verify that a direct client reaches a target without a tunnel route."""

    def __init__(
        self,
        *,
        system_proxy_reader: Callable[[], Mapping[str, Any]] = (
            read_macos_system_proxy_flags
        ),
        route_reader: Callable[[str, str], str] = read_route_interface,
        resolver: Callable[[str], str] = resolve_host,
        rejected_interface_prefixes: tuple[str, ...] = (
            "utun",
            "ppp",
            "ipsec",
            "tun",
        ),
    ) -> None:
        self._system_proxy_reader = system_proxy_reader
        self._route_reader = route_reader
        self._resolver = resolver
        self._rejected_interface_prefixes = tuple(
            prefix.lower() for prefix in rejected_interface_prefixes
        )

    def check(
        self,
        host: str,
        *,
        proxy_bypass_enforced: bool = True,
        allow_rejected_interface: bool = False,
    ) -> RouteEvidence:
        if not proxy_bypass_enforced:
            raise RuntimeError("proxy bypass is not enforced")

        raw_flags = self._system_proxy_reader()
        enabled_types = tuple(
            proxy_type
            for flag, proxy_type in SYSTEM_PROXY_TYPES.items()
            if str(raw_flags.get(flag, "0")).strip() == "1"
        )
        resolved_ip = self._resolver(host)
        interface = self._route_reader(host, resolved_ip)
        lowered = interface.lower()
        rejected_interface = any(
            lowered.startswith(prefix)
            for prefix in self._rejected_interface_prefixes
        )
        if rejected_interface and not allow_rejected_interface:
            raise RuntimeError(f"rejected route interface {interface}")

        checked_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return RouteEvidence(
            host=host,
            resolved_ip=resolved_ip,
            interface=interface,
            system_proxy_enabled=bool(enabled_types),
            enabled_system_proxy_types=enabled_types,
            proxy_bypass_enforced=True,
            rejected_interface=rejected_interface,
            checked_at_utc=checked_at,
        )


class DirectHttpClient:
    """HTTP client that cannot inherit application proxy configuration."""

    def __init__(
        self,
        timeout_seconds: float = 60,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.trust_env = False
        self.explicit_proxy = None
        self.follow_redirects = False
        self._client = httpx.Client(
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": "green-absorptive-debt/0.1",
            },
        )

    def stream(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> Any:
        return self._client.stream(method, url, **kwargs)

    def close(self) -> None:
        self._client.close()
