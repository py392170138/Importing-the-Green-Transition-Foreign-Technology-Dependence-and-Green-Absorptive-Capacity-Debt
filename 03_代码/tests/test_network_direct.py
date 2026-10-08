import os

import pytest

from green_debt.network import (
    DirectHttpClient,
    DirectRouteGuard,
    direct_environment,
)


PROXY_KEYS = {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
}


def test_direct_environment_removes_every_proxy_variable() -> None:
    base = {key: "http://127.0.0.1:9999" for key in PROXY_KEYS}
    base["UNCHANGED"] = "value"

    env = direct_environment(base)

    assert PROXY_KEYS.isdisjoint(env)
    assert env["NO_PROXY"] == "*"
    assert env["no_proxy"] == "*"
    assert env["UNCHANGED"] == "value"


def test_http_client_never_trusts_proxy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")

    client = DirectHttpClient(timeout_seconds=30)

    assert client.trust_env is False
    assert client.explicit_proxy is None
    assert client.follow_redirects is False
    client.close()


def test_route_guard_records_bypassed_system_proxy() -> None:
    guard = DirectRouteGuard(
        system_proxy_reader=lambda: {"HTTPEnable": "1", "HTTPSEnable": "1"},
        route_reader=lambda host, ip: "en0",
        resolver=lambda host: "203.0.113.10",
        rejected_interface_prefixes=("utun", "ppp", "ipsec", "tun"),
    )

    evidence = guard.check("www.cepii.fr", proxy_bypass_enforced=True)

    assert evidence.system_proxy_enabled is True
    assert evidence.enabled_system_proxy_types == ("HTTP", "HTTPS")
    assert evidence.proxy_bypass_enforced is True
    assert evidence.interface == "en0"
    assert evidence.resolved_ip == "203.0.113.10"


def test_route_guard_rejects_tunnel_interface() -> None:
    guard = DirectRouteGuard(
        system_proxy_reader=lambda: {"HTTPEnable": "0", "HTTPSEnable": "0"},
        route_reader=lambda host, ip: "utun4",
        resolver=lambda host: "203.0.113.10",
        rejected_interface_prefixes=("utun", "ppp", "ipsec", "tun"),
    )

    with pytest.raises(RuntimeError, match="rejected route interface utun4"):
        guard.check("api.openalex.org", proxy_bypass_enforced=True)


def test_route_guard_rejects_client_without_proxy_bypass() -> None:
    guard = DirectRouteGuard(
        system_proxy_reader=lambda: {"HTTPEnable": "0"},
        route_reader=lambda host, ip: "en0",
        resolver=lambda host: "203.0.113.10",
        rejected_interface_prefixes=("utun",),
    )

    with pytest.raises(RuntimeError, match="proxy bypass is not enforced"):
        guard.check("api.worldbank.org", proxy_bypass_enforced=False)


def test_direct_environment_does_not_mutate_process_environment() -> None:
    before = dict(os.environ)
    direct_environment(os.environ)
    assert dict(os.environ) == before
