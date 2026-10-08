import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from green_debt.cli import build_parser
from green_debt.network import PROXY_ENVIRONMENT_KEYS
from green_debt.r_environment import (
    CRAN_INDEX_URL,
    FWILDCLUSTERBOOT_SOURCE_URL,
    GITHUB_API_PREFLIGHT_URL,
    R_PROJECTED_WORKING_BYTES,
    run_r_dependency_action,
    run_r_dependency_verification,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_direct_gate():
    calls: list[tuple[str, int, str | None]] = []

    def check(
        url,
        projected_additional_bytes,
        route_exception_authorization=None,
    ):
        calls.append(
            (url, projected_additional_bytes, route_exception_authorization)
        )
        return SimpleNamespace(
            route_interface="en0",
            system_proxy_enabled=False,
            route_exception_authorization=route_exception_authorization,
            route_exception_used=False,
        )

    return SimpleNamespace(check=check, calls=calls)


def test_r_restore_uses_proxy_free_child_environment(fake_direct_gate) -> None:
    captured: dict[str, object] = {}

    receipt = run_r_dependency_action(
        gate=fake_direct_gate,
        action="restore",
        project_root=ROOT,
        rscript=Path("/usr/local/bin/Rscript"),
        base_environment={
            "HTTPS_PROXY": "http://proxy.invalid:9000",
            "http_proxy": "http://proxy.invalid:9001",
            "PATH": "/usr/bin",
        },
        executor=lambda args, env: captured.update(args=args, env=env),
    )

    environment = captured["env"]
    assert PROXY_ENVIRONMENT_KEYS.isdisjoint(environment)
    assert environment["NO_PROXY"] == "*"
    assert environment["no_proxy"] == "*"
    assert environment["RENV_CONFIG_AUTOLOADER_ENABLED"] == "FALSE"
    assert captured["args"][-1] == "restore"
    assert fake_direct_gate.calls == [
        (CRAN_INDEX_URL, R_PROJECTED_WORKING_BYTES, None),
        (GITHUB_API_PREFLIGHT_URL, R_PROJECTED_WORKING_BYTES, None),
        (FWILDCLUSTERBOOT_SOURCE_URL, R_PROJECTED_WORKING_BYTES, None),
    ]
    assert receipt.route_interface == "en0"
    assert receipt.proxy_bypass_enforced is True


def test_r_dependency_action_rejects_unknown_action_before_gate(
    fake_direct_gate,
) -> None:
    with pytest.raises(ValueError, match="initialize or restore"):
        run_r_dependency_action(
            gate=fake_direct_gate,
            action="verify",
            project_root=ROOT,
            rscript=Path("/usr/local/bin/Rscript"),
            executor=lambda _args, _env: None,
        )

    assert fake_direct_gate.calls == []


def test_r_dependency_action_rejects_enabled_system_proxy_before_child() -> None:
    def check(_url, projected_additional_bytes, route_exception_authorization=None):
        return SimpleNamespace(
            route_interface="en0",
            system_proxy_enabled=True,
            route_exception_authorization=route_exception_authorization,
            route_exception_used=False,
            projected_additional_bytes=projected_additional_bytes,
        )

    gate = SimpleNamespace(
        check=check
    )
    executed = False

    def executor(_args, _env):
        nonlocal executed
        executed = True

    with pytest.raises(RuntimeError, match="system proxy.*manually"):
        run_r_dependency_action(
            gate=gate,
            action="initialize",
            project_root=ROOT,
            rscript=Path("/usr/local/bin/Rscript"),
            executor=executor,
        )

    assert executed is False


def test_r_dependency_action_logs_authorized_proxy_vpn_exception() -> None:
    authorization = "user_authorized_proxy_vpn_2026-08-29"
    captured: dict[str, object] = {}
    calls: list[tuple[str, int, str | None]] = []

    def check(url, projected_additional_bytes, route_exception_authorization=None):
        calls.append(
            (url, projected_additional_bytes, route_exception_authorization)
        )
        return SimpleNamespace(
            route_interface="utun6",
            system_proxy_enabled=True,
            route_exception_authorization=route_exception_authorization,
            route_exception_used=True,
        )

    receipt = run_r_dependency_action(
        gate=SimpleNamespace(check=check),
        action="initialize",
        project_root=ROOT,
        rscript=Path("/usr/local/bin/Rscript"),
        route_exception_authorization=authorization,
        executor=lambda args, env: captured.update(args=args, env=env),
    )

    assert calls == [
        (CRAN_INDEX_URL, R_PROJECTED_WORKING_BYTES, authorization),
        (
            GITHUB_API_PREFLIGHT_URL,
            R_PROJECTED_WORKING_BYTES,
            authorization,
        ),
        (
            FWILDCLUSTERBOOT_SOURCE_URL,
            R_PROJECTED_WORKING_BYTES,
            authorization,
        ),
    ]
    assert captured["args"][-1] == "initialize"
    assert receipt.route_interface == "utun6"
    assert receipt.system_proxy_enabled is True
    assert receipt.route_exception_used is True
    assert receipt.route_exception_authorization == authorization


def test_r_verification_is_offline_and_proxy_sanitized() -> None:
    captured: dict[str, object] = {}

    run_r_dependency_verification(
        project_root=ROOT,
        rscript=Path("/usr/local/bin/Rscript"),
        base_environment={
            "ALL_PROXY": "socks5://proxy.invalid:1080",
            "PATH": "/usr/bin",
        },
        executor=lambda args, env: captured.update(args=args, env=env),
    )

    assert PROXY_ENVIRONMENT_KEYS.isdisjoint(captured["env"])
    assert captured["env"]["RENV_CONFIG_AUTOLOADER_ENABLED"] == "FALSE"
    assert captured["env"]["RENV_CONFIG_OFFLINE"] == "TRUE"
    assert captured["args"][-1] == "verify"


def test_dependency_host_policy_is_exactly_the_three_frozen_source_hosts() -> None:
    text = (ROOT / "config/dependency_hosts.yaml").read_text(encoding="utf-8")
    assert text.strip().splitlines() == [
        "allowed_hosts:",
        "  - api.github.com",
        "  - cloud.r-project.org",
        "  - codeload.github.com",
    ]


def test_analysis_dependency_cli_accepts_explicit_route_exception() -> None:
    authorization = "user_authorized_proxy_vpn_2026-08-29"
    args = build_parser().parse_args(
        [
            "analysis-deps",
            "--initialize",
            "--route-exception-authorization",
            authorization,
        ]
    )

    assert args.dependency_action == "initialize"
    assert args.route_exception_authorization == authorization


def test_generated_lock_freezes_core_packages_and_remote_sources() -> None:
    lock = json.loads((ROOT / "renv.lock").read_text(encoding="utf-8"))
    assert lock["R"]["Version"] == "4.6.1"
    packages = lock["Packages"]
    assert {
        "arrow",
        "data.table",
        "fixest",
        "ivreg",
        "clubSandwich",
        "fwildclusterboot",
        "dqrng",
        "ggplot2",
        "ragg",
        "svglite",
        "jsonlite",
        "digest",
        "modelsummary",
        "testthat",
        "renv",
        "summclust",
    }.issubset(packages)
    assert packages["summclust"]["Version"] == "0.7.2"
    fwild = packages["fwildclusterboot"]
    assert fwild["Version"] == "0.14.3"
    assert fwild["Source"] == "GitHub"
    assert fwild["RemoteUsername"] == "s3alfisc"
    assert fwild["RemoteRepo"] == "fwildclusterboot"
    assert fwild["RemoteSha"] == (
        "336bb574eba169ac0183317f01d0564791d8122f"
    )
