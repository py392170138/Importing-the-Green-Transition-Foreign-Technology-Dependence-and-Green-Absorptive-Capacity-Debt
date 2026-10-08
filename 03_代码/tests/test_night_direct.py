from pathlib import Path
import csv
import json
import subprocess

import pytest

from green_debt.network import DirectRouteGuard, PROXY_ENVIRONMENT_KEYS
from green_debt.night_direct import (
    DataHostPolicy,
    NightDirectGate,
    load_data_host_policy,
    parse_connected_tunnel_services,
    run_project_python_after_check,
    validate_data_url,
)
from green_debt.storage import DiskBudgetGuard, GIB


ROOT = Path(__file__).resolve().parents[2]


def policy() -> DataHostPolicy:
    return DataHostPolicy(
        allowed_hosts=("api.worldbank.org", "www.cepii.fr"),
    )


def gate(
    *,
    connected_tunnels: tuple[str, ...] = (),
    interface: str = "en0",
) -> NightDirectGate:
    return NightDirectGate(
        policy=policy(),
        route_guard=DirectRouteGuard(
            system_proxy_reader=lambda: {
                "HTTPEnable": "1",
                "HTTPSEnable": "1",
            },
            route_reader=lambda host, ip: interface,
            resolver=lambda host: "198.51.100.10",
            rejected_interface_prefixes=("utun", "ppp", "ipsec", "tun"),
        ),
        disk_guard=DiskBudgetGuard(
            project_root=ROOT,
            hard_stop_bytes=120 * GIB,
            reserve_bytes=30 * GIB,
            usage_reader=lambda path: 40 * GIB,
            free_reader=lambda path: 200 * GIB,
        ),
        tunnel_reader=lambda: connected_tunnels,
    )


def test_night_gate_accepts_allowlisted_https_on_physical_route() -> None:
    evidence = gate().check(
        "https://www.cepii.fr/DATA_DOWNLOAD/file.zip",
        projected_additional_bytes=5 * GIB,
    )

    assert evidence.status == "physical_direct_ready"
    assert evidence.host == "www.cepii.fr"
    assert evidence.route_interface == "en0"
    assert evidence.connected_tunnel_count == 0
    assert evidence.proxy_bypass_enforced is True
    assert evidence.automatic_redirects is False
    assert evidence.bytes_downloaded == 0
    assert evidence.projected_peak_bytes == 45 * GIB


def test_night_gate_rejects_any_connected_vpn_before_download() -> None:
    with pytest.raises(RuntimeError, match="close Shadowrocket/VPN"):
        gate(connected_tunnels=("Shadowrocket",)).check(
            "https://www.cepii.fr/DATA_DOWNLOAD/file.zip",
            projected_additional_bytes=5 * GIB,
        )


def test_night_gate_still_rejects_a_tunnel_route_after_vpn_disconnect() -> None:
    with pytest.raises(RuntimeError, match="rejected route interface utun6"):
        gate(interface="utun6").check(
            "https://api.worldbank.org/v2/country",
            projected_additional_bytes=1 * GIB,
        )


def test_data_url_requires_https_and_an_exact_registered_host() -> None:
    assert (
        validate_data_url("https://api.worldbank.org/v2/country", policy())
        == "api.worldbank.org"
    )

    with pytest.raises(ValueError, match="host is not allowlisted"):
        validate_data_url("https://www.worldbank.org/", policy())
    with pytest.raises(ValueError, match="HTTPS is required"):
        validate_data_url("http://www.cepii.fr/file.zip", policy())


def test_checked_in_host_policy_covers_registered_public_sources() -> None:
    project_policy = load_data_host_policy(ROOT / "config" / "data_hosts.yaml")
    registry = ROOT / "07_文献与日志" / "公共数据源登记表_v0.1.csv"
    with registry.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    uncovered: list[str] = []
    for row in rows:
        try:
            validate_data_url(row["direct_or_landing_url"], project_policy)
        except ValueError:
            uncovered.append(row["source_id"])

    assert "www.cepii.fr" in project_policy.allowed_hosts
    assert "api.worldbank.org" in project_policy.allowed_hosts
    assert "api.openalex.org" in project_policy.allowed_hosts
    assert "api.openai.com" not in project_policy.allowed_hosts
    assert uncovered == []


def test_connected_tunnel_parser_ignores_disconnected_services() -> None:
    raw = (
        "Available network connection services in the current set:\n"
        '* (Connected) SECRET-ID IPSec "Shadowrocket" [VPN:Shadowrocket]\n'
        '* (Disconnected) OTHER-ID IPSec "Other VPN" [VPN:Other]\n'
    )

    assert parse_connected_tunnel_services(raw) == ("Shadowrocket",)


def test_run_uses_project_python_with_proxy_free_child_environment() -> None:
    captured: dict[str, object] = {}

    def execute(
        executable: str,
        arguments: list[str],
        environment: dict[str, str],
    ) -> None:
        captured.update(
            executable=executable,
            arguments=arguments,
            environment=environment,
        )

    run_project_python_after_check(
        gate=gate(),
        url="https://www.cepii.fr/DATA_DOWNLOAD/file.zip",
        projected_additional_bytes=5 * GIB,
        python_executable=ROOT / ".venv" / "bin" / "python",
        python_arguments=["-m", "green_debt.cli", "acquire", "baci_hs96"],
        executor=execute,
        base_environment={
            "PATH": "/usr/bin:/bin",
            "HTTPS_PROXY": "http://proxy.invalid:9999",
            "ALL_PROXY": "socks5://proxy.invalid:9999",
        },
    )

    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert PROXY_ENVIRONMENT_KEYS.isdisjoint(environment)
    assert environment["NO_PROXY"] == "*"
    assert environment["no_proxy"] == "*"
    assert captured["executable"] == str(ROOT / ".venv" / "bin" / "python")
    assert captured["arguments"] == [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "green_debt.cli",
        "acquire",
        "baci_hs96",
    ]


def test_run_requires_a_positive_declared_download_budget() -> None:
    with pytest.raises(ValueError, match="projected bytes must be positive"):
        run_project_python_after_check(
            gate=gate(),
            url="https://www.cepii.fr/DATA_DOWNLOAD/file.zip",
            projected_additional_bytes=0,
            python_executable=ROOT / ".venv" / "bin" / "python",
            python_arguments=["-m", "green_debt.cli", "acquire", "baci_hs96"],
            executor=lambda executable, arguments, environment: None,
        )


def test_night_direct_script_exposes_check_and_run_commands() -> None:
    completed = subprocess.run(
        [str(ROOT / "03_代码" / "bin" / "night-direct"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "check" in completed.stdout
    assert "run" in completed.stdout
    assert "Shadowrocket" in completed.stdout


def test_night_direct_script_reports_a_clean_block_without_traceback() -> None:
    completed = subprocess.run(
        [
            str(ROOT / "03_代码" / "bin" / "night-direct"),
            "check",
            "http://www.cepii.fr/file.zip",
            "--projected-bytes",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stderr) == {
        "error": "HTTPS is required",
        "status": "blocked",
    }
    assert "Traceback" not in completed.stderr


def test_night_run_parses_explicit_url_before_project_python_arguments() -> None:
    completed = subprocess.run(
        [
            str(ROOT / "03_代码" / "bin" / "night-direct"),
            "run",
            "--url",
            "https://www.cepii.fr/file.zip",
            "--projected-bytes",
            "0",
            "--",
            "-m",
            "green_debt.cli",
            "config-check",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stderr) == {
        "error": "projected bytes must be positive for a download run",
        "status": "blocked",
    }
