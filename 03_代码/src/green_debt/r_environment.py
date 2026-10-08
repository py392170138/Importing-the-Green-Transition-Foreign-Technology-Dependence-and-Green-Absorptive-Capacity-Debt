"""Gated R bootstrap with audited route exceptions and offline verification."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess

from green_debt.network import direct_environment
from green_debt.night_direct import NightDirectGate
from green_debt.storage import GIB


CRAN_INDEX_URL = "https://cloud.r-project.org/src/contrib/PACKAGES.gz"
FWILDCLUSTERBOOT_COMMIT = "336bb574eba169ac0183317f01d0564791d8122f"
FWILDCLUSTERBOOT_SOURCE_URL = (
    "https://codeload.github.com/s3alfisc/fwildclusterboot/tar.gz/"
    f"{FWILDCLUSTERBOOT_COMMIT}"
)
GITHUB_API_PREFLIGHT_URL = (
    "https://api.github.com/repos/s3alfisc/fwildclusterboot"
)
R_DEPENDENCY_PREFLIGHT_URLS = (
    CRAN_INDEX_URL,
    GITHUB_API_PREFLIGHT_URL,
    FWILDCLUSTERBOOT_SOURCE_URL,
)
R_DEPENDENCY_HOSTS = (
    "api.github.com",
    "cloud.r-project.org",
    "codeload.github.com",
)
R_PROJECTED_WORKING_BYTES = 2 * GIB


@dataclass(frozen=True)
class RDependencyReceipt:
    action: str
    route_interface: str
    projected_working_bytes: int
    proxy_bypass_enforced: bool
    system_proxy_enabled: bool
    route_exception_authorization: str | None
    route_exception_used: bool


@dataclass(frozen=True)
class RVerificationReceipt:
    action: str
    network_access: bool
    proxy_bypass_enforced: bool


Executor = Callable[[list[str], dict[str, str]], None]


def _r_child_environment(
    base_environment: Mapping[str, str] | None, *, offline: bool
) -> dict[str, str]:
    environment = direct_environment(base_environment)
    environment["RENV_CONFIG_AUTOLOADER_ENABLED"] = "FALSE"
    if offline:
        environment["RENV_CONFIG_OFFLINE"] = "TRUE"
    else:
        environment.pop("RENV_CONFIG_OFFLINE", None)
    return environment


def _run_r(
    *,
    arguments: list[str],
    project_root: Path,
    environment: dict[str, str],
    executor: Executor | None,
) -> None:
    if executor is None:
        subprocess.run(
            arguments,
            cwd=project_root,
            env=environment,
            check=True,
        )
    else:
        executor(arguments, environment)


def run_r_dependency_action(
    *,
    gate: NightDirectGate,
    action: str,
    project_root: Path,
    rscript: Path,
    base_environment: Mapping[str, str] | None = None,
    route_exception_authorization: str | None = None,
    executor: Executor | None = None,
) -> RDependencyReceipt:
    if action not in {"initialize", "restore"}:
        raise ValueError("network action must be initialize or restore")
    root = project_root.resolve()
    bootstrap = root / "03_代码/R/bootstrap_analysis_env.R"
    if not bootstrap.is_file():
        raise FileNotFoundError(bootstrap)
    evidences = tuple(
        gate.check(
            url,
            projected_additional_bytes=R_PROJECTED_WORKING_BYTES,
            route_exception_authorization=route_exception_authorization,
        )
        for url in R_DEPENDENCY_PREFLIGHT_URLS
    )
    route_interfaces = {evidence.route_interface for evidence in evidences}
    if len(route_interfaces) != 1:
        raise RuntimeError("dependency hosts resolve through different interfaces")
    system_proxy_enabled = any(
        bool(getattr(evidence, "system_proxy_enabled", False))
        for evidence in evidences
    )
    exception_used = any(
        bool(getattr(evidence, "route_exception_used", False))
        for evidence in evidences
    )
    authorizations = {
        getattr(evidence, "route_exception_authorization", None)
        for evidence in evidences
    }
    if len(authorizations) != 1:
        raise RuntimeError("dependency route authorization is inconsistent")
    authorization = authorizations.pop()
    if system_proxy_enabled and not (
        exception_used and authorization == route_exception_authorization
    ):
        raise RuntimeError(
            "macOS system proxy is enabled; disable it manually before R downloads"
        )
    environment = _r_child_environment(base_environment, offline=False)
    arguments = [
        str(rscript),
        "--vanilla",
        str(bootstrap),
        action,
    ]
    _run_r(
        arguments=arguments,
        project_root=root,
        environment=environment,
        executor=executor,
    )
    return RDependencyReceipt(
        action=action,
        route_interface=evidences[0].route_interface,
        projected_working_bytes=R_PROJECTED_WORKING_BYTES,
        proxy_bypass_enforced=True,
        system_proxy_enabled=system_proxy_enabled,
        route_exception_authorization=authorization,
        route_exception_used=exception_used,
    )


def run_r_dependency_verification(
    *,
    project_root: Path,
    rscript: Path,
    base_environment: Mapping[str, str] | None = None,
    executor: Executor | None = None,
) -> RVerificationReceipt:
    root = project_root.resolve()
    bootstrap = root / "03_代码/R/bootstrap_analysis_env.R"
    if not bootstrap.is_file():
        raise FileNotFoundError(bootstrap)
    environment = _r_child_environment(base_environment, offline=True)
    arguments = [str(rscript), "--vanilla", str(bootstrap), "verify"]
    _run_r(
        arguments=arguments,
        project_root=root,
        environment=environment,
        executor=executor,
    )
    return RVerificationReceipt(
        action="verify",
        network_access=False,
        proxy_bypass_enforced=True,
    )
