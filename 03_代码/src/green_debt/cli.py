"""Command-line entry point for the research pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence
from urllib.parse import urlsplit

import httpx
import polars as pl

from green_debt.acquire import AcquisitionRunner, append_acquisition_log
from green_debt.analysis_audit import publish_analysis_audit
from green_debt.analysis_io import (
    AnalysisPaths,
    analysis_preflight,
    bind_analysis_reproduction_context,
    build_run_context,
    cleanup_analysis_reproduction,
    ingest_model_bundle,
    load_stage_a_run_context,
    resolve_authorized_analysis_output,
    run_analysis_command_plan,
    run_analysis_reproduction_check,
    validate_analysis_authority,
)
from green_debt.analysis_spec import load_analysis_spec
from green_debt.artifacts import BuildIdentity, InputArtifact, verify_manifest
from green_debt.artifacts import TableContract, write_authoritative_table
from green_debt.checkpoints import (
    discover_authoritative_manifest_paths,
    run_checkpoint,
    verify_raw_hash_snapshot,
    write_raw_hash_snapshot,
)
from green_debt.config import load_construction_config, load_project_config
from green_debt.economies import (
    audit_economies,
    build_production_economy_crosswalk,
)
from green_debt.diagnostics import run_stage_a_diagnostics
from green_debt.network import (
    DirectHttpClient,
    DirectRouteGuard,
    PROXY_ENVIRONMENT_KEYS,
    read_macos_system_proxy_flags,
)
from green_debt.night_direct import (
    NightDirectGate,
    load_data_host_policy,
    read_connected_tunnel_services,
    run_project_python_after_check,
)
from green_debt.outcomes import audit_outcomes, build_outcome_tables
from green_debt.instruments import (
    audit_instrument_artifacts,
    build_instrument_artifacts,
)
from green_debt.sources import load_source_catalog
from green_debt.sources.oecd_sdmx import (
    BASE_URL as OECD_TIVA_BASE_URL,
    acquire_tiva_supplement,
    audit_tiva_supplement,
    load_tiva_supplements,
)
from green_debt.sources.openalex import (
    acquire_country_year_aggregates,
    audit_country_year_panel,
    audit_topic_registry,
    initialization_supplement_spec,
    load_included_topic_ids,
    validate_initialization_supplement_range,
)
from green_debt.sources.baci import (
    audit_trade_normalization,
    stream_baci_aggregates,
)
from green_debt.sources.irena import (
    audit_irena_table,
    build_irena_table,
)
from green_debt.sources.ilostat import (
    audit_ilostat_table,
    build_ilostat_table,
)
from green_debt.sources.policy import (
    audit_policy_table,
    build_policy_table,
)
from green_debt.sources.tiva import (
    audit_tiva_tables,
    build_tiva_tables,
    production_tiva_specs,
)
from green_debt.sources.wdi import (
    audit_wdi_table,
    build_wdi_table,
)
from green_debt.science import audit_openalex_table, build_gsci, build_openalex_table
from green_debt.sample import (
    audit_analysis_panel_authority,
    audit_provisional_sample,
    build_analysis_panel_authority,
    build_provisional_sample_table,
    build_source_coverage,
    freeze_final_sample_authority,
)
from green_debt.paths import resolve_project_paths
from green_debt.r_environment import (
    R_DEPENDENCY_HOSTS,
    R_PROJECTED_WORKING_BYTES,
    run_r_dependency_action,
    run_r_dependency_verification,
)
from green_debt.storage import (
    DiskBudgetGuard,
    GIB,
    measure_layer_usage,
    sha256_file,
)
from green_debt.taxonomy import audit_taxonomy, build_taxonomy
from green_debt.trade import (
    audit_complexity,
    build_complexity,
    build_supplier_raw,
    build_trade_components,
)
from green_debt.tiva import ActivitySets, build_tiva_measures
from green_debt.scaling import (
    FROZEN_COMPONENT_COLUMNS,
    ScalerAnchor,
    ScalerRegistry,
    apply_scaler,
    build_scaler_authority,
    create_scaler_anchor,
    fit_scaler,
    registry_from_dict,
    scaler_anchor_from_dict,
    verify_frozen_sample_semantics,
    verify_scaler,
    verify_scaler_anchor,
    verify_scaled_authority,
)
from green_debt.gad import (
    FROZEN_SCALER_HASH,
    build_gad_variants,
    gad_output_table,
    registered_gad_specifications,
)
from green_debt.build import (
    BuildController,
    cleanup_reproduction,
    registered_build_graph,
    run_reproduction_check,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "project.yaml"
DEFAULT_DATA_HOST_CONFIG = PROJECT_ROOT / "config" / "data_hosts.yaml"
DEFAULT_DEPENDENCY_HOST_CONFIG = PROJECT_ROOT / "config" / "dependency_hosts.yaml"
DEFAULT_SOURCES_CONFIG = PROJECT_ROOT / "config" / "sources.yaml"
DEFAULT_TIVA_SUPPLEMENTS = PROJECT_ROOT / "config" / "tiva_supplements.yaml"
DEFAULT_CONSTRUCTION_CONFIG = PROJECT_ROOT / "config" / "construction.yaml"
DEFAULT_GAD_FROZEN_UNIVERSE = PROJECT_ROOT / "contracts" / "gad_frozen_universe.json"
DEFAULT_ECONOMY_OVERRIDES = (
    PROJECT_ROOT / "02_数据字典" / "economy_overrides_v1.csv"
)
DEFAULT_ECONOMY_CONTRACT = (
    PROJECT_ROOT / "03_代码" / "contracts" / "economy_crosswalk.json"
)
DEFAULT_INDICATOR_REGISTRY = (
    PROJECT_ROOT / "02_数据字典" / "indicator_registry_v1.csv"
)
DEFAULT_WDI_CONTRACT = PROJECT_ROOT / "03_代码/contracts/wdi_country_year.json"
DEFAULT_IRENA_CONTRACT = PROJECT_ROOT / "03_代码/contracts/irena_country_year.json"
DEFAULT_OPENALEX_CONTRACT = (
    PROJECT_ROOT / "03_代码/contracts/openalex_country_year.json"
)
DEFAULT_ILOSTAT_CONTRACT = PROJECT_ROOT / "03_代码/contracts/ilostat_skill.json"
DEFAULT_POLICY_CONTRACT = (
    PROJECT_ROOT / "03_代码/contracts/policy_country_year.json"
)
DEFAULT_TIVA_CONTRACT = (
    PROJECT_ROOT / "03_代码/contracts/tiva_activity_year.json"
)
DEFAULT_TIVA_WEIGHTS_CONTRACT = (
    PROJECT_ROOT / "03_代码/contracts/tiva_activity_weights.json"
)
DEFAULT_TIVA_MEASURES_CONTRACT = PROJECT_ROOT / "03_代码/contracts/tiva_measures.json"
DEFAULT_TIVA_BOUNDED_CONTRACT = (
    PROJECT_ROOT / "03_代码/contracts/tiva_bounded_robustness.json"
)
DEFAULT_SCALER_COMPONENTS_CONTRACT = (
    PROJECT_ROOT / "03_代码/contracts/gad_scaled_components.json"
)
DEFAULT_SCALER_SAMPLE_SEMANTICS = (
    PROJECT_ROOT / "03_代码/contracts/gad_scaler_sample_semantics.json"
)
DEFAULT_GAD_CONTRACT = PROJECT_ROOT / "03_代码/contracts/gad_country_year.json"
DEFAULT_PROVISIONAL_SAMPLE_CONTRACT = (
    PROJECT_ROOT / "03_代码/contracts/provisional_sample.json"
)
DEFAULT_OPENALEX_REGISTRY = (
    PROJECT_ROOT / "02_数据字典" / "openalex_green_topics_v1.csv"
)
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _night_gate(
    config_path: Path,
    hosts_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> NightDirectGate:
    config = load_project_config(config_path)
    return NightDirectGate(
        policy=load_data_host_policy(hosts_path),
        route_guard=DirectRouteGuard(
            rejected_interface_prefixes=config.network.rejected_interface_prefixes
        ),
        disk_guard=DiskBudgetGuard(
            project_root=project_root,
            hard_stop_bytes=config.storage.hard_stop_gb * GIB,
            reserve_bytes=config.storage.reserve_gb * GIB,
        ),
        tunnel_reader=read_connected_tunnel_services,
        allow_logged_user_route_exception=(
            config.network.allow_logged_user_route_exception
        ),
    )


def _rscript_path() -> Path:
    executable = shutil.which("Rscript")
    if executable is None:
        raise RuntimeError("Rscript is not installed or not on PATH")
    return Path(executable).resolve()


def _code_version() -> str:
    bound = os.environ.get("GREEN_DEBT_IMPLEMENTATION_COMMIT")
    if bound is not None:
        if len(bound) != 40 or any(character not in "0123456789abcdef" for character in bound):
            raise ValueError("GREEN_DEBT_IMPLEMENTATION_COMMIT must be a full lowercase Git hash")
        return bound
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else "uncommitted"


def _acquisition_runner(gate: NightDirectGate) -> AcquisitionRunner:
    return AcquisitionRunner(
        project_root=PROJECT_ROOT,
        gate=gate,
        client=DirectHttpClient(timeout_seconds=300),
        log_path=PROJECT_ROOT / "07_文献与日志" / "下载日志.jsonl",
        code_version=_code_version(),
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with partial.open("xb") as handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _load_table_contract(path: Path) -> TableContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    period_value = payload.get("period")
    return TableContract(
        table_id=str(payload["table_id"]),
        schema_version=str(payload["schema_version"]),
        primary_key=tuple(str(value) for value in payload["primary_key"]),
        columns={str(key): str(value) for key, value in payload["columns"].items()},
        units={str(key): str(value) for key, value in payload["units"].items()},
        period=(int(period_value[0]), int(period_value[1])) if period_value else None,
        zero_semantics={
            str(key): str(value) for key, value in payload.get("zero_semantics", {}).items()
        },
        null_semantics={
            str(key): str(value) for key, value in payload.get("null_semantics", {}).items()
        },
        transformations=tuple(str(value) for value in payload.get("transformations", [])),
    )


def _tiva_activity_sets() -> ActivitySets:
    construction = load_construction_config(DEFAULT_CONSTRUCTION_CONFIG)
    confirmatory = tuple(construction.tiva.confirmatory_activities)
    broad = tuple(dict.fromkeys((*confirmatory, *construction.tiva.broad_additions)))
    equipment_only = tuple(
        activity
        for activity in confirmatory
        if activity not in set(construction.tiva.equipment_only_removals)
    )
    return ActivitySets(confirmatory, broad, equipment_only)


def _build_scaler_matched_frame(paths: object) -> tuple[pl.DataFrame, tuple[Path, ...]]:
    """Return the full provisional-core calendar; unavailable components remain null."""

    # ProjectPaths is intentionally duck-typed here to keep this join side-effect free.
    sample_path = paths.harmonized / "sample/provisional_sample.parquet"  # type: ignore[attr-defined]
    trade_path = paths.measures / "trade/trade_components_raw.parquet"  # type: ignore[attr-defined]
    tiva_path = paths.measures / "tiva/tiva_measures.parquet"  # type: ignore[attr-defined]
    gsci_path = paths.measures / "science/gsci_raw.parquet"  # type: ignore[attr-defined]
    supplier_path = paths.measures / "trade/supplier_raw.parquet"  # type: ignore[attr-defined]
    provisional = pl.read_parquet(sample_path)
    trade = pl.read_parquet(trade_path).filter(pl.col("taxonomy_version") == "main_hs96").select(
        "economy_id", "year", *FROZEN_COMPONENT_COLUMNS[:2], "gnir_raw"
    )
    tiva = pl.read_parquet(tiva_path).filter(
        pl.col("specification_id") == "confirmatory_prod_weight"
    ).select("economy_id", "year", "gfvad_raw")
    gsci = pl.read_parquet(gsci_path).select("economy_id", "year", "gsci_raw")
    supplier = pl.read_parquet(supplier_path).filter(pl.col("taxonomy_version") == "main_hs96").select(
        "economy_id", "year", "gud_raw", "grd_raw"
    )
    authority = build_scaler_authority(
        provisional, trade, tiva, gsci, supplier, years=(1996, 2024)
    )
    return authority, (sample_path, trade_path, tiva_path, gsci_path, supplier_path)


def _verify_scaled_manifest_binding(
    manifest: object,
    registry_path: Path,
    anchor_path: Path,
    implementation_commit: str,
    semantic_anchor_path: Path | None = None,
) -> None:
    """Require the published scaled artifact to name both independent freeze files."""

    if manifest.code_commit != implementation_commit:  # type: ignore[attr-defined]
        raise ValueError("scaled manifest implementation commit is not anchor-bound")
    declared_inputs = {  # type: ignore[attr-defined]
        item.path: item.sha256 for item in manifest.input_artifacts
    }
    def declared_hashes(path: Path, suffix: str) -> set[str]:
        exact = str(path.resolve())
        return {
            declared_sha
            for declared_path, declared_sha in declared_inputs.items()
            if declared_path == exact or Path(declared_path).as_posix().endswith(suffix)
        }

    if declared_hashes(
        registry_path, f"06_结果/{registry_path.name}"
    ) != {sha256_file(registry_path)}:
        raise ValueError("scaled manifest lacks registry binding")
    if declared_hashes(anchor_path, f"06_结果/{anchor_path.name}") != {
        sha256_file(anchor_path)
    }:
        raise ValueError("scaled manifest lacks external anchor binding")
    if semantic_anchor_path is not None:
        semantic_suffix = "03_代码/contracts/gad_scaler_sample_semantics.json"
        if declared_hashes(semantic_anchor_path, semantic_suffix) != {
            sha256_file(semantic_anchor_path)
        }:
            raise ValueError("scaled manifest lacks sample semantic anchor binding")


def _verify_frozen_scaler_for_current_sample(
    *,
    registry: ScalerRegistry,
    anchor: ScalerAnchor,
    registry_path: Path,
    anchor_path: Path,
    sample_manifest_path: Path,
    semantic_anchor_path: Path,
) -> None:
    sample_manifest_payload = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    semantic_anchor = json.loads(semantic_anchor_path.read_text(encoding="utf-8"))
    verify_scaler(
        registry,
        expected_manifest_hash=registry.provisional_sample_manifest_hash,
        expected_columns=FROZEN_COMPONENT_COLUMNS,
        expected_years=(2000, 2004),
    )
    verify_frozen_sample_semantics(
        registry, sample_manifest_payload, semantic_anchor
    )
    verify_scaler_anchor(
        registry,
        anchor,
        registry_file_sha256=sha256_file(registry_path),
        implementation_commit=anchor.implementation_commit,
    )
    if registry.canonical_hash != FROZEN_SCALER_HASH:
        raise ValueError("frozen scaler canonical hash differs from the approved Task 12 hash")


def _read_json_lines(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid acquisition log row {line_number}")
        rows.append(payload)
    return rows


def _audit_initialization_supplements(
    *,
    data_root: Path,
    code_root: Path,
    registry: Path,
    supplement_config: Path,
) -> dict[str, object]:
    paths = resolve_project_paths(code_root, data_root)
    specs = load_tiva_supplements(supplement_config)
    tiva_root = paths.raw / "oecd_tiva/20260823/data"
    tiva_reports = []
    manifests: list[dict[str, object]] = []
    for spec in specs:
        destination = tiva_root / spec.destination
        report = audit_tiva_supplement(destination, spec)
        manifest_path = destination.with_name(f"{destination.name}.manifest.json")
        if not manifest_path.is_file():
            raise RuntimeError(f"missing TiVA supplement manifest: {spec.source_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if destination.stat().st_size != manifest.get("bytes"):
            raise RuntimeError(f"TiVA supplement byte mismatch: {spec.source_id}")
        if sha256_file(destination) != manifest.get("sha256"):
            raise RuntimeError(f"TiVA supplement hash mismatch: {spec.source_id}")
        tiva_reports.append(asdict(report))
        manifests.append(manifest)

    frozen_openalex = initialization_supplement_spec(registry)
    openalex_root = paths.raw / "openalex/20260823/works_aggregate"
    panel = openalex_root / frozen_openalex.destination
    openalex_report = audit_country_year_panel(panel)
    if (openalex_report.start_year, openalex_report.end_year) != (1992, 1995):
        raise RuntimeError("OpenAlex initialization period is not 1992-1995")
    panel_manifest_path = panel.with_name(f"{panel.name}.manifest.json")
    panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    if panel.stat().st_size != panel_manifest.get("bytes"):
        raise RuntimeError("OpenAlex initialization byte count mismatch")
    if sha256_file(panel) != panel_manifest.get("sha256"):
        raise RuntimeError("OpenAlex initialization hash mismatch")
    if panel_manifest.get("topic_ids") is None or len(panel_manifest["topic_ids"]) != 59:
        raise RuntimeError("OpenAlex initialization topic set is not 59")
    if panel_manifest.get("counting_method") != "full_country_participation":
        raise RuntimeError("OpenAlex initialization counting rule changed")
    if panel_manifest.get("include_xpac") is not False:
        raise RuntimeError("OpenAlex initialization include_xpac must be false")
    manifests.append(panel_manifest)

    log_path = paths.data_root / "07_文献与日志/下载日志.jsonl"
    log_rows = _read_json_lines(log_path)
    used_exceptions = 0
    for manifest in manifests:
        if not manifest.get("route_exception_used"):
            continue
        used_exceptions += 1
        source_id = manifest.get("source_id")
        authorization = manifest.get("route_exception_authorization")
        if not authorization:
            raise RuntimeError(f"used route exception lacks authorization: {source_id}")
        if not any(
            row.get("source_id") == source_id
            and row.get("route_exception_authorization") == authorization
            and row.get("route_exception_used") is True
            for row in log_rows
        ):
            raise RuntimeError(f"route exception is absent from acquisition log: {source_id}")

    partials = tuple(str(path) for path in paths.raw.rglob("*.partial"))
    if partials:
        raise RuntimeError(f"raw partial files remain: {len(partials)}")
    usage = measure_layer_usage(paths.data_root, audits_root=paths.audits)
    if usage.project_bytes >= 120 * GIB:
        raise RuntimeError("initialization supplements reach the 120 GB hard stop")

    status_path = paths.raw / "README_数据获取状态_20260823.md"
    status_text = (
        "# 初始化补充数据状态（2026-08-23）\n\n"
        "- OpenAlex 1992—1995：已获取并通过国家-年份唯一性审计。\n"
        "- TiVA DFD_FVA 1995—1999：已获取并通过字段/单位审计。\n"
        "- TiVA FD_VA 1995—2022：已获取并通过字段/单位审计。\n"
        f"- 记录的路由例外：{used_exceptions}。\n"
        f"- 原始数据及构造层占用：{usage.project_bytes} bytes。\n"
        "- 未改动 Shadowrocket 或 macOS 代理配置。\n"
    )
    _write_text_atomic(status_path, status_text)
    return {
        "openalex": asdict(openalex_report),
        "tiva": tiva_reports,
        "route_exceptions_logged": used_exceptions,
        "partial_files": 0,
        "project_bytes": usage.project_bytes,
        "status_document": str(status_path),
        "status": "valid",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="green-debt")
    subparsers = parser.add_subparsers(dest="command", required=True)
    data_root_parent = argparse.ArgumentParser(add_help=False)
    data_root_parent.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT,
        help="shared root containing raw and layered intermediate data",
    )
    config_check = subparsers.add_parser(
        "config-check", help="validate and summarize the frozen project config"
    )
    config_check.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    analysis_preflight_parser = subparsers.add_parser(
        "analysis-preflight",
        help="verify frozen analysis inputs and capacity without writing outputs",
    )
    analysis_preflight_parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="shared root containing the authoritative analysis inputs",
    )
    analysis_preflight_parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "06_结果/analysis",
        help="analysis output root to validate without creating it",
    )
    analysis_diagnostics = subparsers.add_parser(
        "analysis-diagnostics",
        help="write the frozen stage-A samples, diagnostics, and model gates",
    )
    analysis_diagnostics.add_argument("--data-root", type=Path, required=True)
    analysis_diagnostics.add_argument("--output-root", type=Path, required=True)
    analysis_ingest_models = subparsers.add_parser(
        "analysis-ingest-models",
        help="validate and publish one receipt-bound R model bundle",
    )
    analysis_ingest_models.add_argument(
        "--kind", choices=("lp", "threshold-and-iv-audit"), required=True
    )
    analysis_ingest_models.add_argument("--data-root", type=Path, required=True)
    analysis_ingest_models.add_argument(
        "--output-root", type=Path, required=True
    )
    analysis_output_audit = subparsers.add_parser(
        "analysis-output-audit",
        help="audit publication outputs and write the bound success manifest",
    )
    analysis_output_audit.add_argument("--data-root", type=Path, required=True)
    analysis_output_audit.add_argument("--output-root", type=Path, required=True)
    analysis_run = subparsers.add_parser(
        "analysis-run",
        help="execute the frozen offline analysis command plan",
    )
    analysis_run.add_argument("--data-root", type=Path, required=True)
    analysis_run.add_argument("--output-root", type=Path, required=True)
    analysis_reproduce = subparsers.add_parser(
        "analysis-reproduce-check",
        help="run one capability-bound clean-room analysis and compare logical outputs",
    )
    analysis_reproduce.add_argument("--data-root", type=Path, required=True)
    analysis_reproduce.add_argument("--output-root", type=Path, required=True)
    cleanup_analysis_reproduce = subparsers.add_parser(
        "cleanup-analysis-reproduction",
        help="remove only the scratch directory bound by the formal analysis receipt",
    )
    cleanup_analysis_reproduce.add_argument("--receipt", type=Path, required=True)
    analysis_deps = subparsers.add_parser(
        "analysis-deps",
        help="plan, initialize, restore, or offline-verify the frozen R environment",
    )
    dependency_actions = analysis_deps.add_mutually_exclusive_group(required=True)
    dependency_actions.add_argument(
        "--print-plan",
        dest="dependency_action",
        action="store_const",
        const="print-plan",
    )
    dependency_actions.add_argument(
        "--initialize",
        dest="dependency_action",
        action="store_const",
        const="initialize",
    )
    dependency_actions.add_argument(
        "--restore",
        dest="dependency_action",
        action="store_const",
        const="restore",
    )
    dependency_actions.add_argument(
        "--verify",
        dest="dependency_action",
        action="store_const",
        const="verify",
    )
    analysis_deps.add_argument(
        "--route-exception-authorization",
        help=(
            "explicit user authorization recorded when an R dependency "
            "download uses the current proxy or VPN route"
        ),
    )
    network_preflight = subparsers.add_parser(
        "network-preflight",
        help="verify direct-only client, proxy bypass, route, and disk budget",
    )
    network_preflight.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    network_preflight.add_argument("--offline", action="store_true")
    network_preflight.add_argument("--host")
    network_preflight.add_argument("--no-download", action="store_true")
    night_direct = subparsers.add_parser(
        "night-direct",
        help="run project Python only after Shadowrocket/VPN is closed",
        description=(
            "Physical-direct gate for a user-managed download window. "
            "Close Shadowrocket/VPN first; this command never changes it."
        ),
    )
    night_actions = night_direct.add_subparsers(
        dest="night_direct_action",
        required=True,
    )
    night_check = night_actions.add_parser(
        "check",
        help="check one data URL without downloading a response body",
    )
    night_check.add_argument("url")
    night_check.add_argument(
        "--projected-bytes",
        type=int,
        default=0,
    )
    night_check.add_argument(
        "--hosts-config",
        type=Path,
        default=DEFAULT_DATA_HOST_CONFIG,
    )
    night_check.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    night_run = night_actions.add_parser(
        "run",
        help="check, then execute project Python with proxy variables removed",
    )
    night_run.add_argument("--url", required=True)
    night_run.add_argument(
        "--projected-bytes",
        type=int,
        required=True,
    )
    night_run.add_argument(
        "--hosts-config",
        type=Path,
        default=DEFAULT_DATA_HOST_CONFIG,
    )
    night_run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    night_run.add_argument(
        "python_arguments",
        nargs=argparse.REMAINDER,
        help="arguments for the fixed project .venv Python after --",
    )
    source_status = subparsers.add_parser(
        "source-status",
        help="show which registered public sources are enabled",
    )
    source_status.add_argument(
        "--sources-config",
        type=Path,
        default=DEFAULT_SOURCES_CONFIG,
    )
    batch_plan = subparsers.add_parser(
        "batch-plan",
        help="print a frozen batch without network access",
    )
    batch_plan.add_argument("batch")
    batch_plan.add_argument(
        "--sources-config",
        type=Path,
        default=DEFAULT_SOURCES_CONFIG,
    )
    acquire = subparsers.add_parser(
        "acquire",
        help="download one enabled public file through the physical-direct gate",
    )
    acquire.add_argument("source_id")
    acquire.add_argument("--dry-run", action="store_true")
    acquire.add_argument(
        "--sources-config",
        type=Path,
        default=DEFAULT_SOURCES_CONFIG,
    )
    acquire.add_argument(
        "--hosts-config",
        type=Path,
        default=DEFAULT_DATA_HOST_CONFIG,
    )
    acquire.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    acquire_batch = subparsers.add_parser(
        "acquire-batch",
        help="download a frozen batch after proxy-off and capacity checks",
    )
    acquire_batch.add_argument("batch")
    acquire_batch.add_argument("--dry-run", action="store_true")
    acquire_batch.add_argument(
        "--sources-config",
        type=Path,
        default=DEFAULT_SOURCES_CONFIG,
    )
    acquire_batch.add_argument(
        "--hosts-config",
        type=Path,
        default=DEFAULT_DATA_HOST_CONFIG,
    )
    acquire_batch.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    topic_audit = subparsers.add_parser(
        "openalex-topic-audit",
        help="validate the frozen OpenAlex topic registry without network access",
        parents=[data_root_parent],
    )
    topic_audit.add_argument(
        "--registry", type=Path, default=DEFAULT_OPENALEX_REGISTRY
    )
    openalex_acquire = subparsers.add_parser(
        "acquire-openalex-aggregates",
        help="acquire bounded OpenAlex country-year green and total counts",
    )
    openalex_acquire.add_argument(
        "--registry", type=Path, default=DEFAULT_OPENALEX_REGISTRY
    )
    openalex_acquire.add_argument("--output-root", type=Path, required=True)
    openalex_acquire.add_argument("--capacity-root", type=Path, default=PROJECT_ROOT)
    openalex_acquire.add_argument("--start", type=int, default=1996)
    openalex_acquire.add_argument("--end", type=int, default=2024)
    openalex_acquire.add_argument("--response-budget-gb", type=int, default=8)
    openalex_acquire.add_argument(
        "--api-base-url",
        default="https://api.openalex.org/works",
    )
    openalex_acquire.add_argument("--dry-run", action="store_true")
    initialization_acquire = subparsers.add_parser(
        "acquire-initialization-supplements",
        help="acquire only the frozen OpenAlex and two TiVA initialization files",
        parents=[data_root_parent],
    )
    initialization_acquire.add_argument("--dry-run", action="store_true")
    initialization_acquire.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    initialization_acquire.add_argument(
        "--hosts-config", type=Path, default=DEFAULT_DATA_HOST_CONFIG
    )
    initialization_acquire.add_argument(
        "--supplement-config", type=Path, default=DEFAULT_TIVA_SUPPLEMENTS
    )
    initialization_acquire.add_argument(
        "--registry", type=Path, default=DEFAULT_OPENALEX_REGISTRY
    )
    initialization_acquire.add_argument("--route-exception-authorization")
    initialization_audit = subparsers.add_parser(
        "audit-initialization-supplements",
        help="audit exact periods, units, hashes, routes, and raw partial files",
        parents=[data_root_parent],
    )
    initialization_audit.add_argument(
        "--supplement-config", type=Path, default=DEFAULT_TIVA_SUPPLEMENTS
    )
    initialization_audit.add_argument(
        "--registry", type=Path, default=DEFAULT_OPENALEX_REGISTRY
    )
    taxonomy_build = subparsers.add_parser(
        "build-taxonomy",
        help="build frozen HS07 memberships and weighted HS96 taxonomy registries",
        parents=[data_root_parent],
    )
    taxonomy_build.add_argument(
        "--construction-config",
        type=Path,
        default=DEFAULT_CONSTRUCTION_CONFIG,
    )
    subparsers.add_parser(
        "audit-taxonomy",
        help="audit taxonomy counts, mappings, weights, hashes, and duplicate keys",
        parents=[data_root_parent],
    )
    economy_build = subparsers.add_parser(
        "build-economy-crosswalk",
        help="build deterministic source-to-canonical economy mappings",
        parents=[data_root_parent],
    )
    economy_build.add_argument(
        "--overrides", type=Path, default=DEFAULT_ECONOMY_OVERRIDES
    )
    economy_build.add_argument(
        "--contract", type=Path, default=DEFAULT_ECONOMY_CONTRACT
    )
    economy_build.add_argument(
        "--construction-config", type=Path, default=DEFAULT_CONSTRUCTION_CONFIG
    )
    economy_audit = subparsers.add_parser(
        "audit-economies",
        help="audit economy key uniqueness, resolution, exclusions, and hub flags",
        parents=[data_root_parent],
    )
    economy_audit.add_argument(
        "--contract", type=Path, default=DEFAULT_ECONOMY_CONTRACT
    )
    normalize_baci = subparsers.add_parser(
        "normalize-baci",
        help="stream BACI ZIP members into bounded annual Parquet aggregates",
        parents=[data_root_parent],
    )
    normalize_baci.add_argument("--revision", choices=("HS96", "HS07"), required=True)
    subparsers.add_parser(
        "audit-trade-normalization",
        help="audit BACI periods, keys, weights, hashes, and no-expansion invariant",
        parents=[data_root_parent],
    )
    complexity_build = subparsers.add_parser(
        "build-complexity",
        help="construct annual all-product PCI and registry-filtered GPCI",
        parents=[data_root_parent],
    )
    complexity_build.add_argument("--taxonomy", required=True)
    trade_components_build = subparsers.add_parser(
        "build-trade-components",
        help="construct raw green import intensity, lagged complexity, and GNIR",
        parents=[data_root_parent],
    )
    trade_components_build.add_argument("--taxonomy", required=True)
    subparsers.add_parser(
        "build-gsci",
        help="construct complete-window green science capability",
        parents=[data_root_parent],
    )
    supplier_build = subparsers.add_parser(
        "build-supplier-raw",
        help="construct annual raw supplier capability without persisting product proximity",
        parents=[data_root_parent],
    )
    supplier_build.add_argument("--taxonomy", required=True)
    subparsers.add_parser(
        "audit-complexity",
        help="audit annual PCI orientation, finite values, and GPCI lag discipline",
        parents=[data_root_parent],
    )
    subparsers.add_parser(
        "normalize-wdi",
        help="normalize approved WDI responses with exact units and null semantics",
        parents=[data_root_parent],
    )
    subparsers.add_parser(
        "normalize-irena",
        help="normalize IRENA JSON-stat2 dimensions and capacity additions",
        parents=[data_root_parent],
    )
    subparsers.add_parser(
        "normalize-openalex",
        help="union and normalize the frozen 1992-2024 OpenAlex aggregates",
        parents=[data_root_parent],
    )
    subparsers.add_parser(
        "normalize-ilostat",
        help="extract ILOSTAT RDS tables and construct source-consistent shares",
        parents=[data_root_parent],
    )
    subparsers.add_parser(
        "normalize-policy",
        help="normalize robustness-only OECD EPS and IFCMA controls",
        parents=[data_root_parent],
    )
    subparsers.add_parser(
        "normalize-tiva",
        help="normalize all frozen TiVA files and baseline activity weights",
        parents=[data_root_parent],
    )
    subparsers.add_parser(
        "build-provisional-sample",
        help="build the outcome-free Checkpoint 1 sample and flow",
        parents=[data_root_parent],
    )
    subparsers.add_parser(
        "build-tiva-measures",
        help="construct frozen-weight TiVA raw measures and bounded robustness specification",
        parents=[data_root_parent],
    )
    subparsers.add_parser(
        "fit-gad-scaler",
        help="fit and freeze the outcome-free 2000-2004 robust GAD scaler",
        parents=[data_root_parent],
    )
    subparsers.add_parser(
        "apply-gad-scaler",
        help="apply the approved frozen GAD scaler after semantic sample verification",
        parents=[data_root_parent],
    )
    subparsers.add_parser(
        "verify-scaler",
        help="verify frozen scaler hash, sample lineage, columns, years, and finite scales",
        parents=[data_root_parent],
    )
    build_gad = subparsers.add_parser(
        "build-gad",
        help="construct every frozen GAD variant from the anchored scaled authority",
        parents=[data_root_parent],
    )
    build_gad.add_argument("--all-registered-variants", action="store_true", required=True)
    subparsers.add_parser(
        "build-outcomes",
        help="construct independent annual outcome bases and product RCA primitives",
        parents=[data_root_parent],
    )
    subparsers.add_parser(
        "audit-outcomes",
        help="audit outcome units, rawness, mappings, and mechanical-overlap lineage",
        parents=[data_root_parent],
    )
    instruments_build = subparsers.add_parser(
        "build-instruments",
        help="construct fixed-share leave-one-destination supply instruments",
        parents=[data_root_parent],
    )
    instruments_build.add_argument("--taxonomy", required=True)
    subparsers.add_parser(
        "audit-instruments",
        help="audit fixed shares, exclusion lineage, interactions, and CMZ",
        parents=[data_root_parent],
    )
    subparsers.add_parser(
        "freeze-samples",
        help="freeze final Core/Lite structural sample flags",
        parents=[data_root_parent],
    )
    subparsers.add_parser(
        "build-analysis-panels",
        help="construct timing-safe complete-case and bounded-control L4 panels",
        parents=[data_root_parent],
    )
    subparsers.add_parser(
        "audit-analysis-panels",
        help="independently audit L4 timing, mappings, overlap, and leakage",
        parents=[data_root_parent],
    )
    source_audit = subparsers.add_parser(
        "audit-source",
        help="audit one normalized source table against its frozen contract",
        parents=[data_root_parent],
    )
    source_audit.add_argument(
        "--source",
        choices=("wdi", "irena", "openalex", "ilostat", "policy", "tiva"),
        required=True,
    )
    raw_hash_snapshot = subparsers.add_parser(
        "raw-hash-snapshot",
        help="write a self-excluding SHA-256 snapshot of all immutable raw files",
        parents=[data_root_parent],
    )
    raw_hash_snapshot.add_argument("--output", type=Path, required=True)
    science_audit = subparsers.add_parser(
        "science-audit",
        help="validate downloaded OpenAlex country-year count output",
        parents=[data_root_parent],
    )
    science_audit.add_argument("--counts-csv", type=Path, required=True)
    raw_hash_verify = subparsers.add_parser(
        "raw-hash-verify",
        help="verify immutable raw files against a SHA-256 snapshot",
        parents=[data_root_parent],
    )
    raw_hash_verify.add_argument("--snapshot", type=Path, required=True)
    subparsers.add_parser(
        "capacity-report",
        help="report separate raw, layered, scratch, and audit byte counts",
        parents=[data_root_parent],
    )
    verify_artifacts = subparsers.add_parser(
        "verify-artifacts",
        help="verify every authoritative manifest in one construction layer",
        parents=[data_root_parent],
    )
    verify_artifacts.add_argument(
        "--layer",
        choices=("normalized", "harmonized", "measures", "analysis"),
        required=True,
    )
    checkpoint = subparsers.add_parser(
        "checkpoint",
        help="run one frozen construction review gate and write its receipt",
        parents=[data_root_parent],
    )
    checkpoint.add_argument("--number", type=int, choices=(1, 2, 3), required=True)
    checkpoint.add_argument(
        "--verify-only",
        action="store_true",
        help="validate an existing checkpoint receipt without rewriting evidence",
    )
    build = subparsers.add_parser(
        "build",
        help="verify current nodes and rebuild only stale descendants",
        parents=[data_root_parent],
    )
    build.add_argument(
        "--target", choices=("checkpoint1", "checkpoint2", "checkpoint3"), required=True
    )
    subparsers.add_parser(
        "build-status",
        help="explain current, stale, or blocked state for every construction node",
        parents=[data_root_parent],
    )
    reproduce = subparsers.add_parser(
        "reproduce-check",
        help="compare deterministic canonical fingerprints in an exact scratch child",
        parents=[data_root_parent],
    )
    reproduce.add_argument("--scratch-root", type=Path, required=True)
    cleanup = subparsers.add_parser(
        "cleanup-reproduction",
        help="remove only a receipt-bound reproduce.* scratch child",
    )
    cleanup.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analysis-deps":
        if args.dependency_action == "print-plan":
            policy = load_data_host_policy(DEFAULT_DEPENDENCY_HOST_CONFIG)
            if policy.allowed_hosts != R_DEPENDENCY_HOSTS:
                raise ValueError("analysis dependency host policy drift")
            print(
                json.dumps(
                    {
                        "downloads_response_bodies": False,
                        "hosts": list(policy.allowed_hosts),
                        "projected_working_bytes": R_PROJECTED_WORKING_BYTES,
                        "status": "plan_only",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.dependency_action in {"initialize", "restore"}:
            receipt = run_r_dependency_action(
                gate=_night_gate(
                    DEFAULT_CONFIG,
                    DEFAULT_DEPENDENCY_HOST_CONFIG,
                    project_root=PROJECT_ROOT,
                ),
                action=args.dependency_action,
                project_root=PROJECT_ROOT,
                rscript=_rscript_path(),
                route_exception_authorization=(
                    args.route_exception_authorization
                ),
            )
            print(
                json.dumps(
                    {**asdict(receipt), "status": "complete"},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.dependency_action == "verify":
            receipt = run_r_dependency_verification(
                project_root=PROJECT_ROOT,
                rscript=_rscript_path(),
            )
            print(
                json.dumps(
                    {**asdict(receipt), "status": "valid"},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        raise RuntimeError(
            f"unsupported analysis dependency action: {args.dependency_action}"
        )
    if args.command == "analysis-preflight":
        report = analysis_preflight(
            code_root=PROJECT_ROOT,
            data_root=args.data_root,
            output_root=args.output_root,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "analysis-diagnostics":
        analysis_preflight(
            code_root=PROJECT_ROOT,
            data_root=args.data_root,
            output_root=args.output_root,
        )
        spec = load_analysis_spec(PROJECT_ROOT / "config/analysis.yaml")
        authority = validate_analysis_authority(PROJECT_ROOT, args.data_root)
        context = build_run_context(spec, authority, PROJECT_ROOT)
        context = bind_analysis_reproduction_context(
            code_root=PROJECT_ROOT,
            data_root=args.data_root,
            output_root=args.output_root,
            context=context,
        )
        bundle = run_stage_a_diagnostics(
            AnalysisPaths(
                code_root=PROJECT_ROOT,
                data_root=args.data_root.resolve(),
                output_root=args.output_root.resolve(),
            ),
            spec,
            context,
        )
        print(
            json.dumps(
                {
                    "status": "valid",
                    "run_id": context.run_id,
                    "cell_count": bundle.cell_count,
                    "table_paths": [str(path) for path in bundle.table_paths],
                    "gate_path": str(bundle.gate_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "analysis-ingest-models":
        analysis_preflight(
            code_root=PROJECT_ROOT,
            data_root=args.data_root,
            output_root=args.output_root,
        )
        output_root = resolve_authorized_analysis_output(
            PROJECT_ROOT, args.data_root, args.output_root
        )
        spec = load_analysis_spec(PROJECT_ROOT / "config/analysis.yaml")
        context = load_stage_a_run_context(output_root)
        if context.spec_id != spec.spec_id or context.seed != spec.seed:
            raise ValueError("stage A run context does not match analysis spec")
        published = ingest_model_bundle(
            output_root / "_staging" / context.run_id / args.kind,
            output_root / "models",
            context,
            spec,
        )
        print(
            json.dumps(
                {
                    "status": "valid",
                    "kind": args.kind,
                    "run_id": context.run_id,
                    "table_paths": [str(path) for path in published],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "analysis-output-audit":
        payload = publish_analysis_audit(
            code_root=PROJECT_ROOT,
            data_root=args.data_root,
            output_root=args.output_root,
        )
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "analysis-run":
        run_analysis_command_plan(
            code_root=PROJECT_ROOT,
            data_root=args.data_root,
            output_root=args.output_root,
        )
        return 0
    if args.command == "analysis-reproduce-check":
        receipt = run_analysis_reproduction_check(
            code_root=PROJECT_ROOT,
            data_root=args.data_root,
            formal_output_root=args.output_root,
        )
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "status": "matched",
                    "receipt": str(receipt.resolve()),
                    "execution_id": receipt_payload["execution_id"],
                    "formal_run_id": receipt_payload["formal_run_id"],
                    "scratch_root": receipt_payload["scratch_root"],
                    "max_scaled_float_error": receipt_payload[
                        "comparison_results"
                    ]["max_scaled_float_error"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "cleanup-analysis-reproduction":
        removed = cleanup_analysis_reproduction(args.receipt)
        print(
            json.dumps(
                {"removed_path": str(removed), "status": "cleaned"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build":
        graph = registered_build_graph(
            code_root=PROJECT_ROOT,
            data_root=args.data_root.resolve(),
        )
        controller = BuildController(
            graph,
            code_root=PROJECT_ROOT,
            data_root=args.data_root.resolve(),
            implementation_commit=_code_version(),
        )
        report = controller.build(args.target)
        print(
            json.dumps(
                {
                    "target": args.target,
                    "nodes": [asdict(item) for item in report],
                    "status": "valid",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-status":
        graph = registered_build_graph(
            code_root=PROJECT_ROOT,
            data_root=args.data_root.resolve(),
        )
        controller = BuildController(
            graph,
            code_root=PROJECT_ROOT,
            data_root=args.data_root.resolve(),
            implementation_commit=_code_version(),
        )
        report = controller.status("checkpoint3")
        print(
            json.dumps(
                {
                    "target": "checkpoint3",
                    "nodes": [asdict(item) for item in report],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "reproduce-check":
        receipt = run_reproduction_check(
            code_root=PROJECT_ROOT,
            data_root=args.data_root.resolve(),
            scratch_root=args.scratch_root,
        )
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "receipt": str(receipt.resolve()),
                    "artifact_count": payload["artifact_count"],
                    "raw_tree_sha256": payload["raw_tree_sha256_after"],
                    "status": "matched",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "cleanup-reproduction":
        removed = cleanup_reproduction(args.receipt)
        print(
            json.dumps(
                {"removed_path": str(removed), "status": "removed"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-taxonomy":
        report = build_taxonomy(
            code_root=PROJECT_ROOT,
            data_root=args.data_root.resolve(),
            construction_config_path=args.construction_config,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "audit-taxonomy":
        report = audit_taxonomy(
            code_root=PROJECT_ROOT,
            data_root=args.data_root.resolve(),
            write_audit=True,
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "build-economy-crosswalk":
        construction = load_construction_config(args.construction_config)
        report = build_production_economy_crosswalk(
            code_root=PROJECT_ROOT,
            data_root=args.data_root.resolve(),
            overrides_path=args.overrides,
            contract_path=args.contract,
            minimum_population=construction.sample.minimum_population,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "audit-economies":
        report = audit_economies(
            code_root=PROJECT_ROOT,
            contract_path=args.contract,
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "normalize-openalex":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        data_paths = (
            paths.raw
            / "openalex/20260823/works_aggregate/"
            "openalex_country_year_counts_1992_1995.csv",
            paths.raw
            / "openalex/20260822/works_aggregate/"
            "openalex_country_year_counts_1996_2024.csv",
        )
        crosswalk_path = PROJECT_ROOT / "02_数据字典/economy_crosswalk_v1.csv"
        economies = pl.read_csv(
            crosswalk_path,
            schema_overrides={"source_code": pl.String},
            null_values="",
        ).filter(pl.col("source_id") == "openalex")
        destination = paths.normalized / "openalex/openalex_country_year.parquet"
        input_paths = (*data_paths, DEFAULT_OPENALEX_REGISTRY, crosswalk_path)
        report = build_openalex_table(
            data_paths=data_paths,
            economies=economies,
            destination=destination,
            contract_path=DEFAULT_OPENALEX_CONTRACT,
            inputs=tuple(InputArtifact.from_path(path) for path in input_paths),
            build=BuildIdentity(
                command=(
                    "python -m green_debt.cli normalize-openalex "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "normalize-ilostat":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        source_root = paths.raw / "ilostat/20260821"
        primary_path = source_root / "data/EMP_TEMP_SEX_OC2_NB_A.rds"
        icls19_path = source_root / "data/EMP_5EMP_SEX_OC2_NB_A.rds"
        stem_path = source_root / "data/EMP_STEM_SEX_OC2_NB_A.rds"
        dictionary_paths = (
            source_root / "dictionaries/classif1_en.rds",
            source_root / "dictionaries/ref_area_en.rds",
            source_root / "dictionaries/sex_en.rds",
            source_root / "metadata/table_of_contents_en.rds",
        )
        r_script = PROJECT_ROOT / "03_代码/R/extract_ilostat_rds.R"
        crosswalk_path = PROJECT_ROOT / "02_数据字典/economy_crosswalk_v1.csv"
        economies = pl.read_csv(
            crosswalk_path,
            schema_overrides={"source_code": pl.String},
            null_values="",
        ).filter(pl.col("source_id") == "ilostat")
        input_paths = (
            primary_path,
            icls19_path,
            stem_path,
            *dictionary_paths,
            r_script,
            DEFAULT_INDICATOR_REGISTRY,
            crosswalk_path,
        )
        destination = paths.normalized / "ilostat/ilostat_skill.parquet"
        report = build_ilostat_table(
            primary_path=primary_path,
            icls19_path=icls19_path,
            stem_path=stem_path,
            r_script=r_script,
            temporary_directory=paths.scratch / "ilostat",
            economies=economies,
            destination=destination,
            contract_path=DEFAULT_ILOSTAT_CONTRACT,
            inputs=tuple(InputArtifact.from_path(path) for path in input_paths),
            build=BuildIdentity(
                command=(
                    "python -m green_debt.cli normalize-ilostat "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "normalize-policy":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        eps_root = paths.raw / "oecd_eps/20260821"
        eps_composite = (
            eps_root / "data/EPS_composite_all_countries_1990-2020.csv"
        )
        eps_components = (
            eps_root / "data/EPS_all_policy_components_all_countries_1990-2020.csv"
        )
        eps_structure = (
            eps_root / "metadata/DSD_EPS_DF_EPS.structure.json"
        )
        ifcma_path = (
            paths.raw
            / "oecd_ifcma/202604/"
            "IFCMA_ClimatePolicyDatabase_Data_April_2026.csv"
        )
        crosswalk_path = PROJECT_ROOT / "02_数据字典/economy_crosswalk_v1.csv"
        crosswalk = pl.read_csv(
            crosswalk_path,
            schema_overrides={"source_code": pl.String},
            null_values="",
        )
        eps_economies = crosswalk.filter(pl.col("source_id") == "oecd_eps")
        ifcma_economies = crosswalk.filter(pl.col("source_id") == "oecd_ifcma")
        input_paths = (
            eps_composite,
            eps_components,
            eps_structure,
            ifcma_path,
            DEFAULT_INDICATOR_REGISTRY,
            crosswalk_path,
        )
        destination = paths.normalized / "policy/policy_country_year.parquet"
        report = build_policy_table(
            eps_composite_path=eps_composite,
            eps_components_path=eps_components,
            ifcma_path=ifcma_path,
            eps_economies=eps_economies,
            ifcma_economies=ifcma_economies,
            destination=destination,
            contract_path=DEFAULT_POLICY_CONTRACT,
            inputs=tuple(InputArtifact.from_path(path) for path in input_paths),
            build=BuildIdentity(
                command=(
                    "python -m green_debt.cli normalize-policy "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "normalize-tiva":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        specs = production_tiva_specs(paths.raw)
        structure_workbook = (
            paths.raw
            / "oecd_tiva/20260821/documentation/"
            "TiVA_2025_structure_coverage.xlsx"
        )
        crosswalk_path = PROJECT_ROOT / "02_数据字典/economy_crosswalk_v1.csv"
        economies = pl.read_csv(
            crosswalk_path,
            schema_overrides={"source_code": pl.String},
            null_values="",
        ).filter(pl.col("source_id") == "oecd_tiva")
        construction = load_construction_config(DEFAULT_CONSTRUCTION_CONFIG)
        confirmatory = construction.tiva.confirmatory_activities
        broad = tuple(
            dict.fromkeys((*confirmatory, *construction.tiva.broad_additions))
        )
        equipment_only = tuple(
            activity
            for activity in confirmatory
            if activity not in set(construction.tiva.equipment_only_removals)
        )
        input_paths = tuple(
            dict.fromkeys(
                (
                    *(spec.path for spec in specs),
                    structure_workbook,
                    crosswalk_path,
                    DEFAULT_INDICATOR_REGISTRY,
                    DEFAULT_CONSTRUCTION_CONFIG,
                )
            )
        )
        destination = paths.normalized / "tiva/tiva_activity_year.parquet"
        weights_destination = paths.harmonized / "tiva/tiva_activity_weights.parquet"
        report = build_tiva_tables(
            specs=specs,
            structure_workbook=structure_workbook,
            economies=economies,
            destination=destination,
            contract_path=DEFAULT_TIVA_CONTRACT,
            weights_destination=weights_destination,
            weights_contract_path=DEFAULT_TIVA_WEIGHTS_CONTRACT,
            weight_variants={
                "confirmatory_prod_weight": confirmatory,
                "broad_prod_weight": broad,
                "equipment_only_prod_weight": equipment_only,
            },
            equal_variant_activities=confirmatory,
            inputs=tuple(InputArtifact.from_path(path) for path in input_paths),
            weight_inputs=(InputArtifact.from_path(DEFAULT_CONSTRUCTION_CONFIG),),
            build=BuildIdentity(
                command=(
                    "python -m green_debt.cli normalize-tiva "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "build-tiva-measures":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        source_path = paths.normalized / "tiva/tiva_activity_year.parquet"
        destination = paths.measures / "tiva/tiva_measures.parquet"
        bounded_destination = paths.measures / "tiva/tiva_bounded_robustness.parquet"
        result = build_tiva_measures(pl.read_parquet(source_path), _tiva_activity_sets())
        manifest = write_authoritative_table(
            result.table,
            _load_table_contract(DEFAULT_TIVA_MEASURES_CONTRACT),
            destination,
            (
                InputArtifact.from_path(source_path),
                InputArtifact.from_path(DEFAULT_CONSTRUCTION_CONFIG),
            ),
            BuildIdentity(
                command=(
                    "python -m green_debt.cli build-tiva-measures "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        bounded_manifest = write_authoritative_table(
            result.bounded_confirmatory,
            _load_table_contract(DEFAULT_TIVA_BOUNDED_CONTRACT),
            bounded_destination,
            (
                InputArtifact.from_path(source_path),
                InputArtifact.from_path(DEFAULT_CONSTRUCTION_CONFIG),
            ),
            BuildIdentity(
                command=(
                    "python -m green_debt.cli build-tiva-measures "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        audit_path = paths.audits / "TiVA行业权重审计_v1.csv"
        _write_text_atomic(audit_path, result.weights.write_csv())
        print(
            json.dumps(
                {
                    "rows": manifest.rows,
                    "bounded_rows": bounded_manifest.rows,
                    "gfvad_years": [1995, 2022],
                    "construction_use_years": [1997, 2022],
                    "weight_groups": result.weights.group_by("economy_id", "weight_version").len().height,
                    "invalid_gfvad_denominators": result.table.filter(
                        pl.col("gfvad_missing_reason") == "invalid_fd_va_denominator"
                    ).height,
                    "gfvad_out_of_range": result.table.filter(pl.col("gfvad_out_of_range")).height,
                    "dvashare_out_of_range": result.table.filter(pl.col("dvashare_out_of_range")).height,
                    "output_path": str(destination),
                    "bounded_output_path": str(bounded_destination),
                    "audit_path": str(audit_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "fit-gad-scaler":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        authority, input_paths = _build_scaler_matched_frame(paths)
        sample_manifest = input_paths[0].with_name(f"{input_paths[0].name}.manifest.json")
        verify_manifest(sample_manifest)
        baseline = authority.filter(
            pl.col("year").is_between(2000, 2004)
            & pl.all_horizontal(
                [
                    pl.col(column).is_not_null() & pl.col(column).is_finite()
                    for column in FROZEN_COMPONENT_COLUMNS
                ]
            )
        )
        registry = fit_scaler(
            baseline,
            FROZEN_COMPONENT_COLUMNS,
            (2000, 2004),
            provisional_sample_manifest_hash=sha256_file(sample_manifest),
        )
        scaled = apply_scaler(authority, registry)
        implementation_commit = _code_version()
        registry_path = paths.audits / "GAD固定缩放器_v1.json"
        _write_text_atomic(
            registry_path,
            json.dumps(registry.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        anchor_path = paths.audits / "GAD固定缩放器_v1.manifest.json"
        anchor = create_scaler_anchor(
            registry,
            registry_file_sha256=sha256_file(registry_path),
            implementation_commit=implementation_commit,
        )
        _write_text_atomic(
            anchor_path,
            json.dumps(anchor.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        destination = paths.measures / "gad/gad_scaled_components.parquet"
        manifest = write_authoritative_table(
            scaled,
            _load_table_contract(DEFAULT_SCALER_COMPONENTS_CONTRACT),
            destination,
            (
                *(InputArtifact.from_path(path) for path in input_paths),
                InputArtifact.from_path(registry_path),
                InputArtifact.from_path(anchor_path),
            ),
            BuildIdentity(
                command=(
                    "python -m green_debt.cli fit-gad-scaler "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=implementation_commit,
            ),
        )
        print(
            json.dumps(
                {
                    "canonical_hash": registry.canonical_hash,
                    "matched_rows": baseline.height,
                    "matched_economies": baseline.get_column("economy_id").n_unique(),
                    "authority_rows": authority.height,
                    "sample_manifest_hash": registry.provisional_sample_manifest_hash,
                    "scaled_output": str(destination),
                    "scaled_rows": manifest.rows,
                    "registry_path": str(registry_path),
                    "anchor_path": str(anchor_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "apply-gad-scaler":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        authority, input_paths = _build_scaler_matched_frame(paths)
        sample_manifest = input_paths[0].with_name(f"{input_paths[0].name}.manifest.json")
        verify_manifest(sample_manifest)
        registry_path = paths.audits / "GAD固定缩放器_v1.json"
        anchor_path = paths.audits / "GAD固定缩放器_v1.manifest.json"
        registry = registry_from_dict(json.loads(registry_path.read_text(encoding="utf-8")))
        anchor = scaler_anchor_from_dict(json.loads(anchor_path.read_text(encoding="utf-8")))
        _verify_frozen_scaler_for_current_sample(
            registry=registry,
            anchor=anchor,
            registry_path=registry_path,
            anchor_path=anchor_path,
            sample_manifest_path=sample_manifest,
            semantic_anchor_path=DEFAULT_SCALER_SAMPLE_SEMANTICS,
        )
        scaled = apply_scaler(authority, registry)
        destination = paths.measures / "gad/gad_scaled_components.parquet"
        manifest = write_authoritative_table(
            scaled,
            _load_table_contract(DEFAULT_SCALER_COMPONENTS_CONTRACT),
            destination,
            (
                *(InputArtifact.from_path(path) for path in input_paths),
                InputArtifact.from_path(registry_path),
                InputArtifact.from_path(anchor_path),
                InputArtifact.from_path(DEFAULT_SCALER_SAMPLE_SEMANTICS),
            ),
            BuildIdentity(
                command=(
                    "python -m green_debt.cli apply-gad-scaler "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(
            json.dumps(
                {
                    "action": "applied_existing_frozen_scaler",
                    "canonical_hash": registry.canonical_hash,
                    "approved_sample_manifest_hash": registry.provisional_sample_manifest_hash,
                    "current_sample_manifest_hash": sha256_file(sample_manifest),
                    "semantic_anchor_path": str(DEFAULT_SCALER_SAMPLE_SEMANTICS),
                    "scaled_output": str(destination),
                    "scaled_rows": manifest.rows,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "verify-scaler":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        sample_path = paths.harmonized / "sample/provisional_sample.parquet"
        sample_manifest = sample_path.with_name(f"{sample_path.name}.manifest.json")
        verify_manifest(sample_manifest)
        registry_path = paths.audits / "GAD固定缩放器_v1.json"
        anchor_path = paths.audits / "GAD固定缩放器_v1.manifest.json"
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        registry = registry_from_dict(payload)
        anchor = scaler_anchor_from_dict(json.loads(anchor_path.read_text(encoding="utf-8")))
        _verify_frozen_scaler_for_current_sample(
            registry=registry,
            anchor=anchor,
            registry_path=registry_path,
            anchor_path=anchor_path,
            sample_manifest_path=sample_manifest,
            semantic_anchor_path=DEFAULT_SCALER_SAMPLE_SEMANTICS,
        )
        scaled_path = paths.measures / "gad/gad_scaled_components.parquet"
        scaled_manifest = verify_manifest(
            scaled_path.with_name(f"{scaled_path.name}.manifest.json")
        )
        _verify_scaled_manifest_binding(
            scaled_manifest,
            registry_path,
            anchor_path,
            scaled_manifest.code_commit,
            DEFAULT_SCALER_SAMPLE_SEMANTICS,
        )
        verify_scaled_authority(pl.read_parquet(scaled_path), registry)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "canonical_hash": registry.canonical_hash,
                    "sample_manifest_hash": registry.provisional_sample_manifest_hash,
                    "scaled_rows": scaled_manifest.rows,
                    "scaled_code_commit": scaled_manifest.code_commit,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-gad":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        sample_path = paths.harmonized / "sample/provisional_sample.parquet"
        scaled_path = paths.measures / "gad/gad_scaled_components.parquet"
        sample_manifest = sample_path.with_name(f"{sample_path.name}.manifest.json")
        scaled_manifest_path = scaled_path.with_name(f"{scaled_path.name}.manifest.json")
        registry_path = paths.audits / "GAD固定缩放器_v1.json"
        anchor_path = paths.audits / "GAD固定缩放器_v1.manifest.json"
        verify_manifest(sample_manifest)
        scaled_manifest = verify_manifest(scaled_manifest_path)
        registry = registry_from_dict(json.loads(registry_path.read_text(encoding="utf-8")))
        anchor = scaler_anchor_from_dict(json.loads(anchor_path.read_text(encoding="utf-8")))
        _verify_frozen_scaler_for_current_sample(
            registry=registry,
            anchor=anchor,
            registry_path=registry_path,
            anchor_path=anchor_path,
            sample_manifest_path=sample_manifest,
            semantic_anchor_path=DEFAULT_SCALER_SAMPLE_SEMANTICS,
        )
        _verify_scaled_manifest_binding(
            scaled_manifest,
            registry_path,
            anchor_path,
            scaled_manifest.code_commit,
            DEFAULT_SCALER_SAMPLE_SEMANTICS,
        )
        scaled = pl.read_parquet(scaled_path)
        verify_scaled_authority(scaled, registry)
        authority, initialization = build_gad_variants(
            scaled,
            pl.read_parquet(sample_path),
            scaler_hash=registry.canonical_hash,
        )
        destination = paths.measures / "gad/gad_country_year.parquet"
        manifest = write_authoritative_table(
            gad_output_table(authority),
            _load_table_contract(DEFAULT_GAD_CONTRACT),
            destination,
            (
                InputArtifact.from_path(scaled_path),
                InputArtifact.from_path(sample_path),
                InputArtifact.from_path(registry_path),
                InputArtifact.from_path(anchor_path),
                InputArtifact.from_path(DEFAULT_CONSTRUCTION_CONFIG),
                InputArtifact.from_path(DEFAULT_GAD_CONTRACT),
                InputArtifact.from_path(DEFAULT_GAD_FROZEN_UNIVERSE),
            ),
            BuildIdentity(
                command=(
                    "python -m green_debt.cli build-gad --all-registered-variants "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(
            json.dumps(
                {
                    "rows": manifest.rows,
                    "economies": authority.get_column("economy_id").n_unique(),
                    "specifications": [item.specification_id for item in registered_gad_specifications()],
                    "scaler_hash": registry.canonical_hash,
                    "output_path": str(destination),
                    "output_sha256": manifest.output_sha256,
                    "initialization_economies": initialization.height,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-outcomes":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        report = build_outcome_tables(
            paths,
            build=BuildIdentity(
                command=(
                    "python -m green_debt.cli build-outcomes "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "audit-outcomes":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        report = audit_outcomes(
            paths,
            audit_path=paths.audits / "结果变量覆盖审计_v1.csv",
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "build-instruments":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        report = build_instrument_artifacts(
            paths,
            taxonomy=args.taxonomy,
            build=BuildIdentity(
                command=(
                    "python -m green_debt.cli build-instruments "
                    f"--taxonomy {args.taxonomy} --data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "audit-instruments":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        report = audit_instrument_artifacts(
            paths,
            audit_path=paths.audits / "工具变量构造审计_v1.csv",
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "freeze-samples":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        report = freeze_final_sample_authority(
            paths,
            build=BuildIdentity(
                command=(
                    "python -m green_debt.cli freeze-samples "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "build-analysis-panels":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        report = build_analysis_panel_authority(
            paths,
            build=BuildIdentity(
                command=(
                    "python -m green_debt.cli build-analysis-panels "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "audit-analysis-panels":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        report = audit_analysis_panel_authority(
            paths,
            audit_path=paths.audits / "面板时序与重叠审计_v1.csv",
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "build-provisional-sample":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        crosswalk_path = PROJECT_ROOT / "02_数据字典/economy_crosswalk_v1.csv"
        crosswalk = pl.read_csv(
            crosswalk_path,
            schema_overrides={"source_code": pl.String},
            null_values="",
        )
        wdi_path = paths.normalized / "wdi/wdi_country_year.parquet"
        openalex_path = paths.normalized / "openalex/openalex_country_year.parquet"
        tiva_path = paths.normalized / "tiva/tiva_activity_year.parquet"
        totals_paths = tuple(
            paths.normalized
            / "baci/economy_year_totals"
            / f"year={year}"
            / "taxonomy_version=all_hs96.parquet"
            for year in range(1996, 2000)
        )
        green_paths = tuple(
            paths.normalized
            / "baci/green_economy_product"
            / f"year={year}"
            / "taxonomy_version=main_hs96.parquet"
            for year in range(1996, 2000)
        )
        coverage = build_source_coverage(
            crosswalk=crosswalk,
            wdi=pl.read_parquet(wdi_path),
            baci_totals=pl.concat(
                [pl.read_parquet(path) for path in totals_paths]
            ),
            green_imports=pl.concat(
                [pl.read_parquet(path) for path in green_paths]
            ),
            openalex=pl.read_parquet(openalex_path),
            tiva=pl.read_parquet(tiva_path),
        )
        input_paths = (
            crosswalk_path,
            wdi_path,
            openalex_path,
            tiva_path,
            *totals_paths,
            *green_paths,
            DEFAULT_CONSTRUCTION_CONFIG,
        )
        construction = load_construction_config(DEFAULT_CONSTRUCTION_CONFIG)
        destination = paths.harmonized / "sample/provisional_sample.parquet"
        flow_path = paths.audits / "检查点1_样本流_v1.csv"
        report = build_provisional_sample_table(
            coverage=coverage,
            destination=destination,
            contract_path=DEFAULT_PROVISIONAL_SAMPLE_CONTRACT,
            sample_flow_path=flow_path,
            inputs=tuple(InputArtifact.from_path(path) for path in input_paths),
            build=BuildIdentity(
                command=(
                    "python -m green_debt.cli build-provisional-sample "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
            minimum_population=construction.sample.minimum_population,
            minimum_positive_baseline_years=(
                construction.sample.minimum_positive_import_baseline_years
            ),
        )
        audit_provisional_sample(
            manifest_path=destination.with_name(f"{destination.name}.manifest.json"),
            sample_flow_path=flow_path,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "normalize-wdi":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        source_root = paths.raw / "wdi/20260821"
        registry = pl.read_csv(DEFAULT_INDICATOR_REGISTRY)
        approved = registry.filter(
            (pl.col("source_id") == "wdi")
            & ~pl.col("status").str.starts_with("rejected")
        ).sort("source_field")
        data_paths = tuple(
            source_root / f"{row['source_field']}.1996-2024.json"
            for row in approved.iter_rows(named=True)
        )
        metadata_paths = tuple(
            source_root / "metadata" / f"{row['source_field']}.metadata.json"
            for row in approved.iter_rows(named=True)
        )
        crosswalk_path = PROJECT_ROOT / "02_数据字典/economy_crosswalk_v1.csv"
        economies = pl.read_csv(
            crosswalk_path,
            schema_overrides={"source_code": pl.String},
            null_values="",
        ).filter(pl.col("source_id") == "wdi")
        input_paths = (
            *data_paths,
            *metadata_paths,
            DEFAULT_INDICATOR_REGISTRY,
            crosswalk_path,
        )
        destination = paths.normalized / "wdi/wdi_country_year.parquet"
        report = build_wdi_table(
            data_paths=data_paths,
            metadata_paths=metadata_paths,
            registry=registry,
            economies=economies,
            destination=destination,
            contract_path=DEFAULT_WDI_CONTRACT,
            inputs=tuple(InputArtifact.from_path(path) for path in input_paths),
            build=BuildIdentity(
                command=(
                    "python -m green_debt.cli normalize-wdi "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "normalize-irena":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        source_root = paths.raw / "irena/20260821"
        data_directory = source_root / "data"
        capacity_paths = tuple(
            sorted(data_directory.glob("Country_ELECCAP_2026_H1.*.jsonstat2.json"))
        )
        generation_paths = tuple(
            sorted(data_directory.glob("Country_ELECGEN_2025_H2.*.jsonstat2.json"))
        )
        share_paths = tuple(
            sorted(data_directory.glob("RE-SHARE_2026_H1.*.jsonstat2.json"))
        )
        capacity_metadata = (
            source_root / "metadata/Country_ELECCAP_2026_H1.metadata.json"
        )
        generation_metadata = (
            source_root / "metadata/Country_ELECGEN_2025_H2.metadata.json"
        )
        share_metadata = source_root / "metadata/RE-SHARE_2026_H1.metadata.json"
        registry = pl.read_csv(DEFAULT_INDICATOR_REGISTRY)
        crosswalk_path = PROJECT_ROOT / "02_数据字典/economy_crosswalk_v1.csv"
        economies = pl.read_csv(
            crosswalk_path,
            schema_overrides={"source_code": pl.String},
            null_values="",
        ).filter(pl.col("source_id") == "irena")
        input_paths = (
            *capacity_paths,
            *generation_paths,
            *share_paths,
            capacity_metadata,
            generation_metadata,
            share_metadata,
            DEFAULT_INDICATOR_REGISTRY,
            crosswalk_path,
        )
        destination = paths.normalized / "irena/irena_country_year.parquet"
        report = build_irena_table(
            capacity_paths=capacity_paths,
            generation_paths=generation_paths,
            share_paths=share_paths,
            capacity_metadata_path=capacity_metadata,
            generation_metadata_path=generation_metadata,
            share_metadata_path=share_metadata,
            registry=registry,
            economies=economies,
            destination=destination,
            contract_path=DEFAULT_IRENA_CONTRACT,
            inputs=tuple(InputArtifact.from_path(path) for path in input_paths),
            build=BuildIdentity(
                command=(
                    "python -m green_debt.cli normalize-irena "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "audit-source":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        registry = pl.read_csv(DEFAULT_INDICATOR_REGISTRY)
        if args.source == "wdi":
            destination = paths.normalized / "wdi/wdi_country_year.parquet"
            report = audit_wdi_table(
                manifest_path=destination.with_name(
                    f"{destination.name}.manifest.json"
                ),
                registry=registry,
            )
        elif args.source == "irena":
            destination = paths.normalized / "irena/irena_country_year.parquet"
            report = audit_irena_table(
                manifest_path=destination.with_name(
                    f"{destination.name}.manifest.json"
                ),
                registry=registry,
            )
        elif args.source == "openalex":
            destination = paths.normalized / "openalex/openalex_country_year.parquet"
            report = audit_openalex_table(
                manifest_path=destination.with_name(
                    f"{destination.name}.manifest.json"
                )
            )
        elif args.source == "ilostat":
            destination = paths.normalized / "ilostat/ilostat_skill.parquet"
            report = audit_ilostat_table(
                manifest_path=destination.with_name(
                    f"{destination.name}.manifest.json"
                )
            )
        elif args.source == "policy":
            destination = paths.normalized / "policy/policy_country_year.parquet"
            report = audit_policy_table(
                manifest_path=destination.with_name(
                    f"{destination.name}.manifest.json"
                )
            )
        else:
            destination = paths.normalized / "tiva/tiva_activity_year.parquet"
            weights = paths.harmonized / "tiva/tiva_activity_weights.parquet"
            report = audit_tiva_tables(
                activity_manifest_path=destination.with_name(
                    f"{destination.name}.manifest.json"
                ),
                weights_manifest_path=weights.with_name(
                    f"{weights.name}.manifest.json"
                ),
            )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "normalize-baci":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        revision = args.revision.upper()
        archive = (
            paths.raw
            / "baci/202601"
            / f"BACI_{revision}_V202601.zip"
        )
        crosswalk_path = PROJECT_ROOT / "02_数据字典/economy_crosswalk_v1.csv"
        crosswalk = pl.read_csv(
            crosswalk_path,
            schema_overrides={"source_code": pl.String},
            null_values="",
        )
        economies = crosswalk.filter(pl.col("source_id") == "baci").select(
            "source_code", "economy_id", "exclusion_reason"
        )
        if revision == "HS96":
            taxonomy_path = PROJECT_ROOT / "02_数据字典/product_registry_hs96_v1.parquet"
            taxonomy = (
                pl.read_parquet(taxonomy_path)
                .filter(pl.col("green_weight") > 0.0)
                .select(
                    pl.col("hs96").alias("hs6"),
                    "green_weight",
                    (pl.col("list_name") + pl.lit("_hs96")).alias(
                        "taxonomy_version"
                    ),
                )
            )
        else:
            taxonomy_path = PROJECT_ROOT / "02_数据字典/product_registry_hs07_v1.csv"
            taxonomy = (
                pl.read_csv(
                    taxonomy_path,
                    schema_overrides={"hs07": pl.String},
                    null_values="",
                )
                .filter(pl.col("list_name") == "main")
                .select(
                    pl.col("hs07").alias("hs6"),
                    pl.lit(1.0, dtype=pl.Float64).alias("green_weight"),
                    pl.lit("hs07_native", dtype=pl.String).alias(
                        "taxonomy_version"
                    ),
                )
            )
        inputs = (
            InputArtifact.from_path(archive),
            InputArtifact.from_path(taxonomy_path),
            InputArtifact.from_path(crosswalk_path),
        )

        def emit_progress(payload: dict[str, object]) -> None:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)

        report = stream_baci_aggregates(
            archive,
            revision,
            taxonomy,
            economies,
            paths.normalized / "baci",
            scratch_root=paths.scratch / "baci",
            input_artifacts=inputs,
            build_identity=BuildIdentity(
                command=(
                    f"python -m green_debt.cli normalize-baci --revision {revision} "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
            progress=emit_progress,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "audit-trade-normalization":
        report = audit_trade_normalization(data_root=args.data_root.resolve())
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "build-complexity":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        report = build_complexity(
            paths,
            taxonomy=args.taxonomy,
            build=BuildIdentity(
                command=(
                    "python -m green_debt.cli build-complexity "
                    f"--taxonomy {args.taxonomy} --data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "build-trade-components":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        report = build_trade_components(
            paths,
            taxonomy=args.taxonomy,
            build=BuildIdentity(
                command=(
                    "python -m green_debt.cli build-trade-components "
                    f"--taxonomy {args.taxonomy} --data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "build-gsci":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        report = build_gsci(
            paths,
            build=BuildIdentity(
                command=(
                    "python -m green_debt.cli build-gsci "
                    f"--data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "build-supplier-raw":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root.resolve())
        report = build_supplier_raw(
            paths,
            taxonomy=args.taxonomy,
            build=BuildIdentity(
                command=(
                    "python -m green_debt.cli build-supplier-raw "
                    f"--taxonomy {args.taxonomy} --data-root {paths.data_root}"
                ),
                code_commit=_code_version(),
            ),
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "audit-complexity":
        report = audit_complexity(
            data_root=args.data_root.resolve(),
            code_root=PROJECT_ROOT,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "acquire-initialization-supplements":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root)
        specs = load_tiva_supplements(args.supplement_config)
        openalex_spec = initialization_supplement_spec(args.registry)
        validate_initialization_supplement_range(
            openalex_spec.start_year, openalex_spec.end_year
        )
        response_budget_bytes = 128 * 1024**2
        projected_bytes = (
            sum(spec.expected_max_bytes for spec in specs) + response_budget_bytes
        )
        source_ids = [
            *(spec.source_id for spec in specs),
            "openalex_initialization_aggregates",
        ]
        usage = measure_layer_usage(paths.data_root, audits_root=paths.audits)
        if usage.project_bytes + projected_bytes >= 120 * GIB:
            raise RuntimeError("initialization projection reaches 120 GB hard stop")
        if usage.filesystem_free_bytes < projected_bytes + 30 * GIB:
            raise RuntimeError("filesystem cannot preserve 30 GB reserve")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "downloads_response_bodies": False,
                        "openalex_range": [
                            openalex_spec.start_year,
                            openalex_spec.end_year,
                        ],
                        "projected_additional_bytes": projected_bytes,
                        "source_ids": source_ids,
                        "status": "dry_run_ready",
                        "tiva_requests": [
                            {
                                "key": spec.key,
                                "measure": spec.measure,
                                "query": spec.query,
                            }
                            for spec in specs
                        ],
                        "topic_count": openalex_spec.topic_count,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        gate = _night_gate(
            args.config,
            args.hosts_config,
            project_root=paths.data_root,
        )
        authorization = args.route_exception_authorization
        oecd_evidence = gate.check(
            OECD_TIVA_BASE_URL,
            projected_additional_bytes=projected_bytes,
            route_exception_authorization=authorization,
        )
        openalex_evidence = gate.check(
            "https://api.openalex.org/works",
            projected_additional_bytes=0,
            route_exception_authorization=authorization,
        )
        log_path = paths.data_root / "07_文献与日志/下载日志.jsonl"
        tiva_root = paths.raw / "oecd_tiva/20260823/data"
        client = DirectHttpClient(timeout_seconds=300)
        try:
            tiva_records = [
                acquire_tiva_supplement(
                    spec,
                    client=client,
                    output_root=tiva_root,
                    route_exception_authorization=(
                        oecd_evidence.route_exception_authorization
                    ),
                    route_exception_used=oecd_evidence.route_exception_used,
                    audit_log_path=log_path,
                )
                for spec in specs
            ]
        finally:
            client.close()

        topic_ids = load_included_topic_ids(args.registry)
        openalex_root = paths.raw / "openalex/20260823/works_aggregate"
        client = DirectHttpClient(timeout_seconds=300)
        try:
            openalex_record = acquire_country_year_aggregates(
                client=client,
                base_url="https://api.openalex.org/works",
                topic_ids=topic_ids,
                start_year=openalex_spec.start_year,
                end_year=openalex_spec.end_year,
                api_key=os.environ.get("OPENALEX_API_KEY") or None,
                output_root=openalex_root,
                response_budget_bytes=response_budget_bytes,
                route_exception_authorization=(
                    openalex_evidence.route_exception_authorization
                ),
                route_exception_used=openalex_evidence.route_exception_used,
                source_id="openalex_initialization_aggregates",
            )
        finally:
            client.close()
        append_acquisition_log(
            log_path,
            {
                "automatic_redirects": False,
                "bytes": openalex_record.bytes,
                "completed_at_utc": openalex_evidence.checked_at_utc,
                "connected_tunnel_count": (
                    openalex_evidence.connected_tunnel_count
                ),
                "destination": str(
                    openalex_record.panel_path.relative_to(paths.data_root)
                ),
                "proxy_bypass_enforced": True,
                "route_exception_authorization": (
                    openalex_evidence.route_exception_authorization
                ),
                "route_exception_used": openalex_evidence.route_exception_used,
                "route_interface": openalex_evidence.route_interface,
                "sha256": sha256_file(openalex_record.panel_path),
                "source_id": "openalex_initialization_aggregates",
                "source_version": "20260823_1992_1995_topic_registry_v1",
                "status": "downloaded_or_verified",
            },
        )
        print(
            json.dumps(
                {
                    "downloads_response_bodies": True,
                    "openalex": {
                        **asdict(openalex_record),
                        "panel_path": str(openalex_record.panel_path),
                    },
                    "source_ids": source_ids,
                    "status": "complete",
                    "tiva": [asdict(record) for record in tiva_records],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "audit-initialization-supplements":
        report = _audit_initialization_supplements(
            data_root=args.data_root,
            code_root=PROJECT_ROOT,
            registry=args.registry,
            supplement_config=args.supplement_config,
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "raw-hash-snapshot":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root)
        output = args.output
        if not output.is_absolute():
            output = paths.data_root / output
        report = write_raw_hash_snapshot(paths.raw, output)
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "raw-hash-verify":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root)
        snapshot = args.snapshot
        if not snapshot.is_absolute():
            snapshot = paths.data_root / snapshot
            if not snapshot.exists():
                snapshot = paths.raw / args.snapshot.name
        report = verify_raw_hash_snapshot(snapshot, paths.raw)
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0 if report.mismatch_count == 0 else 2
    if args.command == "capacity-report":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root)
        usage = measure_layer_usage(paths.data_root, audits_root=paths.audits)
        payload = {
            **asdict(usage),
            "intermediate_bytes": usage.intermediate_bytes,
            "project_bytes": usage.project_bytes,
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "verify-artifacts":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root)
        layer_root = getattr(paths, args.layer).resolve()
        verified = []
        for manifest_path in discover_authoritative_manifest_paths(paths):
            if manifest_path.is_relative_to(layer_root):
                manifest = verify_manifest(manifest_path)
                verified.append(manifest.table_id)
        print(
            json.dumps(
                {
                    "layer": args.layer,
                    "manifest_count": len(verified),
                    "table_ids": verified,
                    "status": "valid",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "checkpoint":
        paths = resolve_project_paths(PROJECT_ROOT, args.data_root)
        receipt = run_checkpoint(
            args.number,
            paths,
            write_evidence=not args.verify_only,
            approval_gate_verification=args.number in {1, 2},
        )
        print(json.dumps(asdict(receipt), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "config-check":
        config = load_project_config(args.config)
        print(
            json.dumps(
                {
                    "main_period": (
                        f"{config.period.main[0]}-{config.period.main[1]}"
                    ),
                    "hard_stop_gb": config.storage.hard_stop_gb,
                    "direct_only": config.network.direct_only,
                    "bypass_system_proxy": config.network.bypass_system_proxy,
                    "rho": config.gad.rho,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "network-preflight":
        config = load_project_config(args.config)
        if args.offline and args.host:
            raise ValueError("--offline and --host cannot be used together")
        if not args.offline and not args.host:
            raise ValueError("provide --offline or --host")

        client = DirectHttpClient()
        budget = DiskBudgetGuard(
            project_root=PROJECT_ROOT,
            hard_stop_bytes=config.storage.hard_stop_gb * GIB,
            reserve_bytes=config.storage.reserve_gb * GIB,
        ).check(projected_additional_bytes=0)
        proxy_environment_names = sorted(
            key for key in PROXY_ENVIRONMENT_KEYS if key in os.environ
        )

        payload: dict[str, object] = {
            "bytes_downloaded": 0,
            "hard_stop_gb": config.storage.hard_stop_gb,
            "project_bytes": budget.current_project_bytes,
            "proxy_bypass_enforced": (
                client.trust_env is False and client.explicit_proxy is None
            ),
            "proxy_environment": (
                "clear"
                if not proxy_environment_names
                else "present_but_ignored"
            ),
            "proxy_environment_names": proxy_environment_names,
        }
        if args.offline:
            flags = read_macos_system_proxy_flags()
            enabled = sorted(
                key.removesuffix("Enable")
                for key, value in flags.items()
                if value == "1"
            )
            payload.update(
                {
                    "mode": "offline",
                    "route_interface": None,
                    "system_proxy": (
                        "enabled_but_bypassed" if enabled else "disabled"
                    ),
                    "system_proxy_types": enabled,
                }
            )
        else:
            evidence = DirectRouteGuard(
                rejected_interface_prefixes=(
                    config.network.rejected_interface_prefixes
                )
            ).check(
                args.host,
                proxy_bypass_enforced=bool(payload["proxy_bypass_enforced"]),
            )
            payload.update(
                {
                    "mode": "target_route",
                    "host": evidence.host,
                    "resolved_ip": evidence.resolved_ip,
                    "route_interface": evidence.interface,
                    "system_proxy": (
                        "enabled_but_bypassed"
                        if evidence.system_proxy_enabled
                        else "disabled"
                    ),
                    "system_proxy_types": list(
                        evidence.enabled_system_proxy_types
                    ),
                }
            )
        client.close()
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "night-direct":
        gate = _night_gate(args.config, args.hosts_config)
        if args.night_direct_action == "check":
            evidence = gate.check(
                args.url,
                projected_additional_bytes=args.projected_bytes,
            )
            print(
                json.dumps(
                    asdict(evidence),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.night_direct_action == "run":
            python_arguments = list(args.python_arguments)
            if python_arguments[:1] == ["--"]:
                python_arguments = python_arguments[1:]
            run_project_python_after_check(
                gate=gate,
                url=args.url,
                projected_additional_bytes=args.projected_bytes,
                python_executable=PROJECT_PYTHON,
                python_arguments=python_arguments,
            )
            return 0
        raise RuntimeError(
            f"unsupported night-direct action: {args.night_direct_action}"
        )
    if args.command == "source-status":
        catalog = load_source_catalog(args.sources_config, PROJECT_ROOT)
        payload = {
            source_id: {
                "enabled": entry.enabled,
                "kind": entry.kind,
                "reason": entry.reason,
                "version": entry.version,
            }
            for source_id, entry in catalog.sources.items()
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "openalex-topic-audit":
        report = audit_topic_registry(args.registry)
        if (
            report.included_topics == 0
            or report.duplicate_topic_ids
            or report.unreviewed_candidates
            or report.invalid_included_rows
        ):
            raise RuntimeError(f"OpenAlex topic registry audit failed: {report}")
        print(
            json.dumps(
                {**asdict(report), "status": "ready"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "acquire-openalex-aggregates":
        topic_ids = load_included_topic_ids(args.registry)
        if args.response_budget_gb <= 0:
            raise ValueError("response budget must be positive")
        response_budget_bytes = args.response_budget_gb * GIB
        config = load_project_config(DEFAULT_CONFIG)
        budget = DiskBudgetGuard(
            project_root=args.capacity_root,
            hard_stop_bytes=config.storage.hard_stop_gb * GIB,
            reserve_bytes=config.storage.reserve_gb * GIB,
        ).check(projected_additional_bytes=response_budget_bytes)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "downloads_response_bodies": False,
                        "end_year": args.end,
                        "projected_peak_bytes": budget.projected_peak_bytes,
                        "response_budget_bytes": response_budget_bytes,
                        "start_year": args.start,
                        "status": "dry_run_ready",
                        "topic_count": len(topic_ids),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        client = DirectHttpClient(timeout_seconds=300)
        try:
            report = acquire_country_year_aggregates(
                client=client,
                base_url=args.api_base_url,
                topic_ids=topic_ids,
                start_year=args.start,
                end_year=args.end,
                api_key=os.environ.get("OPENALEX_API_KEY") or None,
                output_root=args.output_root,
                response_budget_bytes=response_budget_bytes,
            )
        finally:
            client.close()
        print(
            json.dumps(
                {
                    **asdict(report),
                    "panel_path": str(report.panel_path),
                    "status": "downloaded_or_verified",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "science-audit":
        report = audit_country_year_panel(args.counts_csv)
        print(
            json.dumps(
                {**asdict(report), "status": "valid"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "batch-plan":
        catalog = load_source_catalog(args.sources_config, PROJECT_ROOT)
        specs = catalog.batch_specs(args.batch)
        print(
            json.dumps(
                {
                    "batch": args.batch,
                    "destinations": [
                        str(spec.destination.relative_to(PROJECT_ROOT))
                        for spec in specs
                    ],
                    "downloads_response_bodies": False,
                    "projected_working_bytes": sum(
                        spec.projected_working_bytes for spec in specs
                    ),
                    "source_ids": [spec.source_id for spec in specs],
                    "urls": [spec.url for spec in specs],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "acquire":
        catalog = load_source_catalog(args.sources_config, PROJECT_ROOT)
        spec = catalog.download_spec(args.source_id)
        gate = _night_gate(args.config, args.hosts_config)
        if args.dry_run:
            evidence = gate.check(
                spec.url,
                projected_additional_bytes=spec.projected_working_bytes,
            )
            print(
                json.dumps(
                    {
                        "downloads_response_bodies": False,
                        "evidence": asdict(evidence),
                        "source_id": spec.source_id,
                        "status": "dry_run_ready",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        runner = _acquisition_runner(gate)
        try:
            record = runner.acquire(spec)
        finally:
            runner.close()
        print(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "acquire-batch":
        catalog = load_source_catalog(args.sources_config, PROJECT_ROOT)
        specs = catalog.batch_specs(args.batch)
        gate = _night_gate(args.config, args.hosts_config)
        missing_specs = tuple(spec for spec in specs if not spec.destination.exists())
        if missing_specs:
            total_projected = sum(
                spec.projected_working_bytes for spec in missing_specs
            )
            checked_hosts: set[str] = set()
            for spec in missing_specs:
                host = urlsplit(spec.url).hostname or ""
                if host in checked_hosts:
                    continue
                gate.check(
                    spec.url,
                    projected_additional_bytes=(
                        total_projected if not checked_hosts else 0
                    ),
                )
                checked_hosts.add(host)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "batch": args.batch,
                        "downloads_response_bodies": False,
                        "missing_source_ids": [
                            spec.source_id for spec in missing_specs
                        ],
                        "status": "dry_run_ready",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        runner = _acquisition_runner(gate)
        records = []
        try:
            for index, spec in enumerate(specs, start=1):
                print(
                    json.dumps(
                        {
                            "batch": args.batch,
                            "position": index,
                            "source_id": spec.source_id,
                            "status": "starting",
                            "total": len(specs),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
                record = runner.acquire(spec)
                records.append(record)
                print(
                    json.dumps(asdict(record), ensure_ascii=False, sort_keys=True),
                    flush=True,
                )
        finally:
            runner.close()
        print(
            json.dumps(
                {
                    "batch": args.batch,
                    "bytes": sum(record.bytes for record in records),
                    "source_ids": [record.source_id for record in records],
                    "status": "complete",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    raise RuntimeError(f"unsupported command: {args.command}")


def entrypoint() -> int:
    try:
        return main()
    except (RuntimeError, ValueError, OSError, httpx.HTTPError) as exc:
        print(
            json.dumps(
                {"status": "blocked", "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(entrypoint())
