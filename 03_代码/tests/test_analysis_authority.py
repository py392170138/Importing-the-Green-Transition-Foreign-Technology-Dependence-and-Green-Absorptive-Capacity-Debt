import json

import pytest

import green_debt.analysis_io as analysis_io
from analysis_fixtures import AUTHORITY_TABLE_PATHS, rebind_adjacent_manifest_to_bundle
from green_debt.analysis_io import validate_analysis_authority


def test_authority_binds_three_tables_sidecars_contracts_and_configs(
    project_fixture, monkeypatch
) -> None:
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)

    authority = validate_analysis_authority(
        project_fixture.code_root, project_fixture.data_root
    )

    assert authority.table_id == "model_panel"
    assert authority.rows == 1
    assert len(authority.input_hashes) == 15
    assert {name for name, _ in authority.table_hashes} == {
        "model_panel",
        "iv_baseline_shares",
        "iv_partner_shocks",
    }
    assert {name for name, _ in authority.input_hashes} == {
        "model_panel.output",
        "model_panel.schema",
        "model_panel.manifest",
        "model_panel.contract",
        "iv_baseline_shares.output",
        "iv_baseline_shares.schema",
        "iv_baseline_shares.manifest",
        "iv_baseline_shares.contract",
        "iv_partner_shocks.output",
        "iv_partner_shocks.schema",
        "iv_partner_shocks.manifest",
        "iv_partner_shocks.contract",
        "config.project",
        "config.outcome_gad_map",
        "config.analysis",
    }


def test_authority_checks_live_panel_when_manifest_points_to_bundle(
    project_fixture, monkeypatch
) -> None:
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)
    rebind_adjacent_manifest_to_bundle(project_fixture, "model_panel")
    panel = project_fixture.data_root / AUTHORITY_TABLE_PATHS["model_panel"]
    panel.write_bytes(panel.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="live output hash mismatch"):
        validate_analysis_authority(
            project_fixture.code_root, project_fixture.data_root
        )


def test_authority_fails_after_current_contract_tamper(
    project_fixture, monkeypatch
) -> None:
    monkeypatch.setattr(analysis_io, "FROZEN_MODEL_PANEL_ROWS", 1)
    contract = project_fixture.code_root / "03_代码/contracts/model_panel.json"
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["schema_version"] = f'{payload["schema_version"]}.tampered'
    contract.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="contract-manifest mismatch"):
        validate_analysis_authority(
            project_fixture.code_root, project_fixture.data_root
        )
