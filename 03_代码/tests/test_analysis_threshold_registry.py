from copy import deepcopy

import pytest

from green_debt.analysis_io import (
    canonical_threshold_registry_hash,
    verify_threshold_registry_payload,
)


def threshold_registry_payload(spec, run_context) -> dict[str, object]:
    cell = spec.threshold_cell()
    payload: dict[str, object] = {
        "registry_id": "threshold_registry_v1",
        "selection_outcome": cell.outcome_id,
        "horizon": cell.horizon,
        "gad_version": cell.gad_version,
        "sample_version": cell.sample_version,
        "criterion": spec.threshold.criterion,
        "quantile_type": spec.threshold.quantile_type,
        "tie_break": spec.threshold.tie_break,
        "seed": spec.seed,
        "sample_hash": "d" * 64,
        "input_authority_hash": run_context.input_authority_hash,
        "q": 0.42,
        "percentile": 50,
        "low_share": 0.50,
        "high_share": 0.50,
        "bootstrap_draws": 999,
        "bootstrap_valid_draws": 999,
        "bootstrap_failed_draws": 0,
        "bootstrap_status": "available",
        "percentile_conf_low": 45.0,
        "percentile_conf_high": 55.0,
        "q_conf_low": 0.30,
        "q_conf_high": 0.54,
        "candidates": [
            {
                "percentile": 50,
                "q": 0.42,
                "low_share": 0.50,
                "high_share": 0.50,
                "eligible": True,
                "ssr": 1.0,
                "n": 100,
                "status": "estimated",
            }
        ],
        "created_at_utc": "2026-08-29T00:00:00Z",
    }
    payload["registry_hash"] = canonical_threshold_registry_hash(payload)
    return payload


def test_registry_hash_excludes_only_hash_and_timestamp(spec, run_context) -> None:
    payload = threshold_registry_payload(spec, run_context)
    original_hash = payload["registry_hash"]
    timestamp_changed = deepcopy(payload)
    timestamp_changed["created_at_utc"] = "2026-08-30T00:00:00Z"
    assert canonical_threshold_registry_hash(timestamp_changed) == original_hash
    q_changed = deepcopy(payload)
    q_changed["q"] = 0.43
    assert canonical_threshold_registry_hash(q_changed) != original_hash


def test_registry_rejects_retargeting_and_hash_tamper(spec, run_context) -> None:
    payload = threshold_registry_payload(spec, run_context)
    verify_threshold_registry_payload(payload, spec, run_context)
    retargeted = deepcopy(payload)
    retargeted["selection_outcome"] = "green_export_complexity"
    retargeted["registry_hash"] = canonical_threshold_registry_hash(retargeted)
    with pytest.raises(ValueError, match="selection outcome"):
        verify_threshold_registry_payload(retargeted, spec, run_context)
    tampered = deepcopy(payload)
    tampered["q"] = 0.43
    with pytest.raises(ValueError, match="registry hash"):
        verify_threshold_registry_payload(tampered, spec, run_context)


def test_registry_rejects_invalid_bootstrap_accounting(spec, run_context) -> None:
    payload = threshold_registry_payload(spec, run_context)
    payload["bootstrap_failed_draws"] = 1
    payload["registry_hash"] = canonical_threshold_registry_hash(payload)
    with pytest.raises(ValueError, match="bootstrap draw accounting"):
        verify_threshold_registry_payload(payload, spec, run_context)


def test_registry_rejects_fractional_integer_fields(spec, run_context) -> None:
    payload = threshold_registry_payload(spec, run_context)
    payload["percentile"] = 50.5
    payload["registry_hash"] = canonical_threshold_registry_hash(payload)
    with pytest.raises(ValueError, match="percentile.*integer"):
        verify_threshold_registry_payload(payload, spec, run_context)
