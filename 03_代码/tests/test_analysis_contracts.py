import json
from pathlib import Path

from jsonschema import validate

from green_debt.analysis_io import load_table_contract


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = ROOT / "03_代码/contracts/analysis"
PROVENANCE = {
    "run_id",
    "spec_id",
    "input_authority_hash",
    "git_commit",
    "renv_lock_sha256",
    "created_at_utc",
}
RESULT_METADATA = {
    "evidence_policy_sha256",
    "year_min",
    "year_max",
    "r_version",
    "python_version",
    "julia_version",
    "package_versions_json",
    "random_seed",
    "ssc_config",
    "reference_distribution",
    "reference_df",
}
EXPECTED = {
    "diagnostic_cell": {
        "primary_key": (
            "run_id",
            "analysis_family",
            "outcome_id",
            "horizon",
            "gad_version",
            "sample_version",
        ),
        "required": {
            "role",
            "n",
            "economies",
            "clusters",
            "year_min",
            "year_max",
            "rank",
            "instrument_rank",
            "cross_moment_rank",
            "condition_number",
            "partial_r2_gimc",
            "partial_r2_interaction",
            "effective_f_gimc",
            "effective_f_interaction",
            "first_stage_status",
            "gate_status",
        },
    },
    "model_estimate": {
        "primary_key": (
            "run_id",
            "estimator",
            "analysis_family",
            "outcome_id",
            "horizon",
            "gad_version",
            "sample_version",
            "term",
        ),
        "required": {
            "estimate",
            "std_error",
            "conf_low",
            "conf_high",
            "p_value",
            "wild_conf_low",
            "wild_conf_high",
            "wild_p_value",
            "wild_draws",
            "wild_seed",
            "n",
            "economies",
            "clusters",
            "first_stage_status",
            "inference_status",
        },
    },
    "marginal_effect": {
        "primary_key": (
            "run_id",
            "estimator",
            "analysis_family",
            "outcome_id",
            "horizon",
            "gad_version",
            "sample_version",
            "gad_quantile",
        ),
        "required": {
            "gad_value",
            "estimate",
            "std_error",
            "conf_low",
            "conf_high",
            "p_value",
            "wild_conf_low",
            "wild_conf_high",
            "wild_p_value",
            "wild_draws",
            "wild_seed",
        },
    },
    "weak_iv_set": {
        "primary_key": (
            "run_id",
            "outcome_id",
            "horizon",
            "gad_version",
            "sample_version",
        ),
        "required": {
            "beta_low",
            "beta_high",
            "theta_low",
            "theta_high",
            "status",
            "expansions",
            "accepted_points",
            "accepted_hash",
            "conventional_point_accepted",
        },
    },
    "shift_share_weight": {
        "primary_key": (
            "run_id",
            "analysis_family",
            "outcome_id",
            "horizon",
            "gad_version",
            "sample_version",
            "shock_id",
        ),
        "required": {
            "exporter",
            "hs6",
            "year",
            "shock_cluster_id",
            "signed_weight",
            "absolute_weight",
            "absolute_rank",
        },
    },
    "sample_cell": {
        "primary_key": (
            "run_id",
            "analysis_family",
            "outcome_id",
            "horizon",
            "gad_version",
            "sample_version",
        ),
        "required": {
            "role",
            "candidate_rows",
            "n",
            "sample_loss",
            "sample_loss_share",
            "economies",
            "clusters",
            "year_min",
            "year_max",
        },
    },
    "missingness_cell": {
        "primary_key": (
            "run_id",
            "analysis_family",
            "outcome_id",
            "horizon",
            "gad_version",
            "sample_version",
            "variable",
        ),
        "required": {"role", "candidate_rows", "null_count", "null_share"},
    },
    "distribution": {
        "primary_key": (
            "run_id",
            "gad_version",
            "sample_version",
            "variable",
        ),
        "required": {
            "n",
            "mean",
            "std",
            "min",
            "p01",
            "p25",
            "p50",
            "p75",
            "p99",
            "max",
        },
    },
    "correlation": {
        "primary_key": (
            "run_id",
            "gad_version",
            "sample_version",
            "variable_x",
            "variable_y",
        ),
        "required": {"n", "correlation"},
    },
    "exposure_concentration": {
        "primary_key": ("run_id", "importer"),
        "required": {
            "taxonomy_version",
            "share_version",
            "exposure_units",
            "hhi",
            "effective_units",
            "top1_share",
            "top5_share",
        },
    },
    "descriptive_path": {
        "primary_key": (
            "run_id",
            "outcome_id",
            "horizon",
            "gad_version",
            "sample_version",
        ),
        "required": {
            "path_family",
            "n",
            "economies",
            "year_min",
            "year_max",
            "mean_delta_outcome",
            "interpretation_status",
        },
    },
}


def test_analysis_contracts_freeze_exact_keys_payloads_and_provenance() -> None:
    schema = json.loads(
        (ROOT / "03_代码/contracts/base_table_contract.schema.json").read_text(
            encoding="utf-8"
        )
    )

    for table_id, expected in EXPECTED.items():
        path = CONTRACT_ROOT / f"{table_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate(payload, schema)
        contract = load_table_contract(path)
        assert contract.table_id == table_id
        assert contract.schema_version == "1.0.0"
        assert contract.primary_key == expected["primary_key"]
        assert PROVENANCE <= set(contract.columns)
        assert expected["required"] <= set(contract.columns)
        assert all(contract.columns[name] == "String" for name in PROVENANCE)
        if "horizon" in contract.columns:
            assert contract.columns["horizon"] == "Int16"


def test_analysis_contracts_make_all_float_result_fields_float64() -> None:
    float_fields = {
        "condition_number",
        "partial_r2_gimc",
        "partial_r2_interaction",
        "effective_f_gimc",
        "effective_f_interaction",
        "estimate",
        "std_error",
        "conf_low",
        "conf_high",
        "p_value",
        "wild_conf_low",
        "wild_conf_high",
        "wild_p_value",
        "gad_quantile",
        "gad_value",
        "beta_low",
        "beta_high",
        "theta_low",
        "theta_high",
        "signed_weight",
        "absolute_weight",
        "sample_loss_share",
        "null_share",
        "mean",
        "std",
        "min",
        "p01",
        "p25",
        "p50",
        "p75",
        "p99",
        "max",
        "correlation",
        "hhi",
        "effective_units",
        "top1_share",
        "top5_share",
        "mean_delta_outcome",
    }
    for table_id in EXPECTED:
        contract = load_table_contract(CONTRACT_ROOT / f"{table_id}.json")
        for name in float_fields & set(contract.columns):
            assert contract.columns[name] == "Float64", (table_id, name)


def test_wild_bootstrap_nulls_preserve_finite_and_unbounded_statuses() -> None:
    wild_fields = {
        "wild_conf_low",
        "wild_conf_high",
        "wild_p_value",
        "wild_draws",
        "wild_seed",
    }
    for table_id in ("model_estimate", "marginal_effect"):
        contract = load_table_contract(CONTRACT_ROOT / f"{table_id}.json")
        assert wild_fields <= set(contract.null_semantics)
        for name in wild_fields:
            semantics = contract.null_semantics[name]
            assert "wild_bootstrap_required" in semantics
            assert "wild_bootstrap_unbounded" in semantics


def test_all_machine_readable_model_results_persist_frozen_inference_metadata() -> None:
    for table_id in (
        "model_estimate",
        "model_covariance",
        "marginal_effect",
        "threshold_estimate",
        "weak_iv_set",
        "shift_share_summary",
        "shift_share_weight",
    ):
        contract = load_table_contract(CONTRACT_ROOT / f"{table_id}.json")
        assert RESULT_METADATA <= set(contract.columns), table_id


def test_threshold_contract_freezes_regime_and_high_minus_low_inference() -> None:
    contract = load_table_contract(CONTRACT_ROOT / "threshold_estimate.json")
    required = {
        "wild_estimate",
        "wild_std_error",
        "wild_conf_low",
        "wild_conf_high",
        "wild_p_value",
        "wild_inference_status",
        "wild_draws",
        "wild_seed",
        "difference_estimate",
        "difference_std_error",
        "difference_conf_low",
        "difference_conf_high",
        "difference_p_value",
        "difference_wild_estimate",
        "difference_wild_std_error",
        "difference_wild_conf_low",
        "difference_wild_conf_high",
        "difference_wild_p_value",
        "difference_wild_inference_status",
        "difference_wild_draws",
        "difference_wild_seed",
    }
    assert required <= set(contract.columns)
    for name in required:
        if "status" not in name:
            assert name in contract.null_semantics
