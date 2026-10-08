import polars as pl
import pytest

from green_debt.scaling import (
    FROZEN_COMPONENT_COLUMNS,
    apply_scaler,
    build_scaler_authority,
    create_scaler_anchor,
    fit_scaler,
    verify_frozen_sample_semantics,
    verify_scaler,
    verify_scaler_anchor,
)


def test_scaler_uses_mad_then_iqr_fallback() -> None:
    """Using raw MAD as scale would leave a zero denominator for this valid sample."""
    frame = pl.DataFrame({"economy_id": ["A", "B", "C", "D", "E"], "year": [2000] * 5, "x": [0.0, 0.0, 0.0, 1.0, 2.0]})
    registry = fit_scaler(frame, ("x",), (2000, 2004), provisional_sample_manifest_hash="a" * 64)
    row = registry.rows[0]
    assert row.center == pytest.approx(0.0)
    assert row.scale == pytest.approx(1.0 / 1.349)
    assert row.scale_method == "iqr_div_1_349"
    transformed = apply_scaler(frame, registry)
    assert transformed.columns[-1] == "z0_x"
    assert transformed["x"].to_list() == [0.0, 0.0, 0.0, 1.0, 2.0]


def test_zero_mad_and_zero_iqr_is_nonidentifiable() -> None:
    frame = pl.DataFrame({"economy_id": ["A", "B"], "year": [2000, 2000], "x": [1.0, 1.0]})
    with pytest.raises(ValueError, match="non-identifiable component x"):
        fit_scaler(frame, ("x",), (2000, 2004), provisional_sample_manifest_hash="a" * 64)


def test_scaler_requires_exact_frozen_column_order_and_complete_matched_rows() -> None:
    """Fitting a subset or null-skipping a component would make GAD versions incomparable."""
    columns = {"economy_id": ["A", "B"], "year": [2000, 2001]}
    for index, name in enumerate(FROZEN_COMPONENT_COLUMNS):
        columns[name] = [float(index), float(index + 1)]
    frame = pl.DataFrame(columns)
    with pytest.raises(ValueError, match="exact frozen component order"):
        fit_scaler(frame, tuple(reversed(FROZEN_COMPONENT_COLUMNS)), (2000, 2004), provisional_sample_manifest_hash="a" * 64)
    null_frame = frame.with_columns(pl.lit(None).cast(pl.Float64).alias("gfvad_raw"))
    with pytest.raises(ValueError, match="complete matched rows"):
        fit_scaler(null_frame, FROZEN_COMPONENT_COLUMNS, (2000, 2004), provisional_sample_manifest_hash="a" * 64)


def test_apply_and_verify_scaler_detect_tampered_canonical_payload() -> None:
    frame = pl.DataFrame({"economy_id": ["A", "B", "C"], "year": [2000, 2001, 2002], "x": [1.0, 2.0, 3.0]})
    registry = fit_scaler(frame, ("x",), (2000, 2004), provisional_sample_manifest_hash="b" * 64)
    tampered = registry.__class__(rows=registry.rows, years=registry.years, provisional_sample_manifest_hash=registry.provisional_sample_manifest_hash, canonical_hash="0" * 64)
    with pytest.raises(ValueError, match="canonical hash mismatch"):
        apply_scaler(frame, tampered)
    verify_scaler(registry, expected_manifest_hash="b" * 64, expected_columns=("x",), expected_years=(2000, 2004))


def test_cli_registers_reproducible_tiva_and_scaler_commands() -> None:
    """Removing either production entrypoint would leave the frozen artifacts unrebuildable."""
    from green_debt.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["build-tiva-measures"]).command == "build-tiva-measures"
    assert parser.parse_args(["fit-gad-scaler"]).command == "fit-gad-scaler"
    assert parser.parse_args(["apply-gad-scaler"]).command == "apply-gad-scaler"
    assert parser.parse_args(["verify-scaler"]).command == "verify-scaler"


def _sample_semantics() -> dict[str, object]:
    return {
        "table_id": "provisional_sample",
        "rows": 251,
        "primary_key": ["economy_id"],
        "output_sha256": "1" * 64,
        "schema_sha256": "2" * 64,
        "period": None,
        "duplicate_primary_keys": 0,
        "column_order": ["economy_id", "provisional_core"],
        "columns": {"economy_id": "String", "provisional_core": "Boolean"},
        "null_semantics": {},
        "zero_semantics": {},
        "units": {},
        "transformations": ["outcome_free"],
        "coverage": {"economy_id": {"min": "A", "max": "Z", "unique": 251}},
    }


@pytest.mark.parametrize(
    ("field", "tampered"),
    (
        ("output_sha256", "3" * 64),
        ("rows", 250),
        ("schema_sha256", "4" * 64),
        ("primary_key", ["wrong"]),
        ("period", {"min": 1996, "max": 2024}),
        ("null_semantics", {"provisional_core": "changed"}),
    ),
)
def test_frozen_scaler_applies_only_to_semantically_identical_approved_sample(
    field: str, tampered: object,
) -> None:
    approved_manifest_hash = "a" * 64
    registry = fit_scaler(
        pl.DataFrame(
            {"economy_id": ["A", "B", "C"], "year": [2000] * 3, "x": [1.0, 2.0, 3.0]}
        ),
        ("x",),
        (2000, 2004),
        provisional_sample_manifest_hash=approved_manifest_hash,
    )
    approved = {
        "schema_version": "1.0.0",
        "approved_manifest_sha256": approved_manifest_hash,
        "manifest_semantics": _sample_semantics(),
    }
    current = _sample_semantics()
    verify_frozen_sample_semantics(registry, current, approved)
    current[field] = tampered
    with pytest.raises(ValueError, match="approved frozen-scaler sample semantics mismatch"):
        verify_frozen_sample_semantics(registry, current, approved)


def test_frozen_sample_semantic_anchor_must_match_registry_manifest_identity() -> None:
    registry = fit_scaler(
        pl.DataFrame(
            {"economy_id": ["A", "B", "C"], "year": [2000] * 3, "x": [1.0, 2.0, 3.0]}
        ),
        ("x",),
        (2000, 2004),
        provisional_sample_manifest_hash="a" * 64,
    )
    approved = {
        "schema_version": "1.0.0",
        "approved_manifest_sha256": "b" * 64,
        "manifest_semantics": _sample_semantics(),
    }
    with pytest.raises(ValueError, match="semantic anchor manifest identity mismatch"):
        verify_frozen_sample_semantics(registry, _sample_semantics(), approved)


def test_full_scaler_authority_preserves_initialization_and_lite_years_with_nulls() -> None:
    """Inner joining TiVA would wrongly erase 1996 and post-2022 Lite rows."""
    provisional = pl.DataFrame({"economy_id": ["A"], "provisional_core": [True]})
    years = [1996, 2000, 2023, 2024]
    trade = pl.DataFrame(
        {
            "economy_id": ["A"] * 4,
            "year": years,
            "green_import_intensity_raw": [1.0, 2.0, 3.0, 4.0],
            "green_import_complexity_raw": [1.0, 2.0, 3.0, 4.0],
            "gnir_raw": [0.1, 0.2, 0.3, 0.4],
        }
    )
    tiva = pl.DataFrame({"economy_id": ["A"], "year": [2000], "gfvad_raw": [0.5]})
    gsci = pl.DataFrame({"economy_id": ["A"] * 4, "year": years, "gsci_raw": [5.0, 6.0, 7.0, 8.0]})
    supplier = pl.DataFrame(
        {"economy_id": ["A"] * 4, "year": years, "gud_raw": [1.0, 2.0, 3.0, 4.0], "grd_raw": [0.1, 0.2, 0.3, 0.4]}
    )
    authority = build_scaler_authority(provisional, trade, tiva, gsci, supplier, years=(1996, 2024))
    assert authority.height == 29
    assert authority.filter(pl.col("year") == 1996)["gfvad_raw"].item() is None
    assert authority.filter(pl.col("year") == 2024)["gfvad_raw"].item() is None
    assert authority.filter(pl.col("year") == 2023)["gsci_raw"].item() == 7.0
    assert authority.filter(pl.col("year") == 2024)["gnir_raw"].item() == 0.4


def test_apply_scaler_preserves_authority_rows_and_null_components() -> None:
    """A null raw input is an unavailable component, not a reason to remove its row."""
    baseline = pl.DataFrame({"economy_id": ["A", "B", "C"], "year": [2000, 2000, 2000], "x": [1.0, 2.0, 3.0]})
    registry = fit_scaler(baseline, ("x",), (2000, 2004), provisional_sample_manifest_hash="c" * 64)
    authority = pl.DataFrame({"economy_id": ["A", "A"], "year": [1996, 2024], "x": [None, 4.0]})
    scaled = apply_scaler(authority, registry)
    assert scaled.height == 2
    assert scaled["x"].to_list() == [None, 4.0]
    assert scaled["z0_x"].to_list()[0] is None
    assert scaled["z0_x"].to_list()[1] == pytest.approx(2.0 / 1.4826)


def test_external_anchor_rejects_registry_with_recomputed_internal_hash() -> None:
    """An attacker can recompute a self-declared hash, but cannot rewrite the frozen anchor."""
    frame = pl.DataFrame({"economy_id": ["A", "B", "C"], "year": [2000, 2001, 2002], "x": [1.0, 2.0, 3.0]})
    registry = fit_scaler(frame, ("x",), (2000, 2004), provisional_sample_manifest_hash="d" * 64)
    anchor = create_scaler_anchor(registry, registry_file_sha256="e" * 64, implementation_commit="f" * 40)
    payload = registry.to_dict()
    payload["rows"][0]["center"] = 999.0
    from green_debt.scaling import canonical_registry_hash, registry_from_dict

    payload["canonical_hash"] = canonical_registry_hash({key: value for key, value in payload.items() if key != "canonical_hash"})
    tampered = registry_from_dict(payload)
    with pytest.raises(ValueError, match="external anchor canonical hash mismatch"):
        verify_scaler_anchor(tampered, anchor, registry_file_sha256="e" * 64, implementation_commit="f" * 40)


def test_verify_scaled_authority_rejects_tampered_z0_and_null_mismatch() -> None:
    """A checked registry is insufficient if the published scaled table can drift from it."""
    from green_debt.scaling import verify_scaled_authority

    baseline = pl.DataFrame({"economy_id": ["A", "B", "C"], "year": [2000, 2000, 2000], "x": [1.0, 2.0, 3.0]})
    registry = fit_scaler(baseline, ("x",), (2000, 2004), provisional_sample_manifest_hash="a" * 64)
    scaled = apply_scaler(pl.DataFrame({"economy_id": ["A", "B"], "year": [1996, 2024], "x": [None, 4.0]}), registry)
    verify_scaled_authority(scaled, registry)
    with pytest.raises(ValueError, match="scaled z0 mismatch"):
        verify_scaled_authority(
            scaled.with_columns(
                pl.when(pl.col("x").is_null()).then(pl.lit(None)).otherwise(pl.lit(9.0)).alias("z0_x")
            ),
            registry,
        )
    with pytest.raises(ValueError, match="raw null requires z0 null"):
        verify_scaled_authority(scaled.with_columns(pl.lit(9.0).alias("z0_x")), registry)


def test_cli_scaler_source_loader_keeps_full_provisional_calendar(tmp_path) -> None:
    """The production loader must not turn component availability into row eligibility."""
    from types import SimpleNamespace
    from green_debt.cli import _build_scaler_matched_frame

    harmonized = tmp_path / "harmonized"
    measures = tmp_path / "measures"
    (harmonized / "sample").mkdir(parents=True)
    (measures / "trade").mkdir(parents=True)
    (measures / "tiva").mkdir(parents=True)
    (measures / "science").mkdir(parents=True)
    pl.DataFrame({"economy_id": ["A"], "provisional_core": [True]}).write_parquet(harmonized / "sample/provisional_sample.parquet")
    pl.DataFrame({"taxonomy_version": ["main_hs96"], "economy_id": ["A"], "year": [2000], "green_import_intensity_raw": [1.0], "green_import_complexity_raw": [2.0], "gnir_raw": [0.5]}).write_parquet(measures / "trade/trade_components_raw.parquet")
    pl.DataFrame({"economy_id": ["A"], "year": [2000], "specification_id": ["confirmatory_prod_weight"], "gfvad_raw": [0.4]}).write_parquet(measures / "tiva/tiva_measures.parquet")
    pl.DataFrame({"economy_id": ["A"], "year": [1996], "gsci_raw": [1.0]}).write_parquet(measures / "science/gsci_raw.parquet")
    pl.DataFrame({"taxonomy_version": ["main_hs96"], "economy_id": ["A"], "year": [1996], "gud_raw": [1.0], "grd_raw": [0.2]}).write_parquet(measures / "trade/supplier_raw.parquet")
    authority, _ = _build_scaler_matched_frame(SimpleNamespace(harmonized=harmonized, measures=measures))
    assert authority.height == 29
    assert authority.filter(pl.col("year") == 1996)["gfvad_raw"].item() is None


def test_scaled_manifest_binding_requires_both_registry_and_external_anchor(tmp_path) -> None:
    """A valid scaled Parquet without both lineage files is not reproducibly frozen."""
    from types import SimpleNamespace
    from green_debt.cli import _verify_scaled_manifest_binding

    registry = tmp_path / "registry.json"
    anchor = tmp_path / "anchor.json"
    registry.write_text("registry", encoding="utf-8")
    anchor.write_text("anchor", encoding="utf-8")
    from green_debt.storage import sha256_file

    good = SimpleNamespace(
        code_commit="f" * 40,
        input_artifacts=(
            SimpleNamespace(path=str(registry.resolve()), sha256=sha256_file(registry)),
            SimpleNamespace(path=str(anchor.resolve()), sha256=sha256_file(anchor)),
        ),
    )
    _verify_scaled_manifest_binding(good, registry, anchor, "f" * 40)
    missing_anchor = SimpleNamespace(code_commit="f" * 40, input_artifacts=good.input_artifacts[:1])
    with pytest.raises(ValueError, match="external anchor binding"):
        _verify_scaled_manifest_binding(missing_anchor, registry, anchor, "f" * 40)


def test_scaled_manifest_accepts_current_bundle_paths_but_rejects_wrong_path_identity(
    tmp_path,
) -> None:
    from types import SimpleNamespace
    from green_debt.cli import _verify_scaled_manifest_binding
    from green_debt.storage import sha256_file

    fixed = tmp_path / "fixed"
    bundle = tmp_path / "registry/versions/build/bundle"
    registry = fixed / "06_结果/GAD固定缩放器_v1.json"
    anchor = fixed / "06_结果/GAD固定缩放器_v1.manifest.json"
    semantic = fixed / "03_代码/contracts/gad_scaler_sample_semantics.json"
    for path, content in ((registry, "registry"), (anchor, "anchor"), (semantic, "semantic")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    bundle_registry = bundle / "code/06_结果" / registry.name
    bundle_anchor = bundle / "code/06_结果" / anchor.name
    for source, destination in ((registry, bundle_registry), (anchor, bundle_anchor)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    good = SimpleNamespace(
        code_commit="f" * 40,
        input_artifacts=(
            SimpleNamespace(path=str(bundle_registry), sha256=sha256_file(registry)),
            SimpleNamespace(path=str(bundle_anchor), sha256=sha256_file(anchor)),
            SimpleNamespace(
                path=str(bundle / "code/03_代码/contracts" / semantic.name),
                sha256=sha256_file(semantic),
            ),
        ),
    )
    _verify_scaled_manifest_binding(good, registry, anchor, "f" * 40, semantic)
    wrong_identity = SimpleNamespace(
        code_commit="f" * 40,
        input_artifacts=(
            SimpleNamespace(path=str(bundle / "code/wrong" / registry.name), sha256=sha256_file(registry)),
            *good.input_artifacts[1:],
        ),
    )
    with pytest.raises(ValueError, match="registry binding"):
        _verify_scaled_manifest_binding(
            wrong_identity, registry, anchor, "f" * 40, semantic
        )
