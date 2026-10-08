import polars as pl
import pytest

from green_debt.gad import build_component_indices


def test_component_indices_are_equal_weighted() -> None:
    frame = pl.DataFrame(
        {
            "economy_id": ["AAA"],
            "year": [2000],
            "z0_green_import_intensity_raw": [2.0],
            "z0_green_import_complexity_raw": [0.0],
            "z0_gfvad_raw": [4.0],
            "z0_gsci_raw": [2.0],
            "z0_gud_raw": [0.0],
            "z0_grd_raw": [2.0],
            "z0_gnir_raw": [3.0],
        }
    )

    out = build_component_indices(frame)

    assert out["gimc"].item() == pytest.approx(1.0)
    assert out["supp"].item() == pytest.approx(1.0)
    assert out["external_exposure_core"].item() == pytest.approx(2.5)
    assert out["external_exposure_lite"].item() == pytest.approx(2.0)
    assert out["absorption_core"].item() == pytest.approx(1.5)


def test_leave_one_component_inputs_are_renormalized() -> None:
    frame = pl.DataFrame(
        {
            "economy_id": ["A"],
            "year": [2000],
            "gimc": [1.0],
            "z0_gfvad_raw": [3.0],
            "z0_gnir_raw": [5.0],
            "z0_gsci_raw": [7.0],
            "supp": [9.0],
        }
    )

    out = build_component_indices(frame)

    assert out["external_exposure_core"].item() == pytest.approx(2.0)
    assert out["external_exposure_lite"].item() == pytest.approx(3.0)
    assert out["external_exposure_no_gfvad"].item() == pytest.approx(1.0)
    assert out["absorption_core"].item() == pytest.approx(8.0)
    assert out["absorption_no_supp"].item() == pytest.approx(7.0)
    assert out["absorption_no_gsci"].item() == pytest.approx(9.0)


def test_missing_required_component_remains_missing_not_zero_filled() -> None:
    frame = pl.DataFrame(
        {
            "economy_id": ["A"],
            "year": [2000],
            "z0_green_import_intensity_raw": [1.0],
            "z0_green_import_complexity_raw": [None],
            "z0_gfvad_raw": [2.0],
            "z0_gsci_raw": [3.0],
            "z0_gud_raw": [4.0],
            "z0_grd_raw": [5.0],
            "z0_gnir_raw": [6.0],
        }
    )

    out = build_component_indices(frame)

    assert out["gimc"].item() is None
    assert out["external_exposure_core"].item() is None
