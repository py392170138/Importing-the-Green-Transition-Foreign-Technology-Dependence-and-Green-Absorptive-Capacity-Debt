from pathlib import Path

import pytest

from green_debt.config import load_project_config


ROOT = Path(__file__).resolve().parents[2]


def test_project_config_derives_and_freezes_design_constraints() -> None:
    cfg = load_project_config(ROOT / "config" / "project.yaml")

    assert cfg.period.main == (2000, 2022)
    assert cfg.period.openalex_history == (1992, 1995)
    assert cfg.period.source_trade == (1996, 2024)
    assert cfg.period.tiva_history == (1995, 1999)
    assert cfg.period.initialization == (1997, 1999)
    assert cfg.period.clean_hs07 == (2007, 2022)
    assert cfg.period.lite_extension == (2000, 2024)
    assert cfg.storage.absolute_limit_gb == 150
    assert cfg.storage.hard_stop_gb == 120
    assert cfg.storage.reserve_gb == 30
    assert cfg.gad.half_life_years == 5
    assert cfg.gad.rho == pytest.approx(2 ** (-1 / 5))
    assert cfg.threshold.search_percentiles == (20, 80)
    assert cfg.threshold.minimum_regime_share == 0.20
    assert cfg.network.direct_only is True
    assert cfg.network.bypass_system_proxy is True
    assert cfg.network.require_no_connected_tunnel is True
    assert cfg.network.allow_logged_user_route_exception is True


def test_invalid_capacity_budget_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "period:\n"
        "  initialization: [1996, 1999]\n"
        "  main: [2000, 2022]\n"
        "  clean_hs07: [2007, 2022]\n"
        "  lite_extension: [2000, 2024]\n"
        "storage:\n"
        "  absolute_limit_gb: 150\n"
        "  hard_stop_gb: 140\n"
        "  reserve_gb: 30\n"
        "network:\n"
        "  direct_only: true\n"
        "  bypass_system_proxy: true\n"
        "  rejected_interface_prefixes: [utun]\n"
        "gad:\n"
        "  baseline: [2000, 2004]\n"
        "  half_life_years: 5\n"
        "threshold:\n"
        "  search_percentiles: [20, 80]\n"
        "  minimum_regime_share: 0.20\n"
        "  selection_outcome: green_industrial_upgrading_index\n"
        "horizons:\n"
        "  environmental: [0, 1, 2, 3]\n"
        "  environmental_confirmatory: [1, 2, 3]\n"
        "  industrial: [3, 4, 5, 6, 7, 8]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hard stop plus reserve"):
        load_project_config(path)
