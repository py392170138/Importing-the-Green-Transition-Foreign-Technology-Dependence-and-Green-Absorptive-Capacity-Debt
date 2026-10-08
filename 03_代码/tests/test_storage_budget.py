from pathlib import Path

import pytest

from green_debt.storage import DiskBudgetGuard, sha256_file


GIB = 1024**3


def test_budget_accepts_projected_peak_below_stop(tmp_path: Path) -> None:
    guard = DiskBudgetGuard(
        project_root=tmp_path,
        hard_stop_bytes=120 * GIB,
        reserve_bytes=30 * GIB,
        usage_reader=lambda path: 40 * GIB,
        free_reader=lambda path: 500 * GIB,
    )

    snapshot = guard.check(projected_additional_bytes=20 * GIB)

    assert snapshot.projected_peak_bytes == 60 * GIB
    assert snapshot.current_project_bytes == 40 * GIB


def test_budget_rejects_projected_peak_at_stop(tmp_path: Path) -> None:
    guard = DiskBudgetGuard(
        project_root=tmp_path,
        hard_stop_bytes=120 * GIB,
        reserve_bytes=30 * GIB,
        usage_reader=lambda path: 100 * GIB,
        free_reader=lambda path: 500 * GIB,
    )

    with pytest.raises(RuntimeError, match="120GB hard stop"):
        guard.check(projected_additional_bytes=20 * GIB)


def test_budget_preserves_reserve_on_filesystem(tmp_path: Path) -> None:
    guard = DiskBudgetGuard(
        project_root=tmp_path,
        hard_stop_bytes=120 * GIB,
        reserve_bytes=30 * GIB,
        usage_reader=lambda path: 10 * GIB,
        free_reader=lambda path: 35 * GIB,
    )

    with pytest.raises(RuntimeError, match="30GB reserve"):
        guard.check(projected_additional_bytes=6 * GIB)


def test_sha256_file_uses_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")

    assert (
        sha256_file(path)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
