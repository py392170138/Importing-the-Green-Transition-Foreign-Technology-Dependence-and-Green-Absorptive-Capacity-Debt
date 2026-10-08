"""Build the deterministic two-endogenous-variable LP recovery fixture."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parent
SEED = 20260820


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet(frame: pl.DataFrame, destination: Path) -> None:
    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        frame.write_parquet(partial)
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _atomic_json(payload: dict[str, object], destination: Path) -> None:
    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    try:
        with partial.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    rng = np.random.default_rng(SEED)
    n_economies, n_years = 60, 20
    economy_index = np.repeat(np.arange(n_economies), n_years)
    year_index = np.tile(np.arange(n_years), n_economies)
    gad = np.clip(rng.normal(0.8, 0.5, n_economies * n_years), 0, None)
    z = rng.normal(size=n_economies * n_years)
    first_stage_error = rng.normal(scale=0.7, size=z.size)
    structural_error = rng.normal(scale=0.6, size=z.size)
    country_fe = rng.normal(scale=0.5, size=n_economies)[economy_index]
    year_fe = rng.normal(scale=0.3, size=n_years)[year_index]
    gimc = 0.9 * z + 0.4 * z * gad + first_stage_error
    gimc_gad = gimc * gad
    y = (
        2.0 * gimc
        - 0.5 * gimc_gad
        + 0.3 * gad
        + country_fe
        + year_fe
        + structural_error
    )
    frame = pl.DataFrame(
        {
            "economy_id": [f"E{value:03d}" for value in economy_index],
            "treatment_time": (2000 + year_index).astype(np.int16),
            "delta_outcome": y,
            "gimc_a": gimc,
            "gimc_gad_a": gimc_gad,
            "gad_a": gad,
            "z_a": z,
            "z_gad_a": z * gad,
            "renewable_energy_consumption_share_a": rng.normal(size=z.size),
            "trade_openness_a": rng.normal(size=z.size),
            "industry_value_added_share_a": rng.normal(size=z.size),
            "gdp_per_capita_a": rng.normal(size=z.size),
        }
    ).sort("economy_id", "treatment_time")
    if frame.height != 1_200:
        raise ValueError("synthetic fixture must contain exactly 1,200 rows")
    if frame.select(pl.struct("economy_id", "treatment_time").n_unique()).item() != 1_200:
        raise ValueError("synthetic economy-year key must be unique")

    parquet = ROOT / "synthetic_lp_panel.parquet"
    _atomic_parquet(frame, parquet)
    _atomic_json(
        {
            "beta": 2.0,
            "economies": n_economies,
            "parquet_sha256": _sha256(parquet),
            "rows": frame.height,
            "seed": SEED,
            "theta": -0.5,
            "years": n_years,
        },
        ROOT / "synthetic_truth.json",
    )


if __name__ == "__main__":
    main()
