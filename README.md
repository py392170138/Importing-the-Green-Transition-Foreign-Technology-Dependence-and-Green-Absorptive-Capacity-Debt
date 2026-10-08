# Importing the Green Transition: Foreign Technology Dependence and Green Absorptive-Capacity Debt

This repository contains the **data acquisition, harmonisation, variable construction, and statistical analysis code** for the study. It is a code release; third-party raw data, derived country panels, model outputs, and manuscript files are not redistributed here.

## Data and scope

The study combines CEPII BACI trade data, OECD TiVA, OpenAlex country-year aggregates, IRENA renewable-energy statistics, World Bank WDI, ILOSTAT, OECD climate-policy data, and public product classifications. The frozen local acquisition used approximately 4.3 GB of raw files; the main analysis period is 2000–2022. The locally built model panel contains 55,514 rows and 52 distinct economies across its sample versions; this is the realised analysis coverage, not the 80-economy coverage of the broader OECD source. See [DATA_SOURCES.md](DATA_SOURCES.md) for source URLs, frozen versions, roles, and data caveats. The project configuration enforces a 150 GB total storage ceiling with a 120 GB stop threshold.

## Code layout

- `03_代码/src/green_debt/`: Python acquisition, cleaning, construction, checks, diagnostics, and analysis orchestration.
- `03_代码/R/`: fixed-effect, local-projection, threshold, weak-IV, shift-share, and reporting code.
- `03_代码/tests/` and `03_代码/tests_r/`: tests and small synthetic fixtures.
- `config/`, `contracts/`, `03_代码/contracts/`: frozen specifications, data contracts, and source definitions.
- `02_数据字典/`: small frozen crosswalks and product/topic registries needed by the build.
- `pyproject.toml`, `uv.lock`, `renv.lock`, `julia/`: dependency specifications.

## Reproduce

Python 3.12 is required. The acquisition route guard and analysis reproduction broker use macOS/Darwin facilities; run the complete pipeline on macOS. Obtain the original data from their providers and place them under `04_原始数据/` in the layout expected by `config/sources.yaml`. No network access is required for analysis after the data and dependencies have been installed.

```sh
uv sync --frozen
.venv/bin/python -m green_debt.cli config-check
.venv/bin/python -m green_debt.cli build-status --data-root "$PWD"
.venv/bin/python -m green_debt.cli build --target checkpoint3 --data-root "$PWD"
03_代码/bin/run-analysis --data-root "$PWD"
.venv/bin/python -m green_debt.cli analysis-output-audit \
  --data-root "$PWD" --output-root "06_结果/analysis"
```

The frozen R environment is specified in `renv.lock`; Julia dependencies are in `julia/`. The runner verifies dependency versions and stops if the frozen inputs or environment are unavailable. See `03_代码/R/bootstrap_analysis_env.R` for the R bootstrap. The full build creates `05_中间数据/` and `06_结果/`; these directories are intentionally absent from this release.

`07_文献与日志/公共数据源登记表_v0.1.csv` is an initial planning registry retained for source-catalog tests. Its original “not downloaded” entries are historical planning statuses; [DATA_SOURCES.md](DATA_SOURCES.md) describes the later local acquisition snapshot.

## Inference status

The local analysis completed an output audit, but the evidence policy classified all 39 confirmatory cells as **exploratory** because the exposure-concentration cutoff was not preregistered in executable form. The breadth assessments for H1–H3 were mixed or inconclusive. The code and this repository should not be read as confirming a causal debt-trap claim.

## Reuse

Source data remain subject to each provider's terms. No software license has been assigned to this code release; request permission from the rights holder before reuse beyond GitHub's default permissions.
