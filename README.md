# M2R Langevin MCMC Experiments

This repository contains reproducible experiments comparing Langevin-type MCMC
methods for an M2R project. The planned package will move sampler, target,
diagnostic, plotting, and experiment code out of the exploratory notebook and
into command-line reproducible modules.

## Status

Phase 6 is in progress. The package scaffold is in place, the core ULA, MALA,
and KLMC samplers have been extracted into reusable package code, the main
targets are reusable constructors, central diagnostics define ESS and Gaussian
moment errors, and experiment scripts now write CSV summaries from YAML configs.
Plotting helpers generate PNG report figures from the saved CSVs.

## Quick Start

From this repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -c "import m2r_langevin; print(m2r_langevin.__version__)"
pytest
```

On macOS or Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Planned Outputs

- `results/summary/`: aggregated CSV summaries for report figures.
- `results/references/`: reference-chain summaries.
- `figures/report/`: report-ready figures.
- `tables/tex/`: LaTeX tables.

## Running Experiments

Run one configured experiment:

```powershell
python scripts/run_experiment.py --config configs/gaussian_2d.yaml
```

Run the configured suite:

```powershell
python scripts/run_all.py
```

Generate report figures from saved CSV outputs:

```powershell
python scripts/make_figures.py
```

The current report figures focus on mean error, covariance error, minimum ESS,
and ESS per work unit, where one work unit is one gradient or log-density
evaluation.

## Repository Layout

```text
configs/                 Experiment configuration files
src/m2r_langevin/        Python package
scripts/                 Command-line entry points
notebooks/               Archived and demonstration notebooks
results/                 Generated result data
figures/                 Draft and report figures
tables/                  CSV and LaTeX tables
report/                  Report source and synced report figures
tests/                   Pytest test suite
```

## Notes

The current scientific message to preserve is that ULA can mix quickly but has
finite-step bias, MALA controls that bias through the Metropolis correction, and
KLMC can be competitive but tuning-sensitive.
