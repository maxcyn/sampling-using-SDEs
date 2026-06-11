# Langevin MCMC Experiments

This repository contains the code, configuration files, saved outputs, and
figure-generation scripts for my report on Langevin-type Markov chain Monte
Carlo methods. It is intended as a reproducibility companion: readers should be
able to rerun the numerical experiments from the report and regenerate the CSV
summaries and figures used there.

The experiments compare the unadjusted Langevin algorithm (ULA), the
Metropolis-adjusted Langevin algorithm (MALA), and kinetic Langevin Monte Carlo
(KLMC) on Gaussian, Bayesian linear regression, and Bayesian logistic
regression targets.

## Requirements

- Python 3.10 or later
- A shell from the repository root

Create and activate a virtual environment, then install the project in editable
mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS or Linux, use this activation command instead:

```bash
source .venv/bin/activate
```

Check that the package and test suite are available:

```powershell
python -c "import m2r_langevin; print(m2r_langevin.__version__)"
pytest
```

## Reproducing the Report Outputs

All experiment settings used by the command-line scripts are stored in
`configs/`. To rerun the full configured suite and then regenerate the report
figures, run:

```powershell
python scripts/run_all.py
python scripts/make_figures.py
```

The first command writes summary CSV files under `results/summary/` and, for
the logistic-regression experiment, a reference-chain summary under
`results/references/`. The second command rebuilds the report figures under
`fig/`, including the Gaussian trace/autocorrelation diagnostic.

The scripts use fixed random seeds from the YAML configuration files, so reruns
should be directly comparable with the checked-in outputs. Small numerical
differences can still occur across Python, NumPy, or BLAS/LAPACK versions.

## Running Individual Experiments

Run a single experiment by passing its configuration file:

```powershell
python scripts/run_experiment.py --config configs/gaussian_2d.yaml
```

The configured experiments are:

| Configuration | Purpose | Main output |
| --- | --- | --- |
| `configs/gaussian_2d.yaml` | Step-size sweep on a strongly correlated 2D Gaussian target | `results/summary/gaussian_2d_sweep.csv` |
| `configs/linear_regression.yaml` | Cost-aware comparison on Bayesian linear regression | `results/summary/linear_regression_cost_aware.csv` |
| `configs/logistic_regression.yaml` | Cost-aware comparison on Bayesian logistic regression, scored against a long MALA reference chain | `results/summary/logistic_regression_cost_aware.csv` |
| `configs/dimension_scaling.yaml` | Scaling comparison across Gaussian target dimensions | `results/summary/dimension_scaling.csv` |
| `configs/precision_scaling.yaml` | Accuracy-versus-budget comparison for increasing work budgets | `results/summary/precision_scaling_raw.csv` |
| `configs/klmc_friction_sweep.yaml` | KLMC friction-parameter sweep | `results/summary/klmc_friction_sweep.csv` |

After running one or more experiments, regenerate figures from the available
CSV files:

```powershell
python scripts/make_figures.py --allow-missing
```

Use `--allow-missing` when only a subset of the CSV files has been regenerated.

## Generated Artifacts

- `results/summary/`: aggregated numerical summaries used for plots and tables.
- `results/references/`: reference-chain summaries for non-Gaussian targets.
- `fig/`: PNG figures included in the report.
- `tables/`: table outputs used during report preparation.

These outputs are checked into the repository so readers can inspect the
reported results without rerunning every experiment. To reproduce them from
scratch, rerun the commands in "Reproducing the Report Outputs".

## Repository Layout

```text
configs/                 YAML experiment configurations
src/m2r_langevin/        Samplers, targets, diagnostics, plotting, and results code
src/m2r_langevin/experiments/
                         Experiment implementations called by the scripts
scripts/                 Command-line entry points for experiments and figures
notebooks/               Archived exploratory notebooks
results/                 Generated CSV outputs
fig/                     Report-ready PNG figures
report/                  Report assets
tests/                   Pytest test suite
```

## Notes for Readers

The cost-aware plots use a simple work count based on gradient and log-density
evaluations rather than wall-clock time. The logistic-regression errors are
computed against a long MALA reference chain, while the Gaussian and linear
regression experiments use exact moment information where available.
