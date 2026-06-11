from pathlib import Path

import pandas as pd

from m2r_langevin.experiments import (
    dimension_scaling,
    gaussian_sweep,
    klmc_friction,
    linear_regression,
    logistic_regression,
    precision_scaling,
)
from m2r_langevin.results import write_outputs

REQUIRED_SUMMARY_COLUMNS = {
    "experiment",
    "sampler",
    "setting",
    "seed",
    "h",
    "n_samples",
    "burn",
    "post_burn_samples",
    "dimension",
    "cov_error",
    "min_ess",
    "grad_evals",
    "logprob_evals",
    "work_units",
    "ess_per_work",
    "acceptance_rate",
}


def assert_summary_output(df: pd.DataFrame) -> None:
    assert not df.empty
    assert REQUIRED_SUMMARY_COLUMNS <= set(df.columns)
    assert (df["post_burn_samples"] >= 2).all()
    assert (df["min_ess"] > 0.0).all()


def assert_writes_outputs(
    module, config: dict, tmp_path: Path
) -> dict[str, pd.DataFrame]:
    outputs = module.run(config)
    written = write_outputs(outputs, module.output_paths(config))

    assert set(written) == set(outputs)
    for path in written.values():
        assert path.exists()
        assert path.is_file()
    return outputs


def test_gaussian_sweep_smoke(tmp_path: Path) -> None:
    config = {
        "output_path": tmp_path / "gaussian.csv",
        "target": {"d": 2, "rho": 0.5},
        "samplers": ["ULA", "MALA", "KLMC"],
        "step_grid": [0.01],
        "seeds": [0],
        "n_samples": 50,
        "burn": 10,
        "max_lag": 20,
    }

    outputs = assert_writes_outputs(gaussian_sweep, config, tmp_path)

    assert_summary_output(outputs["summary"])
    assert set(outputs["summary"]["sampler"]) == {"ULA", "MALA", "KLMC"}


def test_linear_regression_smoke(tmp_path: Path) -> None:
    config = {
        "output_path": tmp_path / "linear.csv",
        "target": {"n": 30, "d": 2, "sigma": 1.0, "tau": 3.0, "seed": 0},
        "samplers": ["ULA", "MALA", "KLMC"],
        "step_grid": [0.001],
        "seeds": [1],
        "n_samples": 50,
        "burn": 10,
        "max_lag": 20,
    }

    outputs = assert_writes_outputs(linear_regression, config, tmp_path)

    assert_summary_output(outputs["summary"])


def test_logistic_regression_smoke(tmp_path: Path) -> None:
    config = {
        "output_paths": {
            "summary": tmp_path / "logistic.csv",
            "reference": tmp_path / "logistic_reference.csv",
        },
        "target": {"n": 40, "d": 2, "tau": 3.0, "seed": 0},
        "reference": {"step_size": 0.002, "n_samples": 70, "burn": 10, "seed": 2},
        "samplers": ["ULA", "MALA", "KLMC"],
        "step_grid": [0.001],
        "seeds": [3],
        "n_samples": 50,
        "burn": 10,
        "max_lag": 20,
    }

    outputs = assert_writes_outputs(logistic_regression, config, tmp_path)

    assert_summary_output(outputs["summary"])
    assert not outputs["reference"].empty
    assert {"laplace_mean_diff", "laplace_cov_diff"} <= set(
        outputs["reference"].columns
    )


def test_dimension_scaling_smoke(tmp_path: Path) -> None:
    config = {
        "output_path": tmp_path / "dimension.csv",
        "target": {"rho": 0.4},
        "dimension_grid": [2, 3],
        "samplers": ["ULA", "MALA", "KLMC"],
        "seeds": [0],
        "n_samples": 50,
        "burn": 10,
        "max_lag": 20,
    }

    outputs = assert_writes_outputs(dimension_scaling, config, tmp_path)

    assert_summary_output(outputs["summary"])
    assert set(outputs["summary"]["dimension"]) == {2, 3}


def test_precision_scaling_smoke(tmp_path: Path) -> None:
    config = {
        "output_paths": {
            "raw": tmp_path / "precision_raw.csv",
            "first_budget": tmp_path / "precision_first.csv",
        },
        "target": {"d": 2, "rho": 0.4},
        "samplers": ["ULA", "MALA", "KLMC"],
        "budgets": [40, 60],
        "tolerances": [10.0],
        "burn_fraction": 0.2,
        "max_lag": 20,
    }

    outputs = assert_writes_outputs(precision_scaling, config, tmp_path)

    assert_summary_output(outputs["raw"])
    assert not outputs["first_budget"].empty
    assert {"tolerance", "budget", "mean_error", "cov_error"} <= set(
        outputs["first_budget"].columns
    )


def test_klmc_friction_smoke(tmp_path: Path) -> None:
    config = {
        "output_path": tmp_path / "friction.csv",
        "target": {"d": 2, "rho": 0.4},
        "gamma_grid": [0.5, 2.0],
        "n_samples": 50,
        "burn": 10,
        "max_lag": 20,
    }

    outputs = assert_writes_outputs(klmc_friction, config, tmp_path)

    assert_summary_output(outputs["summary"])
    assert set(outputs["summary"]["sampler"]) == {"KLMC"}
