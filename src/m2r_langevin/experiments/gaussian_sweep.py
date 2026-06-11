"""Gaussian step-size sweep experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from m2r_langevin.experiments.common import (
    config_value,
    nested_config,
    run_sampler,
    sampler_names,
    summarise_sampler_result,
    validate_burn,
)
from m2r_langevin.results import resolve_output_paths, rows_to_dataframe
from m2r_langevin.targets import make_correlated_gaussian

DEFAULT_OUTPUT_PATHS = {"summary": "results/summary/gaussian_2d_sweep.csv"}


def run(config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Run ULA, MALA, and KLMC on an AR(1)-correlated Gaussian target."""

    target_config = nested_config(config, "target", {"d": 2, "rho": 0.9})
    d = int(target_config.get("d", 2))
    rho = float(target_config.get("rho", 0.9))
    target = make_correlated_gaussian(d=d, rho=rho)

    step_grid = [float(h) for h in config_value(config, "step_grid", [0.005, 0.01])]
    seeds = [int(seed) for seed in config_value(config, "seeds", [0])]
    n_samples = int(config_value(config, "n_samples", 12_000))
    burn = int(config_value(config, "burn", 2_000))
    gamma = float(config_value(config, "gamma", 2.0))
    max_lag = int(config_value(config, "max_lag", 1000))
    setting = str(config_value(config, "setting", "gaussian"))
    validate_burn(n_samples, burn)

    rows: list[dict[str, object]] = []
    for h in step_grid:
        for seed in seeds:
            for sampler in sampler_names(config):
                result = run_sampler(
                    sampler,
                    target.log_prob,
                    target.grad_log_prob,
                    target.true_mean,
                    h,
                    n_samples,
                    seed,
                    gamma=gamma,
                )
                rows.append(
                    summarise_sampler_result(
                        result,
                        experiment="gaussian_sweep",
                        setting=setting,
                        seed=seed,
                        burn=burn,
                        dimension=d,
                        gamma=gamma if sampler == "KLMC" else None,
                        reference_mean=target.true_mean,
                        reference_cov=target.true_cov,
                        max_lag=max_lag,
                        extra={"rho": rho},
                    )
                )

    return {"summary": rows_to_dataframe(rows)}


def output_paths(config: Mapping[str, Any]) -> dict[str, object]:
    return resolve_output_paths(config, DEFAULT_OUTPUT_PATHS)


__all__ = ["DEFAULT_OUTPUT_PATHS", "output_paths", "run"]
