"""Bayesian linear-regression cost-aware experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
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
from m2r_langevin.targets import make_linear_regression

DEFAULT_OUTPUT_PATHS = {"summary": "results/summary/linear_regression_cost_aware.csv"}


def run(config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Run the main sampler comparison on Bayesian linear regression."""

    target_config = nested_config(config, "target", {})
    target = make_linear_regression(
        n=int(target_config.get("n", 200)),
        d=int(target_config.get("d", 5)),
        sigma=float(target_config.get("sigma", 1.0)),
        tau=float(target_config.get("tau", 10.0)),
        seed=int(target_config.get("seed", 0)),
    )
    d = int(target.posterior_mean.size)

    step_grid = [
        float(h) for h in config_value(config, "step_grid", [0.0005, 0.001, 0.002])
    ]
    seeds = [int(seed) for seed in config_value(config, "seeds", [0])]
    n_samples = int(config_value(config, "n_samples", 12_000))
    burn = int(config_value(config, "burn", 2_000))
    gamma = float(config_value(config, "gamma", 2.0))
    max_lag = int(config_value(config, "max_lag", 1000))
    setting = str(config_value(config, "setting", "linear"))
    validate_burn(n_samples, burn)

    rows: list[dict[str, object]] = []
    x0 = np.zeros(d)
    for h in step_grid:
        for seed in seeds:
            for sampler in sampler_names(config):
                result = run_sampler(
                    sampler,
                    target.log_prob,
                    target.grad_log_prob,
                    x0,
                    h,
                    n_samples,
                    seed,
                    gamma=gamma,
                )
                rows.append(
                    summarise_sampler_result(
                        result,
                        experiment="linear_regression",
                        setting=setting,
                        seed=seed,
                        burn=burn,
                        dimension=d,
                        gamma=gamma if sampler == "KLMC" else None,
                        reference_mean=target.posterior_mean,
                        reference_cov=target.posterior_cov,
                        max_lag=max_lag,
                        extra={
                            "target_n": int(target.X.shape[0]),
                            "sigma": float(target_config.get("sigma", 1.0)),
                            "tau": float(target_config.get("tau", 10.0)),
                        },
                    )
                )

    return {"summary": rows_to_dataframe(rows)}


def output_paths(config: Mapping[str, Any]) -> dict[str, object]:
    return resolve_output_paths(config, DEFAULT_OUTPUT_PATHS)


__all__ = ["DEFAULT_OUTPUT_PATHS", "output_paths", "run"]
