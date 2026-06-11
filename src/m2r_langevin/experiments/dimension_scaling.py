"""Dimension-scaling experiment."""

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
from m2r_langevin.targets import make_correlated_gaussian

DEFAULT_OUTPUT_PATHS = {"summary": "results/summary/dimension_scaling.csv"}


def run(config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Run dimension scaling on AR(1)-correlated Gaussian targets."""

    target_config = nested_config(config, "target", {})
    rho = float(target_config.get("rho", 0.8))
    dimensions = [int(d) for d in config_value(config, "dimension_grid", [2, 5, 10])]
    step_scales = nested_config(
        config,
        "step_scales",
        {"ULA": 0.45, "MALA": 0.75, "KLMC": 0.60},
    )
    seeds = [int(seed) for seed in config_value(config, "seeds", [0])]
    n_samples = int(config_value(config, "n_samples", 12_000))
    burn = int(config_value(config, "burn", 2_000))
    gamma = float(config_value(config, "gamma", 2.0))
    max_lag = int(config_value(config, "max_lag", 1000))
    validate_burn(n_samples, burn)

    rows: list[dict[str, object]] = []
    for d in dimensions:
        target = make_correlated_gaussian(d=d, rho=rho)
        lipschitz = float(np.linalg.eigvalsh(target.precision).max())
        for seed in seeds:
            for sampler in sampler_names(config):
                step_scale = float(step_scales.get(sampler, 0.5))
                h = step_scale / lipschitz
                result = run_sampler(
                    sampler,
                    target.log_prob,
                    target.grad_log_prob,
                    np.zeros(d),
                    h,
                    n_samples,
                    seed + d,
                    gamma=gamma,
                )
                rows.append(
                    summarise_sampler_result(
                        result,
                        experiment="dimension_scaling",
                        setting=f"d={d}",
                        seed=seed + d,
                        burn=burn,
                        dimension=d,
                        gamma=gamma if sampler == "KLMC" else None,
                        reference_mean=target.true_mean,
                        reference_cov=target.true_cov,
                        max_lag=max_lag,
                        extra={
                            "rho": rho,
                            "lipschitz": lipschitz,
                            "step_scale": step_scale,
                        },
                    )
                )

    return {"summary": rows_to_dataframe(rows)}


def output_paths(config: Mapping[str, Any]) -> dict[str, object]:
    return resolve_output_paths(config, DEFAULT_OUTPUT_PATHS)


__all__ = ["DEFAULT_OUTPUT_PATHS", "output_paths", "run"]
