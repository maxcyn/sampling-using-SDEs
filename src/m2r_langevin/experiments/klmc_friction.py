"""KLMC friction-sweep experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from m2r_langevin.experiments.common import (
    config_value,
    nested_config,
    run_sampler,
    summarise_sampler_result,
    validate_burn,
)
from m2r_langevin.results import resolve_output_paths, rows_to_dataframe
from m2r_langevin.targets import make_correlated_gaussian

DEFAULT_OUTPUT_PATHS = {"summary": "results/summary/klmc_friction_sweep.csv"}


def run(config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Run KLMC across a grid of friction values."""

    target_config = nested_config(config, "target", {})
    d = int(target_config.get("d", 10))
    rho = float(target_config.get("rho", 0.8))
    target = make_correlated_gaussian(d=d, rho=rho)
    lipschitz = float(np.linalg.eigvalsh(target.precision).max())

    gamma_grid = [
        float(gamma)
        for gamma in config_value(config, "gamma_grid", [0.25, 0.5, 1.0, 2.0])
    ]
    h = (
        float(config["h"])
        if "h" in config
        else float(config_value(config, "h_scale", 0.60)) / lipschitz
    )
    seed = int(config_value(config, "seed", 900))
    n_samples = int(config_value(config, "n_samples", 16_000))
    burn = int(config_value(config, "burn", 3_000))
    max_lag = int(config_value(config, "max_lag", 1000))
    validate_burn(n_samples, burn)

    rows: list[dict[str, object]] = []
    for gamma in gamma_grid:
        result = run_sampler(
            "KLMC",
            target.log_prob,
            target.grad_log_prob,
            np.zeros(d),
            h,
            n_samples,
            seed,
            gamma=gamma,
        )
        rows.append(
            summarise_sampler_result(
                result,
                experiment="klmc_friction",
                setting=f"gamma={gamma}",
                seed=seed,
                burn=burn,
                dimension=d,
                gamma=gamma,
                reference_mean=target.true_mean,
                reference_cov=target.true_cov,
                max_lag=max_lag,
                extra={"rho": rho, "lipschitz": lipschitz},
            )
        )

    return {"summary": rows_to_dataframe(rows)}


def output_paths(config: Mapping[str, Any]) -> dict[str, object]:
    return resolve_output_paths(config, DEFAULT_OUTPUT_PATHS)


__all__ = ["DEFAULT_OUTPUT_PATHS", "output_paths", "run"]
