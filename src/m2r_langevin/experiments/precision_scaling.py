"""Precision/budget-scaling experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import math

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

DEFAULT_OUTPUT_PATHS = {
    "raw": "results/summary/precision_scaling_raw.csv",
    "first_budget": "results/summary/precision_scaling_first_budget.csv",
}


def run(config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Run cost-budget scaling and first-budget-to-tolerance summaries."""

    target_config = nested_config(config, "target", {})
    d = int(target_config.get("d", 10))
    rho = float(target_config.get("rho", 0.8))
    target = make_correlated_gaussian(d=d, rho=rho)
    lipschitz = float(np.linalg.eigvalsh(target.precision).max())

    step_scales = nested_config(
        config,
        "step_scales",
        {"ULA": 0.45, "MALA": 0.75, "KLMC": 0.60},
    )
    budgets = [int(budget) for budget in config_value(config, "budgets", [3000, 6000])]
    tolerances = [
        float(tolerance)
        for tolerance in config_value(config, "tolerances", [0.35, 0.25])
    ]
    tolerance_metric = str(config_value(config, "tolerance_metric", "cov_error"))
    burn_fraction = float(config_value(config, "burn_fraction", 0.20))
    seed = int(config_value(config, "seed", 500))
    gamma = float(config_value(config, "gamma", 2.0))
    max_lag = int(config_value(config, "max_lag", 1000))

    rows: list[dict[str, object]] = []
    for sampler in sampler_names(config):
        step_scale = float(step_scales.get(sampler, 0.5))
        h = step_scale / lipschitz
        for budget in budgets:
            burn = int(burn_fraction * budget)
            validate_burn(budget, burn)
            run_seed = seed + budget + len(sampler)
            result = run_sampler(
                sampler,
                target.log_prob,
                target.grad_log_prob,
                np.zeros(d),
                h,
                budget,
                run_seed,
                gamma=gamma,
            )
            rows.append(
                summarise_sampler_result(
                    result,
                    experiment="precision_scaling",
                    setting=f"N={budget}",
                    seed=run_seed,
                    burn=burn,
                    dimension=d,
                    gamma=gamma if sampler == "KLMC" else None,
                    reference_mean=target.true_mean,
                    reference_cov=target.true_cov,
                    max_lag=max_lag,
                    extra={
                        "budget": budget,
                        "rho": rho,
                        "lipschitz": lipschitz,
                        "step_scale": step_scale,
                    },
                )
            )

    raw = rows_to_dataframe(rows)
    summary_rows: list[dict[str, object]] = []
    for sampler in sampler_names(config):
        sampler_rows = raw[raw["sampler"] == sampler].sort_values("budget")
        if tolerance_metric not in sampler_rows.columns:
            raise ValueError(f"unknown tolerance_metric: {tolerance_metric}")
        h_values = sampler_rows["h"].unique()
        h_value = float(h_values[0]) if len(h_values) else math.nan
        for tolerance in tolerances:
            reached = sampler_rows[sampler_rows[tolerance_metric] <= tolerance]
            if reached.empty:
                summary_rows.append(
                    {
                        "experiment": "precision_scaling",
                        "sampler": sampler,
                        "setting": f"tol={tolerance}",
                        "tolerance": tolerance,
                        "h": h_value,
                        "dimension": d,
                        "tolerance_metric": tolerance_metric,
                        "budget": np.nan,
                        "grad_evals": np.nan,
                        "mean_error": np.nan,
                        "cov_error": np.nan,
                    }
                )
            else:
                first = reached.iloc[0].to_dict()
                summary_rows.append(
                    {
                        "experiment": "precision_scaling",
                        "sampler": sampler,
                        "setting": f"tol={tolerance}",
                        "tolerance": tolerance,
                        "h": h_value,
                        "dimension": d,
                        "tolerance_metric": tolerance_metric,
                        "budget": first["budget"],
                        "grad_evals": first["grad_evals"],
                        "mean_error": first["mean_error"],
                        "cov_error": first["cov_error"],
                    }
                )

    return {
        "raw": raw,
        "first_budget": rows_to_dataframe(summary_rows),
    }


def output_paths(config: Mapping[str, Any]) -> dict[str, object]:
    return resolve_output_paths(config, DEFAULT_OUTPUT_PATHS)


__all__ = ["DEFAULT_OUTPUT_PATHS", "output_paths", "run"]
