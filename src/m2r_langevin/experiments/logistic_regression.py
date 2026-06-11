"""Bayesian logistic-regression cost-aware experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from m2r_langevin.costs import sampler_cost_fields
from m2r_langevin.diagnostics import summarise_chain
from m2r_langevin.experiments.common import (
    config_value,
    nested_config,
    run_sampler,
    sampler_names,
    summarise_sampler_result,
    validate_burn,
)
from m2r_langevin.results import resolve_output_paths, rows_to_dataframe
from m2r_langevin.samplers import mala
from m2r_langevin.targets import (
    logistic_laplace_approximation,
    make_logistic_regression,
)

DEFAULT_OUTPUT_PATHS = {
    "summary": "results/summary/logistic_regression_cost_aware.csv",
    "reference": "results/references/logistic_reference_chain_summary.csv",
}


def run(config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Run logistic sampler comparisons against a long MALA reference chain."""

    target_config = nested_config(config, "target", {})
    target = make_logistic_regression(
        n=int(target_config.get("n", 400)),
        d=int(target_config.get("d", 5)),
        tau=float(target_config.get("tau", 10.0)),
        seed=int(target_config.get("seed", 0)),
    )
    d = int(target.X.shape[1])
    laplace = logistic_laplace_approximation(
        target.log_prob,
        target.grad_log_prob,
        target.X,
        target.y,
        target.tau,
    )

    reference_config = nested_config(
        config,
        "reference",
        {"step_size": 0.004, "n_samples": 80_000, "burn": 10_000, "seed": 123},
    )
    ref_n_samples = int(reference_config.get("n_samples", 80_000))
    ref_burn = int(reference_config.get("burn", 10_000))
    ref_step_size = float(reference_config.get("step_size", 0.004))
    ref_seed = int(reference_config.get("seed", 123))
    max_lag = int(config_value(config, "max_lag", 1000))
    validate_burn(ref_n_samples, ref_burn)

    reference_result = mala(
        target.log_prob,
        target.grad_log_prob,
        laplace.beta_map,
        ref_step_size,
        ref_n_samples,
        rng=ref_seed,
    )
    reference_samples = reference_result.samples[ref_burn:]
    reference_mean = reference_samples.mean(axis=0)
    reference_cov = np.asarray(
        np.cov(reference_samples, rowvar=False),
        dtype=float,
    ).reshape(d, d)
    reference_summary = dict(summarise_chain(reference_samples, max_lag=max_lag))
    reference_post_burn_samples = int(reference_summary.pop("n_samples"))
    reference_summary.pop("dimension")
    reference_row: dict[str, object] = {
        "experiment": "logistic_reference",
        "sampler": "MALA",
        "setting": "reference",
        "seed": ref_seed,
        "h": ref_step_size,
        "gamma": np.nan,
        "n_samples": ref_n_samples,
        "burn": ref_burn,
        "post_burn_samples": reference_post_burn_samples,
        "dimension": d,
        "acceptance_rate": reference_result.acceptance_rate,
        "laplace_mean_diff": float(np.linalg.norm(laplace.beta_map - reference_mean)),
        "laplace_cov_diff": float(
            np.linalg.norm(laplace.laplace_covariance - reference_cov, ord="fro")
        ),
        "laplace_success": bool(laplace.optimizer_result.success),
        **reference_summary,
    }
    reference_row.update(
        sampler_cost_fields(
            reference_result, min_ess=float(reference_summary["min_ess"])
        )
    )

    step_grid = [
        float(h)
        for h in config_value(
            config,
            "step_grid",
            [0.001, 0.002, 0.004, 0.006],
        )
    ]
    seeds = [int(seed) for seed in config_value(config, "seeds", [0])]
    n_samples = int(config_value(config, "n_samples", 30_000))
    burn = int(config_value(config, "burn", 5_000))
    gamma = float(config_value(config, "gamma", 2.0))
    setting = str(config_value(config, "setting", "logistic"))
    validate_burn(n_samples, burn)

    rows: list[dict[str, object]] = []
    for h in step_grid:
        for seed in seeds:
            for sampler in sampler_names(config):
                result = run_sampler(
                    sampler,
                    target.log_prob,
                    target.grad_log_prob,
                    laplace.beta_map,
                    h,
                    n_samples,
                    seed,
                    gamma=gamma,
                )
                rows.append(
                    summarise_sampler_result(
                        result,
                        experiment="logistic_regression",
                        setting=setting,
                        seed=seed,
                        burn=burn,
                        dimension=d,
                        gamma=gamma if sampler == "KLMC" else None,
                        reference_mean=reference_mean,
                        reference_cov=reference_cov,
                        max_lag=max_lag,
                        extra={
                            "target_n": int(target.X.shape[0]),
                            "tau": target.tau,
                            "reference_seed": ref_seed,
                            "reference_h": ref_step_size,
                        },
                    )
                )

    return {
        "summary": rows_to_dataframe(rows),
        "reference": rows_to_dataframe([reference_row]),
    }


def output_paths(config: Mapping[str, Any]) -> dict[str, object]:
    return resolve_output_paths(config, DEFAULT_OUTPUT_PATHS)


__all__ = ["DEFAULT_OUTPUT_PATHS", "output_paths", "run"]
