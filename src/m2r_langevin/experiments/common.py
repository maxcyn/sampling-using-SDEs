"""Shared helpers for experiment modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from m2r_langevin.costs import sampler_cost_fields
from m2r_langevin.diagnostics import summarise_chain
from m2r_langevin.samplers import GradLogProb, LogProb, SamplerResult, klmc, mala, ula

SAMPLERS = ("ULA", "MALA", "KLMC")


def config_value(config: Mapping[str, Any], key: str, default: Any) -> Any:
    """Read a config value with a default."""

    return config[key] if key in config else default


def nested_config(
    config: Mapping[str, Any],
    key: str,
    default: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read a nested config mapping as a plain dict."""

    value = config.get(key, default or {})
    if value is None:
        return {}
    return dict(value)


def validate_burn(n_samples: int, burn: int) -> None:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if burn < 0:
        raise ValueError("burn must be non-negative")
    if burn >= n_samples - 1:
        raise ValueError("burn must leave at least two retained samples")


def run_sampler(
    sampler: str,
    log_prob: LogProb,
    grad_log_prob: GradLogProb,
    x0: ArrayLike,
    h: float,
    n_samples: int,
    seed: int,
    gamma: float = 2.0,
) -> SamplerResult:
    """Run one configured sampler."""

    if sampler == "ULA":
        return ula(grad_log_prob, x0, h, n_samples, rng=seed)
    if sampler == "MALA":
        return mala(log_prob, grad_log_prob, x0, h, n_samples, rng=seed)
    if sampler == "KLMC":
        return klmc(grad_log_prob, x0, h, n_samples, gamma=gamma, rng=seed)
    raise ValueError(f"unknown sampler: {sampler}")


def summarise_sampler_result(
    result: SamplerResult,
    experiment: str,
    setting: str,
    seed: int,
    burn: int,
    dimension: int,
    gamma: float | None = None,
    reference_mean: ArrayLike | None = None,
    reference_cov: ArrayLike | None = None,
    max_lag: int | None = 1000,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one flat CSV-ready result row from a sampler run."""

    retained = result.samples[burn:]
    summary = dict(
        summarise_chain(
            retained,
            reference_mean=reference_mean,
            reference_cov=reference_cov,
            max_lag=max_lag,
        )
    )
    post_burn_samples = int(summary.pop("n_samples"))
    summary.pop("dimension")

    row: dict[str, object] = {
        "experiment": experiment,
        "sampler": result.sampler,
        "setting": setting,
        "seed": int(seed),
        "h": float(result.step_size),
        "gamma": np.nan if gamma is None else float(gamma),
        "n_samples": int(result.samples.shape[0]),
        "burn": int(burn),
        "post_burn_samples": post_burn_samples,
        "dimension": int(dimension),
        "acceptance_rate": (
            np.nan if result.acceptance_rate is None else float(result.acceptance_rate)
        ),
        **summary,
    }
    row.update(sampler_cost_fields(result, min_ess=float(summary["min_ess"])))
    if extra:
        row.update(dict(extra))
    return row


def sampler_names(config: Mapping[str, Any]) -> list[str]:
    names = list(config.get("samplers", SAMPLERS))
    unknown = sorted(set(names) - set(SAMPLERS))
    if unknown:
        raise ValueError(f"unknown sampler names: {unknown}")
    return names


__all__ = [
    "SAMPLERS",
    "config_value",
    "nested_config",
    "run_sampler",
    "sampler_names",
    "summarise_sampler_result",
    "validate_burn",
]
