"""Cost-accounting helpers for sampler comparisons."""

from __future__ import annotations

import math

from m2r_langevin.samplers import SamplerResult


def safe_rate(numerator: float, denominator: float | int | None) -> float:
    """Return ``numerator / denominator`` or NaN for unavailable denominators."""

    if denominator is None or denominator <= 0:
        return math.nan
    return float(numerator) / float(denominator)


def cost_rates(
    min_ess: float,
    grad_evals: int,
    logprob_evals: int = 0,
) -> dict[str, float | int]:
    """Compute cost-normalised ESS metrics."""

    work_units = int(grad_evals) + int(logprob_evals)
    return {
        "work_units": work_units,
        "ess_per_work": safe_rate(min_ess, work_units),
    }


def sampler_cost_fields(
    result: SamplerResult, min_ess: float
) -> dict[str, float | int]:
    """Return sampler accounting fields plus derived ESS rates."""

    return {
        "grad_evals": int(result.grad_evals),
        "logprob_evals": int(result.logprob_evals),
        **cost_rates(
            min_ess=min_ess,
            grad_evals=result.grad_evals,
            logprob_evals=result.logprob_evals,
        ),
    }


__all__ = ["cost_rates", "safe_rate", "sampler_cost_fields"]
