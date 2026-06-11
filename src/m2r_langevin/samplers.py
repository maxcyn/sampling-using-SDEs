"""Core Langevin sampler implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
LogProb = Callable[[Array], float]
GradLogProb = Callable[[Array], ArrayLike]


@dataclass
class SamplerResult:
    """Samples and accounting metadata returned by every sampler."""

    samples: Array
    sampler: str
    step_size: float
    grad_evals: int
    logprob_evals: int = 0
    acceptance_rate: float | None = None
    runtime_seconds: float | None = None
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_position(x0: ArrayLike) -> Array:
    x = np.asarray(x0, dtype=float)
    if x.ndim == 0:
        x = x.reshape(1)
    if x.ndim != 1:
        raise ValueError("x0 must be a scalar or one-dimensional array")
    return x.copy()


def _as_rng(
    rng: int | np.random.Generator | None,
) -> tuple[np.random.Generator, int | None]:
    if rng is None:
        return np.random.default_rng(), None
    if isinstance(rng, np.random.Generator):
        return rng, None
    seed = int(rng)
    return np.random.default_rng(seed), seed


def _check_sampler_inputs(step_size: float, n_samples: int) -> None:
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")


def ula(
    grad_log_prob: GradLogProb,
    x0: ArrayLike,
    step_size: float,
    n_samples: int,
    rng: int | np.random.Generator | None = None,
) -> SamplerResult:
    """Run the Unadjusted Langevin Algorithm.

    The discretisation is ``x <- x + h * grad_log_prob(x) + sqrt(2h) * noise``.
    """

    _check_sampler_inputs(step_size, n_samples)
    generator, seed = _as_rng(rng)
    x = _as_position(x0)
    d = x.size
    h = float(step_size)
    noise_scale = np.sqrt(2.0 * h)
    samples = np.empty((n_samples, d), dtype=float)

    start = perf_counter()
    for k in range(n_samples):
        grad = np.asarray(grad_log_prob(x), dtype=float)
        x = x + h * grad + noise_scale * generator.standard_normal(d)
        samples[k] = x
    runtime_seconds = perf_counter() - start

    return SamplerResult(
        samples=samples,
        sampler="ULA",
        step_size=h,
        grad_evals=n_samples,
        runtime_seconds=runtime_seconds,
        seed=seed,
    )


def mala(
    log_prob: LogProb,
    grad_log_prob: GradLogProb,
    x0: ArrayLike,
    step_size: float,
    n_samples: int,
    rng: int | np.random.Generator | None = None,
) -> SamplerResult:
    """Run the Metropolis-Adjusted Langevin Algorithm.

    Gradients and log densities at the current state are cached. The reported
    costs count fresh evaluations: one initial evaluation of each, then one
    proposal log-density and gradient evaluation per step.
    """

    _check_sampler_inputs(step_size, n_samples)
    generator, seed = _as_rng(rng)
    x = _as_position(x0)
    d = x.size
    h = float(step_size)
    noise_scale = np.sqrt(2.0 * h)
    samples = np.empty((n_samples, d), dtype=float)
    n_accept = 0

    start = perf_counter()
    lp_x = float(log_prob(x))
    g_x = np.asarray(grad_log_prob(x), dtype=float)
    logprob_evals = 1
    grad_evals = 1

    for k in range(n_samples):
        mean_x = x + h * g_x
        y = mean_x + noise_scale * generator.standard_normal(d)

        lp_y = float(log_prob(y))
        g_y = np.asarray(grad_log_prob(y), dtype=float)
        logprob_evals += 1
        grad_evals += 1

        mean_y = y + h * g_y
        log_q_y_given_x = -np.sum((y - mean_x) ** 2) / (4.0 * h)
        log_q_x_given_y = -np.sum((x - mean_y) ** 2) / (4.0 * h)
        log_alpha = (lp_y - lp_x) + (log_q_x_given_y - log_q_y_given_x)

        if np.log(generator.random()) < log_alpha:
            x = y
            lp_x = lp_y
            g_x = g_y
            n_accept += 1

        samples[k] = x

    runtime_seconds = perf_counter() - start

    return SamplerResult(
        samples=samples,
        sampler="MALA",
        step_size=h,
        grad_evals=grad_evals,
        logprob_evals=logprob_evals,
        acceptance_rate=n_accept / n_samples,
        runtime_seconds=runtime_seconds,
        seed=seed,
    )


def klmc(
    grad_log_prob: GradLogProb,
    x0: ArrayLike,
    step_size: float,
    n_samples: int,
    gamma: float = 2.0,
    rng: int | np.random.Generator | None = None,
) -> SamplerResult:
    """Run first-order kinetic Langevin Monte Carlo with exact OU noise.

    This is the later KLMC implementation from the exploratory notebook,
    standardised as the main kinetic sampler. Only position samples are returned;
    velocity is kept internally.
    """

    _check_sampler_inputs(step_size, n_samples)
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    generator, seed = _as_rng(rng)
    x = _as_position(x0)
    d = x.size
    v = generator.standard_normal(d)
    h = float(step_size)
    gamma = float(gamma)
    samples = np.empty((n_samples, d), dtype=float)

    a = np.exp(-gamma * h)
    psi1 = (1.0 - a) / gamma
    psi2 = (h - psi1) / gamma

    c00 = (1.0 - a**2) / (2.0 * gamma)
    c01 = (1.0 - a) ** 2 / (2.0 * gamma**2)
    c11 = (h - 2.0 * (1.0 - a) / gamma + (1.0 - a**2) / (2.0 * gamma)) / gamma**2
    noise_cov = 2.0 * gamma * np.array([[c00, c01], [c01, c11]], dtype=float)
    noise_chol = np.linalg.cholesky(noise_cov + 1e-15 * np.eye(2))

    start = perf_counter()
    for k in range(n_samples):
        grad = np.asarray(grad_log_prob(x), dtype=float)
        z = generator.standard_normal((2, d))
        noise = noise_chol @ z
        v_new = a * v + psi1 * grad + noise[0]
        x_new = x + psi1 * v + psi2 * grad + noise[1]
        v = v_new
        x = x_new
        samples[k] = x
    runtime_seconds = perf_counter() - start

    return SamplerResult(
        samples=samples,
        sampler="KLMC",
        step_size=h,
        grad_evals=n_samples,
        runtime_seconds=runtime_seconds,
        seed=seed,
        metadata={"gamma": gamma},
    )


__all__ = ["SamplerResult", "klmc", "mala", "ula"]
