"""Chain diagnostics and error summaries."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def _as_1d(x: ArrayLike) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if x.size == 0:
        raise ValueError("x must contain at least one value")
    return x


def _as_samples(samples: ArrayLike) -> Array:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    if samples.ndim != 2:
        raise ValueError("samples must be a one- or two-dimensional array")
    if samples.shape[0] < 2:
        raise ValueError("samples must contain at least two draws")
    return samples


def _as_covariance(cov: ArrayLike, d: int) -> Array:
    cov = np.asarray(cov, dtype=float)
    if d == 1 and cov.ndim == 0:
        cov = cov.reshape(1, 1)
    if cov.shape != (d, d):
        raise ValueError(f"covariance must have shape ({d}, {d})")
    return cov


def _sample_covariance(samples: Array) -> Array:
    d = samples.shape[1]
    return np.asarray(np.cov(samples, rowvar=False), dtype=float).reshape(d, d)


def autocorr_fft(x: ArrayLike, max_lag: int | None = None) -> Array:
    """Estimate the normalised autocorrelation function using an FFT.

    Returns ``rho_0, rho_1, ...`` with ``rho_0 = 1``. ``max_lag`` is the number
    of returned lags, not the largest lag index.
    """

    x = _as_1d(x)
    n = x.size
    if max_lag is None:
        max_lag = n
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")
    max_lag = min(int(max_lag), n)

    centered = x - x.mean()
    total_variation = float(centered @ centered)
    if total_variation == 0.0:
        autocorr = np.zeros(max_lag, dtype=float)
        autocorr[0] = 1.0
        return autocorr

    fft_length = 1 << (2 * n - 1).bit_length()
    transformed = np.fft.rfft(centered, fft_length)
    autocov = np.fft.irfft(transformed * np.conjugate(transformed), fft_length)[:n].real
    return autocov[:max_lag] / autocov[0]


def integrated_autocorr_time(x: ArrayLike, max_lag: int | None = None) -> float:
    """Estimate integrated autocorrelation time with Geyer's truncation rule."""

    x = _as_1d(x)
    if x.size == 1 or float(np.var(x)) == 0.0:
        return 1.0
    if max_lag is None:
        max_lag = min(1000, x.size)

    autocorr = autocorr_fft(x, max_lag=max_lag)
    if autocorr.size < 3:
        return 1.0

    tau = 1.0
    for lag in range(1, autocorr.size - 1, 2):
        pair_sum = autocorr[lag] + autocorr[lag + 1]
        if pair_sum <= 0.0:
            break
        tau += 2.0 * pair_sum
    return max(float(tau), 1.0)


def effective_sample_size(x: ArrayLike, max_lag: int | None = None) -> float:
    """Estimate the effective sample size of a one-dimensional chain."""

    x = _as_1d(x)
    if x.size == 1 or float(np.var(x)) == 0.0:
        return 1.0

    tau = integrated_autocorr_time(x, max_lag=max_lag)
    return min(float(x.size), max(1.0, float(x.size / tau)))


def min_coordinate_ess(samples: ArrayLike, max_lag: int | None = None) -> float:
    """Return the worst coordinate-wise ESS for a multidimensional chain."""

    samples = _as_samples(samples)
    return min(
        effective_sample_size(samples[:, coordinate], max_lag=max_lag)
        for coordinate in range(samples.shape[1])
    )


def gaussian_errors(
    samples: ArrayLike,
    true_mean: ArrayLike,
    true_cov: ArrayLike,
) -> dict[str, float]:
    """Compute mean and covariance errors against a Gaussian reference."""

    samples = _as_samples(samples)
    d = samples.shape[1]
    true_mean = np.asarray(true_mean, dtype=float)
    if true_mean.shape != (d,):
        raise ValueError(f"true_mean must have shape ({d},)")
    true_cov = _as_covariance(true_cov, d)

    sample_mean = samples.mean(axis=0)
    sample_cov = _sample_covariance(samples)
    return {
        "mean_error": float(np.linalg.norm(sample_mean - true_mean)),
        "cov_error": float(np.linalg.norm(sample_cov - true_cov, ord="fro")),
    }


def summarise_chain(
    samples: ArrayLike,
    reference_mean: ArrayLike | None = None,
    reference_cov: ArrayLike | None = None,
    max_lag: int | None = 1000,
) -> dict[str, float | int]:
    """Summarise one chain in a flat dictionary suitable for a CSV row."""

    samples = _as_samples(samples)
    summary: dict[str, float | int] = {
        "n_samples": int(samples.shape[0]),
        "dimension": int(samples.shape[1]),
        "min_ess": float(min_coordinate_ess(samples, max_lag=max_lag)),
    }

    if reference_mean is not None and reference_cov is not None:
        summary.update(gaussian_errors(samples, reference_mean, reference_cov))
    elif reference_mean is not None or reference_cov is not None:
        raise ValueError("reference_mean and reference_cov must be provided together")

    return summary


__all__ = [
    "autocorr_fft",
    "effective_sample_size",
    "gaussian_errors",
    "integrated_autocorr_time",
    "min_coordinate_ess",
    "summarise_chain",
]
