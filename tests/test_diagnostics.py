import numpy as np
import pytest

from m2r_langevin.diagnostics import (
    autocorr_fft,
    effective_sample_size,
    gaussian_errors,
    integrated_autocorr_time,
    min_coordinate_ess,
    summarise_chain,
)


def make_ar1_chain(phi: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    chain = np.empty(n, dtype=float)
    chain[0] = noise[0]
    for i in range(1, n):
        chain[i] = phi * chain[i - 1] + noise[i]
    return chain


def test_autocorr_fft_returns_normalised_lags() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])

    autocorr = autocorr_fft(x, max_lag=3)

    assert autocorr.shape == (3,)
    assert autocorr[0] == pytest.approx(1.0)
    assert np.all(np.isfinite(autocorr))


def test_iid_gaussian_ess_is_close_to_sample_size() -> None:
    rng = np.random.default_rng(123)
    samples = rng.standard_normal(5000)

    ess = effective_sample_size(samples, max_lag=500)

    assert ess > 4000


def test_ar1_chain_has_much_smaller_ess_than_iid_chain() -> None:
    iid = np.random.default_rng(123).standard_normal(5000)
    ar1 = make_ar1_chain(phi=0.95, n=5000, seed=123)

    iid_ess = effective_sample_size(iid, max_lag=500)
    ar1_ess = effective_sample_size(ar1, max_lag=500)

    assert ar1_ess < iid_ess / 5.0
    assert integrated_autocorr_time(ar1, max_lag=500) > 5.0


def test_min_coordinate_ess_returns_worst_coordinate() -> None:
    rng = np.random.default_rng(42)
    iid = rng.standard_normal(3000)
    ar1 = make_ar1_chain(phi=0.9, n=3000, seed=42)
    samples = np.column_stack([iid, ar1])

    result = min_coordinate_ess(samples, max_lag=400)

    assert result == pytest.approx(effective_sample_size(ar1, max_lag=400))


def test_gaussian_errors_are_small_for_large_iid_reference_sample() -> None:
    rng = np.random.default_rng(1)
    mean = np.array([0.5, -0.25])
    cov = np.array([[1.0, 0.3], [0.3, 2.0]])
    samples = rng.multivariate_normal(mean, cov, size=30_000)

    errors = gaussian_errors(samples, mean, cov)

    assert errors["mean_error"] < 0.03
    assert errors["cov_error"] < 0.06


def test_summarise_chain_returns_flat_csv_ready_dictionary() -> None:
    rng = np.random.default_rng(2)
    mean = np.zeros(2)
    cov = np.eye(2)
    samples = rng.multivariate_normal(mean, cov, size=1000)

    summary = summarise_chain(samples, mean, cov, max_lag=200)

    assert set(summary) == {
        "n_samples",
        "dimension",
        "min_ess",
        "mean_error",
        "cov_error",
    }
    assert summary["n_samples"] == 1000
    assert summary["dimension"] == 2
    assert summary["min_ess"] > 0.0
