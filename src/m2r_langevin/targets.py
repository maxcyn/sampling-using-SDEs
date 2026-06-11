"""Target distribution builders used by the Langevin experiments."""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit

Array = NDArray[np.float64]
LogProb = Callable[[Array], float]
GradLogProb = Callable[[Array], Array]


class GaussianTarget(NamedTuple):
    log_prob: LogProb
    grad_log_prob: GradLogProb
    true_mean: Array
    true_cov: Array
    precision: Array


class LinearRegressionTarget(NamedTuple):
    log_prob: LogProb
    grad_log_prob: GradLogProb
    posterior_mean: Array
    posterior_cov: Array
    X: Array
    y: Array


class LogisticRegressionTarget(NamedTuple):
    log_prob: LogProb
    grad_log_prob: GradLogProb
    X: Array
    y: Array
    tau: float
    beta_true: Array


class LogisticLaplaceApproximation(NamedTuple):
    beta_map: Array
    laplace_covariance: Array
    optimizer_result: OptimizeResult


def _as_vector(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("parameter vector must be one-dimensional")
    return x


def make_correlated_gaussian(d: int = 2, rho: float = 0.9) -> GaussianTarget:
    """Create a zero-mean AR(1)-correlated Gaussian target.

    The covariance matrix has entries ``cov[i, j] = rho ** abs(i - j)``.
    For ``d=2`` and ``rho=0.9`` this matches the strongly correlated Gaussian
    target used in the exploratory notebook.
    """

    if d <= 0:
        raise ValueError("d must be positive")
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie strictly between -1 and 1")

    idx = np.arange(d)
    true_cov = rho ** np.abs(idx[:, None] - idx[None, :])
    precision = np.linalg.inv(true_cov)
    true_mean = np.zeros(d, dtype=float)

    def log_prob(x: Array) -> float:
        x = _as_vector(x)
        return -0.5 * float(x @ precision @ x)

    def grad_log_prob(x: Array) -> Array:
        x = _as_vector(x)
        return -precision @ x

    return GaussianTarget(log_prob, grad_log_prob, true_mean, true_cov, precision)


def make_linear_regression(
    n: int = 200,
    d: int = 5,
    sigma: float = 1.0,
    tau: float = 10.0,
    seed: int = 0,
) -> LinearRegressionTarget:
    """Create a conjugate Bayesian linear-regression target.

    The synthetic model is ``y = X beta + N(0, sigma^2 I)`` with prior
    ``beta ~ N(0, tau^2 I)``. The posterior is Gaussian and is returned exactly
    for later scoring.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if d <= 0:
        raise ValueError("d must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    beta_true = rng.standard_normal(d)
    y = X @ beta_true + sigma * rng.standard_normal(n)

    posterior_precision = X.T @ X / sigma**2 + np.eye(d) / tau**2
    c_factor = cho_factor(posterior_precision, lower=True, check_finite=False)
    posterior_cov = cho_solve(c_factor, np.eye(d), check_finite=False)
    posterior_mean = cho_solve(c_factor, X.T @ y / sigma**2, check_finite=False)

    def log_prob(beta: Array) -> float:
        beta = _as_vector(beta)
        residual = y - X @ beta
        likelihood = -0.5 * float(residual @ residual) / sigma**2
        prior = -0.5 * float(beta @ beta) / tau**2
        return likelihood + prior

    def grad_log_prob(beta: Array) -> Array:
        beta = _as_vector(beta)
        return X.T @ (y - X @ beta) / sigma**2 - beta / tau**2

    return LinearRegressionTarget(
        log_prob,
        grad_log_prob,
        posterior_mean,
        posterior_cov,
        X,
        y,
    )


def make_logistic_regression(
    n: int = 400,
    d: int = 5,
    tau: float = 10.0,
    seed: int = 0,
) -> LogisticRegressionTarget:
    """Create a Bayesian logistic-regression target with a Gaussian prior."""

    if n <= 0:
        raise ValueError("n must be positive")
    if d <= 0:
        raise ValueError("d must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    beta_true = rng.standard_normal(d)
    probabilities = expit(X @ beta_true)
    y = (rng.random(n) < probabilities).astype(float)

    def log_prob(beta: Array) -> float:
        beta = _as_vector(beta)
        z = X @ beta
        log_likelihood = float(np.sum(y * z - np.logaddexp(0.0, z)))
        prior = -0.5 * float(beta @ beta) / tau**2
        return log_likelihood + prior

    def grad_log_prob(beta: Array) -> Array:
        beta = _as_vector(beta)
        return X.T @ (y - expit(X @ beta)) - beta / tau**2

    return LogisticRegressionTarget(
        log_prob, grad_log_prob, X, y, float(tau), beta_true
    )


def logistic_laplace_approximation(
    log_prob: LogProb,
    grad_log_prob: GradLogProb,
    X: Array,
    y: Array,
    tau: float,
) -> LogisticLaplaceApproximation:
    """Approximate a logistic posterior by a Gaussian around its MAP.

    The returned covariance is the inverse Hessian of the negative log posterior
    at the maximum a posteriori estimate.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if y.shape != (X.shape[0],):
        raise ValueError("y must have shape (n,)")
    if tau <= 0:
        raise ValueError("tau must be positive")

    d = X.shape[1]
    result = minimize(
        lambda beta: -float(log_prob(beta)),
        np.zeros(d, dtype=float),
        jac=lambda beta: -np.asarray(grad_log_prob(beta), dtype=float),
        method="L-BFGS-B",
    )

    beta_map = np.asarray(result.x, dtype=float)
    probabilities = expit(X @ beta_map)
    weights = probabilities * (1.0 - probabilities)
    neg_log_posterior_hessian = X.T @ (weights[:, None] * X) + np.eye(d) / tau**2
    c_factor = cho_factor(neg_log_posterior_hessian, lower=True, check_finite=False)
    laplace_covariance = cho_solve(c_factor, np.eye(d), check_finite=False)

    return LogisticLaplaceApproximation(beta_map, laplace_covariance, result)


__all__ = [
    "GaussianTarget",
    "LinearRegressionTarget",
    "LogisticLaplaceApproximation",
    "LogisticRegressionTarget",
    "make_correlated_gaussian",
    "make_linear_regression",
    "make_logistic_regression",
    "logistic_laplace_approximation",
]
