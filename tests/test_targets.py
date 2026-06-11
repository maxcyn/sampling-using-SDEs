import numpy as np
import pytest

from m2r_langevin.targets import (
    logistic_laplace_approximation,
    make_correlated_gaussian,
    make_linear_regression,
    make_logistic_regression,
)


def finite_difference_gradient(log_prob, x, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    grad = np.empty_like(x)
    for j in range(x.size):
        step = np.zeros_like(x)
        step[j] = eps
        grad[j] = (log_prob(x + step) - log_prob(x - step)) / (2.0 * eps)
    return grad


def assert_symmetric_positive_definite(matrix: np.ndarray) -> None:
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(matrix) > 0.0)


def test_correlated_gaussian_matches_expected_2d_covariance() -> None:
    target = make_correlated_gaussian(d=2, rho=0.9)

    np.testing.assert_allclose(target.true_mean, np.zeros(2))
    np.testing.assert_allclose(target.true_cov, np.array([[1.0, 0.9], [0.9, 1.0]]))
    np.testing.assert_allclose(
        target.precision @ target.true_cov, np.eye(2), atol=1e-12
    )


def test_correlated_gaussian_gradient_matches_finite_difference() -> None:
    target = make_correlated_gaussian(d=4, rho=0.8)
    x = np.array([0.2, -0.4, 0.1, 0.3])

    expected = finite_difference_gradient(target.log_prob, x)

    np.testing.assert_allclose(target.grad_log_prob(x), expected, rtol=1e-5, atol=1e-6)


def test_linear_regression_posterior_matches_closed_form_formula() -> None:
    target = make_linear_regression(n=40, d=3, sigma=1.2, tau=4.0, seed=11)

    X = target.X
    y = target.y
    sigma = 1.2
    tau = 4.0
    precision = X.T @ X / sigma**2 + np.eye(3) / tau**2
    expected_cov = np.linalg.inv(precision)
    expected_mean = expected_cov @ (X.T @ y) / sigma**2

    np.testing.assert_allclose(
        target.posterior_cov, expected_cov, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        target.posterior_mean,
        expected_mean,
        rtol=1e-10,
        atol=1e-10,
    )
    assert_symmetric_positive_definite(target.posterior_cov)


def test_linear_regression_gradient_matches_finite_difference() -> None:
    target = make_linear_regression(n=50, d=4, sigma=0.8, tau=3.0, seed=3)
    beta = np.array([0.1, -0.2, 0.3, -0.4])

    expected = finite_difference_gradient(target.log_prob, beta)

    np.testing.assert_allclose(
        target.grad_log_prob(beta), expected, rtol=1e-5, atol=1e-6
    )


def test_logistic_regression_gradient_matches_finite_difference() -> None:
    target = make_logistic_regression(n=60, d=4, tau=5.0, seed=7)
    beta = np.array([0.2, -0.1, 0.4, -0.3])

    expected = finite_difference_gradient(target.log_prob, beta)

    np.testing.assert_allclose(
        target.grad_log_prob(beta), expected, rtol=1e-5, atol=1e-6
    )


def test_logistic_laplace_covariance_is_symmetric_positive_definite() -> None:
    target = make_logistic_regression(n=80, d=5, tau=4.0, seed=13)

    laplace = logistic_laplace_approximation(
        target.log_prob,
        target.grad_log_prob,
        target.X,
        target.y,
        target.tau,
    )

    assert laplace.optimizer_result.success
    assert laplace.beta_map.shape == (5,)
    assert laplace.laplace_covariance.shape == (5, 5)
    assert_symmetric_positive_definite(laplace.laplace_covariance)
    np.testing.assert_allclose(
        target.grad_log_prob(laplace.beta_map),
        np.zeros(5),
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "factory, kwargs, match",
    [
        (make_correlated_gaussian, {"d": 0}, "d"),
        (make_correlated_gaussian, {"rho": 1.0}, "rho"),
        (make_linear_regression, {"n": 0}, "n"),
        (make_linear_regression, {"sigma": 0.0}, "sigma"),
        (make_logistic_regression, {"tau": 0.0}, "tau"),
    ],
)
def test_target_factories_validate_inputs(factory, kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory(**kwargs)
