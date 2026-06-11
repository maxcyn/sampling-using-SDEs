import numpy as np
import pytest

from m2r_langevin.samplers import SamplerResult, klmc, mala, ula


def standard_normal_log_prob(x: np.ndarray) -> float:
    return -0.5 * float(np.dot(x, x))


def standard_normal_grad_log_prob(x: np.ndarray) -> np.ndarray:
    return -x


def test_ula_returns_result_with_expected_shape_and_costs() -> None:
    result = ula(standard_normal_grad_log_prob, np.zeros(3), 0.1, 8, rng=0)

    assert isinstance(result, SamplerResult)
    assert result.samples.shape == (8, 3)
    assert result.sampler == "ULA"
    assert result.grad_evals == 8
    assert result.logprob_evals == 0
    assert result.acceptance_rate is None
    assert result.runtime_seconds is not None
    assert result.seed == 0


def test_mala_returns_result_with_expected_shape_acceptance_and_costs() -> None:
    result = mala(
        standard_normal_log_prob,
        standard_normal_grad_log_prob,
        np.zeros(2),
        0.2,
        10,
        rng=1,
    )

    assert result.samples.shape == (10, 2)
    assert result.sampler == "MALA"
    assert result.grad_evals == 11
    assert result.logprob_evals == 11
    assert result.acceptance_rate is not None
    assert 0.0 <= result.acceptance_rate <= 1.0
    assert result.runtime_seconds is not None
    assert result.seed == 1


def test_klmc_returns_result_with_expected_shape_and_one_gradient_per_step() -> None:
    result = klmc(standard_normal_grad_log_prob, np.zeros(4), 0.1, 6, gamma=2.5, rng=2)

    assert result.samples.shape == (6, 4)
    assert result.sampler == "KLMC"
    assert result.grad_evals == 6
    assert result.logprob_evals == 0
    assert result.acceptance_rate is None
    assert result.metadata["gamma"] == 2.5


@pytest.mark.parametrize("sampler_name", ["ula", "mala", "klmc"])
def test_samplers_are_reproducible_with_fixed_integer_seed(sampler_name: str) -> None:
    x0 = np.array([0.2, -0.1])

    if sampler_name == "ula":
        first = ula(standard_normal_grad_log_prob, x0, 0.05, 12, rng=123)
        second = ula(standard_normal_grad_log_prob, x0, 0.05, 12, rng=123)
    elif sampler_name == "mala":
        first = mala(
            standard_normal_log_prob,
            standard_normal_grad_log_prob,
            x0,
            0.05,
            12,
            rng=123,
        )
        second = mala(
            standard_normal_log_prob,
            standard_normal_grad_log_prob,
            x0,
            0.05,
            12,
            rng=123,
        )
    else:
        first = klmc(standard_normal_grad_log_prob, x0, 0.05, 12, rng=123)
        second = klmc(standard_normal_grad_log_prob, x0, 0.05, 12, rng=123)

    np.testing.assert_allclose(first.samples, second.samples)


def test_samplers_validate_positive_settings() -> None:
    with pytest.raises(ValueError, match="step_size"):
        ula(standard_normal_grad_log_prob, np.zeros(1), 0.0, 1)

    with pytest.raises(ValueError, match="n_samples"):
        mala(
            standard_normal_log_prob, standard_normal_grad_log_prob, np.zeros(1), 0.1, 0
        )

    with pytest.raises(ValueError, match="gamma"):
        klmc(standard_normal_grad_log_prob, np.zeros(1), 0.1, 1, gamma=0.0)
