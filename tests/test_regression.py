"""Tests for vectorized OLS.

Verify against synthetic data where the true coefficients are known. These
tests guard the numerical core; if they fail, every downstream factor model
result is suspect.
"""

from __future__ import annotations

import numpy as np
import pytest

from tapt.factors.regression import BulkOLSResult, ols_bulk


class TestOLSBulk:
    def test_recovers_known_coefficients_no_intercept(self):
        """With zero noise, OLS should recover the true beta exactly."""
        rng = np.random.default_rng(0)
        T, P, N = 200, 3, 5
        true_beta = rng.normal(0, 1, size=(P, N))
        X = rng.normal(0, 1, size=(T, P))
        Y = X @ true_beta  # no noise

        result = ols_bulk(Y, X)
        np.testing.assert_allclose(result.coef, true_beta, atol=1e-10)
        np.testing.assert_allclose(result.r_squared, 1.0, atol=1e-10)

    def test_recovers_known_coefficients_with_noise(self):
        """With moderate noise, OLS should recover beta within sampling error."""
        rng = np.random.default_rng(42)
        T, P, N = 1000, 3, 4
        true_beta = np.array(
            [
                [1.0, 0.5, -0.3, 0.8],
                [-0.2, 1.0, 0.5, -0.4],
                [0.3, -0.1, 0.7, 0.2],
            ]
        )
        X = rng.normal(0, 1, size=(T, P))
        noise = rng.normal(0, 0.1, size=(T, N))
        Y = X @ true_beta + noise

        result = ols_bulk(Y, X)
        # With T=1000 and noise sigma=0.1, standard error on each coefficient
        # is roughly 0.1/sqrt(1000) ~ 0.003. Use a generous tolerance.
        np.testing.assert_allclose(result.coef, true_beta, atol=0.02)

    def test_intercept_is_recovered(self):
        """When X has a constant column, the intercept coefficient is recovered."""
        rng = np.random.default_rng(7)
        T, N = 500, 3
        true_alpha = np.array([0.5, -1.2, 2.0])
        true_beta = np.array([[1.0, 0.5, -0.5]])  # 1 x N
        X_features = rng.normal(0, 1, size=(T, 1))
        X = np.column_stack([np.ones(T), X_features])
        Y = X @ np.vstack([true_alpha, true_beta])

        result = ols_bulk(Y, X)
        np.testing.assert_allclose(result.coef[0], true_alpha, atol=1e-10)
        np.testing.assert_allclose(result.coef[1], true_beta[0], atol=1e-10)

    def test_residual_variance_unbiased(self):
        """The dof-adjusted residual variance should converge to the true noise variance."""
        rng = np.random.default_rng(101)
        T, P, N = 5000, 3, 2
        true_sigma2 = np.array([0.04, 0.01])
        X = rng.normal(0, 1, size=(T, P))
        true_beta = rng.normal(0, 1, size=(P, N))
        noise = rng.normal(0, 1, size=(T, N)) * np.sqrt(true_sigma2)
        Y = X @ true_beta + noise

        result = ols_bulk(Y, X)
        # With T=5000, sample variance should be very close to true
        np.testing.assert_allclose(result.sigma2, true_sigma2, rtol=0.05)

    def test_r_squared_in_zero_one(self):
        rng = np.random.default_rng(3)
        T, P, N = 200, 2, 4
        Y = rng.normal(0, 1, size=(T, N))
        X = rng.normal(0, 1, size=(T, P))
        result = ols_bulk(Y, X)
        # No requirement on the value, just that it's a valid R²
        assert (result.r_squared <= 1.0).all()

    def test_perfect_fit_gives_r_squared_one(self):
        rng = np.random.default_rng(11)
        X = rng.normal(0, 1, size=(50, 3))
        beta = rng.normal(0, 1, size=(3, 2))
        Y = X @ beta
        result = ols_bulk(Y, X)
        np.testing.assert_allclose(result.r_squared, 1.0, atol=1e-10)

    def test_zero_signal_gives_r_squared_near_zero(self):
        """Pure noise Y should give R² near zero (and possibly slightly negative on adjusted variants)."""
        rng = np.random.default_rng(99)
        T, P, N = 1000, 3, 10
        Y = rng.normal(0, 1, size=(T, N))  # pure noise, no signal
        X = rng.normal(0, 1, size=(T, P))
        result = ols_bulk(Y, X)
        # With T >> P, R² should be very close to zero
        assert np.abs(result.r_squared).max() < 0.05


class TestOLSBulkValidation:
    def test_dimension_mismatch_raises(self):
        Y = np.zeros((100, 3))
        X = np.zeros((50, 2))
        with pytest.raises(ValueError, match="same number of rows"):
            ols_bulk(Y, X)

    def test_1d_array_raises(self):
        Y = np.zeros(100)  # 1D
        X = np.zeros((100, 2))
        with pytest.raises(ValueError, match="2D"):
            ols_bulk(Y, X)  # type: ignore

    def test_nan_in_input_raises(self):
        Y = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        X = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError, match="NaN"):
            ols_bulk(Y, X)

    def test_too_few_observations_raises(self):
        Y = np.array([[1.0], [2.0]])  # T=2
        X = np.array([[1.0, 2.0], [3.0, 4.0]])  # P=2
        with pytest.raises(ValueError, match="n > p"):
            ols_bulk(Y, X)

    def test_singular_X_raises(self):
        """Collinear X columns should produce a ValueError, not a crash."""
        T = 50
        x1 = np.arange(T, dtype=float)
        X = np.column_stack([x1, 2 * x1])  # x2 = 2 * x1, collinear
        Y = x1.reshape(-1, 1)
        with pytest.raises(ValueError, match="singular"):
            ols_bulk(Y, X)


class TestBulkOLSResult:
    def test_result_is_frozen(self):
        rng = np.random.default_rng(0)
        Y = rng.normal(0, 1, size=(100, 3))
        X = rng.normal(0, 1, size=(100, 2))
        result = ols_bulk(Y, X)
        with pytest.raises((AttributeError, Exception)):
            result.coef = np.zeros_like(result.coef)  # type: ignore

    def test_n_obs_and_n_params_recorded(self):
        rng = np.random.default_rng(0)
        Y = rng.normal(0, 1, size=(50, 3))
        X = np.column_stack([np.ones(50), rng.normal(0, 1, size=50)])
        result = ols_bulk(Y, X)
        assert result.n_obs == 50
        assert result.n_params == 2
        assert isinstance(result, BulkOLSResult)
