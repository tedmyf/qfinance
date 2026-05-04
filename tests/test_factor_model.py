"""Tests for the factor model.

The most important test in this module is the synthetic-data recovery test:
we generate asset returns from a known factor model with known loadings, fit
the model, and verify that the estimated loadings recover the true loadings
within sampling error. If this test passes, the entire downstream pipeline
(covariance, expected returns, optimization) is built on a sound foundation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tapt.factors import estimate_factor_model, to_excess_returns

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def make_synthetic_factor_data(
    *,
    n_periods: int = 200,
    n_assets: int = 10,
    factor_names: tuple[str, ...] = ("Mkt-RF", "SMB", "HML"),
    factor_means: tuple[float, ...] = (0.005, 0.001, 0.002),
    factor_vols: tuple[float, ...] = (0.04, 0.025, 0.03),
    idio_vol: float = 0.02,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic factor return panel with known loadings.

    Returns
    -------
    asset_returns : pd.DataFrame
        Shape (T, N). Excess asset returns.
    factor_returns : pd.DataFrame
        Shape (T, K). Factor returns.
    true_loadings : pd.DataFrame
        Shape (N, K). The B matrix that generated the data.
    """
    rng = np.random.default_rng(seed)
    K = len(factor_names)

    # Generate factor returns
    F = rng.normal(loc=factor_means, scale=factor_vols, size=(n_periods, K))
    factor_returns = pd.DataFrame(
        F,
        columns=list(factor_names),
        index=pd.date_range("2010-01-31", periods=n_periods, freq="ME"),
    )
    factor_returns.index.name = "date"

    # Generate true loadings: market betas centered at 1.0, others around 0
    true_B = rng.normal(0, 0.4, size=(n_assets, K))
    if "Mkt-RF" in factor_names:
        mkt_idx = factor_names.index("Mkt-RF")
        true_B[:, mkt_idx] = rng.normal(1.0, 0.3, size=n_assets)

    asset_names = [f"A{i:02d}" for i in range(n_assets)]
    true_loadings = pd.DataFrame(true_B, index=asset_names, columns=list(factor_names))

    # Generate asset returns: R = B @ F + epsilon (no alpha)
    epsilon = rng.normal(0, idio_vol, size=(n_periods, n_assets))
    R = F @ true_B.T + epsilon

    asset_returns = pd.DataFrame(R, columns=asset_names, index=factor_returns.index)
    asset_returns.index.name = "date"

    return asset_returns, factor_returns, true_loadings


# ---------------------------------------------------------------------------
# Recovery tests
# ---------------------------------------------------------------------------


class TestEstimateFactorModelRecovery:
    def test_recovers_known_loadings_within_sampling_error(self):
        asset_returns, factor_returns, true_B = make_synthetic_factor_data(
            n_periods=1200, n_assets=10, idio_vol=0.02, seed=42
        )
        fit = estimate_factor_model(
            asset_returns, factor_returns, window=1200, min_obs=120
        )
        # With T=1200 and idio_vol=0.02, std error per beta is on the order of
        # 0.014-0.023. A 0.07 tolerance is roughly 3 sigma, which gives a
        # very high pass rate while still catching real bugs.
        np.testing.assert_allclose(
            fit.loadings.values, true_B.values, atol=0.07
        )

    def test_recovered_alpha_is_near_zero_for_zero_alpha_dgp(self):
        """When the data-generating process has alpha=0, estimated alpha should be near zero."""
        asset_returns, factor_returns, _ = make_synthetic_factor_data(
            n_periods=240, n_assets=10, idio_vol=0.02, seed=99
        )
        fit = estimate_factor_model(
            asset_returns, factor_returns, window=240, min_obs=120
        )
        # Alpha standard error roughly idio_vol / sqrt(T) = 0.02 / sqrt(240) ~ 0.0013
        assert fit.alpha.abs().max() < 0.005

    def test_idiosyncratic_variance_recovers_true_value(self):
        idio_vol = 0.025
        asset_returns, factor_returns, _ = make_synthetic_factor_data(
            n_periods=500, n_assets=8, idio_vol=idio_vol, seed=7
        )
        fit = estimate_factor_model(
            asset_returns, factor_returns, window=500, min_obs=120
        )
        true_var = idio_vol**2
        np.testing.assert_allclose(fit.idio_variance.values, true_var, rtol=0.15)

    def test_factor_covariance_recovers_true_diagonal(self):
        """For independent factors, the estimated factor covariance should be approximately diagonal
        with diagonal entries close to factor_vols^2."""
        factor_vols = (0.04, 0.025, 0.03)
        asset_returns, factor_returns, _ = make_synthetic_factor_data(
            n_periods=500, factor_vols=factor_vols, seed=11
        )
        fit = estimate_factor_model(
            asset_returns, factor_returns, window=500, min_obs=120
        )
        true_var = np.array(factor_vols) ** 2
        np.testing.assert_allclose(
            np.diag(fit.factor_covariance.values), true_var, rtol=0.15
        )


# ---------------------------------------------------------------------------
# Asset covariance properties
# ---------------------------------------------------------------------------


class TestAssetCovariance:
    def test_covariance_is_symmetric(self):
        asset_returns, factor_returns, _ = make_synthetic_factor_data(seed=1)
        fit = estimate_factor_model(asset_returns, factor_returns, window=200)
        Sigma = fit.asset_covariance()
        np.testing.assert_allclose(Sigma.values, Sigma.values.T, atol=1e-12)

    def test_covariance_is_positive_semidefinite(self):
        asset_returns, factor_returns, _ = make_synthetic_factor_data(seed=2)
        fit = estimate_factor_model(asset_returns, factor_returns, window=200)
        Sigma = fit.asset_covariance()
        eigenvalues = np.linalg.eigvalsh(Sigma.values)
        # All eigenvalues should be non-negative (allow tiny numerical noise)
        assert eigenvalues.min() > -1e-10

    def test_covariance_has_correct_shape_and_index(self):
        asset_returns, factor_returns, _ = make_synthetic_factor_data(
            n_assets=15, seed=3
        )
        fit = estimate_factor_model(asset_returns, factor_returns, window=200)
        Sigma = fit.asset_covariance()
        assert Sigma.shape == (15, 15)
        assert list(Sigma.index) == fit.assets
        assert list(Sigma.columns) == fit.assets

    def test_covariance_diagonal_dominates_factor_term(self):
        """Total variance should equal systematic + idiosyncratic for each asset."""
        asset_returns, factor_returns, _ = make_synthetic_factor_data(
            n_periods=500, idio_vol=0.02, seed=4
        )
        fit = estimate_factor_model(asset_returns, factor_returns, window=500)
        Sigma = fit.asset_covariance()

        # Diagonal of Sigma = systematic variance per asset + idio variance
        systematic = np.diag(fit.loadings @ fit.factor_covariance @ fit.loadings.T)
        total = np.diag(Sigma.values)
        np.testing.assert_allclose(total, systematic + fit.idio_variance.values, atol=1e-12)


# ---------------------------------------------------------------------------
# Expected returns
# ---------------------------------------------------------------------------


class TestExpectedReturns:
    def test_default_uses_window_mean(self):
        asset_returns, factor_returns, _ = make_synthetic_factor_data(seed=5)
        fit = estimate_factor_model(asset_returns, factor_returns, window=200)
        mu = fit.expected_returns()
        # Manual calculation: B @ mean(F)
        expected = fit.loadings @ fit.factor_returns_window.mean()
        pd.testing.assert_series_equal(mu, expected.rename("expected_return"))

    def test_custom_factor_premia_works(self):
        asset_returns, factor_returns, _ = make_synthetic_factor_data(seed=6)
        fit = estimate_factor_model(asset_returns, factor_returns, window=200)
        custom_premia = pd.Series(
            {"Mkt-RF": 0.006, "SMB": 0.002, "HML": 0.001}
        )
        mu = fit.expected_returns(factor_premia=custom_premia)
        expected = fit.loadings @ custom_premia
        pd.testing.assert_series_equal(mu, expected.rename("expected_return"))

    def test_include_alpha_adds_intercept(self):
        asset_returns, factor_returns, _ = make_synthetic_factor_data(seed=7)
        fit = estimate_factor_model(asset_returns, factor_returns, window=200)
        mu_no_alpha = fit.expected_returns(include_alpha=False)
        mu_with_alpha = fit.expected_returns(include_alpha=True)
        np.testing.assert_allclose(
            (mu_with_alpha - mu_no_alpha).values, fit.alpha.values
        )

    def test_missing_factor_in_premia_raises(self):
        asset_returns, factor_returns, _ = make_synthetic_factor_data(seed=8)
        fit = estimate_factor_model(asset_returns, factor_returns, window=200)
        bad_premia = pd.Series({"Mkt-RF": 0.005, "SMB": 0.001})  # missing HML
        with pytest.raises(ValueError, match="missing factors"):
            fit.expected_returns(factor_premia=bad_premia)


# ---------------------------------------------------------------------------
# Window and as_of discipline
# ---------------------------------------------------------------------------


class TestWindowAndAsOf:
    def test_as_of_truncates_fit_window(self):
        asset_returns, factor_returns, _ = make_synthetic_factor_data(
            n_periods=200, seed=10
        )
        # Fit with as_of in the middle of the data
        as_of = asset_returns.index[100]
        fit = estimate_factor_model(
            asset_returns, factor_returns, window=60, as_of=as_of
        )
        assert fit.fit_end <= as_of
        # Window of 60 should give us 60 observations
        assert fit.n_obs == 60

    def test_window_size_respected(self):
        asset_returns, factor_returns, _ = make_synthetic_factor_data(
            n_periods=200, seed=11
        )
        fit = estimate_factor_model(asset_returns, factor_returns, window=36)
        assert fit.n_obs == 36

    def test_insufficient_observations_raises(self):
        asset_returns, factor_returns, _ = make_synthetic_factor_data(
            n_periods=20, seed=12
        )
        with pytest.raises(ValueError, match="Insufficient observations"):
            estimate_factor_model(
                asset_returns, factor_returns, window=20, min_obs=24
            )

    def test_no_overlapping_dates_raises(self):
        asset_returns = pd.DataFrame(
            {"A": [0.01, 0.02]},
            index=pd.date_range("2024-01-31", periods=2, freq="ME"),
        )
        factor_returns = pd.DataFrame(
            {"Mkt-RF": [0.005, 0.003]},
            index=pd.date_range("2010-01-31", periods=2, freq="ME"),
        )
        with pytest.raises(ValueError, match="No overlapping"):
            estimate_factor_model(asset_returns, factor_returns, window=2, min_obs=2)

    def test_as_of_before_data_raises(self):
        asset_returns, factor_returns, _ = make_synthetic_factor_data(seed=13)
        with pytest.raises(ValueError, match="No data on or before"):
            estimate_factor_model(
                asset_returns, factor_returns, window=24, as_of="1990-01-01"
            )

    def test_rf_column_is_dropped_if_present(self):
        """If the user passes the full FF dataset including RF, RF should not be a factor."""
        asset_returns, factor_returns, _ = make_synthetic_factor_data(seed=14)
        # Add an RF column
        factor_returns_with_rf = factor_returns.copy()
        factor_returns_with_rf["RF"] = 0.0001
        fit = estimate_factor_model(
            asset_returns, factor_returns_with_rf, window=100, min_obs=24
        )
        assert "RF" not in fit.factors


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


class TestFactorModelFitStructure:
    def test_fit_is_frozen(self):
        asset_returns, factor_returns, _ = make_synthetic_factor_data(seed=15)
        fit = estimate_factor_model(asset_returns, factor_returns, window=200)
        with pytest.raises((AttributeError, Exception)):
            fit.loadings = pd.DataFrame()  # type: ignore

    def test_assets_and_factors_properties_match_dataframes(self):
        asset_returns, factor_returns, _ = make_synthetic_factor_data(
            n_assets=7, seed=16
        )
        fit = estimate_factor_model(asset_returns, factor_returns, window=100)
        assert fit.n_assets == 7
        assert fit.n_factors == 3
        assert fit.assets == list(fit.loadings.index)
        assert fit.factors == list(fit.loadings.columns)

    def test_repr_contains_useful_info(self):
        asset_returns, factor_returns, _ = make_synthetic_factor_data(seed=17)
        fit = estimate_factor_model(asset_returns, factor_returns, window=100)
        r = repr(fit)
        assert "n_assets" in r
        assert "n_factors" in r


# ---------------------------------------------------------------------------
# Excess returns helper
# ---------------------------------------------------------------------------


class TestToExcessReturns:
    def test_subtracts_per_period_rate(self):
        # Annualized 5.04% over 252 days = 0.02% per day
        rate = pd.Series(
            [0.0504] * 5,
            index=pd.date_range("2024-01-01", periods=5),
            name="DGS3MO",
        )
        returns = pd.DataFrame(
            {"A": [0.01, 0.01, 0.01, 0.01, 0.01]},
            index=rate.index,
        )
        excess = to_excess_returns(returns, rate, rate_periods_per_year=252)
        expected_period_rate = 0.0504 / 252
        expected = 0.01 - expected_period_rate
        np.testing.assert_allclose(excess["A"].values, expected, atol=1e-10)

    def test_forward_fills_rate_on_missing_dates(self):
        rate = pd.Series(
            [0.0504, 0.0504],
            index=pd.to_datetime(["2024-01-01", "2024-01-04"]),
        )
        returns = pd.DataFrame(
            {"A": [0.01] * 5},
            index=pd.date_range("2024-01-01", periods=5),
        )
        excess = to_excess_returns(returns, rate, rate_periods_per_year=252)
        assert not excess.isna().any().any()

    def test_monthly_conversion(self):
        rate = pd.Series(
            [0.06], index=pd.to_datetime(["2024-01-31"]), name="DGS3MO"
        )
        returns = pd.DataFrame(
            {"A": [0.05]}, index=pd.to_datetime(["2024-01-31"])
        )
        excess = to_excess_returns(returns, rate, rate_periods_per_year=12)
        # 6% annualized / 12 = 0.5% monthly
        np.testing.assert_allclose(excess["A"].iloc[0], 0.05 - 0.005)
