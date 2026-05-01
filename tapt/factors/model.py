"""Factor model estimation and asset covariance construction.

A factor model decomposes asset returns into systematic exposures to a set of
common factors plus idiosyncratic noise:

    R_i,t = alpha_i + sum_k B_i,k * F_k,t + eps_i,t

The implied asset covariance matrix is:

    Sigma = B * Cov(F) * B' + D

where B is the N-by-K matrix of factor loadings, Cov(F) is the K-by-K factor
covariance, and D is a diagonal matrix of idiosyncratic variances.

This decomposition is the standard approach in institutional risk modeling.
It produces a well-conditioned covariance matrix even when N exceeds T, and
it cleanly separates systematic risk (low-rank, factor-driven) from
idiosyncratic risk.

Workflow
--------
>>> from tapt.factors import estimate_factor_model
>>> fit = estimate_factor_model(
...     excess_asset_returns=asset_excess,
...     factor_returns=ff5_factors,
...     window=60,
...     as_of="2024-12-31",
... )
>>> Sigma = fit.asset_covariance()
>>> mu = fit.expected_returns()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tapt.factors.regression import ols_bulk


@dataclass(frozen=True)
class FactorModelFit:
    """Frozen result of fitting a factor model on a single window.

    All attributes are pandas objects with informative indices and column
    names. The fit is immutable: use the methods below to derive downstream
    quantities (covariance, expected returns) without mutating state.

    Attributes
    ----------
    loadings : pd.DataFrame
        Shape (N, K). Factor loadings B. Index = asset names, columns = factors.
    alpha : pd.Series
        Shape (N,). Estimated intercept per asset. Zero if the regression
        was fit without an intercept.
    idio_variance : pd.Series
        Shape (N,). Residual variance per asset, with degrees-of-freedom
        adjustment.
    r_squared : pd.Series
        Shape (N,). Per-asset coefficient of determination.
    factor_returns_window : pd.DataFrame
        Shape (T, K). The factor return panel used for the fit. Retained so
        downstream methods (expected_returns) can use historical means as a
        default factor premia estimate.
    factor_covariance : pd.DataFrame
        Shape (K, K). Sample covariance of factor returns over the fit window.
    fit_start, fit_end : pd.Timestamp
        Inclusive boundaries of the window used.
    n_obs : int
        Number of observations in the fit window after dropping NaN rows.
    """

    loadings: pd.DataFrame
    alpha: pd.Series
    idio_variance: pd.Series
    r_squared: pd.Series
    factor_returns_window: pd.DataFrame
    factor_covariance: pd.DataFrame
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    n_obs: int

    @property
    def assets(self) -> list[str]:
        """Names of fitted assets, in loading-matrix row order."""
        return list(self.loadings.index)

    @property
    def factors(self) -> list[str]:
        """Names of factors, in loading-matrix column order."""
        return list(self.loadings.columns)

    @property
    def n_assets(self) -> int:
        return len(self.loadings)

    @property
    def n_factors(self) -> int:
        return self.loadings.shape[1]

    def asset_covariance(self) -> pd.DataFrame:
        """Construct the asset covariance matrix Sigma = B * Cov(F) * B' + D.

        Returns
        -------
        pd.DataFrame
            Shape (N, N). Symmetric and positive semi-definite by construction.
            Indexed by asset names on both axes.
        """
        B = self.loadings.values
        F = self.factor_covariance.values
        D = self.idio_variance.values

        Sigma = B @ F @ B.T + np.diag(D)

        # Symmetrize to clean up tiny floating-point asymmetries
        Sigma = 0.5 * (Sigma + Sigma.T)

        return pd.DataFrame(
            Sigma,
            index=self.loadings.index,
            columns=self.loadings.index,
        )

    def expected_returns(
        self,
        factor_premia: pd.Series | None = None,
        include_alpha: bool = False,
    ) -> pd.Series:
        """Compute factor-model implied expected returns.

        Parameters
        ----------
        factor_premia : pd.Series, optional
            Per-factor premium (same units as factor returns). If None,
            defaults to the historical mean of factor returns over the fit
            window. The Series index must match the model's factor names.
        include_alpha : bool
            If True, add the per-asset alpha to the factor-implied return.
            Default False because alpha estimates are noisy and using them
            in mean-variance optimization tends to overfit historical data.

        Returns
        -------
        pd.Series
            Shape (N,). Expected excess returns per asset (or total returns
            if the asset returns passed to the fit were not excess of RF).

        Raises
        ------
        ValueError
            If factor_premia is provided but its index does not align with
            the model's factors.
        """
        if factor_premia is None:
            premia = self.factor_returns_window.mean()
        else:
            premia = factor_premia.reindex(self.factors)
            if premia.isna().any():
                missing = premia.index[premia.isna()].tolist()
                raise ValueError(f"factor_premia missing factors: {missing}")

        mu = self.loadings @ premia
        if include_alpha:
            mu = mu + self.alpha
        mu.name = "expected_return"
        return mu

    def __repr__(self) -> str:
        return (
            f"FactorModelFit(n_assets={self.n_assets}, n_factors={self.n_factors}, "
            f"n_obs={self.n_obs}, window=[{self.fit_start.date()}, {self.fit_end.date()}])"
        )


def estimate_factor_model(
    excess_asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    *,
    window: int = 60,
    min_obs: int = 24,
    as_of: str | pd.Timestamp | None = None,
    include_intercept: bool = True,
) -> FactorModelFit:
    """Fit a Fama-French style factor model on the most recent window.

    The function selects the most recent ``window`` periods of data ending at
    or before ``as_of`` (or at the end of the data if ``as_of`` is None),
    aligns asset and factor returns by date, drops rows with any NaN, and
    runs a single vectorized OLS regression to estimate factor loadings.

    Parameters
    ----------
    excess_asset_returns : pd.DataFrame
        Shape (T, N). Asset returns in excess of the risk-free rate. Caller
        is responsible for the conversion. We do not subtract RF here because
        the conversion depends on annualization conventions that the caller
        knows better than we do.
    factor_returns : pd.DataFrame
        Shape (T, K). Factor returns (e.g., Mkt-RF, SMB, HML, RMW, CMA).
        Mkt-RF is already excess by definition. Other factors are
        long-short returns and don't need RF subtraction.
        The "RF" column in Kenneth French's data should NOT be passed as a
        factor; it's used only to construct excess returns upstream.
    window : int
        Number of periods to use for the fit. Typically 36 to 60 months for
        monthly data, 252 to 504 days for daily data.
    min_obs : int
        Minimum required observations after NaN dropping. If the available
        clean data is less than this, raises an error rather than producing
        an unreliable estimate.
    as_of : str or pd.Timestamp, optional
        Point-in-time cutoff. The fit window ends at the latest observation
        on or before this date. None means use the latest available data.
    include_intercept : bool
        If True, include an alpha intercept in the regression. The standard
        Fama-French regression includes alpha (it is interpreted as the
        risk-adjusted excess return). For risk-only modeling, alpha doesn't
        enter the covariance matrix, so excluding it has no effect on Sigma
        but does affect the residuals (and thus idiosyncratic variance).

    Returns
    -------
    FactorModelFit

    Raises
    ------
    ValueError
        If insufficient clean observations after alignment, or if the asset
        and factor frames have no overlapping dates.
    """
    # Drop the "RF" column if the user passed the full FF dataset by mistake
    factor_cols = [c for c in factor_returns.columns if c.upper() != "RF"]
    factors = factor_returns[factor_cols].copy()

    # Align by intersection of dates
    aligned = excess_asset_returns.join(factors, how="inner", lsuffix="_asset")
    if aligned.empty:
        raise ValueError(
            "No overlapping dates between excess_asset_returns and factor_returns"
        )

    # Apply as_of cutoff
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        aligned = aligned.loc[aligned.index <= cutoff]
        if aligned.empty:
            raise ValueError(f"No data on or before as_of={as_of}")

    # Take the trailing window
    aligned = aligned.tail(window)

    # Drop rows with any NaN
    aligned = aligned.dropna(how="any")
    if len(aligned) < min_obs:
        raise ValueError(
            f"Insufficient observations after cleaning: {len(aligned)} < min_obs={min_obs}"
        )

    asset_names = [
        c[:-6] if c.endswith("_asset") else c
        for c in aligned.columns
        if c not in factors.columns
    ]
    # Reconstruct asset and factor matrices from the aligned frame
    asset_panel = aligned.iloc[:, : len(asset_names)]
    asset_panel.columns = asset_names
    factor_panel = aligned[factor_cols]

    Y = asset_panel.to_numpy(dtype=float)
    F = factor_panel.to_numpy(dtype=float)

    if include_intercept:
        X = np.column_stack([np.ones(len(F)), F])
        result = ols_bulk(Y, X)
        alpha_arr = result.coef[0, :]
        loadings_arr = result.coef[1:, :].T  # (N, K)
    else:
        result = ols_bulk(Y, F)
        alpha_arr = np.zeros(Y.shape[1])
        loadings_arr = result.coef.T  # (N, K)

    loadings = pd.DataFrame(
        loadings_arr,
        index=asset_names,
        columns=factor_cols,
    )
    alpha = pd.Series(alpha_arr, index=asset_names, name="alpha")
    idio_variance = pd.Series(result.sigma2, index=asset_names, name="idio_variance")
    r_squared = pd.Series(result.r_squared, index=asset_names, name="r_squared")

    factor_cov = factor_panel.cov()

    return FactorModelFit(
        loadings=loadings,
        alpha=alpha,
        idio_variance=idio_variance,
        r_squared=r_squared,
        factor_returns_window=factor_panel,
        factor_covariance=factor_cov,
        fit_start=aligned.index[0],
        fit_end=aligned.index[-1],
        n_obs=len(aligned),
    )


def to_excess_returns(
    asset_returns: pd.DataFrame,
    risk_free_rate: pd.Series,
    rate_periods_per_year: int = 252,
    return_periods_per_year: int = 252,
) -> pd.DataFrame:
    """Convert total returns to excess returns by subtracting the risk-free rate.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Shape (T, N). Total returns in decimal form (0.01 = 1%).
    risk_free_rate : pd.Series
        Annualized risk-free rate in decimal form (0.05 = 5% per year).
        Indexed by date.
    rate_periods_per_year : int
        How to convert the annualized rate to per-period. 252 for daily,
        12 for monthly.
    return_periods_per_year : int
        Currently unused. Reserved for future scaling logic.

    Returns
    -------
    pd.DataFrame
        Excess returns aligned to ``asset_returns`` index. Dates without a
        risk-free rate observation are forward-filled from the most recent
        prior observation.

    Notes
    -----
    The standard convention for FRED Treasury rates: an annualized rate of
    0.0525 corresponds to a daily rate of approximately 0.0525 / 252 under
    the simple approximation, or (1 + 0.0525)^(1/252) - 1 under compounding.
    We use the simple form because Kenneth French's monthly RF column also
    uses the simple form (the source of truth in factor research).
    """
    rate_aligned = risk_free_rate.reindex(asset_returns.index, method="ffill")
    period_rate = rate_aligned / rate_periods_per_year
    return asset_returns.subtract(period_rate, axis=0)
