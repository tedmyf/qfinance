"""Vectorized OLS helpers for factor model estimation.

When all assets share the same regressor matrix X (typical for cross-sectional
factor models), we can solve all N regressions simultaneously with a single
matrix solve. This is much faster than looping over assets, and the API is
cleaner because we operate on numpy arrays directly.

This module is internal. The public interface is in tapt.factors.model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BulkOLSResult:
    """Result of a vectorized OLS regression of multiple dependent variables.

    Attributes
    ----------
    coef : np.ndarray
        Shape (P, N). Coefficients per regressor per dependent variable.
        If the X matrix included an intercept, the intercept row is the first row.
    residuals : np.ndarray
        Shape (T, N). Residuals per observation per dependent variable.
    sigma2 : np.ndarray
        Shape (N,). Residual variance per dependent variable, computed with
        degrees-of-freedom adjustment (T - P).
    r_squared : np.ndarray
        Shape (N,). Coefficient of determination per dependent variable.
    n_obs : int
        Number of observations T.
    n_params : int
        Number of regressors P (including intercept if present).
    """

    coef: np.ndarray
    residuals: np.ndarray
    sigma2: np.ndarray
    r_squared: np.ndarray
    n_obs: int
    n_params: int


def ols_bulk(Y: np.ndarray, X: np.ndarray) -> BulkOLSResult:
    """Solve N least-squares problems sharing a single regressor matrix.

    Parameters
    ----------
    Y : np.ndarray
        Shape (T, N). Each column is a dependent variable.
    X : np.ndarray
        Shape (T, P). Regressor matrix. Include a column of ones for an intercept.

    Returns
    -------
    BulkOLSResult

    Raises
    ------
    ValueError
        If shapes are incompatible, if T <= P (degrees of freedom), or if X
        is rank-deficient.

    Notes
    -----
    Solves the normal equations X'X beta = X'Y directly via np.linalg.solve.
    For the typical factor model use case (P = 4 to 6, T = 24 to 120, N = 30
    to 500) this is fast and numerically stable.

    NaN handling is the caller's responsibility. Pass complete data only.
    """
    if Y.ndim != 2 or X.ndim != 2:
        raise ValueError(f"Y and X must be 2D; got Y.ndim={Y.ndim}, X.ndim={X.ndim}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have same number of rows; got {X.shape[0]} and {Y.shape[0]}"
        )
    if np.isnan(Y).any() or np.isnan(X).any():
        raise ValueError("ols_bulk does not accept NaN values; clean inputs upstream")

    n, p = X.shape
    if n <= p:
        raise ValueError(f"Need n > p observations, got n={n} p={p}")

    # Solve normal equations
    XtX = X.T @ X
    XtY = X.T @ Y
    try:
        coef = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"X'X is singular; check for collinear regressors: {e}") from e

    fitted = X @ coef
    residuals = Y - fitted

    dof = n - p
    sigma2 = np.sum(residuals**2, axis=0) / dof

    Y_centered = Y - Y.mean(axis=0)
    tss = np.sum(Y_centered**2, axis=0)
    rss = np.sum(residuals**2, axis=0)
    # Avoid divide-by-zero on constant Y columns
    with np.errstate(divide="ignore", invalid="ignore"):
        r_squared = np.where(tss > 0, 1.0 - rss / tss, np.nan)

    return BulkOLSResult(
        coef=coef,
        residuals=residuals,
        sigma2=sigma2,
        r_squared=r_squared,
        n_obs=n,
        n_params=p,
    )
