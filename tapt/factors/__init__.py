"""Factor model construction.

Estimates Fama-French style factor models and produces asset covariance
matrices and expected returns suitable for portfolio optimization.

Public API
----------
estimate_factor_model : Fit a factor model on a window of returns.
FactorModelFit : Frozen result of estimate_factor_model.
to_excess_returns : Convert total returns to excess returns using a rate series.
"""

from tapt.factors.model import (
    FactorModelFit,
    estimate_factor_model,
    to_excess_returns,
)

__all__ = [
    "FactorModelFit",
    "estimate_factor_model",
    "to_excess_returns",
]
