"""Data loading utilities for TAPT.

Provides loaders for equity prices, Fama-French factors, and risk-free rates,
along with a Parquet caching layer and point-in-time discipline helpers.
"""

from tapt.data.loaders import (
    compute_returns,
    load_equity_prices,
    load_fama_french_factors,
    load_risk_free_rate,
)

__all__ = [
    "compute_returns",
    "load_equity_prices",
    "load_fama_french_factors",
    "load_risk_free_rate",
]
