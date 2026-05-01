"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Isolated cache directory for tests."""
    cache = tmp_path / "tapt_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Synthetic daily price data for 3 tickers, 252 trading days.

    The series are generated from independent geometric Brownian motions
    with known parameters, so downstream tests can verify expected return
    and volatility calculations against analytical answers.
    """
    rng = np.random.default_rng(seed=42)
    n_days = 252
    n_assets = 3
    tickers = ["AAA", "BBB", "CCC"]
    mu = np.array([0.10, 0.08, 0.12]) / 252  # daily drift
    sigma = np.array([0.20, 0.15, 0.25]) / np.sqrt(252)  # daily vol

    log_returns = rng.normal(loc=mu, scale=sigma, size=(n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))

    dates = pd.bdate_range("2024-01-02", periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def sample_factor_returns() -> pd.DataFrame:
    """Synthetic Fama-French 3-factor returns, monthly."""
    rng = np.random.default_rng(seed=123)
    n_months = 60
    factors = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.005, 0.04, n_months),
            "SMB": rng.normal(0.002, 0.025, n_months),
            "HML": rng.normal(0.003, 0.03, n_months),
            "RF": rng.normal(0.002, 0.001, n_months),
        },
        index=pd.date_range("2020-01-31", periods=n_months, freq="ME"),
    )
    factors.index.name = "date"
    return factors


@pytest.fixture
def multi_index_panel() -> pd.DataFrame:
    """A long-format panel with (date, ticker) MultiIndex for testing PIT discipline."""
    dates = pd.date_range("2024-01-01", "2024-01-10")
    tickers = ["AAA", "BBB"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(seed=7)
    return pd.DataFrame(
        {"return": rng.normal(0, 0.01, len(idx)), "volume": rng.integers(1000, 10000, len(idx))},
        index=idx,
    )
