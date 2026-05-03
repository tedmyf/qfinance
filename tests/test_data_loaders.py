"""Tests for data loaders.

Unit tests run against synthetic data and mocked external calls. Integration
tests (marked with ``pytest.mark.integration``) hit the real network. Run them
with ``pytest -m integration``. They are excluded from the default CI run.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from tapt.data.loaders import (
    compute_returns,
    load_equity_prices,
    load_fama_french_factors,
    load_risk_free_rate,
    _parse_french_csv,
)


# ---------------------------------------------------------------------------
# yfinance loader (mocked)
# ---------------------------------------------------------------------------


class TestLoadEquityPrices:
    @patch("yfinance.download")
    def test_single_ticker_returns_named_column(self, mock_download, tmp_cache_dir):
        idx = pd.date_range("2024-01-02", periods=5)
        raw = pd.DataFrame({"Adj Close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=idx)
        mock_download.return_value = raw

        with patch("tapt.data.cache.DEFAULT_CACHE_DIR", tmp_cache_dir):
            prices = load_equity_prices(
                ["AAPL"], "2024-01-01", "2024-01-10", use_cache=False
            )

        assert list(prices.columns) == ["AAPL"]
        assert len(prices) == 5

    @patch("yfinance.download")
    def test_multi_ticker_returns_wide_frame(self, mock_download, tmp_cache_dir):
        idx = pd.date_range("2024-01-02", periods=3)
        raw = pd.DataFrame(
            {"AAPL": [100.0, 101.0, 102.0], "MSFT": [200.0, 201.0, 202.0]},
            index=idx,
        )
        wrapped = pd.concat({"Adj Close": raw}, axis=1)
        mock_download.return_value = wrapped

        with patch("tapt.data.cache.DEFAULT_CACHE_DIR", tmp_cache_dir):
            prices = load_equity_prices(
                ["AAPL", "MSFT"], "2024-01-01", "2024-01-10", use_cache=False
            )

        assert set(prices.columns) == {"AAPL", "MSFT"}

    @patch("yfinance.download")
    def test_empty_response_raises(self, mock_download, tmp_cache_dir):
        mock_download.return_value = pd.DataFrame()
        with patch("tapt.data.cache.DEFAULT_CACHE_DIR", tmp_cache_dir), pytest.raises(ValueError, match="No price data"):
            load_equity_prices(["FAKE"], "2024-01-01", "2024-01-10", use_cache=False)

    def test_invalid_date_range_raises(self, tmp_cache_dir):
        with patch("tapt.data.cache.DEFAULT_CACHE_DIR", tmp_cache_dir), pytest.raises(ValueError, match="must be before"):
            load_equity_prices(
                ["AAPL"], "2024-12-31", "2024-01-01", use_cache=False
            )

    @patch("yfinance.download")
    def test_as_of_truncates_result(self, mock_download, tmp_cache_dir):
        idx = pd.date_range("2024-01-02", periods=10)
        raw = pd.DataFrame({"Adj Close": list(range(100, 110))}, index=idx)
        mock_download.return_value = raw

        with patch("tapt.data.cache.DEFAULT_CACHE_DIR", tmp_cache_dir):
            prices = load_equity_prices(
                ["AAPL"],
                "2024-01-01",
                "2024-01-31",
                as_of="2024-01-05",
                use_cache=False,
            )

        assert prices.index.max() <= pd.Timestamp("2024-01-05")


# ---------------------------------------------------------------------------
# Fama-French parser (unit tested without network)
# ---------------------------------------------------------------------------


SAMPLE_FF_MONTHLY = """
This file was created by CMPT_ME_BEME_RETS using the 202412 CRSP database.

,Mkt-RF,SMB,HML,RF
202401, 1.50, -0.30,  0.80, 0.42
202402, 2.10,  0.40, -0.50, 0.43
202403, 0.80, -0.20,  0.30, 0.43

  Annual Factors: January-December

,Mkt-RF,SMB,HML,RF
2024,    20.5,  -1.0,   3.4, 5.10

Copyright 2025 Kenneth R. French
"""


SAMPLE_FF_DAILY = """
This file was created by CMPT_ME_BEME_RETS_DAILY using the 202412 CRSP database.

,Mkt-RF,SMB,HML,RF
20240102, 0.50, 0.10,-0.20, 0.02
20240103,-0.30,-0.05, 0.10, 0.02
20240104, 0.10, 0.00, 0.05, 0.02

Copyright 2025 Kenneth R. French
"""


class TestParseFrenchCSV:
    """Robust parsing of the eccentric French CSV format is critical."""

    def test_monthly_layout_parses_correctly(self):
        df = _parse_french_csv(SAMPLE_FF_MONTHLY, "monthly")
        assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
        assert len(df) == 3
        # Monthly index should be month-end
        assert df.index[0] == pd.Timestamp("2024-01-31")

    def test_monthly_skips_annual_table(self):
        """The annual aggregate table after the main one must not be included."""
        df = _parse_french_csv(SAMPLE_FF_MONTHLY, "monthly")
        # If we incorrectly read the annual row, we'd see a year-only date
        assert all(df.index.month != 1) or all(df.index.day > 1) or len(df) == 3
        assert len(df) == 3

    def test_daily_layout_parses_correctly(self):
        df = _parse_french_csv(SAMPLE_FF_DAILY, "daily")
        assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
        assert df.index[0] == pd.Timestamp("2024-01-02")

    def test_malformed_csv_raises(self):
        bad = "no header line here\njust some text\n"
        with pytest.raises(ValueError, match="Could not locate"):
            _parse_french_csv(bad, "monthly")


class TestLoadFamaFrenchFactors:
    def test_invalid_combination_raises(self, tmp_cache_dir):
        with patch("tapt.data.cache.DEFAULT_CACHE_DIR", tmp_cache_dir), pytest.raises(ValueError, match="Unsupported"):
            load_fama_french_factors(
                model="3factor",
                frequency="weekly",  # type: ignore
                use_cache=False,
            )

    @patch("tapt.data.loaders._download_zipped_csv")
    def test_converts_percent_to_decimal(self, mock_download, tmp_cache_dir):
        mock_download.return_value = SAMPLE_FF_MONTHLY

        with patch("tapt.data.cache.DEFAULT_CACHE_DIR", tmp_cache_dir):
            factors = load_fama_french_factors(
                model="3factor", frequency="monthly", start="2024-01-01", use_cache=False
            )

        # 1.5% becomes 0.015
        assert factors["Mkt-RF"].iloc[0] == pytest.approx(0.015)
        assert factors["RF"].iloc[0] == pytest.approx(0.0042)

    @patch("tapt.data.loaders._download_zipped_csv")
    def test_as_of_filters_dates(self, mock_download, tmp_cache_dir):
        mock_download.return_value = SAMPLE_FF_MONTHLY

        with patch("tapt.data.cache.DEFAULT_CACHE_DIR", tmp_cache_dir):
            factors = load_fama_french_factors(
                model="3factor",
                frequency="monthly",
                start="2024-01-01",
                as_of="2024-02-29",
                use_cache=False,
            )

        # February month-end is 2024-02-29; should be included. March excluded.
        assert factors.index.max() <= pd.Timestamp("2024-02-29")
        assert len(factors) == 2


# ---------------------------------------------------------------------------
# FRED loader (mocked)
# ---------------------------------------------------------------------------


SAMPLE_FRED_CSV = """observation_date,DGS3MO
2024-01-02,5.25
2024-01-03,5.27
2024-01-04,.
2024-01-05,5.30
"""


class TestLoadRiskFreeRate:
    @patch("tapt.data.loaders.urlopen")
    def test_converts_percent_to_decimal_and_ffills(self, mock_urlopen, tmp_cache_dir):
        # Build a context manager that returns bytes
        class FakeResponse:
            def read(self):
                return SAMPLE_FRED_CSV.encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        mock_urlopen.return_value = FakeResponse()

        with patch("tapt.data.cache.DEFAULT_CACHE_DIR", tmp_cache_dir):
            rate = load_risk_free_rate("2024-01-01", "2024-01-10", use_cache=False)

        assert rate.iloc[0] == pytest.approx(0.0525)
        # The "." in row 3 should have been treated as NaN and ffilled from prior
        assert not rate.isna().any()


# ---------------------------------------------------------------------------
# compute_returns
# ---------------------------------------------------------------------------


class TestComputeReturns:
    def test_simple_returns(self):
        prices = pd.DataFrame(
            {"A": [100.0, 110.0, 121.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        returns = compute_returns(prices, method="simple")
        assert returns["A"].iloc[0] == pytest.approx(0.10)
        assert returns["A"].iloc[1] == pytest.approx(0.10)

    def test_log_returns(self):
        import numpy as np

        prices = pd.DataFrame(
            {"A": [100.0, 110.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        returns = compute_returns(prices, method="log")
        assert returns["A"].iloc[0] == pytest.approx(np.log(1.10))

    def test_unknown_method_raises(self, sample_prices):
        with pytest.raises(ValueError, match="Unknown method"):
            compute_returns(sample_prices, method="invalid")  # type: ignore

    def test_first_row_dropped(self, sample_prices):
        returns = compute_returns(sample_prices)
        assert len(returns) == len(sample_prices) - 1

    def test_as_of_truncation_via_decorator(self):
        prices = pd.DataFrame(
            {"A": [100.0, 110.0, 121.0, 130.0, 140.0]},
            index=pd.date_range("2024-01-01", periods=5),
        )
        returns = compute_returns(prices, as_of="2024-01-03")
        assert returns.index.max() <= pd.Timestamp("2024-01-03")


# ---------------------------------------------------------------------------
# Integration tests (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    """Smoke tests against real external data sources."""

    def test_load_aapl_prices_real(self):
        prices = load_equity_prices(["AAPL"], "2024-01-01", "2024-03-31")
        assert len(prices) > 30
        assert "AAPL" in prices.columns
        assert prices["AAPL"].min() > 0

    def test_load_ff5_factors_real(self):
        factors = load_fama_french_factors(
            model="5factor", frequency="monthly", start="2020-01-01"
        )
        for col in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]:
            assert col in factors.columns
        assert factors["Mkt-RF"].abs().max() < 0.30

    def test_load_risk_free_rate_real(self):
        rate = load_risk_free_rate("2023-01-01", "2023-12-31")
        assert len(rate) > 200
        assert rate.mean() > 0.03
        assert rate.mean() < 0.07
