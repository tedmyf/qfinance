"""Data loaders for equity prices, Fama-French factors, and risk-free rates.

All loaders accept ``use_cache`` and ``refresh_cache`` keyword arguments
(injected by the ``@cached_parquet`` decorator) so callers can bypass or
refresh the Parquet cache without changing any other call site.

Point-in-time discipline is enforced via the ``as_of`` parameter: pass an
ISO date string to truncate results to data that would have been available
at that date.
"""

from __future__ import annotations

import zipfile
from io import BytesIO, StringIO
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yfinance

from tapt.data.cache import cached_parquet
from tapt.data.point_in_time import enforce_as_of

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FF_URLS: dict[tuple[str, str], str] = {
    ("3factor", "monthly"): (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors.zip"
    ),
    ("3factor", "daily"): (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors_daily.zip"
    ),
    ("5factor", "monthly"): (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_5_Factors_2x3.zip"
    ),
    ("5factor", "daily"): (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_5_Factors_2x3_daily.zip"
    ),
}

_FRED_RF_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"


def _download_zipped_csv(url: str) -> str:
    """Download a zip file from ``url`` and return the first CSV entry as text."""
    with urlopen(url) as resp:
        data = resp.read()
    with zipfile.ZipFile(BytesIO(data)) as zf:
        name = zf.namelist()[0]
        return zf.read(name).decode("latin-1")


def _parse_french_csv(text: str, frequency: str) -> pd.DataFrame:
    """Parse the idiosyncratic CSV format used by Kenneth French's data library.

    Parameters
    ----------
    text : str
        Raw file contents (already decoded).
    frequency : str
        ``"monthly"`` or ``"daily"``. Controls date parsing.

    Returns
    -------
    pd.DataFrame
        Values in percentage units (not yet converted to decimal).

    Raises
    ------
    ValueError
        If the header row cannot be located.
    """
    lines = text.strip().splitlines()

    # Locate the first header row: starts with "," and has non-numeric columns.
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith(","):
            continue
        cols = [c.strip() for c in stripped.split(",")[1:]]
        if cols and not cols[0].lstrip("-").replace(".", "").isdigit():
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            "Could not locate the data header in the French CSV file; "
            "check that the input is a valid French data library file."
        )

    columns = [c.strip() for c in lines[header_idx].strip().split(",")[1:]]

    rows: list[tuple[str, list[float]]] = []
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        parts = [p.strip() for p in stripped.split(",")]
        date_str = parts[0]
        if not date_str.lstrip("-").isdigit():
            break
        # Monthly files have a trailing annual-aggregates section with 4-digit years.
        if frequency == "monthly" and len(date_str) == 4:
            break
        values = [float(v) for v in parts[1:] if v != ""]
        if len(values) != len(columns):
            continue
        rows.append((date_str, values))

    if not rows:
        raise ValueError("Could not locate any data rows in the French CSV file.")

    if frequency == "monthly":
        index = pd.to_datetime([r[0] for r in rows], format="%Y%m") + pd.offsets.MonthEnd(0)
    elif frequency == "daily":
        index = pd.to_datetime([r[0] for r in rows], format="%Y%m%d")
    else:
        raise ValueError(f"Unknown frequency: {frequency!r}")

    df = pd.DataFrame([r[1] for r in rows], index=index, columns=columns)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


@cached_parquet("equity_prices")
def load_equity_prices(
    tickers: list[str],
    start: str,
    end: str,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load adjusted closing prices from Yahoo Finance.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.
    start, end : str
        ISO date strings for the query range (inclusive).
    as_of : str or Timestamp, optional
        Truncate result to rows on or before this date.

    Returns
    -------
    pd.DataFrame
        Wide frame with tickers as columns and dates as the index.

    Raises
    ------
    ValueError
        If ``start >= end`` or if the download returns no data.
    """
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError(
            f"start={start!r} must be before end={end!r}"
        )

    raw = yfinance.download(tickers, start=start, end=end, progress=False, auto_adjust=False)

    if raw.empty:
        raise ValueError(f"No price data returned for tickers={tickers}")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Adj Close"].copy()
    elif "Adj Close" in raw.columns:
        prices = raw[["Adj Close"]].copy()
        prices.columns = list(tickers)
    else:
        prices = raw.copy()

    if as_of is not None:
        prices = prices[prices.index <= pd.Timestamp(as_of)]

    return prices


@cached_parquet("ff_factors")
def load_fama_french_factors(
    model: str = "3factor",
    frequency: str = "monthly",
    start: str | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load Fama-French factor returns from Kenneth French's data library.

    Parameters
    ----------
    model : str
        ``"3factor"`` or ``"5factor"``.
    frequency : str
        ``"monthly"`` or ``"daily"``.
    start : str, optional
        Keep only rows on or after this ISO date.
    as_of : str or Timestamp, optional
        Truncate result to rows on or before this date.

    Returns
    -------
    pd.DataFrame
        Factor returns in decimal form (e.g., 0.01 = 1% per period).

    Raises
    ------
    ValueError
        If ``model``/``frequency`` combination is not supported.
    """
    key = (model, frequency)
    if key not in _FF_URLS:
        raise ValueError(
            f"Unsupported combination model={model!r}, frequency={frequency!r}. "
            f"Supported: {sorted(_FF_URLS)}"
        )

    text = _download_zipped_csv(_FF_URLS[key])
    df = _parse_french_csv(text, frequency)
    df = df / 100.0  # percent to decimal

    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]

    if as_of is not None:
        df = df[df.index <= pd.Timestamp(as_of)]

    return df


@cached_parquet("risk_free_rate")
def load_risk_free_rate(
    start: str,
    end: str,
    as_of: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Load the 3-month T-bill rate from FRED as a daily annualized series.

    Missing observations (FRED uses ``"."`` for non-reporting days) are
    forward-filled from the prior trading day.

    Parameters
    ----------
    start, end : str
        ISO date strings defining the range to return.
    as_of : str or Timestamp, optional
        Truncate to rows on or before this date.

    Returns
    -------
    pd.Series
        Annualized risk-free rate in decimal form (e.g., 0.05 = 5%).
    """
    with urlopen(_FRED_RF_URL) as resp:
        raw_bytes = resp.read()

    text = raw_bytes.decode("utf-8")
    df = pd.read_csv(StringIO(text), index_col=0, parse_dates=True, na_values=".")
    rate = df.iloc[:, 0] / 100.0
    rate = rate.ffill()

    rate = rate[rate.index >= pd.Timestamp(start)]
    rate = rate[rate.index <= pd.Timestamp(end)]

    if as_of is not None:
        rate = rate[rate.index <= pd.Timestamp(as_of)]

    return rate


@enforce_as_of()
def compute_returns(
    prices: pd.DataFrame,
    method: str = "simple",
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute period returns from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide price frame (dates x tickers).
    method : str
        ``"simple"`` for arithmetic returns, ``"log"`` for log returns.
    as_of : str or Timestamp, optional
        Truncate result to rows on or before this date (applied by decorator).

    Returns
    -------
    pd.DataFrame
        Returns frame with the first row dropped.

    Raises
    ------
    ValueError
        If ``method`` is not ``"simple"`` or ``"log"``.
    """
    if method == "simple":
        return prices.pct_change().iloc[1:]
    if method == "log":
        return np.log(prices / prices.shift(1)).iloc[1:]
    raise ValueError(f"Unknown method={method!r}; choose 'simple' or 'log'")
