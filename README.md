# Tax-Aware Portfolio Toolkit (TAPT)

Factor-based portfolio optimization with integrated tax-loss harvesting.
Built on cvxpy, pandas, and Kenneth French's data library.

[![CI](https://github.com/tedmyf/tax-aware-portfolio-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/tedmyf/tax-aware-portfolio-toolkit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What this is

A research toolkit that combines four capabilities into a single cohesive pipeline:

1. Fama-French factor model construction with proper point-in-time discipline
2. cvxpy-based portfolio optimization (mean-variance, minimum tracking error, Black-Litterman)
3. Walk-forward backtesting with realistic transaction cost modeling
4. Lot-level tax-loss harvesting overlay with wash-sale-aware optimization

The integration is the point. Existing open-source libraries cover one or two
of these well. None combine all four with the rigor required for institutional
research.

## Why it exists

Direct indexing and tax-managed SMA strategies are a rapidly growing segment
of the wealth management industry. The mechanics are well understood at major
asset managers (Aperio, Parametric, Vanguard Personalized Indexing) but the
public open-source ecosystem lags behind. This toolkit is a transparent
reference implementation of the core methodology.

## Status

This is an active research project. The roadmap below reflects the planned
build order. Modules marked complete have unit-tested implementations and
working examples.

| Module             | Status         | Description                                            |
|--------------------|----------------|--------------------------------------------------------|
| `data`             | Complete       | Loaders, point-in-time discipline, parquet caching     |
| `factors`          | Complete       | Fama-French model construction, factor covariance      |
| `optimization`     | Planned        | cvxpy optimizers (MVO, min-TE, Black-Litterman)        |
| `backtest`         | Planned        | Walk-forward engine with transaction costs             |
| `analytics`        | Planned        | Performance metrics and factor attribution             |
| `tax`              | Planned        | Lot-level cost basis, wash-sale detection              |
| `harvesting`       | Planned        | TLH overlay integrated with the optimizer              |

## Quick start

```bash
git clone https://github.com/tedmyf/tax-aware-portfolio-toolkit.git
cd tax-aware-portfolio-toolkit
pip install -e ".[dev]"
pytest -m "not integration"
```

```python
from tapt.data import load_equity_prices, load_fama_french_factors, load_risk_free_rate
from tapt.data.loaders import compute_returns
from tapt.factors import estimate_factor_model, to_excess_returns

prices = load_equity_prices(
    tickers=["AAPL", "MSFT", "JPM", "JNJ", "PG"],
    start="2018-01-01",
    end="2024-12-31",
)
factors = load_fama_french_factors(model="5factor", frequency="monthly")
rf = load_risk_free_rate(start="2018-01-01", end="2024-12-31")

monthly_returns = compute_returns(prices.resample("ME").last())
excess = to_excess_returns(monthly_returns, rf.resample("ME").last(), rate_periods_per_year=12)

fit = estimate_factor_model(
    excess_asset_returns=excess,
    factor_returns=factors,
    window=60,
    as_of="2024-12-31",
)

Sigma = fit.asset_covariance()      # N x N covariance, ready for cvxpy
mu = fit.expected_returns()         # Factor-implied expected excess returns
```

See `notebooks/01_factor_model_demo.ipynb` for a fully worked example.

## Methodology

The toolkit implements a methodology consistent with public descriptions of
how institutional direct indexing platforms operate:

1. Estimate a factor risk model on a rolling window using Fama-French factors.
2. Construct a portfolio that minimizes tracking error to a benchmark, subject
   to constraints on weights, sector exposures, and turnover.
3. At each rebalance date, identify positions with unrealized losses, sell
   them to harvest the tax benefit, and substitute with replacement securities
   selected to maintain factor exposure within a tracking error budget.
4. Account for the wash-sale rule by excluding recently sold securities (and
   substantially identical replacements) from the buy universe for thirty days.

The trade-off between tax alpha and tracking error is the central object of
study. We characterize this frontier across realistic parameter ranges.

## Limitations

This toolkit is a research tool, not investment advice or production software.

- Universe construction uses yfinance, which is not survivorship-bias-free.
  Results on long historical windows will overstate performance versus a
  CRSP-based study.
- Transaction cost models are calibrated to liquid US equities. Small-cap and
  international applications would require recalibration.
- Tax accounting models US federal long-term and short-term capital gains
  rates. State variation is parameterized but not modeled in detail. Estate
  planning interactions, AMT, and net investment income tax are not modeled.
- The wash-sale rule implementation handles the most common cases. Complex
  patterns (substantially identical securities across funds, married couple
  cross-account violations) are flagged but not fully resolved.

## References

- Aperio Group. "Personalized Indexing." BlackRock white papers.
- Vanguard. "Personalized Indexing: A Portfolio Construction Plan." (2022)
- Parametric Portfolio Associates. Research on after-tax optimization.
- Fama, E. F. and French, K. R. (2015). "A Five-Factor Asset Pricing Model."
- Elm Wealth. "Robbing Peter to Pay Paul: Long/Short Direct Index Tax-Loss
  Harvesting."
- Boyd, S. et al. "Multi-Period Trading via Convex Optimization." (cvxportfolio).

## License

MIT. See [LICENSE](LICENSE).

## Author

Ted (Yufei) Ma. Investment Decision Analyst at Morningstar. CFA Level 2
candidate. University of Chicago Booth School of Business, MFA 2025.
