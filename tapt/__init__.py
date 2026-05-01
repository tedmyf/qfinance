"""Tax-Aware Portfolio Toolkit (TAPT).

A research toolkit for factor-based portfolio optimization with integrated
tax-loss harvesting. Built on cvxpy, pandas, and Kenneth French's data library.

Key modules
-----------
data : Loaders for prices, factor returns, and risk-free rates with point-in-time discipline.
factors : Fama-French factor model construction and covariance estimation.
optimization : cvxpy-based portfolio optimizers (MVO, min-TE, Black-Litterman).
backtest : Walk-forward backtesting engine with realistic transaction costs.
analytics : Performance metrics and factor-based attribution.
tax : Lot-level cost basis accounting and wash-sale detection.
harvesting : Tax-loss harvesting overlay integrated with the optimizer.
"""

__version__ = "0.1.0"
__author__ = "Ted (Yufei) Ma"
