# Methodology

This document describes the methodology implemented by the toolkit. It is
intended as a reference for users and as a checklist for self-review during
development. All claims here should be verifiable against the source code.

## 1. Point-in-time data discipline

Look-ahead bias is the most common error in factor research and backtesting.
It arises when data that would not have been available at a given historical
date is used to inform a decision dated as if it were made on that date.

We mitigate this risk in two layers:

1. Every loader function in `tapt.data.loaders` accepts an optional `as_of`
   parameter. When set, the function returns only data with index values
   less than or equal to `as_of`.
2. The `enforce_as_of` decorator wraps functions and applies the same
   truncation defensively after the function returns. This catches the
   common case where the inner function forgets to honor `as_of`.

Backtest loops should pass the current rebalance date as `as_of` to every
data access. The walk-forward engine (planned, week 3) enforces this.

## 2. Factor risk model (planned: week 1, days 4-5)

We use the Fama-French five-factor model (Mkt-RF, SMB, HML, RMW, CMA) plus
optional momentum (UMD) as the asset return generating process.

For an asset universe with returns `R` (T-by-N matrix, T dates, N assets) and
factor returns `F` (T-by-K matrix, K factors), we estimate factor loadings
`B` (N-by-K) by rolling-window OLS:

    R_t = B * F_t + epsilon_t

Idiosyncratic variance `D` (N-by-N diagonal) is the residual variance from
the regression for each asset.

The asset covariance matrix is then:

    Sigma = B * Cov(F) * B' + D

where `Cov(F)` is the K-by-K factor covariance estimated on the same window.

This decomposition has two practical advantages over a sample covariance:

- It is well-conditioned even when N exceeds the number of observations.
- It separates systematic risk (low-rank, from `B * Cov(F) * B'`) from
  idiosyncratic risk (diagonal `D`), enabling clean factor exposure analysis.

## 3. Portfolio optimization (planned: week 2)

We implement three optimization formulations using cvxpy:

### Mean-variance optimization

    minimize    w' Sigma w - lambda * mu' w
    subject to  sum(w) = 1
                w >= 0          (long-only)
                w_min <= w <= w_max  (per-name bounds)
                |w - w_prev|_1 <= turnover_budget

Here `lambda` is a risk-aversion parameter, `mu` is a vector of expected
returns (which can come from historical means, factor model expected returns,
or a Black-Litterman posterior).

### Minimum tracking error to a benchmark

    minimize    (w - w_bench)' Sigma (w - w_bench)
    subject to  sum(w) = 1
                w >= 0
                w_i <= max_weight
                exposure constraints

This is the standard formulation for direct indexing.

### Black-Litterman posterior

Given equilibrium implied returns `pi` (computed from market weights and a
risk-aversion parameter via reverse optimization), and a set of investor
views `Q` with confidence matrix `Omega`, the posterior expected return is:

    mu_BL = ((tau * Sigma)^-1 + P' * Omega^-1 * P)^-1 *
            ((tau * Sigma)^-1 * pi + P' * Omega^-1 * Q)

where `P` is the view-asset incidence matrix and `tau` is a scalar that
controls the weight on the prior. The posterior is then fed into either
mean-variance or minimum-tracking-error optimization.

## 4. Backtesting (planned: week 3)

The backtest engine is walk-forward by construction:

1. Choose a rebalance schedule (e.g., monthly or quarterly).
2. At each rebalance date `t`:
   - Fetch all data with `as_of = t`.
   - Estimate the factor model on the most recent K periods.
   - Solve the optimization given the current portfolio and a turnover budget.
   - Apply transaction costs to the implied trade.
   - Update the portfolio.
3. Accumulate returns from `t` to the next rebalance using realized prices.
4. Compute performance and attribution at the end of the run.

Transaction cost model:

    cost_i = (spread_i / 2) * |trade_i| + impact_coeff * (|trade_i| / ADV_i)^0.5 * |trade_i|

The square-root form is the standard market impact model from the empirical
microstructure literature.

## 5. Tax-loss harvesting (planned: week 4)

The TLH overlay operates at the lot level. Each buy creates a new lot
recorded with date, share count, and price. Each sell consumes lots according
to a configurable selection rule: FIFO, HIFO (highest in first out, which
maximizes loss harvesting), or specific identification.

At each rebalance date:

1. Scan all lots for unrealized losses exceeding a threshold.
2. For each loss-making lot, identify a replacement security that:
   - Maintains factor exposure within tracking error budget.
   - Is not "substantially identical" to the sold security per IRS rules.
   - Was not bought or sold within the past 30 days (wash-sale window).
3. Sell the loss lot, book the loss, buy the replacement.
4. Track replacement chains so the wash-sale rule is enforced across
   subsequent rebalance dates.

The wash-sale rule is implemented as a binary constraint in the cvxpy
optimizer: securities recently transacted are excluded from the buy or sell
universe for thirty calendar days. This integration is the technically
interesting part of the project. Existing open-source TLH scripts treat the
wash-sale rule as a post-hoc filter, which can produce infeasible portfolios.

## 6. After-tax accounting

Each realized gain or loss is classified as short-term (held one year or less)
or long-term (held more than one year). Federal long-term capital gains rates
are applied at user-specified marginal rates (default: 23.8%, top federal LTCG
plus net investment income tax). Short-term gains are taxed at user-specified
ordinary income rates. State tax rates can be added on top.

The reported metric is **after-tax annualized return**, computed by reducing
each year's realized gain by the applicable tax. Unrealized gains accumulate
as a deferred tax liability that is released at the end of the simulation
horizon, so the headline number is **after-tax wealth**, not just **after-tax
realized return**.

Tax alpha is reported as the difference between the after-tax return of the
TLH-overlay portfolio and the after-tax return of an equivalent portfolio
without the overlay, holding pre-tax tracking error roughly constant.

## 7. What this methodology does not capture

Honest limitations:

- AMT and net investment income tax interactions
- Estate planning step-up basis (which affects optimal harvest decisions)
- Multi-account wash-sale rule (a married couple's combined accounts)
- Dividends and qualified dividend income classification
- Foreign tax credits and PFIC complications
- Trader vs. investor tax status distinctions

Some of these would matter materially for a real implementation. They are
deliberately out of scope for a research toolkit.
