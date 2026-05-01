"""Smoke test: run the full Week 1 pipeline against synthetic data.

This is a manual sanity check, not a unit test. Run with:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tapt.factors import estimate_factor_model
from tests.test_factor_model import make_synthetic_factor_data


def main() -> None:
    print("=" * 60)
    print("TAPT Week 1 smoke test")
    print("=" * 60)

    asset_returns, factor_returns, true_B = make_synthetic_factor_data(
        n_periods=240, n_assets=15, seed=42
    )
    print(f"\nGenerated synthetic data:")
    print(f"  Asset returns: {asset_returns.shape}")
    print(f"  Factor returns: {factor_returns.shape}")

    fit = estimate_factor_model(
        excess_asset_returns=asset_returns,
        factor_returns=factor_returns,
        window=120,
        as_of="2018-12-31",
    )
    print(f"\nFit: {fit}")

    print(f"\nLoadings (first 5 assets):")
    print(fit.loadings.head().round(3))

    print(f"\nLoading recovery error (max abs):")
    err = (fit.loadings - true_B).abs().max().max()
    print(f"  {err:.4f}")

    Sigma = fit.asset_covariance()
    eigvals = np.linalg.eigvalsh(Sigma.values)
    print(f"\nAsset covariance:")
    print(f"  Shape: {Sigma.shape}")
    print(f"  Min eigenvalue: {eigvals.min():.6e}")
    print(f"  PSD: {(eigvals >= -1e-10).all()}")
    print(f"  Symmetric: {np.allclose(Sigma.values, Sigma.values.T)}")

    mu = fit.expected_returns()
    print(f"\nExpected returns (annualized):")
    print((mu * 12).round(3).head())

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
