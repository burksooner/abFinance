"""
Momentum-based trading strategies.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def calculate_momentum(
    prices: pd.DataFrame, lookback_period: int = 252, skip_recent: int = 21
) -> pd.DataFrame:
    """
    Calculate momentum for a set of assets.

    Args:
        prices: DataFrame of asset prices (columns are assets)
        lookback_period: Number of periods to look back for momentum calculation
        skip_recent: Number of recent periods to skip (to avoid short-term reversals)

    Returns:
        DataFrame of momentum scores
    """
    # Calculate returns
    returns = prices.pct_change()

    # Calculate momentum (return over lookback period, excluding recent periods)
    momentum = (
        prices.shift(skip_recent) / prices.shift(lookback_period + skip_recent)
    ) - 1

    return momentum


def momentum_rank_strategy(
    prices: pd.DataFrame,
    lookback_period: int = 252,
    skip_recent: int = 21,
    n_top: int = 5,
    rebalance_freq: str = "M",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Implement a momentum ranking strategy.

    Args:
        prices: DataFrame of asset prices (columns are assets)
        lookback_period: Number of periods to look back for momentum calculation
        skip_recent: Number of recent periods to skip
        n_top: Number of top momentum assets to select
        rebalance_freq: Rebalancing frequency ('D' for daily, 'W' for weekly, 'M' for monthly)

    Returns:
        Tuple of (weights DataFrame, momentum scores DataFrame)
    """
    # Resample prices to the rebalancing frequency
    prices_rebal = prices.resample(rebalance_freq).last()

    # Calculate momentum at each rebalancing date
    momentum = calculate_momentum(prices_rebal, lookback_period, skip_recent)

    # Rank assets by momentum and select top n
    weights = pd.DataFrame(0, index=momentum.index, columns=momentum.columns)

    for date in momentum.index:
        if momentum.loc[date].count() > 0:  # Check if we have data
            # Rank assets by momentum
            ranked = momentum.loc[date].sort_values(ascending=False)

            # Select top n assets
            top_assets = ranked.iloc[:n_top].index.tolist()

            # Equal weight the top assets
            if top_assets:
                weights.loc[date, top_assets] = 1.0 / len(top_assets)

    # Forward fill weights for all dates in the original price series
    weights = weights.reindex(prices.index, method="ffill")

    return weights, momentum


def dual_momentum_strategy(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    lookback_period: int = 252,
    skip_recent: int = 21,
    n_top: int = 5,
    rebalance_freq: str = "M",
) -> pd.DataFrame:
    """
    Implement a dual momentum strategy (combining absolute and relative momentum).

    Args:
        prices: DataFrame of asset prices (columns are assets)
        benchmark_prices: Series of benchmark prices (e.g., risk-free asset or market index)
        lookback_period: Number of periods to look back for momentum calculation
        skip_recent: Number of recent periods to skip
        n_top: Number of top momentum assets to select
        rebalance_freq: Rebalancing frequency ('D' for daily, 'W' for weekly, 'M' for monthly)

    Returns:
        DataFrame of portfolio weights
    """
    # Resample prices to the rebalancing frequency
    prices_rebal = prices.resample(rebalance_freq).last()
    benchmark_rebal = benchmark_prices.resample(rebalance_freq).last()

    # Calculate relative momentum (asset vs asset)
    rel_momentum = calculate_momentum(prices_rebal, lookback_period, skip_recent)

    # Calculate absolute momentum (asset vs benchmark)
    abs_momentum = pd.DataFrame(index=rel_momentum.index, columns=rel_momentum.columns)
    for col in rel_momentum.columns:
        # Calculate asset momentum relative to benchmark
        asset_vs_bench = (
            prices_rebal[col].shift(skip_recent)
            / prices_rebal[col].shift(lookback_period + skip_recent)
        ) - (
            benchmark_rebal.shift(skip_recent)
            / benchmark_rebal.shift(lookback_period + skip_recent)
        )
        abs_momentum[col] = asset_vs_bench

    # Initialize weights
    weights = pd.DataFrame(0, index=rel_momentum.index, columns=rel_momentum.columns)

    for date in rel_momentum.index:
        if rel_momentum.loc[date].count() > 0:  # Check if we have data
            # Filter for assets with positive absolute momentum
            positive_momentum = abs_momentum.loc[date] > 0

            # Rank assets by relative momentum among those with positive absolute momentum
            if positive_momentum.sum() > 0:
                candidates = rel_momentum.loc[date][positive_momentum]
                ranked = candidates.sort_values(ascending=False)

                # Select top n assets
                top_assets = ranked.iloc[: min(n_top, len(ranked))].index.tolist()

                # Equal weight the top assets
                if top_assets:
                    weights.loc[date, top_assets] = 1.0 / len(top_assets)

    # Forward fill weights for all dates in the original price series
    weights = weights.reindex(prices.index, method="ffill")

    return weights
