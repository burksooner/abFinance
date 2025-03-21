"""Functions for analyzing equity ETFs including tracking error, dividend metrics, and factor exposures."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .performance import calculate_beta
from .statistics import covariance


def calculate_tracking_error(
    etf_returns: pd.Series, index_returns: pd.Series, annualize: bool = True
) -> float:
    """Calculate the tracking error between an ETF and its underlying index.

    Args:
        etf_returns: Daily returns of the ETF
        index_returns: Daily returns of the underlying index
        annualize: Whether to annualize the tracking error (default: True)

    Returns:
        float: The tracking error value, annualized if specified

    Notes:
        Tracking error measures how closely the ETF follows its benchmark index.
        Lower values indicate better index replication.
    """
    tracking_diff = etf_returns - index_returns
    te = np.std(tracking_diff)
    if annualize:
        te *= np.sqrt(252)  # Annualize assuming daily data
    return te


def analyze_expense_ratio_impact(
    etf_returns: pd.Series, expense_ratio: float
) -> Dict[str, float]:
    """Analyze the impact of expense ratio on ETF performance.

    Args:
        etf_returns: Daily returns of the ETF
        expense_ratio: Annual expense ratio as a decimal (e.g., 0.0009 for 9 bps)

    Returns:
        Dict containing:
            - daily_cost: Average daily performance impact
            - annual_cost: Annual performance impact
            - cumulative_cost: Total cumulative cost over the period
    """
    daily_cost = expense_ratio / 252
    periods = len(etf_returns)
    years = periods / 252

    return {
        "daily_cost": daily_cost,
        "annual_cost": expense_ratio,
        "cumulative_cost": expense_ratio * years,
    }


def calculate_dividend_metrics(
    prices: pd.Series, dividends: pd.Series
) -> Dict[str, float]:
    """Calculate key dividend metrics for an equity ETF.

    Args:
        prices: Daily closing prices of the ETF
        dividends: Dividend payments series with same index as prices

    Returns:
        Dict containing:
            - dividend_yield: Trailing 12-month dividend yield
            - dividend_growth: YoY dividend growth rate
            - payout_ratio: If available for the ETF
            - distribution_frequency: Number of distributions per year
    """
    annual_div = dividends.rolling(window=252).sum()
    current_price = prices.iloc[-1]

    # Calculate trailing 12-month dividend yield
    ttm_div_yield = (annual_div.iloc[-1] / current_price) if current_price > 0 else 0

    # Calculate YoY dividend growth
    current_year_div = dividends.tail(252).sum()
    prev_year_div = dividends.tail(504).head(252).sum()
    div_growth = ((current_year_div / prev_year_div) - 1) if prev_year_div > 0 else 0

    # Calculate distribution frequency
    dist_frequency = len(dividends[dividends > 0]) / (len(dividends) / 252)

    return {
        "dividend_yield": ttm_div_yield,
        "dividend_growth": div_growth,
        "distribution_frequency": dist_frequency,
    }


def analyze_sector_exposure(
    holdings: pd.DataFrame, sector_column: str = "sector", weight_column: str = "weight"
) -> pd.Series:
    """Analyze sector exposure of an equity ETF.

    Args:
        holdings: DataFrame containing holdings with sector and weight columns
        sector_column: Name of the sector column
        weight_column: Name of the weight column

    Returns:
        pd.Series: Sector weights sorted by exposure
    """
    return (
        holdings.groupby(sector_column)[weight_column]
        .sum()
        .sort_values(ascending=False)
    )


def calculate_factor_exposures(
    etf_returns: pd.Series, factor_returns: pd.DataFrame
) -> Dict[str, float]:
    """Calculate factor exposures using regression against common equity factors.

    Args:
        etf_returns: Daily returns of the ETF
        factor_returns: DataFrame of factor returns (e.g., market, size, value, momentum)

    Returns:
        Dict containing factor betas (exposures) for each factor
    """
    exposures = {}
    for factor in factor_returns.columns:
        factor_beta = calculate_beta(etf_returns, factor_returns[factor])
        exposures[factor] = factor_beta
    return exposures


def analyze_liquidity_metrics(
    volume: pd.Series, prices: pd.Series, shares_outstanding: float
) -> Dict[str, float]:
    """Calculate liquidity metrics for an equity ETF.

    Args:
        volume: Daily trading volume
        prices: Daily closing prices
        shares_outstanding: Current shares outstanding

    Returns:
        Dict containing:
            - avg_daily_volume: Average daily trading volume
            - avg_daily_value_traded: Average daily dollar volume
            - turnover_ratio: Annual turnover ratio
            - bid_ask_spread: Average bid-ask spread if available
    """
    avg_daily_volume = volume.mean()
    avg_daily_value = (volume * prices).mean()
    turnover_ratio = (volume.sum() / shares_outstanding) * (252 / len(volume))

    return {
        "avg_daily_volume": avg_daily_volume,
        "avg_daily_value_traded": avg_daily_value,
        "turnover_ratio": turnover_ratio,
    }


def calculate_creation_redemption_metrics(
    nav: pd.Series, market_price: pd.Series
) -> Dict[str, float]:
    """Calculate metrics related to the ETF creation/redemption mechanism.

    Args:
        nav: Daily Net Asset Value per share
        market_price: Daily market price

    Returns:
        Dict containing:
            - avg_premium_discount: Average premium/discount to NAV
            - premium_discount_volatility: Volatility of premium/discount
            - days_at_premium: Percentage of days trading at premium
    """
    premium_discount = (market_price - nav) / nav
    avg_premium = premium_discount.mean()
    premium_vol = premium_discount.std()
    days_at_premium = (premium_discount > 0).mean() * 100

    return {
        "avg_premium_discount": avg_premium,
        "premium_discount_volatility": premium_vol,
        "days_at_premium": days_at_premium,
    }


def plot_spread_analysis(
    etf_data: pd.DataFrame,
    plot_type: str = "both",
    max_expense_ratio: float = 0.30,
    max_spread: float = 0.20,
    max_volume: float = 40.0,
    figsize: tuple = (12, 10),
) -> None:
    """Create bid-ask spread analysis plots for ETFs.

    Args:
        etf_data: DataFrame containing ETF data with columns:
            - expense_ratio: Expense ratio as percentage
            - bid_ask_spread: 30-day average bid-ask spread as percentage
            - avg_daily_volume: 30-day average daily volume in millions of shares
            - etf_name: Name or ticker of the ETF (optional)
        plot_type: Type of plot to create ('expense', 'volume', or 'both')
        max_expense_ratio: Maximum expense ratio to display (default: 0.30)
        max_spread: Maximum bid-ask spread to display (default: 0.20)
        max_volume: Maximum volume to display in millions (default: 40.0)
        figsize: Figure size as (width, height) tuple

    Notes:
        Creates scatter plots showing the relationships between:
        1. Bid-ask spread vs expense ratio
        2. Bid-ask spread vs trading volume
        Similar to institutional ETF analysis charts
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set the style
    plt.style.use("seaborn")
    sns.set_palette("husl")

    if plot_type in ["expense", "both"]:
        # Create figure for expense ratio plot
        fig1, ax1 = plt.subplots(figsize=(figsize[0], figsize[1] // 2))

        # Plot expense ratio vs spread
        sns.scatterplot(
            data=etf_data, x="expense_ratio", y="bid_ask_spread", alpha=0.6, ax=ax1
        )

        # Add trend line
        sns.regplot(
            data=etf_data,
            x="expense_ratio",
            y="bid_ask_spread",
            scatter=False,
            color="gray",
            ax=ax1,
        )

        # Customize the plot
        ax1.set_xlim(0, max_expense_ratio)
        ax1.set_ylim(0, max_spread)
        ax1.set_xlabel("Expense Ratio (%)")
        ax1.set_ylabel("30-Day Average Bid-Ask (%)")
        ax1.set_title("Bid-Ask Spread Versus Expense Ratio")

        # Add R-squared value
        from scipy import stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            etf_data["expense_ratio"], etf_data["bid_ask_spread"]
        )
        r_squared = r_value**2
        ax1.text(
            0.95,
            0.95,
            f"R² = {r_squared:.4f}",
            transform=ax1.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
        )

        plt.tight_layout()

    if plot_type in ["volume", "both"]:
        # Create figure for volume plot
        fig2, ax2 = plt.subplots(figsize=(figsize[0], figsize[1] // 2))

        # Plot volume vs spread
        sns.scatterplot(
            data=etf_data, x="avg_daily_volume", y="bid_ask_spread", alpha=0.6, ax=ax2
        )

        # Customize the plot
        ax2.set_xlim(0, max_volume)
        ax2.set_ylim(0, max_spread)
        ax2.set_xlabel("30-Day Average Daily Volume (Shares) (Millions)")
        ax2.set_ylabel("30-Day Average Bid-Ask ($)")
        ax2.set_title("Bid-Ask Spread Versus Trading Volume")

        plt.tight_layout()


def analyze_liquidity_comparison(
    etf_list: List[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """Analyze and compare liquidity metrics for a list of ETFs.

    Args:
        etf_list: List of ETF tickers to analyze
        start_date: Start date for analysis (YYYY-MM-DD)
        end_date: End date for analysis (YYYY-MM-DD)

    Returns:
        DataFrame containing liquidity metrics for comparison:
            - expense_ratio
            - bid_ask_spread (30-day average)
            - avg_daily_volume (30-day average)
            - avg_daily_value_traded
            - median_spread
            - spread_volatility
    """
    metrics = []

    for etf in etf_list:
        # Calculate 30-day averages and other metrics
        # Note: This is a placeholder - actual implementation would
        # need to fetch real market data
        metrics.append(
            {
                "etf": etf,
                "expense_ratio": 0.0,  # To be filled with actual data
                "bid_ask_spread": 0.0,  # To be filled with actual data
                "avg_daily_volume": 0.0,  # To be filled with actual data
                "avg_daily_value_traded": 0.0,  # To be filled with actual data
                "median_spread": 0.0,  # To be filled with actual data
                "spread_volatility": 0.0,  # To be filled with actual data
            }
        )

    return pd.DataFrame(metrics)


def calculate_relative_liquidity_score(
    etf_data: pd.DataFrame, weights: Dict[str, float] = None
) -> pd.Series:
    """Calculate a relative liquidity score for ETFs based on multiple metrics.

    Args:
        etf_data: DataFrame containing ETF metrics
        weights: Dictionary of weights for each metric (optional)
                Default weights prioritize spread and volume equally

    Returns:
        Series of liquidity scores (higher is better)
    """
    if weights is None:
        weights = {
            "bid_ask_spread": -0.4,  # Negative because lower is better
            "avg_daily_volume": 0.4,
            "expense_ratio": -0.2,  # Negative because lower is better
        }

    # Normalize each metric to 0-1 scale
    normalized = pd.DataFrame()
    for col in weights.keys():
        if col in ["bid_ask_spread", "expense_ratio"]:
            # For these metrics, lower is better
            normalized[col] = 1 - (etf_data[col] - etf_data[col].min()) / (
                etf_data[col].max() - etf_data[col].min()
            )
        else:
            # For these metrics, higher is better
            normalized[col] = (etf_data[col] - etf_data[col].min()) / (
                etf_data[col].max() - etf_data[col].min()
            )

    # Calculate weighted score
    score = pd.Series(0, index=etf_data.index)
    for col, weight in weights.items():
        score += normalized[col] * abs(weight)

    return score
