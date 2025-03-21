"""Functions for analyzing fixed income ETFs including yield metrics, duration analysis, and credit risk."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .performance import calculate_beta
from .statistics import covariance


def calculate_yield_metrics(
    holdings: pd.DataFrame, market_price: float
) -> Dict[str, float]:
    """Calculate various yield metrics for a fixed income ETF.

    Args:
        holdings: DataFrame containing bond holdings with yield and weight columns
        market_price: Current market price of the ETF

    Returns:
        Dict containing:
            - yield_to_maturity: Weighted average YTM of holdings
            - distribution_yield: TTM distribution yield
            - sec_yield: 30-day SEC yield if available
            - yield_to_worst: Weighted average YTW of holdings
    """
    ytm = (holdings["ytm"] * holdings["weight"]).sum()
    ytw = (
        (holdings["ytw"] * holdings["weight"]).sum()
        if "ytw" in holdings.columns
        else None
    )

    return {"yield_to_maturity": ytm, "yield_to_worst": ytw if ytw is not None else ytm}


def analyze_duration_metrics(holdings: pd.DataFrame) -> Dict[str, float]:
    """Calculate duration-related metrics for the ETF.

    Args:
        holdings: DataFrame containing bond holdings with duration and weight columns

    Returns:
        Dict containing:
            - effective_duration: Weighted average effective duration
            - modified_duration: Weighted average modified duration
            - convexity: Portfolio convexity measure
    """
    eff_duration = (holdings["effective_duration"] * holdings["weight"]).sum()
    mod_duration = (
        (holdings["modified_duration"] * holdings["weight"]).sum()
        if "modified_duration" in holdings.columns
        else eff_duration
    )
    convexity = (
        (holdings["convexity"] * holdings["weight"]).sum()
        if "convexity" in holdings.columns
        else None
    )

    return {
        "effective_duration": eff_duration,
        "modified_duration": mod_duration,
        "convexity": convexity,
    }


def calculate_credit_metrics(
    holdings: pd.DataFrame,
) -> Dict[str, Union[float, pd.Series]]:
    """Analyze credit quality distribution and metrics.

    Args:
        holdings: DataFrame containing bond holdings with credit ratings and weights

    Returns:
        Dict containing:
            - credit_distribution: Series of weights by credit rating
            - average_credit_quality: Weighted average credit rating
            - high_yield_exposure: Percentage in high yield bonds
            - investment_grade_exposure: Percentage in investment grade bonds
    """
    credit_dist = holdings.groupby("credit_rating")["weight"].sum()

    # Map ratings to numeric scores (higher = better)
    rating_scores = {
        "AAA": 1,
        "AA+": 2,
        "AA": 3,
        "AA-": 4,
        "A+": 5,
        "A": 6,
        "A-": 7,
        "BBB+": 8,
        "BBB": 9,
        "BBB-": 10,
        "BB+": 11,
        "BB": 12,
        "BB-": 13,
        "B+": 14,
        "B": 15,
        "B-": 16,
        "CCC+": 17,
        "CCC": 18,
        "CCC-": 19,
        "CC": 20,
        "C": 21,
        "D": 22,
    }

    holdings["rating_score"] = holdings["credit_rating"].map(rating_scores)
    avg_rating_score = (holdings["rating_score"] * holdings["weight"]).sum()

    # Calculate high yield exposure
    high_yield_mask = holdings["credit_rating"].apply(
        lambda x: x.startswith("B") or x.startswith("C")
    )
    high_yield_exposure = holdings[high_yield_mask]["weight"].sum()

    return {
        "credit_distribution": credit_dist,
        "average_credit_score": avg_rating_score,
        "high_yield_exposure": high_yield_exposure,
        "investment_grade_exposure": 1 - high_yield_exposure,
    }


def analyze_maturity_profile(
    holdings: pd.DataFrame,
) -> Dict[str, Union[float, pd.Series]]:
    """Analyze the maturity distribution of the ETF holdings.

    Args:
        holdings: DataFrame containing bond holdings with maturity dates and weights

    Returns:
        Dict containing:
            - weighted_average_maturity: Weighted average years to maturity
            - maturity_distribution: Series of weights by maturity bucket
            - key_rate_durations: Duration contribution by maturity bucket
    """
    wam = (holdings["years_to_maturity"] * holdings["weight"]).sum()

    # Define maturity buckets
    buckets = [0, 1, 3, 5, 7, 10, 20, float("inf")]
    labels = ["0-1Y", "1-3Y", "3-5Y", "5-7Y", "7-10Y", "10-20Y", "20Y+"]

    holdings["maturity_bucket"] = pd.cut(
        holdings["years_to_maturity"], bins=buckets, labels=labels, right=False
    )

    maturity_dist = holdings.groupby("maturity_bucket")["weight"].sum()

    return {"weighted_average_maturity": wam, "maturity_distribution": maturity_dist}


def calculate_interest_rate_sensitivity(
    holdings: pd.DataFrame, rate_changes: List[float] = [-1, -0.5, 0.5, 1]
) -> Dict[str, pd.Series]:
    """Calculate the ETF's sensitivity to interest rate changes.

    Args:
        holdings: DataFrame containing bond holdings with duration and convexity
        rate_changes: List of rate changes to analyze (in percentage points)

    Returns:
        Dict containing estimated price changes for different rate scenarios
    """
    duration = analyze_duration_metrics(holdings)
    eff_duration = duration["effective_duration"]
    convexity = duration["convexity"]

    price_changes = {}
    for rate_change in rate_changes:
        # First-order approximation with duration
        price_change = -eff_duration * rate_change

        # Second-order approximation with convexity if available
        if convexity is not None:
            price_change += 0.5 * convexity * (rate_change**2)

        price_changes[f"{rate_change}%"] = price_change

    return {"price_impact": pd.Series(price_changes)}


def analyze_income_stability(monthly_distributions: pd.Series) -> Dict[str, float]:
    """Analyze the stability of the ETF's income distributions.

    Args:
        monthly_distributions: Series of monthly distribution amounts

    Returns:
        Dict containing:
            - distribution_volatility: Standard deviation of monthly distributions
            - distribution_growth: YoY growth in distributions
            - distribution_consistency: Percentage of months with stable/growing distributions
    """
    dist_vol = (
        monthly_distributions.std() / monthly_distributions.mean()
    )  # Coefficient of variation

    # Calculate YoY growth
    current_year = monthly_distributions.tail(12).sum()
    prev_year = monthly_distributions.tail(24).head(12).sum()
    growth = ((current_year / prev_year) - 1) if prev_year > 0 else 0

    # Calculate consistency
    stable_months = (
        monthly_distributions >= monthly_distributions.shift(1)
    ).mean() * 100

    return {
        "distribution_volatility": dist_vol,
        "distribution_growth": growth,
        "distribution_consistency": stable_months,
    }


def calculate_spread_metrics(holdings: pd.DataFrame) -> Dict[str, float]:
    """Calculate spread-related metrics for the fixed income ETF.

    Args:
        holdings: DataFrame containing bond holdings with spread data

    Returns:
        Dict containing:
            - option_adjusted_spread: Weighted average OAS
            - spread_duration: Weighted average spread duration
            - spread_carry: Expected return from spread carry
    """
    oas = (holdings["option_adjusted_spread"] * holdings["weight"]).sum()
    spread_duration = (
        (holdings["spread_duration"] * holdings["weight"]).sum()
        if "spread_duration" in holdings.columns
        else None
    )

    return {
        "option_adjusted_spread": oas,
        "spread_duration": spread_duration,
        "spread_carry": oas
        * (
            1
            - holdings["default_probability"].fillna(0)
            * holdings["loss_given_default"].fillna(0.6)
        ).mean(),
    }


def calculate_sec_yield(
    interest_earned: float,
    expenses: float,
    average_net_assets: float,
    days_in_period: int = 30,
) -> float:
    """Calculate the 30-day SEC yield for a fixed income ETF.

    The SEC yield is calculated by dividing the aggregate net investment income
    over a 30-day period by the average net assets during that same period,
    and annualizing the result.

    Args:
        interest_earned: Total interest earned during the period
        expenses: Fund expenses during the period
        average_net_assets: Average daily net assets during the period
        days_in_period: Number of days in the period (default: 30)

    Returns:
        float: Annualized 30-day SEC yield as a percentage

    Notes:
        - The SEC yield is calculated according to the formula:
          SEC Yield = 2 * ((Interest Earned - Expenses) / (Average Net Assets)) * (365 / days)
        - This is a standardized calculation required by the SEC for bond funds
        - The result is typically expressed as a percentage
        - A 30-day period is standard, but the function allows for different periods
    """
    if average_net_assets <= 0:
        return 0.0

    # Calculate net earnings (interest earned minus expenses)
    net_earnings = interest_earned - expenses

    # Calculate the base yield for the period
    period_yield = net_earnings / average_net_assets

    # Annualize the yield (multiply by 2 per SEC formula)
    annual_yield = 2 * period_yield * (365 / days_in_period)

    # Convert to percentage
    return annual_yield * 100


def calculate_sec_yield_from_daily(
    daily_interest: pd.Series,
    daily_expenses: pd.Series,
    daily_net_assets: pd.Series,
    lookback_days: int = 30,
) -> float:
    """Calculate the 30-day SEC yield using daily data.

    Args:
        daily_interest: Series of daily interest earned
        daily_expenses: Series of daily fund expenses
        daily_net_assets: Series of daily net assets
        lookback_days: Number of days to look back (default: 30)

    Returns:
        float: Annualized 30-day SEC yield as a percentage

    Notes:
        - This function aggregates daily data to calculate the SEC yield
        - All input series should have the same index and length
        - The most recent lookback_days will be used for calculation
    """
    # Validate inputs
    if not (len(daily_interest) == len(daily_expenses) == len(daily_net_assets)):
        raise ValueError("All input series must have the same length")

    # Get the last lookback_days of data
    interest = daily_interest.tail(lookback_days).sum()
    expenses = daily_expenses.tail(lookback_days).sum()
    avg_assets = daily_net_assets.tail(lookback_days).mean()

    return calculate_sec_yield(
        interest_earned=interest,
        expenses=expenses,
        average_net_assets=avg_assets,
        days_in_period=lookback_days,
    )
