"""
Performance analysis functions for financial assets and portfolios.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .statistics import covariance, mean, std


def calculate_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    """
    Calculate returns from a price series.

    Args:
        prices: Series of prices
        method: Return calculation method ('simple' or 'log')

    Returns:
        Series of returns
    """
    if method.lower() == "simple":
        return prices.pct_change().dropna()
    elif method.lower() == "log":
        return np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("Method must be 'simple' or 'log'")


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Calculate cumulative returns from a return series.

    Args:
        returns: Series of returns

    Returns:
        Series of cumulative returns
    """
    return (1 + returns).cumprod() - 1


def calculate_annualized_return(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """
    Calculate annualized return from a return series.

    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly, etc.)

    Returns:
        Annualized return as a decimal
    """
    cumulative_return = calculate_cumulative_returns(returns).iloc[-1]
    n_periods = len(returns)
    return (1 + cumulative_return) ** (periods_per_year / n_periods) - 1


def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility from a return series.

    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly, etc.)

    Returns:
        Annualized volatility as a decimal
    """
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio from a return series.

    Args:
        returns: Series of returns
        risk_free_rate: Annualized risk-free rate as a decimal
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly, etc.)

    Returns:
        Sharpe ratio
    """
    excess_return = (
        calculate_annualized_return(returns, periods_per_year) - risk_free_rate
    )
    volatility = calculate_volatility(returns, periods_per_year)
    return excess_return / volatility if volatility > 0 else 0


def calculate_sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio from a return series.

    Args:
        returns: Series of returns
        risk_free_rate: Annualized risk-free rate as a decimal
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly, etc.)

    Returns:
        Sortino ratio
    """
    excess_return = (
        calculate_annualized_return(returns, periods_per_year) - risk_free_rate
    )
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
    return excess_return / downside_deviation if downside_deviation > 0 else 0


def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Calculate Omega ratio from a return series.

    Parameters
    ----------
    returns : pd.Series
        Series of returns
    threshold : float, optional
        Minimum acceptable return (default is 0.0)

    Returns
    -------
    float
        The Omega ratio
        Returns inf if there are no returns below threshold

    Notes
    -----
    - Ratio of gains to losses above/below threshold
    - More comprehensive than Sharpe ratio
    - Considers entire return distribution
    - Higher values indicate better performance
    """
    excess_returns = returns - threshold
    positive_returns = excess_returns[excess_returns > 0].sum()
    negative_returns = abs(excess_returns[excess_returns < 0].sum())
    return (
        float("inf") if negative_returns == 0 else positive_returns / negative_returns
    )


def calculate_treynor_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Treynor ratio from return series.

    Parameters
    ----------
    returns : pd.Series
        Series of strategy returns
    benchmark_returns : pd.Series
        Series of benchmark returns
    risk_free_rate : float, optional
        Risk-free rate (default is 0.0)
    periods_per_year : int, optional
        Number of periods in a year (default is 252)

    Returns
    -------
    float
        The Treynor ratio
        Returns nan if beta is zero

    Notes
    -----
    - Measures excess return per unit of systematic risk
    - Similar to Sharpe but uses beta instead of volatility
    - Higher values indicate better risk-adjusted performance
    """
    excess_return = (
        calculate_annualized_return(returns, periods_per_year) - risk_free_rate
    )
    beta = calculate_beta(returns, benchmark_returns)
    return excess_return / beta if beta != 0 else float("nan")


def calculate_information_ratio(
    returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252
) -> float:
    """
    Calculate Information ratio from return series.

    Parameters
    ----------
    returns : pd.Series
        Series of strategy returns
    benchmark_returns : pd.Series
        Series of benchmark returns
    periods_per_year : int, optional
        Number of periods in a year (default is 252)

    Returns
    -------
    float
        The Information ratio
        Returns inf if tracking error is zero

    Notes
    -----
    - Measures active return per unit of active risk
    - Active returns = Strategy returns - Benchmark returns
    - Higher values indicate better active management
    - Similar to Sharpe ratio for active returns
    """
    active_returns = returns - benchmark_returns
    active_return = calculate_annualized_return(active_returns, periods_per_year)
    tracking_error = calculate_volatility(active_returns, periods_per_year)
    return active_return / tracking_error if tracking_error > 0 else float("inf")


def calculate_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate beta of returns relative to benchmark.

    Parameters
    ----------
    returns : pd.Series
        Series of strategy returns
    benchmark_returns : pd.Series
        Series of benchmark returns

    Returns
    -------
    float
        The beta value
        Returns 0 if benchmark variance is zero

    Notes
    -----
    - Measures systematic risk relative to benchmark
    - Beta > 1 indicates higher volatility than benchmark
    - Beta < 1 indicates lower volatility than benchmark
    - Beta = 1 indicates same volatility as benchmark
    """
    cov = returns.cov(benchmark_returns)
    var = benchmark_returns.var()
    return cov / var if var != 0 else 0


def calculate_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Jensen's alpha from return series.

    Parameters
    ----------
    returns : pd.Series
        Series of strategy returns
    benchmark_returns : pd.Series
        Series of benchmark returns
    risk_free_rate : float, optional
        Risk-free rate (default is 0.0)
    periods_per_year : int, optional
        Number of periods in a year (default is 252)

    Returns
    -------
    float
        The alpha value

    Notes
    -----
    - Measures excess return relative to CAPM prediction
    - Positive alpha indicates outperformance
    - Negative alpha indicates underperformance
    - Adjusted for systematic risk (beta)
    """
    strategy_return = calculate_annualized_return(returns, periods_per_year)
    market_return = calculate_annualized_return(benchmark_returns, periods_per_year)
    beta = calculate_beta(returns, benchmark_returns)
    return strategy_return - (risk_free_rate + beta * (market_return - risk_free_rate))


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from a return series.

    Args:
        returns: Series of returns

    Returns:
        Maximum drawdown as a decimal
    """
    cumulative_returns = calculate_cumulative_returns(returns)
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / (1 + peak)
    return drawdown.min()


def calculate_performance_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Calculate a comprehensive set of performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Series of returns
    benchmark_returns : pd.Series, optional
        Series of benchmark returns (default is None)
    risk_free_rate : float, optional
        Annualized risk-free rate as a decimal (default is 0.0)
    periods_per_year : int, optional
        Number of periods in a year (default is 252)

    Returns
    -------
    Dict[str, float]
        Dictionary of performance metrics

    Notes
    -----
    - Includes both absolute and relative performance metrics
    - Benchmark-related metrics only if benchmark_returns provided
    - All metrics are annualized where applicable
    - Returns are assumed to be in decimal form
    """
    metrics = {
        "Annualized Return": calculate_annualized_return(returns, periods_per_year),
        "Annualized Volatility": calculate_volatility(returns, periods_per_year),
        "Sharpe Ratio": calculate_sharpe_ratio(
            returns, risk_free_rate, periods_per_year
        ),
        "Sortino Ratio": calculate_sortino_ratio(
            returns, risk_free_rate, periods_per_year
        ),
        "Omega Ratio": calculate_omega_ratio(returns),
        "Maximum Drawdown": calculate_max_drawdown(returns),
        "Positive Periods": (returns > 0).sum() / len(returns),
        "Negative Periods": (returns < 0).sum() / len(returns),
    }

    if benchmark_returns is not None:
        metrics.update(
            {
                "Beta": calculate_beta(returns, benchmark_returns),
                "Alpha": calculate_alpha(
                    returns, benchmark_returns, risk_free_rate, periods_per_year
                ),
                "Information Ratio": calculate_information_ratio(
                    returns, benchmark_returns, periods_per_year
                ),
                "Treynor Ratio": calculate_treynor_ratio(
                    returns, benchmark_returns, risk_free_rate, periods_per_year
                ),
            }
        )

    return metrics
