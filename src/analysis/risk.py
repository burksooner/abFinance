"""
Risk metrics for financial analysis.
"""

from typing import List, Optional

from .statistics import mean, percentile, std


def value_at_risk(
    returns: List[float], confidence_level: float = 0.95
) -> Optional[float]:
    """
    Calculates Value at Risk (VaR) at the given confidence level.

    Parameters
    ----------
    returns : list[float]
        List of period returns
    confidence_level : float, optional
        Confidence level for VaR calculation (default is 0.95)

    Returns
    -------
    float or None
        The VaR value at specified confidence level

    Notes
    -----
    - Historical VaR calculation method
    - Represents potential loss at given confidence level
    - Uses percentile function for calculation
    - Common confidence levels: 0.95, 0.99
    """
    percentile_value = (1 - confidence_level) * 100
    return percentile(returns, percentile_value)


def conditional_var(
    returns: List[float], confidence_level: float = 0.95
) -> Optional[float]:
    """
    Calculates Conditional Value at Risk (CVaR) or Expected Shortfall.

    Parameters
    ----------
    returns : list[float]
        List of period returns
    confidence_level : float, optional
        Confidence level for CVaR calculation (default is 0.95)

    Returns
    -------
    float or None
        The CVaR value at specified confidence level

    Notes
    -----
    - Also known as Expected Shortfall
    - Average loss beyond VaR
    - More sensitive to tail risk than VaR
    - Returns VaR if no returns beyond VaR threshold
    """
    var_val = value_at_risk(returns, confidence_level)
    if var_val is None:
        return None
    tail_losses = [r for r in returns if r <= var_val]
    if tail_losses:
        return mean(tail_losses)
    return var_val


def beta(strategy_returns: List[float], benchmark_returns: List[float]) -> float:
    """
    Calculates beta, measuring systematic risk relative to the benchmark.

    Parameters
    ----------
    strategy_returns : list[float]
        List of strategy returns
    benchmark_returns : list[float]
        List of benchmark returns

    Returns
    -------
    float
        The beta value
        Returns 0 if benchmark variance is zero

    Notes
    -----
    - Formula: Covariance(strategy, benchmark) / Variance(benchmark)
    - Beta > 1 indicates higher volatility than benchmark
    - Beta < 1 indicates lower volatility than benchmark
    - Beta = 1 indicates same volatility as benchmark
    """
    if not strategy_returns or not benchmark_returns:
        return 0

    benchmark_var = sum(
        (x - mean(benchmark_returns)) ** 2 for x in benchmark_returns
    ) / len(benchmark_returns)
    if benchmark_var == 0:
        return 0

    covar = sum(
        (s - mean(strategy_returns)) * (b - mean(benchmark_returns))
        for s, b in zip(strategy_returns, benchmark_returns)
    ) / len(strategy_returns)
    return covar / benchmark_var


def max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculates the maximum peak-to-trough decline in the equity curve.

    Parameters
    ----------
    equity_curve : list[float]
        List of equity values over time

    Returns
    -------
    float
        The maximum drawdown as a negative decimal

    Notes
    -----
    - Represents worst historical loss from peak to trough
    - Important measure of downside risk
    - Always negative or zero
    - Calculated using running maximum approach
    """
    if not equity_curve:
        return 0

    peak = equity_curve[0]
    max_dd = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (equity - peak) / peak
        if drawdown < max_dd:
            max_dd = drawdown
    return max_dd


def longest_drawdown_days(equity_curve: List[float]) -> int:
    """
    Calculates the longest consecutive period of drawdown in days.

    Parameters
    ----------
    equity_curve : list[float]
        List of equity values over time

    Returns
    -------
    int
        Number of consecutive days in longest drawdown period

    Notes
    -----
    - Measures persistence of drawdowns
    - Counts consecutive days below previous peak
    - Important for understanding recovery periods
    - Complements maximum drawdown metric
    """
    if not equity_curve:
        return 0

    peak = equity_curve[0]
    longest = 0
    current = 0
    for equity in equity_curve:
        if equity < peak:
            current += 1
        else:
            current = 0
            peak = equity
        longest = max(longest, current)
    return longest


def average_drawdown_percentage(equity_curve: List[float]) -> float:
    """
    Calculates the average drawdown percentage during drawdown periods.

    Parameters
    ----------
    equity_curve : list[float]
        List of equity values over time

    Returns
    -------
    float
        Average drawdown as a percentage
        Returns 0 if no drawdowns

    Notes
    -----
    - Provides typical drawdown magnitude
    - Includes all drawdown periods
    - Returns percentage value
    - Less sensitive to outliers than max drawdown
    """
    if not equity_curve:
        return 0

    peak = equity_curve[0]
    drawdowns = []
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak
        if dd < 0:
            drawdowns.append(dd)
    return mean(drawdowns) * 100 if drawdowns else 0


def time_under_water_percentage(equity_curve: List[float]) -> float:
    """
    Calculates the percentage of time the equity curve is below its peak.

    Parameters
    ----------
    equity_curve : list[float]
        List of equity values over time

    Returns
    -------
    float
        Percentage of time spent in drawdown (0-100)

    Notes
    -----
    - Measures frequency of drawdowns
    - Important for psychological impact assessment
    - Returns percentage value (0-100)
    - Complements drawdown magnitude metrics
    """
    if not equity_curve:
        return 0

    peak = equity_curve[0]
    under_water = 0
    for equity in equity_curve:
        if equity < peak:
            under_water += 1
        else:
            peak = equity
    return (under_water / len(equity_curve)) * 100
