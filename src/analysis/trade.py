"""
Trade-specific metrics for financial analysis.
"""

from typing import List

from .statistics import mean, std


def profit_factor(trade_results: List[float]) -> float:
    """
    Calculates the Profit Factor: ratio of total profits to total losses.

    Parameters
    ----------
    trade_results : list[float]
        List of individual trade profits/losses

    Returns
    -------
    float
        The profit factor
        Returns inf if there are no losing trades

    Notes
    -----
    - Formula: sum(profits) / sum(|losses|)
    - Values > 1 indicate overall profitability
    - Higher values indicate better performance
    - Infinity indicates no losing trades
    """
    sum_wins = sum(trade for trade in trade_results if trade > 0)
    sum_losses = sum(abs(trade) for trade in trade_results if trade < 0)
    if sum_losses == 0:
        return float("inf")
    return sum_wins / sum_losses


def awal_ratio(trade_results: List[float]) -> float:
    """
    Calculates the Average Win to Average Loss (AWAL) Ratio.

    Parameters
    ----------
    trade_results : list[float]
        List of individual trade profits/losses

    Returns
    -------
    float
        The AWAL ratio
        Returns inf if there are no losing trades

    Notes
    -----
    - Formula: average_win / |average_loss|
    - Different from profit factor (uses averages instead of sums)
    - Higher values indicate better risk/reward
    - Infinity indicates no losing trades
    """
    wins = [trade for trade in trade_results if trade > 0]
    losses = [trade for trade in trade_results if trade < 0]
    if not losses:
        return float("inf")
    avg_win = mean(wins) if wins else 0
    avg_loss = mean(losses) if losses else 0
    if avg_loss == 0:
        return float("inf")
    return avg_win / abs(avg_loss)


def calculate_expectancy(trade_results: List[float]) -> float:
    """
    Calculates the expected value per trade.

    Parameters
    ----------
    trade_results : list[float]
        List of individual trade profits/losses

    Returns
    -------
    float
        The expected value per trade

    Notes
    -----
    - Formula: (win_prob * avg_win) + ((1 - win_prob) * avg_loss)
    - Combines win rate with average win/loss
    - Key metric for system profitability
    - Returns 0 for empty trade list
    """
    if not trade_results:
        return 0

    wins = [trade for trade in trade_results if trade > 0]
    losses = [trade for trade in trade_results if trade < 0]
    win_prob = len(wins) / len(trade_results) if trade_results else 0
    avg_win = mean(wins) if wins else 0
    avg_loss = mean(losses) if losses else 0
    return win_prob * avg_win + (1 - win_prob) * avg_loss


def rina_index(trade_results: List[float], max_drawdown: float) -> float:
    """
    Calculates the RINA Index, relating expectancy to maximum drawdown.

    Parameters
    ----------
    trade_results : list[float]
        List of individual trade profits/losses
    max_drawdown : float
        Maximum drawdown value (as a negative decimal)

    Returns
    -------
    float
        The RINA Index
        Returns inf if max_drawdown is zero

    Notes
    -----
    - Formula: Expectancy / |MaxDrawdown|
    - Relates system profitability to risk
    - Higher values indicate better risk-adjusted performance
    - Infinity indicates no drawdown
    """
    exp = calculate_expectancy(trade_results)
    if max_drawdown == 0:
        return float("inf")
    return exp / abs(max_drawdown)


def average_trade(trade_results: List[float]) -> float:
    """
    Calculates the arithmetic mean of all trade results.

    Parameters
    ----------
    trade_results : list[float]
        List of individual trade profits/losses

    Returns
    -------
    float
        The average trade result
        Returns 0 for empty trade list

    Notes
    -----
    - Simple arithmetic mean of all trades
    - Includes both winning and losing trades
    - Positive value indicates overall profitability
    - Uses mean function
    """
    return mean(trade_results)


def winning_percentage(trade_results: List[float]) -> float:
    """
    Calculates the percentage of winning trades.

    Parameters
    ----------
    trade_results : list[float]
        List of individual trade profits/losses

    Returns
    -------
    float
        The winning percentage (0-100)
        Returns 0 if no trades

    Notes
    -----
    - Formula: (number of winning trades / total trades) * 100
    - Winning trade defined as profit > 0
    - Returns percentage value (0-100)
    - Important but should not be viewed in isolation
    """
    if not trade_results:
        return 0
    wins = len([trade for trade in trade_results if trade > 0])
    return wins / len(trade_results) * 100


def max_consecutive_loss(trade_results: List[float]) -> int:
    """
    Calculates the maximum number of consecutive losing trades.

    Parameters
    ----------
    trade_results : list[float]
        List of individual trade profits/losses

    Returns
    -------
    int
        Maximum number of consecutive losing trades

    Notes
    -----
    - Important for risk management
    - Helps in sizing position and risk limits
    - Indicates worst historical losing streak
    - Useful for psychological preparation
    """
    max_losses = 0
    current_losses = 0
    for trade in trade_results:
        if trade < 0:
            current_losses += 1
            max_losses = max(max_losses, current_losses)
        else:
            current_losses = 0
    return max_losses


def std_dev_trades(trade_results: List[float]) -> float:
    """
    Calculates the standard deviation of trade outcomes.

    Parameters
    ----------
    trade_results : list[float]
        List of individual trade profits/losses

    Returns
    -------
    float
        Standard deviation of trade results
        Returns 0 for empty trade list

    Notes
    -----
    - Measures consistency of trading results
    - Uses population standard deviation
    - Lower values indicate more consistent returns
    - Uses std function
    """
    return std(trade_results)
