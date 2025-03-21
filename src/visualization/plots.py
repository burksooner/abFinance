"""
Plotting functions for financial data visualization.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_returns(
    returns: pd.Series,
    title: str = "Returns",
    figsize: Tuple[int, int] = (12, 6),
    color: str = "blue",
    show_mean: bool = True,
) -> plt.Figure:
    """
    Plot a time series of returns.

    Args:
        returns: Series of returns
        title: Plot title
        figsize: Figure size
        color: Line color
        show_mean: Whether to show the mean return line

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    returns.plot(ax=ax, color=color)

    if show_mean:
        mean_return = returns.mean()
        ax.axhline(
            mean_return,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {mean_return:.4f}",
        )

    ax.set_title(title)
    ax.set_ylabel("Return")
    ax.legend()

    return fig


def plot_cumulative_returns(
    returns: Union[pd.Series, pd.DataFrame],
    title: str = "Cumulative Returns",
    figsize: Tuple[int, int] = (12, 6),
    colors: Optional[List[str]] = None,
    baseline: bool = True,
) -> plt.Figure:
    """
    Plot cumulative returns over time.

    Args:
        returns: Series or DataFrame of returns
        title: Plot title
        figsize: Figure size
        colors: List of colors for multiple return series
        baseline: Whether to show a baseline at 0

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(returns, pd.Series):
        cumulative = (1 + returns).cumprod() - 1
        cumulative.plot(ax=ax, color=colors[0] if colors else "blue")
    else:
        cumulative = (1 + returns).cumprod() - 1
        cumulative.plot(ax=ax, color=colors)

    if baseline:
        ax.axhline(0, color="black", linestyle="-", alpha=0.2)

    ax.set_title(title)
    ax.set_ylabel("Cumulative Return")
    ax.legend()

    return fig


def plot_drawdowns(
    returns: pd.Series,
    title: str = "Drawdowns",
    figsize: Tuple[int, int] = (12, 6),
    color: str = "red",
    threshold: Optional[float] = -0.1,
) -> plt.Figure:
    """
    Plot drawdowns over time.

    Args:
        returns: Series of returns
        title: Plot title
        figsize: Figure size
        color: Line color
        threshold: Optional threshold to highlight significant drawdowns

    Returns:
        Matplotlib figure
    """
    cumulative_returns = (1 + returns).cumprod() - 1
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / (1 + peak)

    fig, ax = plt.subplots(figsize=figsize)
    drawdown.plot(ax=ax, color=color)

    if threshold is not None:
        ax.axhline(
            threshold,
            color="black",
            linestyle="--",
            alpha=0.7,
            label=f"Threshold: {threshold:.1%}",
        )

    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.legend()

    return fig


def plot_rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    metrics: List[str] = ["return", "volatility", "sharpe"],
    figsize: Tuple[int, int] = (12, 8),
    colors: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot rolling financial metrics.

    Args:
        returns: Series of returns
        window: Rolling window size
        metrics: List of metrics to plot
        figsize: Figure size
        colors: List of colors for the metrics

    Returns:
        Matplotlib figure
    """
    if colors is None:
        colors = ["blue", "green", "red", "purple", "orange"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        if metric.lower() == "return":
            rolling_data = returns.rolling(window).mean() * 252  # Annualized
            label = "Rolling Annualized Return"
        elif metric.lower() == "volatility":
            rolling_data = returns.rolling(window).std() * np.sqrt(252)  # Annualized
            label = "Rolling Annualized Volatility"
        elif metric.lower() == "sharpe":
            rolling_return = returns.rolling(window).mean() * 252
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            rolling_data = rolling_return / rolling_vol
            label = "Rolling Sharpe Ratio"
        else:
            continue

        rolling_data.plot(ax=axes[i], color=colors[i % len(colors)], label=label)
        axes[i].set_ylabel(label)
        axes[i].legend()

    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    returns: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "coolwarm",
) -> plt.Figure:
    """
    Plot a correlation matrix of asset returns.

    Args:
        returns: DataFrame of returns (columns are assets)
        title: Plot title
        figsize: Figure size
        cmap: Colormap for the heatmap

    Returns:
        Matplotlib figure
    """
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=True,
        fmt=".2f",
        ax=ax,
    )

    ax.set_title(title)

    return fig


def plot_performance_summary(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Create a comprehensive performance summary plot.

    Args:
        returns: Series of strategy returns
        benchmark_returns: Optional series of benchmark returns
        risk_free_rate: Annualized risk-free rate
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    # Create a 2x2 grid of subplots
    gs = fig.add_gridspec(2, 2)

    # Cumulative returns plot
    ax1 = fig.add_subplot(gs[0, :])
    cum_returns = (1 + returns).cumprod() - 1
    cum_returns.plot(ax=ax1, color="blue", label="Strategy")

    if benchmark_returns is not None:
        cum_bench = (1 + benchmark_returns).cumprod() - 1
        cum_bench.plot(ax=ax1, color="gray", label="Benchmark", alpha=0.7)

    ax1.axhline(0, color="black", linestyle="-", alpha=0.2)
    ax1.set_title("Cumulative Returns")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend()

    # Drawdown plot
    ax2 = fig.add_subplot(gs[1, 0])
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / (1 + peak)
    drawdown.plot(ax=ax2, color="red", label="Strategy")

    if benchmark_returns is not None:
        bench_peak = cum_bench.cummax()
        bench_dd = (cum_bench - bench_peak) / (1 + bench_peak)
        bench_dd.plot(ax=ax2, color="gray", label="Benchmark", alpha=0.7)

    ax2.set_title("Drawdowns")
    ax2.set_ylabel("Drawdown")
    ax2.legend()

    # Rolling Sharpe ratio
    ax3 = fig.add_subplot(gs[1, 1])
    window = min(
        252, len(returns) // 2
    )  # Use half the data length or 252, whichever is smaller

    rolling_return = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_return - risk_free_rate) / rolling_vol
    rolling_sharpe.plot(ax=ax3, color="green", label="Strategy")

    if benchmark_returns is not None:
        bench_rolling_return = benchmark_returns.rolling(window).mean() * 252
        bench_rolling_vol = benchmark_returns.rolling(window).std() * np.sqrt(252)
        bench_rolling_sharpe = (
            bench_rolling_return - risk_free_rate
        ) / bench_rolling_vol
        bench_rolling_sharpe.plot(ax=ax3, color="gray", label="Benchmark", alpha=0.7)

    ax3.axhline(0, color="black", linestyle="-", alpha=0.2)
    ax3.set_title(f"Rolling Sharpe Ratio ({window} days)")
    ax3.set_ylabel("Sharpe Ratio")
    ax3.legend()

    plt.tight_layout()
    return fig
