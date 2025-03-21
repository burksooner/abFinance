"""
Data loading utilities for financial data.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import pandas_datareader as pdr
import yfinance as yf


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load financial data from a CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame containing the data
    """
    return pd.read_csv(
        file_path,
        parse_dates=(
            ["Date"] if "Date" in pd.read_csv(file_path, nrows=0).columns else None
        ),
        index_col="Date" if "Date" in pd.read_csv(file_path, nrows=0).columns else None,
    )


def download_stock_data(
    symbols: Union[str, List[str]],
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    interval: str = "1d",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download stock price data from Yahoo Finance.

    Args:
        symbols: Stock symbol or list of symbols
        start_date: Start date for data (default: 1 year ago)
        end_date: End date for data (default: today)
        interval: Data interval (1d, 1wk, 1mo, etc.)
        save_path: Optional path to save the data as CSV

    Returns:
        DataFrame containing the stock data
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    data = yf.download(symbols, start=start_date, end=end_date, interval=interval)

    if save_path is not None:
        data.to_csv(save_path)

    return data


def get_fred_data(
    series_ids: Union[str, List[str]],
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download economic data from FRED (Federal Reserve Economic Data).

    Args:
        series_ids: FRED series ID or list of IDs
        start_date: Start date for data (default: 5 years ago)
        end_date: End date for data (default: today)
        save_path: Optional path to save the data as CSV

    Returns:
        DataFrame containing the economic data
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    data = pdr.get_data_fred(series_ids, start=start_date, end=end_date)

    if save_path is not None:
        data.to_csv(save_path)

    return data
