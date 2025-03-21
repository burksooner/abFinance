"""
Basic statistical functions for financial analysis.
"""

import math
from typing import List, Optional, Union

import numpy as np


def mean(data: List[float]) -> float:
    """
    Calculates the arithmetic mean of a list of numbers.

    Parameters
    ----------
    data : list[float]
        List of numerical values

    Returns
    -------
    float
        The arithmetic mean of the input list
        Returns 0 if the input list is empty

    Notes
    -----
    - Handles empty lists by returning 0
    - Uses simple arithmetic mean calculation: sum(data)/len(data)
    """
    if not data:
        return 0
    return sum(data) / len(data)


def variance(data: List[float]) -> float:
    """
    Calculates the population variance of a list of numbers.

    Parameters
    ----------
    data : list[float]
        List of numerical values

    Returns
    -------
    float
        The population variance of the input list
        Returns 0 if the input list is empty

    Notes
    -----
    - Uses population variance formula: Σ(x - μ)²/N
    - Returns 0 for empty lists
    - Uses arithmetic mean from mean function
    """
    if not data:
        return 0
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)


def std(data: List[float]) -> float:
    """
    Calculates the population standard deviation of a list of numbers.

    Parameters
    ----------
    data : list[float]
        List of numerical values

    Returns
    -------
    float
        The population standard deviation of the input list
        Returns 0 if the input list is empty

    Notes
    -----
    - Calculated as square root of population variance
    - Returns 0 for empty lists
    - Uses variance function for calculation
    """
    if not data:
        return 0
    return math.sqrt(variance(data))


def percentile(data: List[float], percentile: float) -> Optional[float]:
    """
    Computes the percentile of a list of numbers using linear interpolation.

    Parameters
    ----------
    data : list[float]
        List of numerical values
    percentile : float
        Percentile to compute (0-100)

    Returns
    -------
    float or None
        The interpolated value at the specified percentile
        Returns None if the input list is empty

    Notes
    -----
    - Uses linear interpolation between closest ranks
    - Handles edge cases (empty lists, exact matches)
    - Sorts data before computation
    """
    if not data:
        return None
    sorted_data = sorted(data)
    n = len(sorted_data)
    pos = (percentile / 100) * (n - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_data[int(pos)]
    lower_value = sorted_data[lower]
    upper_value = sorted_data[upper]
    weight = pos - lower
    return lower_value + weight * (upper_value - lower_value)


def skewness(data: List[float]) -> float:
    """
    Calculates the skewness of a distribution.

    Parameters
    ----------
    data : list[float]
        List of numerical values

    Returns
    -------
    float
        The skewness value
        Returns 0 for empty lists or zero standard deviation

    Notes
    -----
    - Measures asymmetry of distribution
    - Positive skew indicates right tail (extreme gains)
    - Negative skew indicates left tail (extreme losses)
    - Uses population formula
    """
    m = mean(data)
    s = std(data)
    n = len(data)
    if n == 0 or s == 0:
        return 0
    skew_val = sum((r - m) ** 3 for r in data) / n
    return skew_val / (s**3)


def kurtosis(data: List[float]) -> float:
    """
    Calculates the excess kurtosis of a distribution.

    Parameters
    ----------
    data : list[float]
        List of numerical values

    Returns
    -------
    float
        The excess kurtosis value
        Returns 0 for empty lists or zero standard deviation

    Notes
    -----
    - Measures "tailedness" of distribution
    - Excess kurtosis = (kurtosis - 3) for comparison to normal distribution
    - Higher values indicate more extreme outliers
    - Uses population formula
    """
    m = mean(data)
    s = std(data)
    n = len(data)
    if n == 0 or s == 0:
        return 0
    kurt = sum((r - m) ** 4 for r in data) / n
    return kurt / (s**4) - 3


def covariance(x: List[float], y: List[float]) -> float:
    """
    Calculates the population covariance between two lists.

    Parameters
    ----------
    x : list[float]
        First list of values
    y : list[float]
        Second list of values

    Returns
    -------
    float
        The covariance between x and y
        Returns 0 if lists are empty or of different lengths

    Notes
    -----
    - Formula: E[(X - μx)(Y - μy)]
    - Measures linear relationship between variables
    - Uses population formula
    - Requires equal length inputs
    """
    if len(x) != len(y) or not x:
        return 0
    m_x = mean(x)
    m_y = mean(y)
    return sum((xi - m_x) * (yi - m_y) for xi, yi in zip(x, y)) / len(x)
