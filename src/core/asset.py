"""
Base classes for financial assets.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union


@dataclass
class Asset:
    """Base class for all financial assets."""

    symbol: str
    name: Optional[str] = None
    asset_type: Optional[str] = None
    currency: str = "USD"

    def __str__(self) -> str:
        return (
            f"{self.symbol}: {self.name or 'Unnamed'} ({self.asset_type or 'Unknown'})"
        )


@dataclass
class Stock(Asset):
    """Class representing a stock."""

    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None

    def __post_init__(self):
        self.asset_type = "Stock"


@dataclass
class Bond(Asset):
    """Class representing a bond."""

    maturity_date: Optional[datetime] = None
    coupon_rate: Optional[float] = None
    face_value: Optional[float] = None

    def __post_init__(self):
        self.asset_type = "Bond"


@dataclass
class ETF(Asset):
    """Class representing an ETF."""

    holdings: Optional[Dict[str, float]] = None
    expense_ratio: Optional[float] = None

    def __post_init__(self):
        self.asset_type = "ETF"
