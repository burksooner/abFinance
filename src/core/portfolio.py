"""
Portfolio management functionality.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd

from .asset import Asset


@dataclass
class Position:
    """A position in a portfolio, representing ownership of an asset."""

    asset: Asset
    quantity: float
    purchase_price: float
    purchase_date: datetime

    @property
    def cost_basis(self) -> float:
        """Calculate the total cost basis of the position."""
        return self.quantity * self.purchase_price


@dataclass
class Portfolio:
    """A collection of positions representing a portfolio."""

    name: str
    positions: Dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    currency: str = "USD"

    def add_position(self, position: Position) -> None:
        """Add a position to the portfolio."""
        self.positions[position.asset.symbol] = position

    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove a position from the portfolio."""
        return self.positions.pop(symbol, None)

    def total_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate the total value of the portfolio.

        Args:
            prices: Dictionary mapping symbols to current prices

        Returns:
            Total portfolio value including cash
        """
        position_value = sum(
            pos.quantity * prices.get(pos.asset.symbol, 0)
            for pos in self.positions.values()
        )
        return position_value + self.cash

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the portfolio to a pandas DataFrame."""
        data = []
        for symbol, position in self.positions.items():
            data.append(
                {
                    "Symbol": symbol,
                    "Name": position.asset.name,
                    "Type": position.asset.asset_type,
                    "Quantity": position.quantity,
                    "Purchase Price": position.purchase_price,
                    "Cost Basis": position.cost_basis,
                    "Purchase Date": position.purchase_date,
                }
            )
        return pd.DataFrame(data)
