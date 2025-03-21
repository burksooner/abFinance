# abFinance

A comprehensive Python library for financial analysis, quantitative trading, and portfolio management. This library provides tools for calculating various financial metrics, analyzing market data, and implementing trading strategies.

## Features

- Performance Metrics: Sharpe ratio, Sortino ratio, Omega ratio, and more
- Risk Analysis: Value at Risk (VaR), Expected Shortfall, Beta calculation
- Portfolio Management: Portfolio optimization, risk allocation
- Technical Analysis: Moving averages, momentum indicators, volatility measures
- Data Integration: Support for various data sources including CSV files and financial APIs

## Project Structure

```
abFinance/
├── data/               # Data storage
│   ├── raw/           # Original, immutable data
│   └── processed/     # Cleaned and processed data
├── src/               # Source code
│   ├── core/          # Core functionality and data models
│   ├── analysis/      # Financial analysis modules
│   ├── strategies/    # Trading and investment strategies
│   ├── utils/         # Utility functions and helpers
│   └── visualization/ # Data visualization tools
├── notebooks/         # Jupyter notebooks for exploration and examples
├── config/           # Configuration files
├── tests/            # Unit and integration tests
└── docs/             # Documentation
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/abFinance.git
   cd abFinance
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

```python
from abFinance.analysis import performance, risk
import pandas as pd

# Calculate Sharpe Ratio
returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
sharpe = performance.calculate_sharpe_ratio(returns)

# Calculate Value at Risk
var = risk.calculate_var(returns, confidence_level=0.95)
```

## Documentation

Detailed documentation for each module and function can be found in the `docs/` directory.

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Future Development

- Integration with additional financial data providers
- Web dashboard for real-time analytics
- Machine learning models for market prediction
- Automated trading system integration
- Enhanced documentation and examples

## Contact

For questions and feedback, please [open an issue](https://github.com/yourusername/abFinance/issues) on GitHub. 