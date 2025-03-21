# Higham's Nearest Correlation Matrix Algorithm

This directory contains an implementation of Higham's algorithm for finding the nearest correlation matrix to a given matrix.

## Overview

In finance and statistical analysis, it's common to work with correlation matrices. However, empirical correlation matrices estimated from data often have issues:

1. They may not be positive semidefinite due to noise, missing data, or estimation errors
2. They may have values outside the valid range of [-1, 1]
3. The diagonal elements may not be exactly 1

Higham's algorithm provides a principled way to find the nearest valid correlation matrix to a given approximation, using alternating projections.

## Files

- `nearest_correlation_matrix.py`: Implementation of Higham's algorithm
- `examples.py`: Various examples and use cases of the algorithm

## Usage

```python
import numpy as np
from nearest_correlation_matrix import nearest_correlation_matrix

# Create an invalid correlation matrix
A = np.array([
    [ 2, -1,  0,  0], 
    [-1,  2, -1,  0],
    [ 0, -1,  2, -1], 
    [ 0,  0, -1,  2]
])

# Find the nearest valid correlation matrix
B = nearest_correlation_matrix(A)
```

## Examples

Run the examples file to see various applications:

```
python examples.py
```

Examples include:

1. Basic matrix correction (from Higham's paper)
2. Handling random symmetric matrices
3. Repairing financial correlation matrices with missing data
4. Visualizing the difference between original and corrected matrices
5. Performance benchmarking for different matrix sizes

## Algorithm Details

The implementation uses the alternating projection method described in:

> Higham, N. J. (2002). "Computing the nearest correlation matrix—A problem from finance." 
> IMA Journal of Numerical Analysis, 22(3), 329-343.

The algorithm works by alternating between two projections:
1. Projecting onto the space of positive semidefinite matrices
2. Enforcing unit diagonal constraints

## Applications

- Portfolio optimization
- Risk management
- Monte Carlo simulations
- Stress testing
- Multi-asset option pricing
- Factor models

## Requirements

- NumPy
- Matplotlib (for visualizations in examples.py)