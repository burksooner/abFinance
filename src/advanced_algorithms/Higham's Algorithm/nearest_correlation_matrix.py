# Source: https://www.sitmo.com/finding-the-nearest-valid-correlation-matrix-with-highams-algorithm/

import numpy as np


def symmetrize_matrix(A):
    """Ensures the matrix is symmetric by averaging it with its transpose."""
    return (A + A.T) / 2


def project_to_positive_semidefinite(A):
    """Projects the matrix onto the space of positive semidefinite matrices by setting negative eigenvalues to zero."""
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    A_psd = (eigenvectors * np.maximum(eigenvalues, 0)).dot(eigenvectors.T)
    return symmetrize_matrix(A_psd)


def nearest_correlation_matrix(A, tol=1e-8, max_iterations=100):
    """
    Computes the nearest valid correlation matrix to a given symmetric matrix.

    This method is known as Higham's Nearest Correlation Matrix Algorithm. It was introduced by
    Nicholas J. Higham in the paper:
        Higham, N. J. (2002).
        "Computing the nearest correlation matrix—A problem from finance."
        IMA Journal of Numerical Analysis, 22(3), 329-343.

    The algorithm is an iterative projection method that alternates between projecting a given
    matrix onto the space of symmetric positive semidefinite matrices and enforcing the unit
    diagonal constraint. The approach is widely used in finance and risk management when working
    with empirical correlation matrices that may not be numerically valid due to noise
    or rounding errors.

    Parameters:
        A (ndarray): Input symmetric matrix.
        tol (float): Convergence tolerance.
        max_iterations (int): Maximum number of iterations.

    Returns:
        ndarray: The nearest valid correlation matrix.
    """
    X = symmetrize_matrix(A)  # Ensure input is symmetric
    correction_matrix = np.zeros_like(
        X
    )  # Stores corrections applied to enforce constraints

    for iteration in range(max_iterations):
        X_old = X.copy()
        residual = X - correction_matrix  # Step towards a valid correlation matrix
        X = project_to_positive_semidefinite(
            residual
        )  # Project onto positive semidefinite matrices
        correction_matrix = X - residual  # Update correction matrix
        np.fill_diagonal(X, 1)  # Ensure diagonal elements are exactly 1

        # Check for convergence
        if np.linalg.norm(X - X_old, "fro") / np.linalg.norm(X, "fro") < tol:
            break

    return X


# ------------------------------------------------------------------------
# Example from Higham's paper, section 4 "Numerical Experiments"
# ------------------------------------------------------------------------

# A is an invalid non-positive definite corr matrix
A = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]])

# fix it!
B = nearest_correlation_matrix(A)
print(B)
