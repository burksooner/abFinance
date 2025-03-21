import matplotlib.pyplot as plt
import numpy as np
from nearest_correlation_matrix import nearest_correlation_matrix

# Example 1: Matrix from Higham's paper
print("Example 1: Matrix from Higham's paper")
A1 = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]])
print("Original matrix:")
print(A1)
print("Eigenvalues:", np.linalg.eigvals(A1))
print("Is positive semidefinite:", np.all(np.linalg.eigvals(A1) >= 0))

B1 = nearest_correlation_matrix(A1)
print("\nNearest correlation matrix:")
print(B1)
print("Eigenvalues:", np.linalg.eigvals(B1))
print("Is positive semidefinite:", np.all(np.linalg.eigvals(B1) >= 0))
print("Has unit diagonal:", np.allclose(np.diag(B1), 1))

# Example 2: Random symmetric matrix with values between -1 and 1
print("\nExample 2: Random symmetric matrix")
np.random.seed(42)
n = 5
random_matrix = np.random.uniform(-1, 1, (n, n))
A2 = (random_matrix + random_matrix.T) / 2  # Make it symmetric
np.fill_diagonal(A2, 1)  # Set diagonal to 1
print("Original matrix:")
print(A2)
print("Eigenvalues:", np.linalg.eigvals(A2))
print("Is positive semidefinite:", np.all(np.linalg.eigvals(A2) >= 0))

B2 = nearest_correlation_matrix(A2)
print("\nNearest correlation matrix:")
print(B2)
print("Eigenvalues:", np.linalg.eigvals(B2))
print("Is positive semidefinite:", np.all(np.linalg.eigvals(B2) >= 0))
print("Has unit diagonal:", np.allclose(np.diag(B2), 1))

# Example 3: Financial application - correlation matrix with missing values
print("\nExample 3: Financial correlation matrix with missing data")
# Create a valid correlation matrix
np.random.seed(123)
n_assets = 6
# Generate random factors
factors = np.random.randn(100, 3)
# Generate asset returns based on factors plus noise
returns = np.zeros((100, n_assets))
for i in range(n_assets):
    factor_loadings = np.random.randn(3)
    returns[:, i] = factors.dot(factor_loadings) + 0.1 * np.random.randn(100)

# Calculate correlation matrix
valid_corr = np.corrcoef(returns.T)

# Introduce some missing values (set to 0)
corrupted_corr = valid_corr.copy()
corrupted_corr[0, 2] = corrupted_corr[2, 0] = 0
corrupted_corr[1, 4] = corrupted_corr[4, 1] = 0
corrupted_corr[3, 5] = corrupted_corr[5, 3] = 0

print("Original valid correlation matrix:")
print(valid_corr)
print("\nCorrupted correlation matrix with missing values:")
print(corrupted_corr)
print("Eigenvalues:", np.linalg.eigvals(corrupted_corr))
print("Is positive semidefinite:", np.all(np.linalg.eigvals(corrupted_corr) >= 0))

repaired_corr = nearest_correlation_matrix(corrupted_corr)
print("\nRepaired correlation matrix:")
print(repaired_corr)
print("Eigenvalues:", np.linalg.eigvals(repaired_corr))
print("Is positive semidefinite:", np.all(np.linalg.eigvals(repaired_corr) >= 0))
print("Has unit diagonal:", np.allclose(np.diag(repaired_corr), 1))

# Example 4: Visualizing the difference between matrices
print("\nExample 4: Visualizing the matrix correction")
A4 = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, 1.0]])
print("Original matrix with high correlations:")
print(A4)
print("Eigenvalues:", np.linalg.eigvals(A4))
print("Is positive semidefinite:", np.all(np.linalg.eigvals(A4) >= 0))

B4 = nearest_correlation_matrix(A4)
print("\nNearest correlation matrix:")
print(B4)
print("Eigenvalues:", np.linalg.eigvals(B4))
print("Difference (original - corrected):")
print(A4 - B4)

# Create heatmap visualizations
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
im0 = axs[0].imshow(A4, cmap="coolwarm", vmin=-1, vmax=1)
axs[0].set_title("Original Matrix")
im1 = axs[1].imshow(B4, cmap="coolwarm", vmin=-1, vmax=1)
axs[1].set_title("Nearest Correlation Matrix")
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("correlation_matrix_comparison.png")
print("\nVisualization saved as 'correlation_matrix_comparison.png'")

# Example 5: Performance benchmark for different matrix sizes
print("\nExample 5: Performance benchmark")
sizes = [10, 20, 50, 100]
times = []

import time

for n in sizes:
    # Create a random matrix that might not be PSD
    np.random.seed(42)
    A5 = np.random.uniform(-0.5, 1, (n, n))
    A5 = (A5 + A5.T) / 2  # Symmetrize
    np.fill_diagonal(A5, 1)  # Set diagonal to 1

    start_time = time.time()
    B5 = nearest_correlation_matrix(A5)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)

    print(f"Size {n}x{n}: {elapsed_time:.4f} seconds")
    print(f"  Min eigenvalue before: {min(np.linalg.eigvals(A5)):.6f}")
    print(f"  Min eigenvalue after: {min(np.linalg.eigvals(B5)):.6f}")

# Plot performance results
plt.figure(figsize=(10, 6))
plt.plot(sizes, times, "o-", linewidth=2)
plt.title("Performance of Nearest Correlation Matrix Algorithm")
plt.xlabel("Matrix Size")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.savefig("performance_benchmark.png")
print("\nPerformance benchmark saved as 'performance_benchmark.png'")

if __name__ == "__main__":
    print("\nAll examples completed successfully")
