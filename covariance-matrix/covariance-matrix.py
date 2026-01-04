import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    """Reference implementation for covariance matrix calculation"""
    X = np.asarray(X, dtype=float)
    
    # Check input dimensions
    if X.ndim != 2:
        return None
    
    N, D = X.shape
    
    # Need at least 2 samples for covariance
    if N < 2:
        return None
    
    # Center the data by subtracting mean of each feature
    mu = np.mean(X, axis=0)  # shape (D,)
    X_centered = X - mu      # shape (N, D)
    
    # Compute covariance matrix: (1/(N-1)) * X_centered.T @ X_centered
    cov_matrix = (1.0 / (N - 1)) * np.dot(X_centered.T, X_centered)
    
    return cov_matrix
