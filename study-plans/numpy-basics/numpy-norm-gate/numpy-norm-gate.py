import numpy as np

def norm_gate(X, W, threshold):
    """Returns: np.ndarray of shape (n, k), gated projection where rows below threshold are zeroed"""
    Z = np.array(X, dtype=np.float64) @ np.array(W, dtype=np.float64)
    norms = np.linalg.norm(Z, axis=1)
    return np.where(norms[:, np.newaxis] >= threshold, Z, 0.0)
