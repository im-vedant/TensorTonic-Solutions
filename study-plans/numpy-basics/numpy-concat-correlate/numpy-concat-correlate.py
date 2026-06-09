import numpy as np

def compare_correlations(a, b):
    """Returns: np.ndarray of shape (3, n, n), stacked correlation matrices"""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    combined = np.concatenate([a, b], axis=0)
    corr_a = np.corrcoef(a.T)
    corr_b = np.corrcoef(b.T)
    corr_both = np.corrcoef(combined.T)
    return np.stack([corr_a, corr_b, corr_both])