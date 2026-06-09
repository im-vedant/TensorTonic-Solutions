import numpy as np

def normalize(data):
    """Returns: np.ndarray of shape (m, n), z-score normalized per column"""
    a = np.array(data, dtype=np.float64)
    mu = np.mean(a, axis=0)
    sigma = np.std(a, axis=0)
    return (a - mu) / sigma
