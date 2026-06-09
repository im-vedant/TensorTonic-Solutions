import numpy as np

def outer_sum(a, b):
    """Returns: np.ndarray of shape (m, n), outer sum where out[i,j] = a[i] + b[j]"""
    x = np.array(a, dtype=np.float64)
    y = np.array(b, dtype=np.float64)
    return x[:, np.newaxis] + y
