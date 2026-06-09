import numpy as np

def sort_with_indices(data, axis):
    """Returns: np.ndarray of shape (2, m, n), stacked sorted values and sort indices"""
    a = np.array(data, dtype=np.float64)
    sorted_vals = np.sort(a, axis=axis)
    sort_idx = np.argsort(a, axis=axis).astype(np.float64)
    return np.stack([sorted_vals, sort_idx])
