import numpy as np

def row_extremes(data):
    """Returns: np.ndarray of shape (4, m), rows are max_val, max_col, min_val, min_col"""
    a = np.array(data, dtype=np.float64)
    m = a.shape[0]
    row_idx = np.arange(m)
    max_cols = np.argmax(a, axis=1)
    min_cols = np.argmin(a, axis=1)
    max_vals = a[row_idx, max_cols]
    min_vals = a[row_idx, min_cols]
    return np.stack([max_vals, max_cols.astype(np.float64), min_vals, min_cols.astype(np.float64)])
