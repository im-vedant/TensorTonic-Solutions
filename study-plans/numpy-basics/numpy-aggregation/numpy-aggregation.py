import numpy as np

def summarize(data, axis):
    """Returns: np.ndarray of shape (4, k), rows are mean, std, min, max"""
    a = np.array(data, dtype=np.float64)
    return np.stack([
        np.mean(a, axis=axis),
        np.std(a, axis=axis),
        np.min(a, axis=axis),
        np.max(a, axis=axis),
    ])
