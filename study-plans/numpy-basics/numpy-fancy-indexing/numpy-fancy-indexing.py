import numpy as np

def select_by_index(arr, indices, axis):
    """
    Returns: 2D ndarray of float64
    """
    a = np.array(arr, dtype=np.float64)
    idx = np.array(indices)
    if axis == 0:
        return a[idx]
    return a[:, idx]
