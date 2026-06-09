import numpy as np

def tile_diff(data, reps):
    """Returns: np.ndarray of shape (2, m*reps, n), stacked tiled array and padded differences"""
    a = np.array(data, dtype=np.float64)
    tiled = np.tile(a, (reps, 1))
    diff = np.diff(tiled, axis=0)
    diff_padded = np.pad(diff, ((0, 1), (0, 0)))
    return np.stack([tiled, diff_padded])
