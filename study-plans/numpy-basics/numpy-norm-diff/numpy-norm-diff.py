import numpy as np

def norm_diff(a, b, lo, hi):
    """Returns: np.ndarray of absolute differences after clipping and rescaling to [0, 1]"""
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    a_clip = np.clip(a_arr, lo, hi)
    b_clip = np.clip(b_arr, lo, hi)
    a_norm = (a_clip - lo) / (hi - lo)
    b_norm = (b_clip - lo) / (hi - lo)
    return np.abs(a_norm - b_norm)
