import numpy as np

def winsorize(data, lo_q, hi_q):
    """Returns: np.ndarray of shape (3, m, n), stacked clipped values, lo_mask, hi_mask"""
    a = np.array(data, dtype=np.float64)
    lo = np.percentile(a, lo_q, axis=0)
    hi = np.percentile(a, hi_q, axis=0)
    clipped = np.clip(a, lo, hi)
    lo_mask = (a < lo).astype(np.float64)
    hi_mask = (a > hi).astype(np.float64)
    return np.stack([clipped, lo_mask, hi_mask])
