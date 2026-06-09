import numpy as np

def quantize_and_frame(data, decimals, pad_width):
    """Returns: np.ndarray of shape (3, m+2p, n+2p), stacked rounded, floored, ceiled with zero-padding"""
    a = np.array(data, dtype=np.float64)
    pad = lambda x: np.pad(x, pad_width, mode='constant', constant_values=0.0)
    rounded = pad(np.round(a, decimals))
    floored = pad(np.floor(a))
    ceiled  = pad(np.ceil(a))
    return np.stack([rounded, floored, ceiled])
