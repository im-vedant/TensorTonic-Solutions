import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
    Wz, Uz, bz = params["Wz"], params["Uz"], params["bz"]
    Wr, Ur, br = params["Wr"], params["Ur"], params["br"]
    Wh, Uh, bh = params["Wh"], params["Uh"], params["bh"]
    x = np.asarray(x, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    D = Wz.shape[0]; H = Wz.shape[1]
    x2d, x_1d = _as2d(x, D)
    h2d, h_1d = _as2d(h_prev, H)
    z = _sigmoid(x2d @ Wz + h2d @ Uz + bz)
    r = _sigmoid(x2d @ Wr + h2d @ Ur + br)
    h_tilde = np.tanh(x2d @ Wh + (r * h2d) @ Uh + bh)
    h = (1.0 - z) * h2d + z * h_tilde
    if h_1d:
        return h.reshape(H)
    return h