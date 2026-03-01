import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    if x.shape != p.shape:
        return None
    if x.ndim != 1:
        return None
    if not np.allclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("probabilities must sum to 1")
    ev = np.sum(x * p)
    return float(ev)
