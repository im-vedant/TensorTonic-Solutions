import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    # Write code here
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    y = np.asarray(y, float)
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("y must be in {0,1}")
    diff = a - b
    if diff.ndim == 1:
        diff = diff[None, :]
    d = np.linalg.norm(diff, axis=1)
    pos = y * (d**2)
    neg = (1.0 - y) * np.maximum(0.0, margin - d)**2
    loss = pos + neg
    return float(loss.mean() if reduction == "mean" else loss.sum())