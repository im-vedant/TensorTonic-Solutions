import numpy as np

def identity_block(x, W1, W2):
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)
    identity = x.copy()
    out = np.maximum(0, x @ W1.T)
    out = out @ W2.T
    result = np.maximum(0, out + identity)
    return [[round(float(v), 3) for v in row] for row in result]

