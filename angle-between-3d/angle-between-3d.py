import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    norm_v = np.sqrt(np.sum(v**2))
    norm_w = np.sqrt(np.sum(w**2))

    cos_theta = np.dot(v, w) / (norm_v * norm_w)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(theta)