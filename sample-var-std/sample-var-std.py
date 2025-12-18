import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation. lfgkljkljkl random shit
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    n = len(x)
    
    if n < 2:
        raise ValueError("Need at least 2 samples for unbiased variance")
    
    # Sample variance with Bessel correction: s^2 = 1/(n-1) * sum((x_i - x_bar)^2)
    mean_x = np.mean(x)
    var = np.sum((x - mean_x) ** 2) / (n - 1)
    std = np.sqrt(var)
    
    return float(var), float(std)
