import numpy as np

def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    x_t = np.array(x_t, dtype=float)
    ep = np.array(epsilon_pred, dtype=float)
    betas = np.array(betas, dtype=float)
    
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)
    alpha_t = alphas[t - 1]
    alpha_bar_t = alpha_bar[t - 1]
    beta_t = betas[t - 1]
    
    coef1 = 1 / np.sqrt(alpha_t)
    coef2 = beta_t / np.sqrt(1 - alpha_bar_t)
    mu = coef1 * (x_t - coef2 * ep)
    
    if t > 1 and z is not None:
        sigma = np.sqrt(beta_t)
        result = mu + sigma * np.array(z, dtype=float)
    else:
        result = mu
    
    def to_list(a):
        if a.ndim == 0: return round(float(a), 4)
        return [to_list(r) for r in a]
    return to_list(result)
