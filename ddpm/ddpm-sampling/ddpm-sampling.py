import numpy as np

def ddpm_sample(x_T, betas, epsilon_preds, z_values):
    x_T = np.array(x_T, dtype=float)
    betas = np.array(betas, dtype=float)
    T = len(betas)
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)
    
    x = x_T.copy()
    for i, t in enumerate(range(T, 0, -1)):
        alpha_t = alphas[t - 1]
        alpha_bar_t = alpha_bar[t - 1]
        beta_t = betas[t - 1]
        
        ep = np.array(epsilon_preds[i], dtype=float)
        coef1 = 1 / np.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / np.sqrt(1 - alpha_bar_t)
        mu = coef1 * (x - coef2 * ep)
        
        if t > 1:
            sigma = np.sqrt(beta_t)
            z = np.array(z_values[i], dtype=float)
            x = mu + sigma * z
        else:
            x = mu
    
    def to_list(a):
        if a.ndim == 0: return round(float(a), 4)
        return [to_list(r) for r in a]
    return to_list(x)
