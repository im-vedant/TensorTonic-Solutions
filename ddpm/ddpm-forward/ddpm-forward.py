import numpy as np

def get_alpha_bar(betas):
    alphas = 1 - np.array(betas, dtype=float)
    return [round(float(v), 6) for v in np.cumprod(alphas)]

def forward_diffusion(x_0, t, betas, epsilon):
    x_0 = np.array(x_0, dtype=float)
    betas = np.array(betas, dtype=float)
    epsilon = np.array(epsilon, dtype=float)
    alpha_bar = np.cumprod(1 - betas)
    abt = alpha_bar[t - 1]
    x_t = np.sqrt(abt) * x_0 + np.sqrt(1 - abt) * epsilon
    
    def to_list(a):
        if a.ndim == 0:
            return round(float(a), 4)
        return [to_list(row) for row in a]
    return to_list(x_t)
