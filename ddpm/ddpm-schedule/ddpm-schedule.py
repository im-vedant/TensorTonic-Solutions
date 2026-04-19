import numpy as np

def linear_beta_schedule(T, beta_1=0.0001, beta_T=0.02):
    result = np.linspace(beta_1, beta_T, T)
    return [round(float(v), 6) for v in result]

def cosine_alpha_bar_schedule(T, s=0.008):
    steps = np.arange(T + 1) / T
    f = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bars = f[1:] / f[0]
    alpha_bars = np.clip(alpha_bars, 0.0001, 0.9999)
    return [round(float(v), 6) for v in alpha_bars]

def alpha_bar_to_betas(alpha_bars):
    ab = np.array(alpha_bars, dtype=float)
    ab_prev = np.concatenate([[1.0], ab[:-1]])
    betas = 1 - ab / ab_prev
    betas = np.clip(betas, 0.0001, 0.9999)
    return [round(float(v), 6) for v in betas]
