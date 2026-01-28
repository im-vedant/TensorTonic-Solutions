import numpy as np

def get_alpha_bar(betas: np.ndarray) -> np.ndarray:
    alphas = 1 - betas
    return np.cumprod(alphas)

def forward_diffusion(
    x_0: np.ndarray,
    t: int,
    betas: np.ndarray
) -> tuple:
    alpha_bar = get_alpha_bar(betas)
    alpha_bar_t = alpha_bar[t - 1]  # t is 1-indexed

    epsilon = np.random.randn(*x_0.shape)

    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * epsilon

    return x_t, epsilon
