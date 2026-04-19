import numpy as np

def compute_ddpm_loss(x_0, betas, t_values, epsilon, epsilon_pred):
    epsilon = np.array(epsilon, dtype=float)
    epsilon_pred = np.array(epsilon_pred, dtype=float)
    loss = np.mean((epsilon - epsilon_pred) ** 2)
    return round(float(loss), 6)
