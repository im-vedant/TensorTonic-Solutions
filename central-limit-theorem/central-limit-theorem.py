import numpy as np
from scipy import stats

def clt_probability(pop_mean, pop_std, sample_size, threshold):
    """
    Apply CLT to compute sampling distribution properties and probabilities.

    Args:
        pop_mean (float): Population mean
        pop_std (float): Population standard deviation
        sample_size (int): Sample size n
        threshold (float): Value to compare sample mean against

    Returns:
        dict: sampling_mean, standard_error, z_score, prob_greater, prob_less
    """
    # Your code here
    sampling_mean = pop_mean
    standard_error = pop_std / np.sqrt(sample_size)
    z_score = (threshold - pop_mean) / standard_error
    prob_less = stats.norm.cdf(z_score)
    prob_greater = 1 - prob_less
    return {
        "sampling_mean": float(sampling_mean),
        "standard_error": float(standard_error),
        "z_score": float(z_score),
        "prob_greater": float(prob_greater),
        "prob_less": float(prob_less)
    }
