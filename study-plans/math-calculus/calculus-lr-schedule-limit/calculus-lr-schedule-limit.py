import numpy as np

def lr_schedule_analysis(alpha_0, k):
    """
    Returns: dict with 'limit' (float), 'sum_diverges' (bool), 'sum_sq_converges' (bool)
    """
    # Limit: as t -> inf, alpha_0 / (1 + k*t) -> 0 if k > 0, else alpha_0
    if k > 0:
        limit = 0.0
    else:
        limit = float(alpha_0)

    # Sum diverges? For alpha_0 > 0:
    #   k > 0: alpha(t) ~ alpha_0/(kt), sum behaves like harmonic series -> diverges
    #   k = 0: alpha(t) = alpha_0 (constant), sum -> infinity
    # For alpha_0 = 0: all terms are 0, sum = 0 (finite, does not diverge)
    sum_diverges = alpha_0 > 0

    # Sum of squares converges? For alpha_0 > 0:
    #   k > 0: alpha(t)^2 ~ alpha_0^2/(kt)^2, sum behaves like p-series p=2 -> converges
    #   k = 0: alpha(t)^2 = alpha_0^2 (constant), sum -> infinity (does not converge)
    # For alpha_0 = 0: sum = 0 (converges trivially)
    sum_sq_converges = (k > 0) or (alpha_0 == 0)

    return {
        'limit': limit,
        'sum_diverges': sum_diverges,
        'sum_sq_converges': sum_sq_converges,
    }
