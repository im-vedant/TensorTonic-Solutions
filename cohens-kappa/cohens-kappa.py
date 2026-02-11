import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    r1 = np.asarray(rater1)
    r2 = np.asarray(rater2)
    n = len(r1)
    
    # Observed agreement
    p_o = np.sum(r1 == r2) / n
    
    # Expected agreement
    labels = np.unique(np.concatenate([r1, r2]))
    p_e = sum((np.sum(r1 == k) / n) * (np.sum(r2 == k) / n) for k in labels)
    
    # Perfect agreement edge case
    if p_e == 1.0:
        return 1.0
    
    return float((p_o - p_e) / (1 - p_e))