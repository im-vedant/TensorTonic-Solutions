def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    n_samples = len(predictions[0])
    result = []
    for i in range(n_samples):
        votes = {}
        for t in range(len(predictions)):
            v = predictions[t][i]
            votes[v] = votes.get(v, 0) + 1
        mx = max(votes.values())
        result.append(min(k for k, v in votes.items() if v == mx))
    return result