import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)

    classes = sorted(set(y.tolist()))
    train_indices = []
    test_indices = []

    for c in classes:
        c_indices = np.where(y == c)[0].copy()
        n_c = len(c_indices)

        if rng is not None:
            c_indices = rng.permutation(c_indices)

        n_test_c = int(round(n_c * test_size))
        if n_test_c >= n_c and n_c > 1:
            n_test_c = n_c - 1

        test_indices.extend(c_indices[:n_test_c].tolist())
        train_indices.extend(c_indices[n_test_c:].tolist())

    train_indices = np.array(sorted(train_indices), dtype=int)
    test_indices = np.array(sorted(test_indices), dtype=int)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]