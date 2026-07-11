def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    for _ in range(steps):
        logits = X @ w + b
        preds = _sigmoid(logits)
        grad_w = X.T @ (preds - y) / N
        grad_b = np.mean(preds - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b
