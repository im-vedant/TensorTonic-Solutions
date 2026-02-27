import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.

    Args:
        x: Input of shape (batch, seq_len, d_model)
        W1: Weight matrix of shape (d_model, d_ff)
        b1: Bias of shape (d_ff,)
        W2: Weight matrix of shape (d_ff, d_model)
        b2: Bias of shape (d_model,)

    Returns:
        Output of shape (batch, seq_len, d_model)
    """

    # First linear transformation
    hidden = x @ W1 + b1

    # ReLU activation
    hidden = np.maximum(0, hidden)

    # Second linear transformation
    output = hidden @ W2 + b2

    return output