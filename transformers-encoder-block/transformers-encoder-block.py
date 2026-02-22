import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ---------------------------
# 1. Layer Normalization
# ---------------------------
def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization over last dimension.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = (x - mean) / np.sqrt(var + eps)

    return gamma * x_norm + beta


# ---------------------------
# 2. Multi-Head Attention
# ---------------------------
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention implementation.
    """

    batch_size, seq_len, d_model = Q.shape
    head_dim = d_model // num_heads

    # Linear projections
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    # Split heads
    def split_heads(x):
        x = x.reshape(batch_size, seq_len, num_heads, head_dim)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, head_dim)

    Q_heads = split_heads(Q_proj)
    K_heads = split_heads(K_proj)
    V_heads = split_heads(V_proj)

    # Scaled dot-product attention
    scores = Q_heads @ K_heads.transpose(0,1,3,2) / np.sqrt(head_dim)
    attn_weights = softmax(scores, axis=-1)
    attn_output = attn_weights @ V_heads  # (batch, heads, seq, head_dim)

    # Concatenate heads
    attn_output = attn_output.transpose(0,2,1,3).reshape(batch_size, seq_len, d_model)

    # Final linear projection
    output = attn_output @ W_o

    return output


# ---------------------------
# 3. Feed Forward Network
# ---------------------------
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    Uses ReLU activation.
    """
    hidden = np.maximum(0, x @ W1 + b1)  # ReLU
    output = hidden @ W2 + b2
    return output


# ---------------------------
# 4. Encoder Block
# ---------------------------
def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block:
    MHA + Residual + LayerNorm
    FFN + Residual + LayerNorm
    """

    # ---- Multi-Head Attention ----
    mha_output = multi_head_attention(x, x, x,
                                      W_q, W_k, W_v,
                                      W_o, num_heads)

    x_res1 = x + mha_output
    x_norm1 = layer_norm(x_res1, gamma1, beta1)

    # ---- Feed Forward ----
    ffn_output = feed_forward(x_norm1, W1, b1, W2, b2)

    x_res2 = x_norm1 + ffn_output
    output = layer_norm(x_res2, gamma2, beta2)

    return output