import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor,
                                 K: torch.Tensor,
                                 V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """

    # d_k = key dimension
    d_k = K.size(-1)

    # Compute attention scores: (batch, seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply softmax over key dimension
    attn_weights = F.softmax(scores, dim=-1)

    # Weighted sum of values: (batch, seq_len_q, d_v)
    output = torch.matmul(attn_weights, V)

    return output