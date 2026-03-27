import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Queries tensor of shape (batch, seq_len_q, d_k)
        K: Keys tensor of shape (batch, seq_len_k, d_k)
        V: Values tensor of shape (batch, seq_len_k, d_v)
    
    Returns:
        Tensor of shape (batch, seq_len_q, d_v)
    """
    # Step 1: Compute attention scores (QK^T)
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len_q, seq_len_k)
    
    # Step 2: Scale by sqrt(d_k)
    d_k = Q.size(-1)
    scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len_q, seq_len_k)
    
    # Step 4: Multiply with V to get output
    output = torch.matmul(attention_weights, V)  # (batch, seq_len_q, d_v)
    
    return output