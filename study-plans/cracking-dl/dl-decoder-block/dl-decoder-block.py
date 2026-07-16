import numpy as np

def decoder_block(x, enc_output, W_q1, W_k1, W_v1, W_o1, W_q2, W_k2, W_v2, W_o2, num_heads,
                 W1, b1, W2, b2, gamma1, beta1, gamma2, beta2, gamma3, beta3, mask=None):
    """
    Returns: Dict with "self_attn", "norm1", "cross_attn", "norm2", "ffn_output", "output".
    """
    def layer_norm(x, gamma, beta, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * gamma + beta

    def softmax(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def mha(Q_in, K_in, V_in, Wq, Wk, Wv, Wo, nh, m=None):
        sq, dm = Q_in.shape; sk = K_in.shape[0]; dh = dm // nh
        Q = (Q_in @ Wq.T).reshape(sq, nh, dh).transpose(1, 0, 2)
        K = (K_in @ Wk.T).reshape(sk, nh, dh).transpose(1, 0, 2)
        V = (K_in @ Wv.T).reshape(sk, nh, dh).transpose(1, 0, 2)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)
        if m is not None:
            scores = np.where(np.array(m)[np.newaxis, :, :], scores, scores - 1e9)
        w = softmax(scores)
        out = (w @ V).transpose(1, 0, 2).reshape(sq, dm)
        return out @ Wo.T

    def to_lol(arr):
        return [[round(float(v), 4) for v in row] for row in arr]

    x = np.array(x, dtype=float); enc = np.array(enc_output, dtype=float)
    W_q1=np.array(W_q1,dtype=float); W_k1=np.array(W_k1,dtype=float)
    W_v1=np.array(W_v1,dtype=float); W_o1=np.array(W_o1,dtype=float)
    W_q2=np.array(W_q2,dtype=float); W_k2=np.array(W_k2,dtype=float)
    W_v2=np.array(W_v2,dtype=float); W_o2=np.array(W_o2,dtype=float)
    W1=np.array(W1,dtype=float); b1=np.array(b1,dtype=float)
    W2=np.array(W2,dtype=float); b2=np.array(b2,dtype=float)
    gamma1=np.array(gamma1,dtype=float); beta1=np.array(beta1,dtype=float)
    gamma2=np.array(gamma2,dtype=float); beta2=np.array(beta2,dtype=float)
    gamma3=np.array(gamma3,dtype=float); beta3=np.array(beta3,dtype=float)

    # Masked self-attention
    sa = mha(x, x, x, W_q1, W_k1, W_v1, W_o1, num_heads, mask)
    n1 = layer_norm(x + sa, gamma1, beta1)

    # Cross-attention (Q from decoder, K/V from encoder)
    ca = mha(n1, enc, enc, W_q2, W_k2, W_v2, W_o2, num_heads)
    n2 = layer_norm(n1 + ca, gamma2, beta2)

    # FFN
    ff = np.maximum(0, n2 @ W1.T + b1) @ W2.T + b2
    out = layer_norm(n2 + ff, gamma3, beta3)

    return {"self_attn": to_lol(sa), "norm1": to_lol(n1), "cross_attn": to_lol(ca),
            "norm2": to_lol(n2), "ffn_output": to_lol(ff), "output": to_lol(out)}
