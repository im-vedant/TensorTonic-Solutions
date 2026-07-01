import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    t = torch.tensor(x, dtype=torch.float32)
    if op == "flatten":
        return t.flatten().tolist()
    elif op == "squeeze":
        return t.squeeze().tolist()
    elif op == "transpose":
        return t.T.tolist()
