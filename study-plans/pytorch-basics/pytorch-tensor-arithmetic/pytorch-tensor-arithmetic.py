import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    t_x = torch.tensor(x, dtype=torch.float32)
    t_y = torch.tensor(y, dtype=torch.float32)

    res = None

    if op == "add":
        return torch.add(t_x, t_y).tolist()
    elif op == "multiply":
        return torch.mul(t_x, t_y).tolist()
    elif op == "matmul":
        return torch.matmul(t_x, t_y).tolist()
    elif op == "power":
        return torch.pow(t_x, t_y).tolist()
    elif op == "max":
        return torch.max(torch.stack([t_x, t_y], dim=0), dim=0).values.tolist()