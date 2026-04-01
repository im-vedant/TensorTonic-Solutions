import numpy as np

class BatchNorm:
    """Batch Normalization layer."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def post_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """Post-activation: Conv -> BN -> ReLU -> Conv -> BN + skip -> ReLU"""
    identity = x
    out = x @ W1
    out = bn1.forward(out, training=True)
    out = relu(out)
    out = out @ W2
    out = bn2.forward(out, training=True)
    out = out + identity  # Skip connection
    out = relu(out)
    return out

def pre_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """Pre-activation: BN -> ReLU -> Conv -> BN -> ReLU -> Conv + skip"""
    identity = x
    out = bn1.forward(x, training=True)
    out = relu(out)
    out = out @ W1
    out = bn2.forward(out, training=True)
    out = relu(out)
    out = out @ W2
    out = out + identity  # Skip connection
    return out
