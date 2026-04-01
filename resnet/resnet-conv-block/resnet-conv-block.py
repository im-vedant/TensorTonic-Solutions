import numpy as np

def relu(x):
    return np.maximum(0, x)

class ConvBlock:
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.W1 = np.random.randn(in_channels, out_channels) * 0.01
        self.W2 = np.random.randn(out_channels, out_channels) * 0.01
        self.Ws = np.random.randn(in_channels, out_channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        shortcut = x @ self.Ws
        out = relu(x @ self.W1)
        out = relu(out @ self.W2)
        return out + shortcut