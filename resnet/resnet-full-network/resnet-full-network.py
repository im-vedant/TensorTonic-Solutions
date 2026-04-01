import numpy as np

def relu(x):
    return np.maximum(0, x)

class BasicBlock:
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        self.downsample = downsample
        self.W1 = np.random.randn(in_ch, out_ch) * 0.01
        self.W2 = np.random.randn(out_ch, out_ch) * 0.01
        self.W_proj = np.random.randn(in_ch, out_ch) * 0.01 if in_ch != out_ch or downsample else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        identity = x if self.W_proj is None else x @ self.W_proj
        out = relu(x @ self.W1)
        out = out @ self.W2
        return relu(out + identity)

class ResNet18:
    def __init__(self, num_classes: int = 10):
        self.conv1 = np.random.randn(3, 64) * 0.01

        self.layer1 = [BasicBlock(64, 64) for _ in range(2)]
        self.layer2 = [BasicBlock(64, 128, downsample=True)] + [BasicBlock(128, 128) for _ in range(1)]
        self.layer3 = [BasicBlock(128, 256, downsample=True)] + [BasicBlock(256, 256) for _ in range(1)]
        self.layer4 = [BasicBlock(256, 512, downsample=True)] + [BasicBlock(512, 512) for _ in range(1)]

        self.fc = np.random.randn(512, num_classes) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = relu(x @ self.conv1)
        for block in self.layer1:
            x = block.forward(x)
        for block in self.layer2:
            x = block.forward(x)
        for block in self.layer3:
            x = block.forward(x)
        for block in self.layer4:
            x = block.forward(x)
        return x @ self.fc