import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.array(x)   # convert list to numpy array
    return 1 / (1 + np.exp(-x))