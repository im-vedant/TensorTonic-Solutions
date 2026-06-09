import numpy as np

def reshape_array(data, operation):
    """
    Returns: ndarray of float64 with shape determined by the operation
    """
    arr = np.array(data, dtype=np.float64)
    if operation == 'flatten':
        return arr.flatten()
    if operation == 'transpose':
        return arr.T.copy()
    return np.expand_dims(arr, axis=0)
