import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    n, m = A.shape
    T = np.zeros((m, n), dtype=A.dtype)
    for i in range(n):
        for j in range(m):
            T[j, i] = A[i, j]
    return T