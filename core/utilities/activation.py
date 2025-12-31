import numpy as np
from numpy.typing import ArrayLike

def ReLU(X: ArrayLike) -> ArrayLike:
    return np.maximum(X, 0)

def identity(X: ArrayLike) -> ArrayLike:
    return X
