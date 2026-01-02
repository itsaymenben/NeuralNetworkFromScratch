import numpy as np
from numpy.typing import NDArray

def ReLU(X: NDArray) -> NDArray:
    return np.maximum(X, 0)

def D_ReLU(X: NDArray) -> NDArray:
    return (X > 0) * np.ones(X.shape)

def sigmoid(X: NDArray) -> NDArray:
    return 1 / (1 + np.exp(- X))

def D_sigmoid(X: NDArray) -> NDArray:
    return X / (1 + np.exp(- X))

def identity(X: NDArray) -> NDArray:
    return X

def D_identity(X: NDArray) -> NDArray:
    return np.ones(X.shape)
