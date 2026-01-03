import numpy as np
from numpy.typing import NDArray

def ReLU(X: NDArray) -> NDArray:
    return np.maximum(X, 0)

def D_ReLU(X: NDArray) -> NDArray:
    return (X > 0) * np.ones(X.shape)

def sigmoid(X: NDArray) -> NDArray:
    return 1 / (1 + np.exp(- X))

def D_sigmoid(X: NDArray) -> NDArray:
    sigmoid_X = sigmoid(X)
    return sigmoid_X * (1 - sigmoid_X)

def tanh(X: NDArray) -> NDArray:
    return (1 - np.exp(- 2 * X)) / (1 + np.exp(- 2 * X))

def D_tanh(X: NDArray) -> NDArray:
    return 1 - tanh(X) ** 2

def identity(X: NDArray) -> NDArray:
    return X

def D_identity(X: NDArray) -> NDArray:
    return np.ones(X.shape)
