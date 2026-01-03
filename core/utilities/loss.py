import numpy as np
from numpy.typing import NDArray

def MSE(y_true: NDArray,
        y_pred: NDArray) -> float:
    return 1 / 2 * np.mean((y_true - y_pred) ** 2) # type: ignore

def D_MSE(y_true: NDArray,
        y_pred: NDArray) -> NDArray:
    return - (y_true - y_pred) / y_true.shape[0]

def RMSE(y_true: NDArray,
        y_pred: NDArray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def MAE(y_true: NDArray,
        y_pred: NDArray) -> float:
    return np.mean(np.abs(y_true - y_pred)) # type: ignore

def binary_crossentropy(y_true: NDArray,
                        y_pred: NDArray) -> float:
    return - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) # type: ignore

def D_binary_crossentropy(y_true: NDArray,
                          y_pred: NDArray) -> NDArray:
    return - (y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]
