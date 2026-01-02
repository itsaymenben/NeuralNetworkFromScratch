import numpy as np
from numpy.typing import NDArray

def MSE(y_true: NDArray,
        y_pred: NDArray) -> float:
    return 1 / 2 * np.mean((y_true - y_pred) ** 2) # type: ignore

def D_MSE(y_true: NDArray,
        y_pred: NDArray) -> NDArray:
    return - (y_true - y_pred)

def RMSE(y_true: NDArray,
        y_pred: NDArray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def MAE(y_true: NDArray,
        y_pred: NDArray) -> float:
    return np.mean(np.abs(y_true - y_pred)) # type: ignore
