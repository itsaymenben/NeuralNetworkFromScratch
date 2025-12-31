import numpy as np
from numpy.typing import ArrayLike

def MSE(y_true: ArrayLike,
        y_pred: ArrayLike) -> float:
    return np.mean((y_true - y_pred) ** 2) # type: ignore

def RMSE(y_true: ArrayLike,
        y_pred: ArrayLike) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) # type: ignore

def MAE(y_true: ArrayLike,
        y_pred: ArrayLike) -> float:
    return np.mean(np.abs(y_true - y_pred)) # type: ignore
