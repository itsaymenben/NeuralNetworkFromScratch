import numpy as np
from core.network.layer import Layer
from core.utilities.loss import MSE, D_MSE, RMSE, MAE
# Logging
import logging
# Type Hinting
from numpy.typing import NDArray
from typing import List, Optional

LOSS_FUNCTIONS = {"MSE": MSE,
                  "RMSE": RMSE,
                  "MAE": MAE,
}

DERIVATIVE_LOSS_FUNCTIONS = {"MSE": D_MSE,
}

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s", 
)
logger = logging.getLogger(__name__)

class NeuralNetwork:
    def __init__(self,
                 input_dim: int,
                 layers: Optional[List[Layer]] = None) -> None:
        if layers is None:
            self._layers = []
        else:
            self._layers = layers
        self.input_dim = input_dim
        self._is_built = False

    def add_layer(self,
                  layer: Layer) -> None:
        self._layers.append(layer)

    def build(self,
              loss: str) -> None:
        # We initialize the weights based on the input_dim and layers' sizes
        if len(self._layers) == 0:
            raise ValueError("Cannot build the model without layers. Add at least one layer before calling 'build()'.")
        self.weights = []
        self.biases = []

        prev_dim = self.input_dim
        for layer in self._layers:
            n_neurons = layer.n_neurons
            self.weights.append(np.random.randn(prev_dim, n_neurons))
            self.biases.append(np.zeros((1, n_neurons)))
            prev_dim = n_neurons

        # We set loss the function for the object
        if not isinstance(loss, str):
            raise TypeError(f"'loss' should be of type {str}")
        if not loss.upper() in LOSS_FUNCTIONS.keys():
            raise ValueError(f"{loss} is not available as a loss function, available ones: {LOSS_FUNCTIONS.keys()}")
        self.loss = LOSS_FUNCTIONS[loss.upper()]
        self.d_loss = DERIVATIVE_LOSS_FUNCTIONS[loss.upper()]

        self._is_built = True

    def forward(self,
                X: NDArray) -> List[NDArray]:
        if not self._is_built:
            raise RuntimeError("The model must be built before calling 'forward(input)', please call 'build()' first.")
        outputs = [X]
        inputs = [X]
        for step, layer in enumerate(self._layers):
            inputs.append(np.dot(outputs[-1], self.weights[step]) + self.biases[step])
            outputs.append(layer.activate(inputs[-1]))
        return inputs, outputs # type: ignore

    def backward(self,
                 X: NDArray,
                 y: NDArray,
                 learning_rate: float) -> None:
        N_obs = X.shape[0]
        gradient_weights = [np.zeros(weight.shape) for weight in self.weights]
        gradient_biases = [np.zeros(bias.shape) for bias in self.biases]
        inputs, outputs = self.forward(X)
        derivative_error = self.d_loss(y, outputs[-1]) * self._layers[-1].d_activate(inputs[-1])   # Dimension (input_dim * n_neurons of Output Layer)
        gradient_biases[-1] = derivative_error.sum(axis=0)  # Dimension (1 * n_neurons of Output Layer)
        gradient_weights[-1] = np.dot(derivative_error.T, outputs[-2]).T / N_obs

        for step in range(len(self._layers) - 2, -1, -1):
            derivative_error = np.dot(derivative_error, self.weights[step + 1].T) * self._layers[step].d_activate(inputs[step + 1])
            gradient_biases[step] = derivative_error.sum(axis=0)
            gradient_weights[step] = np.dot(derivative_error.T, outputs[step]).T / N_obs

        for step in range(len(self._layers)):
            self.weights[step] -= learning_rate * gradient_weights[step]
            self.biases[step] -= learning_rate * gradient_biases[step]

    def fit(self,
            X: NDArray,
            y: NDArray,
            epochs: int,
            learning_rate: float,
            batch_size: int = -1) -> List[float]:
        if (batch_size > X.shape[0] or batch_size <= 0) and batch_size != -1:
            raise ValueError("'batch_size' should be a positive integer smaller than or equal to the number of input observations, or equal to -1.")
        history = []
        N_obs = X.shape[0]
        batch_size = batch_size if batch_size != -1 else N_obs
        for epoch in range(1, epochs + 1):
            for start in range(0, N_obs, batch_size):
                end = min(N_obs, start + batch_size)
                X_batch = X[start:end]
                y_batch = y[start:end]
                self.backward(X_batch, y_batch, learning_rate)
            history.append(self.loss(y, self.predict(X)))
            if epoch % 10 == 0:
                score = self.score(X, y)
                logger.info(f"Epoch {epoch} finished with score {score}")
        return history

    def predict(self,
                X: NDArray) -> NDArray:
        return self.forward(X)[1][-1]

    def score(self,
              X: NDArray,
              y: NDArray) -> float:
        if not self._is_built:
            raise RuntimeError("The model must be built before calling 'forward(input)', please call 'build()' first.")
        y_pred = self.predict(X)
        if y_pred.shape != y.shape:
            raise ValueError(f"Shape mismatch between y_pred {y_pred.shape} and y_true {y.shape}.")

        return self.loss(y, y_pred)


