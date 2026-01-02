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
                input: NDArray) -> List[NDArray]:
        if not self._is_built:
            raise RuntimeError("The model must be built before calling 'forward(input)', please call 'build()' first.")
        outputs = [input]
        inputs = [input]
        for step, layer in enumerate(self._layers):
            inputs.append(np.dot(outputs[-1], self.weights[step]) + self.biases[step])
            outputs.append(layer.activate(inputs[-1]))
        return inputs, outputs # type: ignore

    def backward(self,
                 input: NDArray,
                 y_true: NDArray,
                 learning_rate: float):
        gradient_weights = [np.zeros(weight.shape) for weight in self.weights]
        gradient_biases = [np.zeros(bias.shape) for bias in self.biases]
        inputs, outputs = self.forward(input)
        derivative_error = self.d_loss(y_true, outputs[-1]) * self._layers[-1].d_activate(inputs[-1])   # Dimension (input_dim * n_neurons of Output Layer)
        gradient_biases[-1] = derivative_error.sum(axis=0)  # Dimension (1 * n_neurons of Output Layer)
        gradient_weights[-1] = np.dot(derivative_error.T, outputs[-2]).T

        for step in range(len(self._layers) - 2, -1, -1):
            derivative_error = np.dot(derivative_error, self.weights[step + 1].T) * self._layers[step].d_activate(inputs[step + 1])
            gradient_biases[step] = derivative_error.sum(axis=0)
            gradient_weights[step] = np.dot(derivative_error.T, outputs[step]).T

        for step in range(len(self._layers)):
            self.weights[step] -= learning_rate * gradient_weights[step]
            self.biases[step] -= learning_rate * gradient_biases[step]

    def fit(self,
            input: NDArray,
            y_true: NDArray,
            epochs: int,
            learning_rate: float):
        for epoch in range(1, epochs + 1):
            self.backward(input, y_true, learning_rate)
            if epoch % 10 == 0:
                score = self.score(input, y_true)
                logger.info(f"Epoch {epoch} finished with score {score}")

    def score(self,
              input: NDArray,
              y_true: NDArray) -> float:
        if not self._is_built:
            raise RuntimeError("The model must be built before calling 'forward(input)', please call 'build()' first.")
        y_pred = self.forward(input)[1][-1]
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch between y_pred {y_pred.shape} and y_true {y_true.shape}.")

        return self.loss(y_true, y_pred)

    def predict(self,
                input: NDArray) -> NDArray:
        return self.forward(input)[1][-1]
