import numpy as np
from core.network.layer import Layer
from core.utilities.loss import MSE, RMSE, MAE
# Type Hinting
from numpy.typing import ArrayLike
from typing import List, Optional

LOSS_FUNCTIONS = {"MSE": MSE,
                  "RMSE": RMSE,
                  "MAE": MAE,
}

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

        self.__is_built = True

    def forward(self,
                input: ArrayLike) -> ArrayLike:
        if not self.__is_built:
            raise RuntimeError("The model must be built before calling 'forward(input)', please call 'build()' first.")
        output = input
        for step, layer in enumerate(self._layers):
            output = layer.activate(np.dot(output, self.weights[step]) + self.biases[step])
        return output

    def score(self,
              y_true: ArrayLike,
              input: ArrayLike) -> float:
        if not self.__is_built:
            raise RuntimeError("The model must be built before calling 'forward(input)', please call 'build()' first.")
        y_pred = self.forward(input)
        return self.loss(y_true, y_pred)
