from core.utilities.activation import ReLU, identity

# Type Hinting
from numpy.typing import ArrayLike

ACTIVATION_FUNCTIONS = {"RELU": ReLU,
                        "IDENTITY": identity}

class Layer:
    def __init__(self,
                 n_neurons: int,
                 activation: str = "identity"):

        # Tests for activation function
        if not isinstance(activation, str):
            raise TypeError(f"'activation' should be of type {str}")
        if not activation.upper() in ACTIVATION_FUNCTIONS.keys():
            raise ValueError(f"{activation} is not available as an activation function, available ones: {ACTIVATION_FUNCTIONS.keys()}")
        self.activation = ACTIVATION_FUNCTIONS[activation.upper()]
        self.n_neurons = n_neurons

    def activate(self,
                input: ArrayLike):
        return self.activation(input)
