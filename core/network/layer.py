from core.utilities.activation import ReLU, D_ReLU, sigmoid, D_sigmoid, tanh, D_tanh, identity, D_identity

# Type Hinting
from numpy.typing import NDArray

ACTIVATION_FUNCTIONS = {"RELU": ReLU,
                        "SIGMOID": sigmoid,
                        "TANH": tanh,
                        "IDENTITY": identity}

DERIVATIVE_ACTIVATION_FUNCTIONS = {"RELU": D_ReLU,
                        "SIGMOID": D_sigmoid,
                        "TANH": D_tanh,
                        "IDENTITY": D_identity}

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
        self.d_activation = DERIVATIVE_ACTIVATION_FUNCTIONS[activation.upper()]
        self.n_neurons = n_neurons

    def activate(self,
                input: NDArray):
        return self.activation(input)

    def d_activate(self,
                input: NDArray):
        return self.d_activation(input)
