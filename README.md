# NeuralNetworkFromScratch
## Presentation
This repository implements a simple Feed Forward Neural Network from scratch using only NumPy.

### Goal
The goal of this project is to understand deep learning components from a mathematical perspective:

- Forward Propagation
- Backward Propagation
- Optimization
- Batching

### Results
The results of the implementation are showcased using four classic problems:
- Simple Linear Regression: $y = a \times x + b$
- Multiple Linear Regression: $y = a_1 \times x_1 + a_2 \times x_2 + b$
- Approximating Sinus Function: $y = sin(x)$
- XOR Problem

## 1- Neural Network
Before defining a neural network, it is necessary to define a [Layer](core/network/layer.py) object. It takes a number of neurons 'n_neurons' and an activation function defined in the file [activation.py](core/utilities/activation.py).
```
layer = Layer(n_neurons=10, activation="ReLU")
```
The neural network is defined by a class [NeuralNetwork](core/network/network.py) that takes as input a list of objects "Layer" and the dimension of the input.
```
network = NeuralNetwork(input_dim=X.shape[1],
                        layers=[Layer(n_neurons=1, activation="identity")])
```

Alternatively, it is possible to define an empty NeuralNetwork object and then adding each layer using the add_layer() method.

```
network = NeuralNetwork(input_dim=X.shape[1])
network.add_layer(Layer(n_neurons=1, activation="identity"))
```
Before fitting the model, it is important to call the build method, this method initiates the weights matrix and bias vector for each layer.

*NB: The last layer added to the NeuralNetwork object is automatically considered the output layer, and each layer before it a hidden layer.*

## 2- Forward Propagation
The forward() method on a [NeuralNetwork](core/network/network.py) object does a forward pass using the current weights and biases and takes an array X as input. For each layer "L", it effectively computes its output $A_L$:

$$Z_L = input_L \times W_L + b_L$$
$$A_L = \sigma_L (Z_L)$$

Where, $\sigma_L$ is the layer's activation function, $input_L \in \mathbb{R}^{1 \times m_{L-1}}$ is the output of the previous layer $A_{L-1}$ or the input of the model $X$ if it is the first layer, $W_L \in \mathbb{R}^{m_{L-1} \times m_{L}}$ is the weights matrix, and $b_L \in \mathbb{R}^{1 \times m_{L}}$ the bias vector.

*NB: In the case of multiple observations, we just modify the 1 in the previous expression by N, the number of observations.*

## 3- Backward Propagation

## 4- Optimization

## 5- Batching

## 6- Results

### a- Simple Linear Regression

$y = a \times x + b$

### b- Multiple Linear Regression

$y = a_1 \times x_1 + a_2 \times x_2 + b$

### c- Approximating the Sine Function

![til](./assets/gifs/sinus_approximation.gif)

### d- XOR Problem

| Input 1  | Input 2   |  Output |
|---|---|---|
| 0  | 0  | 0  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 0  |

