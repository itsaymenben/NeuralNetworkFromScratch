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
- Approximating the Sine Function: $y = sin(x)$
- XOR Problem

## 1- Neural Network
Before defining a neural network, it is necessary to define a [`Layer`](core/network/layer.py) object. It takes a number of neurons `n_neurons` and an activation function defined in the file [activation.py](core/utilities/activation.py).

```python
layer = Layer(n_neurons=10, activation="ReLU")
```

The neural network is defined by a class [`NeuralNetwork`](core/network/network.py) that takes as input a list of objects `Layer` and the dimension of the input.

```python
network = NeuralNetwork(input_dim=X.shape[1],
                        layers=[Layer(n_neurons=1, activation="identity")])
```

Alternatively, it is possible to define an empty `NeuralNetwork` object and then adding each layer using the `add_layer()` method.

```python
network = NeuralNetwork(input_dim=X.shape[1])
network.add_layer(Layer(n_neurons=1, activation="identity"))
```

Before fitting the model, it is important to call the build method, this method initiates the weights matrix and bias vector for each layer.

*NB: The last layer added to the `NeuralNetwork` object is automatically considered the output layer, and each layer before it a hidden layer.*

## 2- Forward Propagation
The `forward()` method on a [NeuralNetwork](core/network/network.py) object does a forward pass using the current weights and biases and takes an array X as input. For each layer "L", it effectively computes its output $A_L$:

$$Z^L = input^L \times W^L + b^L$$
$$A^L = \sigma^L (Z^L)$$

Where, $\sigma^L$ is the layer's activation function, $input^L \in \mathbb{R}^{1 \times m_{L-1}}$ is the output of the previous layer $A^{L-1}$ or the input of the model $X$ if it is the first layer, $W^L \in \mathbb{R}^{m_{L-1} \times m_{L}}$ is the weights matrix, and $b^L \in \mathbb{R}^{1 \times m_{L}}$ the bias vector.

*NB: In the case of multiple observations, we just modify the 1 in the previous expression by N, the number of observations.*

## 3- Backward Propagation
In order to optimize the weight matrices and bias vectors of the model, it is essential to compute the partial derivatives of a loss function with respect to each of these parameters. The loss function, in this context, is the function that computes the difference between the network output, obtained using a forward pass, and the expected output. The selection of a loss function in the model is done during the building phase, the argument `loss` is used with the `build()` method on the `NeuralNetwork` object. The available loss functions are those present in the file [loss.py](core/utilities/loss.py).

The partial derivatives are computed using the Backward Propagation algorithm, an efficient method that avoids redundant calculations by going through the network backwards (as its name states). Indeed, the chain rule shows that, when traversing a branch of the network, several intermediate terms in the derivative expressions are repeated, meaning they can be stored and reused during the backward pass.

*Example:* <br>
For a neural network with a single two-neurons hidden layer and one output, the output could be written using the expression:

$$Prediction = A^1_{1, 1} \times  W^1_{1, 1} + A^1_{1, 2} \times  W^1_{2, 1} + b^1$$

Where, $A^1 \in \mathbb{R}^{1 \times 2}$ is the output of the hidden layer (or the input of the output layer), $W^1 \in \mathbb{R}^{2 \times 1}$ the weight matrix and $b^1 \in \mathbb{R}$ the bias of the output layer.

The Mean Squared Error (MSE) loss, in the case of a one observation input, is written as:

$$MSE = \frac{1}{2} \times (Observation - Prediction)^2$$

Using these expressions and the chain rule, the gradients of the loss function with respect to $W^1_{2, 1}$ and $b^1$ are:

$$\frac{\partial MSE}{\partial W^1_{2, 1}} = \frac{\partial MSE}{\partial Prediction} \times \frac{\partial Prediction}{\partial W^1_{2, 1}} = - (Observation - Prediction) \times A^1_{1, 2}$$

$$\frac{\partial MSE}{\partial b^1} = \frac{\partial MSE}{\partial Prediction} \times \frac{\partial Prediction}{\partial b^1} = - (Observation - Prediction) \times 1$$

And using the chain rule, the term $\frac{\partial MSE}{\partial Prediction}$, for example, appears for all the other differentials.

For a more general case, the `backward()` method contains the whole gradient computation logic.

```python
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
```



## 4- Optimization
The original reason for computing the partial derivatives of the loss function is to be able to minimize it. Minimizing the loss function is equivalent to closing the gap between the model output and the expected output. In this regard, different optimization algorithms could be used. The one implemented so far in the model is the Gradient Descent algorithm, it is used automatically after calling the `fit()` method on the `NeuralNetwork` object.

The Gradient Descent algorithm consists in moving the parameter in the opposite direction of the gradient of the loss function. This move lets the parameters approach a set that minimizes the loss function, until a stop condition is met.Gradient Descent is a first-order optimization algorithm.

$$W^L = W^L - \eta \times \frac{\partial loss}{\partial W^L}$$
$$b^L = b^L - \eta \times \frac{\partial loss}{\partial b^L}$$

Where, $W^L \in \mathbb{R}^{m_{L-1} \times m_{L}}$ is still the weights matrix, $b^L \in \mathbb{R}^{1 \times m_{L}}$ the bias vector, and $\eta \in \mathbb{R}$ is the learning rate for the optimization.

In the `backward()` method, the [NeuralNetwork](core/network/network.py) updates its weights after one full backward pass using the Gradient Descent algorithm.

```python
for step in range(len(self._layers)):
    self.weights[step] -= learning_rate * gradient_weights[step]
    self.biases[step] -= learning_rate * gradient_biases[step]
```

The `learning_rate` is taken as a parameter by the `fit()` method.

*NB: The only stop condition implemented so far is the number of `epochs`, which is the number of backward passes on the network. It is also a parameter in the `fit()` method.*

## 5- Batching
Batching is a technique where the input data is separated into chunks of fixed size and fed separately to the model. The smaller data size helps the model learn faster and the variety of the batches makes the model more accurate and generalize better, thanks to the noisier gradients that help the optimization algorithm explore the loss surface better.

Selecting a batch size in the model involves adding it as a model (`batch_size`) parameter during the fitting, it defaults to -1 which takes the whole dataset as a chunk.

Depending on the chosen batch size, different optimization regimes are obtained:
- If `batch_size` is equal to the number of observations, the model uses **Gradient Descent**.
- If `batch_size` is between 1 and the number of observations, the model uses **Mini-batch Gradient Descent**.
- If `batch_size = 1`, the model uses **Stochastic Gradient Descent (SGD)**, where the weights are updated after each individual observation.


## 6- Results
As stated at the beginning, the results are applied on four different classic problems. This [notebook](example/testing_network.ipynb) groups all the results.


### a- Simple Linear Regression
The first example is a sanity check for a simple linear regression with a single 1-neuron layer.

$y = a \times x + b$

```python
network = NeuralNetwork(input_dim=X.shape[1])
network.add_layer(Layer(n_neurons=1, activation="identity"))
network.build(loss="MSE")
history = network.fit(X, y, epochs=15, learning_rate=0.1, batch_size=50, verbose=False)
```

The model is able to learn data within 15 epochs. To make the plot more visible, small noise was added on the data.

![Example 1](./assets/plots/example1_predict.png)

### b- Multiple Linear Regression
The second example is very similar to the first, it is a multiple linear regression to test the ability of the model to handle multi-dimensional inputs.

$y = a_1 \times x_1 + a_2 \times x_2 + b$



![Example 2](./assets/plots/example2_training.png)

This time, the model converges within 150 epochs and gives the correct results.

```python
Weights = [[1.9912418 ]
 [2.99404012]]
 
Bias = [[0.99949735]]
```

### c- Approximating the Sine Function
The third problem is the approximation of the sine function. For this case, a neural network with two hidden layers using a sigmoid activation function is used. The reason behind the activation function is that it is a smooth function, and a weighted sum of sigmoid functions can relatively easily approximate any complex nonlinear functions, such as a sine function. A relatively large network is used to clearly illustrate the convergence.

```python
network = NeuralNetwork(input_dim=X.shape[1])
network.add_layer(Layer(n_neurons=256, activation="sigmoid"))
network.add_layer(Layer(n_neurons=256, activation="sigmoid"))
network.add_layer(Layer(n_neurons=1, activation="identity"))
network.build(loss="MSE")
history = network.fit(X, y, epochs=epochs, learning_rate=0.1, batch_size=50, verbose=False)
```

Below is the training progress for the approximation using 10,000 epochs.

![til](./assets/gifs/sinus_approximation.gif)

### d- XOR Problem
The XOR problem is another classic problem solved by neural networks, it is the gate that activates if and only if one of the two inputs is activated, as such.

| Input 1  | Input 2   |  Output |
|---|---|---|
| 0  | 0  | 0  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 0  |

The reason it can be solved by a neural network is because, using two layers, it is possible to create a decision boundary that uses more than one line.
The model, does in fact find the correct decision boundary after around 10,000 epochs.

![Example 4](./assets/plots/example4_xor_boundary.png)

## 7- Limitations and Future Work