import numpy as np


class Layer:

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, partial_derivative_error_output, learning_rate):
        pass


class Dense(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, partial_derivative_error_output, learning_rate):
        weights_gradient = np.dot(partial_derivative_error_output, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * partial_derivative_error_output
        return np.dot(self.weights.T, partial_derivative_error_output)


class Activation(Layer):

    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, partial_derivative_error_output, learning_rate):
        return np.multiply(partial_derivative_error_output, self.activation_derivative(self.input))


class Tanh(Activation):

    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_derivative = lambda x: 1 - (np.tanh(x) ** 2)
        super().__init__(tanh, tanh_derivative)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


# Initializing Training Data + Targets
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# Neural Network Architecture
neural_network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

# Initializing Hyper-Parameters
epochs = 10000
learning_rate = 0.1

for i in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        # Forward Propagation
        output = x
        for layer in neural_network:
            output = layer.forward(output)

        # Error Calculation
        error += mse(y, output)

        # Backward Propagation
        grad = mse_derivative(y, output)
        for layer in reversed(neural_network):
            grad = layer.backward(grad, learning_rate)

    error /= len(X)
    print('%d/%d, error=%f' % (i + 1, epochs, error))

