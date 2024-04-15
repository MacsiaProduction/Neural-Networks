import numpy as np


class Neuron:
    def __init__(self, input_size):
        # He initialization
        self.weights = np.random.randn(input_size) * np.sqrt(2. / input_size)
        self.bias = 0
        self.output = None
        self.input = None
        self.delta = None

    def activate(self, inputs):
        self.input = inputs
        self.output = self.relu(np.dot(inputs, self.weights) + self.bias)
        return self.output

    def relu(self, x):
        return np.maximum(0, x)

    def derivative_relu(self, x):
        return np.where(x <= 0, 0, 1)

    def calculate_delta(self, error):
        self.delta = error * self.derivative_relu(self.output)


class MultilayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
        self.output_layer = [Neuron(hidden_size) for _ in range(output_size)]

    def predict(self, inputs):
        hidden_outputs = np.array([neuron.activate(inputs) for neuron in self.hidden_layer]).T
        return np.array([neuron.activate(hidden_outputs) for neuron in self.output_layer]).T

    def backpropagation(self, inputs, expected_output, learning_rate):
        inputs = np.array(inputs, dtype=np.float64)
        outputs = self.predict(inputs)
        output_errors = expected_output - outputs

        # delta for output layer
        for i, neuron in enumerate(self.output_layer):
            neuron.calculate_delta(output_errors[i])

        # delta for hidden layer
        hidden_errors = np.dot(np.array([neuron.delta for neuron in self.output_layer]),
                               np.array([neuron.weights for neuron in self.output_layer]))
        for i, neuron in enumerate(self.hidden_layer):
            neuron.calculate_delta(hidden_errors[i])

        # weights for output layer
        for neuron in self.output_layer:
            neuron.weights += learning_rate * neuron.delta * np.array([n.output for n in self.hidden_layer])
            neuron.bias += learning_rate * neuron.delta

        # weights for hidden layer
        for neuron in self.hidden_layer:
            neuron.weights += learning_rate * neuron.delta * inputs
            neuron.bias += learning_rate * neuron.delta

    def train(self, training_inputs, training_outputs, learning_rate, epochs):
        for _ in range(epochs):
            for inputs, expected_output in zip(training_inputs, training_outputs):
                self.backpropagation(inputs, expected_output, learning_rate)