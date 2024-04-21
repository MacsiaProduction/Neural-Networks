import numpy as np
from scipy import signal

from better_perceptron import Layer


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)


class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient


class MaxPooling(Layer):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, input):
        self.input = input
        depth, height, width = input.shape
        pooled_height = height // self.pool_size
        pooled_width = width // self.pool_size
        self.output = np.zeros((depth, pooled_height, pooled_width))

        for d in range(depth):
            for ph in range(pooled_height):
                for pw in range(pooled_width):
                    h_start = ph * self.pool_size
                    h_end = h_start + self.pool_size
                    w_start = pw * self.pool_size
                    w_end = w_start + self.pool_size
                    window = input[d, h_start:h_end, w_start:w_end]
                    self.output[d, ph, pw] = np.max(window)

        return self.output

    def backward(self, output_gradient, learning_rate):
        depth, pooled_height, pooled_width = output_gradient.shape
        input_gradient = np.zeros_like(self.input)

        for d in range(depth):
            for ph in range(pooled_height):
                for pw in range(pooled_width):
                    h_start = ph * self.pool_size
                    h_end = h_start + self.pool_size
                    w_start = pw * self.pool_size
                    w_end = w_start + self.pool_size
                    window = self.input[d, h_start:h_end, w_start:w_end]
                    max_value = np.max(window)
                    mask = (window == max_value)
                    input_gradient[d, h_start:h_end, w_start:w_end] += mask * output_gradient[d, ph, pw]

        return input_gradient
