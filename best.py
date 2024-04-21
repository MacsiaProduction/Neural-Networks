from keras.datasets import mnist
from keras.utils import to_categorical

from CNN import *
from better_perceptron import *


def preprocess_data1(x, y, limit):
    x = x.reshape(x.shape[0], 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data1(x_train, y_train, y_train.shape[0])
x_test, y_test = preprocess_data1(x_test, y_test, y_test.shape[0])
print(y_train.shape, y_test.shape)

best_error = float('inf')
best_network = None
best_params = {}

# Define a grid of hyperparameters to search over
hyperparameters_grid = {
    'learning_rate': [0.1, 0.01]
}

epochs = 20
dense = 32
kernel1 = 5
kernels1 = 5
kernel2 = 5
kernels2 = 3

for learning_rate in hyperparameters_grid['learning_rate']:
    b = (28 - kernel1 + 1) // 2
    c = b - kernel2 + 1
    network = [
        Convolutional((1, 28, 28), kernel1, kernels1),
        Sigmoid(),
        MaxPooling(2),
        Convolutional((kernels1, b, b), kernel2, kernels2),
        Sigmoid(),
        Reshape((kernels2, c, c), (kernels2 * c * c, 1)),
        Dense(kernels2 * c * c, dense),
        Sigmoid(),
        Dense(dense, 10),
        Sigmoid()
    ]

    # Train the network
    train(network, binary_cross_entropy, binary_cross_entropy_prime, x_train, y_train, epochs=epochs,
          learning_rate=learning_rate, verbose=False)

    # Test the network on the validation set
    error, _ = test(network, x_test, y_test, binary_cross_entropy)

    # Check if the current model is the best so far
    if error < best_error:
        best_error = error
        best_network = network
        best_params['kernel1'] = kernel1
        best_params['kernel2'] = kernel2
        best_params['kernels1'] = kernels1
        best_params['kernels2'] = kernels2
        best_params['learning_rate'] = learning_rate
        print(f"error {error}, params {best_params}")

# Test the best model on the test set
final_error, predictions = test(best_network, x_test, y_test, binary_cross_entropy)

print("Best hyperparameters:", best_params)
print("Error on test set:", final_error)
