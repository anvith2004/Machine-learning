import numpy as np


# Define the functions from above

def summation_unit(inputs, weights):
    """Computes the weighted sum of inputs."""
    return np.dot(inputs, weights)


def step_activation(x):
    """Step activation function."""
    return np.where(x > 0, 1, 0)


def bipolar_step_activation(x):
    """Bipolar step activation function."""
    return np.where(x > 0, 1, -1)


def sigmoid_activation(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def tanh_activation(x):
    """Tanh activation function."""
    return np.tanh(x)


def relu_activation(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def leaky_relu_activation(x, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)


def mean_squared_error(y_true, y_pred):
    """Computes the Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def binary_crossentropy(y_true, y_pred):
    """Computes the Binary Cross-Entropy."""
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Initialize Network Parameters
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.05
max_epochs = 1000
error_threshold = 0.002

# XOR Input and Output
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Initialize weights
np.random.seed(0)
W1 = np.random.rand(input_size, hidden_size)  # Weights for input to hidden layer
W2 = np.random.rand(hidden_size, output_size)  # Weights for hidden to output layer


# Training the Neural Network
def train_neural_network(X, y, W1, W2, learning_rate, max_epochs, error_threshold):
    for epoch in range(max_epochs):
        # Forward Pass
        hidden_layer_input = summation_unit(X, W1)
        hidden_layer_output = tanh_activation(hidden_layer_input)

        output_layer_input = summation_unit(hidden_layer_output, W2)
        output_layer_output = sigmoid_activation(output_layer_input)

        # Compute the error
        error = y - output_layer_output
        mean_squared_error_val = mean_squared_error(y, output_layer_output)

        # Backward Pass
        d_output_layer = error * sigmoid_activation(output_layer_input) * (1 - sigmoid_activation(output_layer_input))
        d_hidden_layer = np.dot(d_output_layer, W2.T) * (1 - tanh_activation(hidden_layer_input) ** 2)

        # Update weights
        W2 += learning_rate * np.dot(hidden_layer_output.T, d_output_layer)
        W1 += learning_rate * np.dot(X.T, d_hidden_layer)

        # Check for convergence
        if mean_squared_error_val <= error_threshold:
            print(f"Convergence reached after {epoch + 1} epochs with MSE {mean_squared_error_val}.")
            break
    else:
        print(f"Stopped after {max_epochs} epochs with MSE {mean_squared_error_val} (did not converge).")

    return W1, W2


# Train the neural network
W1, W2 = train_neural_network(X, y, W1, W2, learning_rate, max_epochs, error_threshold)


# Predicting using the trained model
def predict(X, W1, W2):
    hidden_layer_output = tanh_activation(summation_unit(X, W1))
    output_layer_output = sigmoid_activation(summation_unit(hidden_layer_output, W2))
    return np.round(output_layer_output)


# Test the neural network
predictions = predict(X, W1, W2)
print("Predictions:")
print(predictions)

# Check final weights
print("Final weights:")
print("W1 (Input to Hidden):")
print(W1)
print("W2 (Hidden to Output):")
print(W2)
