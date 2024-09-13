import numpy as np


# Activation Functions
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    return x * (1 - x)


def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)


def tanh_derivative(x):
    """Derivative of the tanh function."""
    return 1.0 - np.tanh(x) ** 2


# Initialize Network Parameters
input_size = 2  # Number of input neurons
hidden_size = 4  # Increased number of hidden neurons
output_size = 1  # Number of output neurons
learning_rate = 0.01
max_epochs = 1000
error_threshold = 0.002

# Input data for AND gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target data for AND gate
y = np.array([
    [0],
    [0],
    [0],
    [1]
])

# Initialize weights
np.random.seed(0)
W1 = np.random.rand(input_size, hidden_size)  # Weights for input to hidden layer
W2 = np.random.rand(hidden_size, output_size)  # Weights for hidden to output layer


# Training the Neural Network
def train_neural_network(X, y, W1, W2, learning_rate, max_epochs, error_threshold):
    for epoch in range(max_epochs):
        # Forward Pass
        hidden_layer_input = np.dot(X, W1)
        hidden_layer_output = tanh(hidden_layer_input)  # Using tanh activation

        output_layer_input = np.dot(hidden_layer_output, W2)
        output_layer_output = sigmoid(output_layer_input)

        # Compute the error
        error = y - output_layer_output
        mean_squared_error = np.mean(error ** 2)

        # Backward Pass
        # Calculate the gradients
        d_output_layer = error * sigmoid_derivative(output_layer_output)
        d_hidden_layer = np.dot(d_output_layer, W2.T) * tanh_derivative(hidden_layer_output)

        # Update weights
        W2 += learning_rate * np.dot(hidden_layer_output.T, d_output_layer)
        W1 += learning_rate * np.dot(X.T, d_hidden_layer)

        # Check for convergence
        if mean_squared_error <= error_threshold:
            print(f"Convergence reached after {epoch + 1} epochs with MSE {mean_squared_error}.")
            break
    else:
        print(f"Stopped after {max_epochs} epochs with MSE {mean_squared_error} (did not converge).")

    return W1, W2


# Train the neural network
W1, W2 = train_neural_network(X, y, W1, W2, learning_rate, max_epochs, error_threshold)


# Predicting using the trained model
def predict(X, W1, W2):
    hidden_layer_output = tanh(np.dot(X, W1))
    output_layer_output = sigmoid(np.dot(hidden_layer_output, W2))
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
