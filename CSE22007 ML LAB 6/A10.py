import numpy as np


# Define Activation Functions
def step_activation(x):
    """Step activation function."""
    return np.where(x > 0, 1, 0)


def sigmoid_activation(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    return x * (1 - x)


# Define the XOR Gate Logic with Two Output Nodes
def one_hot_encode(y):
    """Convert binary output to one-hot encoding."""
    return np.array([[1, 0] if val == 0 else [0, 1] for val in y])


# Input Data for AND Gate
X = np.array([
    [1, 0, 0],  # Including bias term as the first feature
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# Target Output Data for AND Gate with One-Hot Encoding
y = one_hot_encode([0, 0, 0, 1])

# Initialize Weights
input_size = X.shape[1]
hidden_size = 2  # For simplicity, we use 2 hidden neurons
output_size = 2  # Two output nodes for one-hot encoding

np.random.seed(0)
W1 = np.random.rand(input_size, hidden_size)  # Weights for input to hidden layer
W2 = np.random.rand(hidden_size, output_size)  # Weights for hidden to output layer


# Training Function
def train_neural_network(X, y, W1, W2, learning_rate, max_epochs):
    for epoch in range(max_epochs):
        # Forward Pass
        hidden_layer_input = np.dot(X, W1)
        hidden_layer_output = sigmoid_activation(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, W2)
        output_layer_output = sigmoid_activation(output_layer_input)

        # Compute the error
        error = y - output_layer_output
        mean_squared_error = np.mean(error ** 2)

        # Backward Pass
        d_output_layer = error * sigmoid_derivative(output_layer_output)
        d_hidden_layer = np.dot(d_output_layer, W2.T) * sigmoid_derivative(hidden_layer_output)

        # Update weights
        W2 += learning_rate * np.dot(hidden_layer_output.T, d_output_layer)
        W1 += learning_rate * np.dot(X.T, d_hidden_layer)

        # Print the epoch and mean squared error
        print(f"Epoch {epoch + 1}, MSE: {mean_squared_error}")

    return W1, W2


# Train the neural network
W1, W2 = train_neural_network(X, y, W1, W2, learning_rate=0.05, max_epochs=1000)


# Predicting using the trained model
def predict(X, W1, W2):
    hidden_layer_output = sigmoid_activation(np.dot(X, W1))
    output_layer_output = sigmoid_activation(np.dot(hidden_layer_output, W2))
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
