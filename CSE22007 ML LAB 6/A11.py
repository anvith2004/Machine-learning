import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def train_and_predict(X, y, hidden_layer_sizes, learning_rate, max_iter):
    """
    Trains an MLPClassifier on the given data and returns predictions and accuracy.

    Parameters:
    - X: Input features
    - y: Target labels
    - hidden_layer_sizes: Tuple specifying the number of neurons in each hidden layer
    - learning_rate: Learning rate for the optimizer
    - max_iter: Maximum number of iterations for training

    Returns:
    - y_pred: Predicted labels
    - accuracy: Accuracy of the model
    - coefs: List of weight matrices (hidden and output layers)
    """
    # Initialize and train the MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu',
                        solver='adam', learning_rate_init=learning_rate, max_iter=max_iter, random_state=0)
    mlp.fit(X, y)

    # Predict and evaluate
    y_pred = mlp.predict(X)
    accuracy = accuracy_score(y, y_pred)

    return y_pred, accuracy, mlp.coefs_


# Input Data and Target Labels for AND Gate
X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_and = np.array([0, 0, 0, 1])

# Input Data and Target Labels for XOR Gate
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_xor = np.array([0, 1, 1, 0])

# Parameters
hidden_layer_sizes = (2,)  # Single hidden layer with 2 neurons
learning_rate = 0.05
max_iter = 1000

# Train and evaluate for AND Gate
print("AND Gate:")
y_pred_and, accuracy_and, coefs_and = train_and_predict(X_and, y_and, hidden_layer_sizes, learning_rate, max_iter)
print("Predictions:", y_pred_and)
print("Accuracy:", accuracy_and)
print("Final Weights:")
print("Hidden Layer Weights:", coefs_and[0])
print("Output Layer Weights:", coefs_and[1])

# Train and evaluate for XOR Gate
print("\nXOR Gate:")
y_pred_xor, accuracy_xor, coefs_xor = train_and_predict(X_xor, y_xor, hidden_layer_sizes, learning_rate, max_iter)
print("Predictions:", y_pred_xor)
print("Accuracy:", accuracy_xor)
print("Final Weights:")
print("Hidden Layer Weights:", coefs_xor[0])
print("Output Layer Weights:", coefs_xor[1])
