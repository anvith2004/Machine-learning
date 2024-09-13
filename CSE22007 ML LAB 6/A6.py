import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# Numerically Stable Sigmoid Activation Function using scipy
def sigmoid_activation(x):
    """Numerically stable sigmoid activation function using scipy."""
    return expit(x)

# Perceptron Training Function with Sigmoid Activation
def train_perceptron(X, y, w, learning_rate, activation_func, max_epochs=1000, error_threshold=0.002):
    n_samples = X.shape[0]
    errors = []
    epochs = 0

    for epoch in range(max_epochs):
        total_error = 0
        for i in range(n_samples):
            # Calculate the linear combination of inputs and weights
            linear_output = np.dot(X[i], w)

            # Apply the activation function
            prediction = activation_func(linear_output)

            # Convert the prediction to a binary class (0 or 1) for classification
            predicted_class = 1 if prediction >= 0.5 else 0

            # Calculate the error
            error = y[i] - predicted_class

            # Update weights based on error
            w += learning_rate * error * X[i]

            # Accumulate the squared error
            total_error += error ** 2

        # Record the sum of squared errors for this epoch
        errors.append(total_error)
        epochs += 1

        # Check for convergence
        if total_error <= error_threshold:
            print(f"Convergence reached after {epochs} epochs with error {total_error}.")
            break
    else:
        print(f"Stopped after {epochs} epochs with error {total_error} (did not converge).")

    return w, errors, epochs

# Input Data (Including bias term as the first feature)
X = np.array([
    [1, 20, 6, 2, 386],  # C_1
    [1, 16, 3, 6, 289],  # C_2
    [1, 27, 6, 2, 393],  # C_3
    [1, 19, 1, 2, 110],  # C_4
    [1, 24, 4, 2, 280],  # C_5
    [1, 22, 1, 5, 167],  # C_6
    [1, 15, 4, 2, 271],  # C_7
    [1, 18, 4, 2, 274],  # C_8
    [1, 21, 1, 4, 148],  # C_9
    [1, 16, 2, 4, 198]  # C_10
])

# Target Labels (1 for "High Value" and 0 for "Low Value")
y = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

# Initial weights (chosen arbitrarily)
initial_weights = np.random.rand(X.shape[1])  # Random initial weights

# Learning rate
learning_rate = 0.01

# Train the perceptron
final_weights, errors, num_epochs = train_perceptron(X, y, initial_weights, learning_rate, sigmoid_activation)

# Plotting epochs vs error
plt.plot(range(num_epochs), errors, label='Sigmoid Activation', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Sum-Squared Error')
plt.title('Epochs vs Sum-Squared Error for Customer Transaction Classification')
plt.grid(True)
plt.show()

print(f"Final weights after training: {final_weights}")
