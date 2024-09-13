import numpy as np
import matplotlib.pyplot as plt


# Activation Functions
def step_activation(x):
    """Step activation function."""
    return 1 if x >= 0 else 0


def bipolar_step_activation(x):
    """Bipolar step activation function."""
    return 1 if x >= 0 else -1


def sigmoid_activation(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def relu_activation(x):
    """ReLU activation function."""
    return max(0, x)


# Perceptron Training Function
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

            # Calculate the error
            error = y[i] - prediction

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


# AND gate inputs and outputs
# Including bias term in inputs (x0 = 1)
X = np.array([
    [1, 0, 0],  # [bias, x1, x2] -> AND(0, 0) = 0
    [1, 0, 1],  # [bias, x1, x2] -> AND(0, 1) = 0
    [1, 1, 0],  # [bias, x1, x2] -> AND(1, 0) = 0
    [1, 1, 1]  # [bias, x1, x2] -> AND(1, 1) = 1
])

y = np.array([0, 0, 0, 1])  # Expected outputs for AND gate

# Initial weights: W0 = 10, W1 = 0.2, W2 = -0.75
initial_weights = np.array([10, 0.2, -0.75])

# Learning rate (α) = 0.05
learning_rate = 0.05

# List of activation functions to compare
activation_functions = {
    "Step": step_activation,
    "Bipolar Step": bipolar_step_activation,
    "Sigmoid": sigmoid_activation,
    "ReLU": relu_activation
}

# Store results for comparison
results = {}

# Train perceptron with each activation function
for name, func in activation_functions.items():
    print(f"Training with {name} activation function...")
    final_weights, errors, num_epochs = train_perceptron(X, y, initial_weights.copy(), learning_rate, func)
    results[name] = (errors, num_epochs)
    print(f"Final weights: {final_weights}\n")

# Plotting the results
plt.figure(figsize=(10, 6))
for name, (errors, epochs) in results.items():
    plt.plot(range(epochs), errors, label=f'{name} (Epochs: {epochs})')

plt.xlabel('Epochs')
plt.ylabel('Sum-Squared Error')
plt.title('Epochs vs Sum-Squared Error for Different Activation Functions')
plt.legend()
plt.grid(True)
plt.show()
