import numpy as np
import matplotlib.pyplot as plt


# Step Activation Function
def step_activation(x):
    """Step activation function."""
    return 1 if x >= 0 else 0


# Perceptron Training Function
def train_perceptron(X, y, w, learning_rate, activation_func, max_epochs=1000, error_threshold=0.002):
    n_samples = X.shape[0]
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

        epochs += 1

        # Check for convergence
        if total_error <= error_threshold:
            break
    return epochs


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

# Learning rates to test
learning_rates = [0.1 * i for i in range(1, 11)]  # [0.1, 0.2, ..., 1.0]

# Store results for comparison
results = {}

# Train perceptron with each learning rate
for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    num_epochs = train_perceptron(X, y, initial_weights.copy(), lr, step_activation)
    results[lr] = num_epochs
    print(f"Learning rate: {lr} took {num_epochs} epochs to converge.\n")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='-', color='blue')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Epochs to Converge')
plt.title('Learning Rate vs Number of Epochs for Convergence')
plt.grid(True)
plt.xticks(learning_rates)
plt.show()
