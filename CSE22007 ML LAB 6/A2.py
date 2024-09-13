import numpy as np
import matplotlib.pyplot as plt


# Define the Step Activation Function
def step_activation(x):
    return 1 if x >= 0 else 0


# Perceptron Training Function
def train_perceptron(X, y, w, learning_rate, max_epochs=1000, error_threshold=0.002):
    n_samples = X.shape[0]
    errors = []
    epochs = 0

    for epoch in range(max_epochs):
        total_error = 0
        for i in range(n_samples):
            # Calculate the linear combination of inputs and weights
            linear_output = np.dot(X[i], w)

            # Apply the step activation function
            prediction = step_activation(linear_output)

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

# Train the perceptron
final_weights, errors, num_epochs = train_perceptron(X, y, initial_weights, learning_rate)

# Plotting epochs vs error
plt.plot(range(num_epochs), errors, color='red')
plt.xlabel('Epochs')
plt.ylabel('Sum-Squared Error')
plt.title('Epochs vs Sum-Squared Error')
plt.grid(True)
plt.show()

print(f"Final weights after training: {final_weights}")
