import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Generate 20 random data points for X and Y, between 1 and 10
X = np.random.uniform(1, 10, 20)
Y = np.random.uniform(1, 10, 20)

# Create a DataFrame
data = pd.DataFrame({'X': X, 'Y': Y})

# Assign classes based on the value of X
# If X > 5, assign class 1 (Red); otherwise, assign class 0 (Blue)
data['Class'] = np.where(data['X'] > 5, 1, 0)

# Create scatter plot
plt.figure(figsize=(8, 6))

# Plot class 0 (Blue)
plt.scatter(data[data['Class'] == 0]['X'], data[data['Class'] == 0]['Y'], color='blue', label='Class 0')

# Plot class 1 (Red)
plt.scatter(data[data['Class'] == 1]['X'], data[data['Class'] == 1]['Y'], color='red', label='Class 1')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Training Data')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
