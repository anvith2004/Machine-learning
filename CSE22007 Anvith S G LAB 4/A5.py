import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Set random seed for reproducibility
np.random.seed(0)

# Generate 20 random training data points for X and Y, between 1 and 10
X_train = np.random.uniform(1, 10, 20)
Y_train = np.random.uniform(1, 10, 20)

# Create training DataFrame
train_data = pd.DataFrame({'X': X_train, 'Y': Y_train})
train_data['Class'] = np.where(train_data['X'] > 5, 1, 0)

# Split data into features (X_train) and target (y_train)
X_train = train_data[['X', 'Y']]
y_train = train_data['Class']

# Generate test data with values of X and Y varying between 0 and 10 with increments of 0.1
x_test = np.arange(0, 10.1, 0.1)
y_test = np.arange(0, 10.1, 0.1)
X_test, Y_test = np.meshgrid(x_test, y_test)
X_test = X_test.ravel()
Y_test = Y_test.ravel()

# Create test DataFrame
test_data = pd.DataFrame({'X': X_test, 'Y': Y_test})

# Define k values to experiment with
k_values = [1, 3, 5, 10]

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, k in enumerate(k_values):
    # Create kNN classifier with current k
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict classes for the test data
    test_data['Predicted_Class'] = knn.predict(test_data[['X', 'Y']])

    # Plot the results
    ax = axes[i]
    scatter = ax.scatter(test_data[test_data['Predicted_Class'] == 0]['X'],
                         test_data[test_data['Predicted_Class'] == 0]['Y'],
                         color='blue', s=1, label='Class 0')
    scatter = ax.scatter(test_data[test_data['Predicted_Class'] == 1]['X'],
                         test_data[test_data['Predicted_Class'] == 1]['Y'],
                         color='red', s=1, label='Class 1')

    ax.set_title(f'k = {k}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

