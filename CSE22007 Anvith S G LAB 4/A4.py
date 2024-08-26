import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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

# Create kNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Generate test data with values of X and Y varying between 0 and 10 with increments of 0.1
x_test = np.arange(0, 10.1, 0.1)
y_test = np.arange(0, 10.1, 0.1)
X_test, Y_test = np.meshgrid(x_test, y_test)
X_test = X_test.ravel()
Y_test = Y_test.ravel()

# Create test DataFrame
test_data = pd.DataFrame({'X': X_test, 'Y': Y_test})

# Predict classes for the test data
test_data['Predicted_Class'] = knn.predict(test_data[['X', 'Y']])

# Plot the test data
plt.figure(figsize=(10, 8))

# Plot test data points with their predicted classes
plt.scatter(test_data[test_data['Predicted_Class'] == 0]['X'],
            test_data[test_data['Predicted_Class'] == 0]['Y'],
            color='blue', s=1, label='Class 0')

plt.scatter(test_data[test_data['Predicted_Class'] == 1]['X'],
            test_data[test_data['Predicted_Class'] == 1]['Y'],
            color='red', s=1, label='Class 1')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Test Data with Predicted Classes')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
