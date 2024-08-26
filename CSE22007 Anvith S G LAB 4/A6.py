import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Set random seed for reproducibility
np.random.seed(0)
df = pd.read_excel('Profitability Ratio.xlsx', sheet_name='page-1_table-1')

# Load the dataset (assuming you have already loaded your project data into a DataFrame named 'df')
# Use 'ROCE' and 'Return on Equity' as features, and 'Recommendation' as the class
features = df[['ROCE', 'Return on Equity']].dropna()
classes = df['Recommendation'].dropna()

# Encode the classes (Highly Recommended = 1, Not Recommended = 0)
classes_encoded = classes.apply(lambda x: 1 if x == 'Highly Recommended' else 0)

# Combine features and classes into a single DataFrame for consistent formatting
train_data = pd.concat([features, classes_encoded], axis=1)

# A3: Scatter Plot of Training Data
plt.figure(figsize=(8, 6))
plt.scatter(train_data[train_data['Recommendation'] == 0]['ROCE'],
            train_data[train_data['Recommendation'] == 0]['Return on Equity'],
            color='blue', label='Not Recommended')
plt.scatter(train_data[train_data['Recommendation'] == 1]['ROCE'],
            train_data[train_data['Recommendation'] == 1]['Return on Equity'],
            color='red', label='Highly Recommended')
plt.xlabel('ROCE')
plt.ylabel('Return on Equity')
plt.title('Scatter Plot of Training Data')
plt.legend(loc='upper left')  # Manually set legend location
plt.grid(True)
plt.show()
print(3)
# A4: Generate test set data and classify using kNN (k=3)
x_test = np.arange(0, 100.1, 0.1)
y_test = np.arange(0, 100.1, 0.1)
X_test, Y_test = np.meshgrid(x_test, y_test)
X_test_flat = X_test.ravel()
Y_test_flat = Y_test.ravel()

# Create test DataFrame
test_data = pd.DataFrame({'ROCE': X_test_flat, 'Return on Equity': Y_test_flat})

# Create kNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data[['ROCE', 'Return on Equity']], train_data['Recommendation'])

# Predict classes for the test data
test_data['Predicted_Class'] = knn.predict(test_data[['ROCE', 'Return on Equity']])

# A4: Plot the test data
plt.figure(figsize=(10, 8))
plt.scatter(test_data[test_data['Predicted_Class'] == 0]['ROCE'],
            test_data[test_data['Predicted_Class'] == 0]['Return on Equity'],
            color='blue', s=1, label='Not Recommended')

plt.scatter(test_data[test_data['Predicted_Class'] == 1]['ROCE'],
            test_data[test_data['Predicted_Class'] == 1]['Return on Equity'],
            color='red', s=1, label='Highly Recommended')

plt.xlabel('ROCE')
plt.ylabel('Return on Equity')
plt.title('Scatter Plot of Test Data with Predicted Classes (k=3)')
plt.legend(loc='upper left')  # Manually set legend location
plt.grid(True)
plt.show()
print(4)
# A5: for various values of k
k_values = [1, 3, 5, 10]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, k in enumerate(k_values):
    # Create kNN classifier with current k
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data[['ROCE', 'Return on Equity']], train_data['Recommendation'])

    # Predict classes for the test data
    test_data['Predicted_Class'] = knn.predict(test_data[['ROCE', 'Return on Equity']])

    # Plot the results
    ax = axes[i]
    ax.scatter(test_data[test_data['Predicted_Class'] == 0]['ROCE'],
               test_data[test_data['Predicted_Class'] == 0]['Return on Equity'],
               color='blue', s=1, label='Not Recommended')
    ax.scatter(test_data[test_data['Predicted_Class'] == 1]['ROCE'],
               test_data[test_data['Predicted_Class'] == 1]['Return on Equity'],
               color='red', s=1, label='Highly Recommended')

    ax.set_title(f'k = {k}')
    ax.set_xlabel('ROCE')
    ax.set_ylabel('Return on Equity')
    ax.legend(loc='upper left')  # Manually set legend location
    ax.grid(True)

plt.tight_layout()
plt.show()