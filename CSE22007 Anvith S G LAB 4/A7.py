import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(0)

# Load the dataset
df = pd.read_excel('Profitability Ratio.xlsx', sheet_name='page-1_table-1')

# Use 'ROCE' and 'Return on Equity' as features, and 'Recommendation' as the class
features = df[['ROCE', 'Return on Equity']].dropna()
classes = df['Recommendation'].dropna()

# Encode the classes (Highly Recommended = 1, Not Recommended = 0)
classes_encoded = classes.apply(lambda x: 1 if x == 'Highly Recommended' else 0)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, classes_encoded, test_size=0.3, random_state=42)

# Define the parameter grid (k values to try)
param_grid = {'n_neighbors': np.arange(1, 31)}

# Instantiate the kNN model
knn = KNeighborsClassifier()

# Set up GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit the model with the grid search
grid_search.fit(X_train, y_train)

# Get the best parameter (k value)
best_k = grid_search.best_params_['n_neighbors']
print(f"The best k value is: {best_k}")

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the best model with k={best_k}: {accuracy:.2f}")