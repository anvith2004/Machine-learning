import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Dataset
data = {
    'age': ['<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40', '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Separate the feature columns (X) from the target column (y)
X = df.drop('buys_computer', axis=1)
y = df['buys_computer']

# Convert categorical features to numeric using LabelEncoder
label_encoder = LabelEncoder()
for column in X.columns:
    X[column] = label_encoder.fit_transform(X[column])

# Encode the target variable (y)
y = label_encoder.fit_transform(y)

# Initialize the DecisionTreeClassifier model
model = DecisionTreeClassifier()

# Fit the model with training data
ml_model = model.fit(X, y)

# Calculate accuracy on the training set
training_accuracy = ml_model.score(X, y)
print(f"Training Set Accuracy: {training_accuracy}")

# Print the depth of the constructed decision tree
tree_depth = ml_model.get_depth()
print(f"Tree Depth: {tree_depth}")

# Visualize the decision tree with smaller nodes and font size
plt.figure(figsize=(12,6))  # Make the figure smaller
plot_tree(ml_model,
          filled=True,
          feature_names=X.columns,
          class_names=['No', 'Yes'],
          rounded=True,
          fontsize=8)  # Reduce font size to make nodes smaller
plt.show()
