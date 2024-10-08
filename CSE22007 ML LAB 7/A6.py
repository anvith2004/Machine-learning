import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Sample data: Replace this with your project data
data = {
    'age': ['<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40', '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Separate features (X) and target variable (y)
X = df.drop('buys_computer', axis=1)
y = df['buys_computer']

# Convert categorical features to numeric using LabelEncoder
label_encoder = LabelEncoder()
for column in X.columns:
    X[column] = label_encoder.fit_transform(X[column])

# Encode target variable (y)
y = label_encoder.fit_transform(y)

# Split data into training and testing sets (80% training, 20% testing)
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Decision Tree with Entropy Criterion ---
print("Decision Tree with Entropy Criterion:")

# Initialize the DecisionTreeClassifier model with the "entropy" criterion
model_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train the model with the training data
ml_model_entropy = model_entropy.fit(Tr_X, Tr_y)

# Evaluate the model's accuracy on the training set
training_accuracy_entropy = ml_model_entropy.score(Tr_X, Tr_y)
print(f"Training Set Accuracy (criterion='entropy'): {training_accuracy_entropy}")

# Evaluate the model's accuracy on the test set
test_accuracy_entropy = ml_model_entropy.score(Te_X, Te_y)
print(f"Test Set Accuracy (criterion='entropy'): {test_accuracy_entropy}")

# Visualize the Decision Tree with Entropy criterion
plt.figure(figsize=(15,8))  # Adjust size to make it visible
plot_tree(ml_model_entropy,
          filled=True,
          feature_names=X.columns,
          class_names=['No', 'Yes'],
          rounded=True,
          fontsize=10)  # Adjust font size to make it readable
plt.title("Decision Tree (Entropy Criterion)")
plt.show()

# --- Summary of Results ---
print("\nSummary of Results:")
print(f"Training Set Accuracy (Entropy): {training_accuracy_entropy}, Test Set Accuracy (Entropy): {test_accuracy_entropy}")
