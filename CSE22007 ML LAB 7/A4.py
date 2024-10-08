import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Sample data: Replace this with your project data (Tr_X, Tr_y)
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

# Initialize the DecisionTreeClassifier model
model = DecisionTreeClassifier()

# Train the model with the training data
ml_model = model.fit(Tr_X, Tr_y)

# Evaluate the model's accuracy on the training set
training_accuracy = ml_model.score(Tr_X, Tr_y)
print(f"Training Set Accuracy: {training_accuracy}")

# Evaluate the model's accuracy on the test set
test_accuracy = ml_model.score(Te_X, Te_y)
print(f"Test Set Accuracy: {test_accuracy}")

# Visualize the Decision Tree
plt.figure(figsize=(15,8))  # Adjust size to make it visible
plot_tree(ml_model,
          filled=True,
          feature_names=X.columns,
          class_names=['No', 'Yes'],
          rounded=True,
          fontsize=10)  # Adjust font size to make it readable
plt.show()
