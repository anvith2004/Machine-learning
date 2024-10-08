import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
df = pd.read_excel(r'Profitability Ratio.xlsx', sheet_name='page-1_table-1')

# Define features and target
features = df[['1 Market Cap', 'EBITDA Margin', 'ROCE', 'Return on Equity',
               'Return on Assets', 'EPS (Q)', 'Net Profit Margin']]
target = df['Recommendation']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print performance metrics
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Draw the decision tree
plt.figure(figsize=(15, 7))  # Set figure size for better visualization
plot_tree(clf,
          feature_names=features.columns,  # Use feature names for better interpretability
          class_names=clf.classes_,        # Use class names from the classifier
          filled=True,                     # Fill nodes with colors
          rounded=True,                    # Rounded corners for better readability
          fontsize=6,)                     # Set font size for clarity
                    # Remove 'gini', 'samples', and 'value' labels
plt.title('Decision Tree Visualization')
plt.show()