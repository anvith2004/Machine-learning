import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Step 1: Load the dataset
file_path = 'Profitability Ratio.xlsx'
df = pd.read_excel(file_path, sheet_name='page-1_table-1')

# Step 2: Encode the target variable ('Recommendation')
label_encoder = LabelEncoder()
df['Recommendation'] = label_encoder.fit_transform(df['Recommendation'])

# Step 3: Select features (excluding 'Sub-Sector') and the target variable
X = df.drop(columns=['Sub-Sector', 'Recommendation'])
y = df['Recommendation']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Normalize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train the MLPClassifier with all improvements
mlp = MLPClassifier(
    hidden_layer_sizes=(50, 30),  # Adjusted hidden layers
    solver='lbfgs',               # Changed solver to 'lbfgs'
    learning_rate_init=0.001,      # Reduced learning rate
    random_state=42,
    max_iter=2000                  # Increased max iterations
)
mlp.fit(X_train_scaled, y_train)

# Step 7: Make predictions and evaluate the model
y_pred = mlp.predict(X_test_scaled)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Print the classification report
print(report)
