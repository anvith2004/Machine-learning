import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_excel('Lab Session Data.xlsx', sheet_name='Purchase data')

df['Customer Type'] = np.where(df['Payment (Rs)'] > 200, 'RICH', 'POOR')

X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].fillna(0)  # Fill NaN values with 0
y = df['Customer Type']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000, class_weight='balanced')  # Adjust max_iter and handle class imbalance
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=1))

print("Class distribution in training set:")
print(pd.Series(y_train).value_counts())

print("Class distribution in test set:")
print(pd.Series(y_test).value_counts())
