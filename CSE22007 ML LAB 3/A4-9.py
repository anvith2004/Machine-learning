import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Data preprocessing
df = pd.read_excel(r'D:\D - Drive\Python\ml_lab_vs\ml_lab3\Profitability Ratio.xlsx', sheet_name='page-1_table-1')
df = df.iloc[:, 1:11]
label_encoders = {}
for column in ['Sub-Sector', 'Recommendation']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

features = df[['1 Market Cap', 'EBITDA Margin', 'ROCE', 'Return on Equity',
               'Return on Assets', 'EPS (Q)', 'Net Profit Margin']]
target = df['Recommendation']

scaler = StandardScaler()
features = scaler.fit_transform(features)

# Applying model
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=10)
for i in range(1, 12):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)

    # Evaluation for test set
    print("For K =", i)
    print("Prediction of X_test[0] -", knn.predict([X_test[0]]))
    print()
    y_pred = knn.predict(X_test)
    print("For Testing:")
    print()
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:", classification_report(y_test, y_pred, zero_division=0))
    print()

    # Evaluation for training set
    y_pred = knn.predict(X_train)
    print("For Training:")
    print()
    print("Accuracy:", accuracy_score(y_train, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_train, y_pred))
    print("Classification Report:", classification_report(y_train, y_pred, zero_division=0))
    print()
    print()
print("The model is regularfit")