import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Load the dataset
file_path = 'Lab Session Data.xlsx'
data = pd.read_excel(file_path, sheet_name='IRCTC Stock Price')

# Extract relevant columns
data = data[['Price', 'Open', 'High', 'Low', 'Volume']]

# Convert 'Volume' to numeric values (e.g., '5.27M' to 5.27 * 1,000,000)
def convert_volume(volume_str):
    if 'M' in volume_str:
        return float(volume_str.replace('M', '')) * 1e6
    elif 'K' in volume_str:
        return float(volume_str.replace('K', '')) * 1e3
    else:
        return float(volume_str)

data['Volume'] = data['Volume'].apply(convert_volume)

# Split data into features (X) and target (y)
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate MSE, RMSE, MAPE, and R² scores
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate the percentage of MSE and RMSE
y_mean = np.mean(y_test)
mse_percentage = (mse / (y_mean ** 2)) * 100
rmse_percentage = (rmse / y_mean) * 100

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")
print(f"R²: {r2}")
print(f"MSE as percentage of mean squared target: {mse_percentage}%")
print(f"RMSE as percentage of mean target: {rmse_percentage}%")
