import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Define the path to the file
file_path = 'Lab Session Data.xlsx'

try:
    # Load the dataset
    df = pd.read_excel(file_path, sheet_name='thyroid0387_UCI')

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Determine which columns need scaling
    # We will use the IQR method to identify outliers
    scaling_cols = []  # Columns that need Min-Max Scaling
    standardizing_cols = []  # Columns that need Standard Scaling

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Check for outliers
        if ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).any():
            standardizing_cols.append(col)  # Columns with outliers
        else:
            scaling_cols.append(col)  # Columns without outliers

    # Apply Min-Max Scaling
    if scaling_cols:
        scaler_minmax = MinMaxScaler()
        df[scaling_cols] = scaler_minmax.fit_transform(df[scaling_cols])

    # Apply Standard Scaling
    if standardizing_cols:
        scaler_std = StandardScaler()
        df[standardizing_cols] = scaler_std.fit_transform(df[standardizing_cols])

    print("Data normalization and scaling completed.")
    print("\nScaled Data:")
    print(df.head())

except FileNotFoundError:
    print(f"Error: The file at path '{file_path}' was not found. Please check the file path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")
