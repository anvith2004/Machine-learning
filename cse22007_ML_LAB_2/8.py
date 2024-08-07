import pandas as pd
import numpy as np

# Define the path to the file
file_path = 'Lab Session Data.xlsx'

try:
    # Load the dataset
    df = pd.read_excel(file_path, sheet_name='thyroid0387_UCI')

    # Identify binary attributes (0/1 values)
    binary_cols = df.columns[df.nunique() == 2]  # Simplistic approach for identifying binary columns

    if len(binary_cols) < 1:
        raise ValueError("No binary columns found in the dataset.")

    # Extract the first two rows
    vector1 = df.loc[0, binary_cols]
    vector2 = df.loc[1, binary_cols]

    # Calculate f11, f10, f01, and f00
    f11 = np.sum((vector1 == 1) & (vector2 == 1))  # Both 1
    f10 = np.sum((vector1 == 1) & (vector2 == 0))  # 1 in vector1 only
    f01 = np.sum((vector1 == 0) & (vector2 == 1))  # 1 in vector2 only
    f00 = np.sum((vector1 == 0) & (vector2 == 0))  # Both 0

    # Calculate Jaccard Coefficient (JC)
    denominator_jc = f01 + f10 + f11
    jc = f11 / denominator_jc if denominator_jc != 0 else float('nan')

    # Calculate Simple Matching Coefficient (SMC)
    denominator_smc = f00 + f01 + f10 + f11
    smc = (f11 + f00) / denominator_smc if denominator_smc != 0 else float('nan')

    print("\nJaccard Coefficient (JC):", jc)
    print("Simple Matching Coefficient (SMC):", smc)

except FileNotFoundError:
    print(f"Error: The file at path '{file_path}' was not found. Please check the file path and try again.")
except ValueError as ve:
    print(f"Value Error: {ve}")
except Exception as e:
    print(f"An error occurred: {e}")
