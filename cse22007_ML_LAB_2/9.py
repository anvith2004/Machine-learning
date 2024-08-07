import pandas as pd
import numpy as np

# Define the path to the file
file_path = 'Lab Session Data.xlsx'

try:
    # Load the dataset
    df = pd.read_excel(file_path, sheet_name='thyroid0387_UCI')

    # Convert all columns to numeric, forcing errors to NaN (Not a Number)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Check for missing values
    if df.isnull().values.any():
        print("Missing values detected. Filling missing values with 0.")
        df = df.fillna(0)  # Fill missing values with 0 or another imputation strategy

    # Extract the complete vectors for the first two observations
    vector1 = df.loc[0].values
    vector2 = df.loc[1].values

    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector1, vector2)

    # Calculate the magnitudes (norms) of the vectors
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    # Calculate the Cosine Similarity
    if norm_vector1 > 0 and norm_vector2 > 0:
        cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
    else:
        cosine_similarity = float('nan')  # Handle the case where one or both vectors are zero vectors

    print("\nCosine Similarity:", cosine_similarity)

except FileNotFoundError:
    print(f"Error: The file at path '{file_path}' was not found. Please check the file path and try again.")
except ValueError as ve:
    print(f"Value Error: {ve}")
except Exception as e:
    print(f"An error occurred: {e}")
