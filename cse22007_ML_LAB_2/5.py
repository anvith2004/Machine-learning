import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

try:
    # Load the dataset
    df = pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')

    # Display basic information about the dataset
    print("Basic Information:")
    print(df.info())
    print("\nFirst few rows of the dataset:")
    print(df.head())

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    print("\nCategorical Columns:")
    print(categorical_cols)

    # Check unique values in each categorical column
    print("\nUnique values in categorical columns:")
    for col in categorical_cols:
        print(f"{col}: {df[col].unique()}")

    # Determine encoding scheme
    # For demonstration purposes, assuming ordinal variables are identified manually
    # Example: Assuming 'Education Level' is ordinal and others are nominal
    ordinal_cols = []  # Example list, empty if no ordinal columns identified
    nominal_cols = list(set(categorical_cols) - set(ordinal_cols))
    print("\nOrdinal Columns:")
    print(ordinal_cols)
    print("\nNominal Columns:")
    print(nominal_cols)

    # Data range for numeric variables
    print("\nNumeric Data Range:")
    print(df.describe())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Identify outliers using IQR method
    numeric_cols = df.select_dtypes(include=['number']).columns
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR)))
    print("\nOutliers count:")
    print(outliers.sum())

    # Box plots for numeric columns to visualize outliers
    print("\nBox Plots for Numeric Columns:")
    for col in numeric_cols:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f'Box Plot for {col}')
        plt.show()

    # Calculate mean and variance (or standard deviation)
    print("\nMean of Numeric Columns:")
    print(df[numeric_cols].mean())

    print("\nVariance of Numeric Columns:")
    print(df[numeric_cols].var())

    print("\nStandard Deviation of Numeric Columns:")
    print(df[numeric_cols].std())

except FileNotFoundError:
    print(f"Error: The file 'Lab Session Data.xlsx' was not found. Please check the file path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")
