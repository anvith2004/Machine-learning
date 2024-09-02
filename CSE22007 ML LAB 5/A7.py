import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load Excel Data
file_path = 'Profitability Ratio.xlsx'

# Load the sheet and drop non-numerical columns ('Sub-Sector' and 'Recommendation')
df = pd.read_excel(file_path, sheet_name='page-1_table-1')
df_numerical = df.drop(columns=['Sub-Sector', 'Recommendation'])

# Handle missing values
df_numerical = df_numerical.dropna()

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numerical)

# List to store inertia (distortions)
distortions = []

# Calculate inertia for k values from 2 to 19
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled_data)
    distortions.append(kmeans.inertia_)

# Plot the elbow plot
plt.figure(figsize=(8, 5))
plt.plot(range(2, 20), distortions, marker='o', color='b')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Distortion)')
plt.grid(True)
plt.show()
