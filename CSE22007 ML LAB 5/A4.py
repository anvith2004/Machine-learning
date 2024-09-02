import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load Excel Data
file_path = 'Profitability Ratio.xlsx'
excel_data = pd.ExcelFile(file_path)

# Load the sheet and drop non-numerical columns ('Sub-Sector' and 'Recommendation')
df = excel_data.parse('page-1_table-1')
df_numerical = df.drop(columns=['Sub-Sector', 'Recommendation'])

# missing values
df_numerical = df_numerical.dropna()

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numerical)

# Perform KMeans clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(scaled_data)

# Output the labels and cluster centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

print("Cluster Labels:\n", labels)
print("\nCluster Centers:\n", centers)
