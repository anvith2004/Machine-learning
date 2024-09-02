import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(scaled_data)

# Calculate Clustering Scores
sil_score = silhouette_score(scaled_data, kmeans.labels_)
ch_score = calinski_harabasz_score(scaled_data, kmeans.labels_)
db_score = davies_bouldin_score(scaled_data, kmeans.labels_)

print(f"Silhouette Score: {sil_score}")
print(f"Calinski-Harabasz Score: {ch_score}")
print(f"Davies-Bouldin Index: {db_score}")
