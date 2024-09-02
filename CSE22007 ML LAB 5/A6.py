import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Load Excel Data
file_path = 'Profitability Ratio.xlsx'
excel_data = pd.ExcelFile(file_path)

# Load the sheet and drop non-numerical columns ('Sub-Sector' and 'Recommendation')
df = excel_data.parse('page-1_table-1')
df_numerical = df.drop(columns=['Sub-Sector', 'Recommendation'])

# Handle missing values
df_numerical = df_numerical.dropna()

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numerical)

# Define a range of k values
k_values = range(2, 11)

# Lists to store scores
sil_scores = []
ch_scores = []
db_scores = []

# Perform KMeans for each k value and compute the evaluation metrics
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled_data)

    # Calculate metrics
    sil_score = silhouette_score(scaled_data, kmeans.labels_)
    ch_score = calinski_harabasz_score(scaled_data, kmeans.labels_)
    db_score = davies_bouldin_score(scaled_data, kmeans.labels_)

    # Append the scores
    sil_scores.append(sil_score)
    ch_scores.append(ch_score)
    db_scores.append(db_score)

# Plot the scores against the k values
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.plot(k_values, sil_scores, marker='o', label='Silhouette Score', color='b')
plt.title('Silhouette Score vs k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(k_values, ch_scores, marker='o', label='CH Score', color='g')
plt.title('Calinski-Harabasz Score vs k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Calinski-Harabasz Score')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(k_values, db_scores, marker='o', label='DB Index', color='r')
plt.title('Davies-Bouldin Index vs k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin Index')
plt.grid(True)

plt.tight_layout()
plt.show()
