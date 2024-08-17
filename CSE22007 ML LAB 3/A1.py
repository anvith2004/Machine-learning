import numpy as np
import pandas as pd

df = pd.read_excel(r'Profitability Ratio.xlsx', sheet_name='page-1_table-1')
df_input = df.drop(['Recommendation', 'Sub-Sector'], axis=1)

df_output = df['Recommendation']
mat_input = df_input.values
mat_output = df_output.values
classes = np.unique(mat_output)

# Calculate mean for each class
centroids = {}
for cls in classes:
    class_data = df_input[mat_output == cls]
    centroids[cls] = np.mean(class_data, axis=0)
print("Centroids:")
print(centroids)
print()

# Calculate spread (standard deviation) for each class
spreads = {}
for cls in classes:
    class_data = df_input[mat_output == cls]
    spreads[cls] = np.std(class_data, axis=0)
print("Class Spreads:")
print(spreads)
print()

# Calculate the distance between mean vectors between classes

for i in range(len(classes)):
    for j in range(i + 1, len(classes)):
        centroid1 = centroids[classes[i]]
        centroid2 = centroids[classes[j]]
        distance = np.linalg.norm(centroid1 - centroid2)
        print(f"Distance between class {classes[i]} and {classes[j]}: {distance}")
