import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import minkowski

df = pd.read_excel(r'Profitability Ratio.xlsx', sheet_name='page-1_table-1')
vector1 = df['Return on Assets'].values
vector2 = df['EPS (Q)'].values


# Calculate Minkowski distances for r from 1 to 10
r_values = range(1, 11)
distances = [minkowski(vector1, vector2, p) for p in r_values]
for i, j in enumerate(distances):
    print('r =', i+1, ', distance =', j)

# Plotting the distances
plt.plot(r_values, distances, marker='o')
plt.title('Minkowski Distance vs. r')
plt.xlabel('r')
plt.ylabel('Minkowski Distance')
plt.grid(True)
plt.show()
