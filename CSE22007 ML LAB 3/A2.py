import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_excel(r'Profitability Ratio.xlsx', sheet_name='page-1_table-1')
mat = df['ROCE'].values

# Example dataset
data = mat  # Replace with your feature data

# Calculate histogram
hist, bins = np.histogram(data, bins=30)  # 30 bins as an example
print("Histogram frequencies - ")
print(hist)
print()
print("Bins x axis points - ")
print(bins)
print()

# Plot histogram
plt.hist(data, bins=30, edgecolor='black')
plt.title('Histogram of Feature')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Calculate mean and variance
mean = np.mean(data)
variance = np.var(data)

print("Mean:", mean)
print("Variance:", variance)
plt.show()
