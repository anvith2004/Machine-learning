import numpy as np
import pandas as pd

df = pd.read_excel('Lab Session Data.xlsx', sheet_name='Purchase data')
data = df.iloc[:,1:5]

# Segregate the data into matrices A and C
A = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
C = df[["Payment (Rs)"]].values

# Compute the pseudo-inverse of A
A_pseudo_inv = np.linalg.pinv(A)

# Compute the model vector X by multiplying the pseudo-inverse of A with C
X = A_pseudo_inv @ C

# Display the model vector X
print("Model vector X (estimated cost of each product):")
print(X)
