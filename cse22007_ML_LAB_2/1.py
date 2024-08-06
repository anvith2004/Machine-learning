import pandas as pd
import numpy as np

df = pd.read_excel('Lab Session Data.xlsx', sheet_name='Purchase data')
data = df.iloc[:,1:5]
print(data)

#Dimensionality
rank = pd.DataFrame(data)
rows, colums = data.shape
print("No of rows :", rows)
print("No of columns : ", colums)

#Rank
rank = np.linalg.matrix_rank(df.iloc[:,1:5])
print("The Dimensionality : ",rank)
num_vectors = data.shape[0]


A = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
C = df[["Payment (Rs)"]].values
print(A)
print(C)

A_pseudo_inv = np.linalg.pinv(A)
cost_vector = A_pseudo_inv @ C
print("Cost of product for Candies, Mangoes, Milk Packets : ")
print(cost_vector)
print("Vectors : ", num_vectors)
