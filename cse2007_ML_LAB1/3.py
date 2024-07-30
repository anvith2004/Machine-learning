import numpy as np
def matrix_power(list, n):
    if n<1:
        return "m is a positive num"
    result = np.linalg.matrix_power(list,n)
    return result

list=np.array([[1,2], [3,4]])
n=3
result = matrix_power(list,n)
print(result)
