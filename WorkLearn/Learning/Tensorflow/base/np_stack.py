import numpy as np

arr1 = np.array([[1, 2, 3],[2,3,4]])
arr2 = np.array([[4, 5, 6],[7,8,9]])
res_v = np.vstack((arr1, arr2))#垂直方向堆叠矩阵
res_h = np.hstack((arr1,arr2))#水平方向堆叠矩阵

print("res_v:\n",res_v)
print("res_h:\n",res_h)