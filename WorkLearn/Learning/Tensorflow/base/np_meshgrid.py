import numpy as np

array1=np.array([1,2,3,4,5,6])
array2=np.array([2,4,6])

res_array1,res_array2 = np.meshgrid(array1,array2)

print("res_array1:\n",res_array1)
print("res_array2:\n",res_array2)