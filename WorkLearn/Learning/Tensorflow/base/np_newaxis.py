import numpy as np

arr1 = np.array([[1,2,3,2,3,4],[2,3,5,4,3,5]])
result=arr1[:,np.newaxis]#维度扩展


print("arr1.shape:\n",arr1.shape)
print("arr1:\n",arr1)
print("result.shape:\n",result.shape)
print("result:\n",result)