import numpy as np

array = np.arange(10)

res_repeat= array.repeat(2)
res_tile=np.tile(array,2)

print("res_repeat:\n",res_repeat)
print("res_tile:\n",res_tile)