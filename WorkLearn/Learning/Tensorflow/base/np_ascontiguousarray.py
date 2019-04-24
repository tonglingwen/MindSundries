import numpy as np

array=np.arange(6).reshape(2,3)

#np.ascontiguousarray(array,dtype=np.float32)


print (array.flags['C_CONTIGUOUS'])