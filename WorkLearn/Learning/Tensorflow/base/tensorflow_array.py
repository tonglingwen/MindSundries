import numpy as np

na=np.array([1,2,3,4,2,4,6,1])
nb=np.array([2,1,-3,-4,2,5,6,-9])

na[nb<0]=0



print(na)
