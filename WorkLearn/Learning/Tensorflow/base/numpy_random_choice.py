import numpy.random as npr

#res = npr.choice(5)

res = npr.choice(5,size=[40,4],replace=True,p=[0.2,0.01,0.01,0.01,1-(0.2+0.01+0.01+0.01)]) #根据要求随机取数
print("res:\n",res)