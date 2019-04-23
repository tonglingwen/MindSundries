import numpy as np

x=np.random.randn(4,4)
print("x:\n",x)
x_result=np.where(x>0,2,-2) #判断x中的所有元素是否>0为真输出2否则输出-2
print("np.where(x>0,2,-2):\n",x_result)
x_result=np.where(x>0)
print("np.where(x>0):\n",x_result)#判断x中的所有元素是否>0为真输出所在索引
x_result=np.where([[True,False],[True,True]],[[1,2],[3,4]],[[9,8],[7,6]])#np.where(x>0,2,-2)的一般情形
print("np.where([[True,False],[True,True]],[[1,2],[3,4]],[[9,8],[7,6]]):\n",x_result)
