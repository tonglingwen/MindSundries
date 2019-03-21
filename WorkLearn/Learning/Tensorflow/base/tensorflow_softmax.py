import tensorflow as tf
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.set_printoptions(threshold=np.inf) 

con= tf.constant([1.3,10,13.3],shape=[3, 3],dtype=tf.float32)
var=tf.Variable(con)

softmax=tf.nn.softmax(var)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
softmax_out,var_out=sess.run([softmax,var])

softmax_numpy_out=np.zeros(var_out.shape)
for i in range(var_out.shape[1]):
	softmax_numpy_out[i]=(np.exp(var_out)/np.sum(np.exp(var_out),axis=1)[i])[i]
	#print(np.exp(var_out[i:])/np.sum(np.exp(var_out),axis=1)[i])

#softmax_numpy_out=np.exp(var_out)/np.sum(np.exp(var_out),axis=1)#softmax的具体实现
print("var:\n",var_out)
print("softmax:\n",softmax_out)
print("softmax_numpy:\n",softmax_numpy_out)
