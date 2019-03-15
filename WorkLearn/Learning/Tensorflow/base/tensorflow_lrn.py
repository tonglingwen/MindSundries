import tensorflow as tf
import numpy as np

con= tf.constant(np.random.randint(-1,3,(1,100)),shape=[1,10,10,1],dtype=tf.float32)
con1=tf.truncated_normal([3,3,1,20], stddev=0.1)
var=tf.Variable(con1)

matual=tf.nn.conv2d(con, var, strides=[1, 4, 4, 1], padding='VALID')

localrn= tf.nn.local_response_normalization(matual,alpha=0.001/9.0,beta=0.75)
lrn= tf.nn.lrn(matual,alpha=0.001/9.0,beta=0.75)
softmax=tf.nn.softmax(var)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
print("localrn",sess.run(localrn))
print("lrn",sess.run(lrn))
#print(sess.run(con1))
