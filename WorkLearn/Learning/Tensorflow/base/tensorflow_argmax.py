import tensorflow as tf
import numpy as np

a=tf.placeholder(tf.float32,shape=[None,2])

result=tf.argmax(a,axis=1)#获取较大值的索引按列axis=0或按行axis=1

sess=tf.Session()
f= np.array([[2,1],[0,10],[10,45]])
print(sess.run(result,feed_dict={a:f}))