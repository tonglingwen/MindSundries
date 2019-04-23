import tensorflow as tf
import numpy as np

def tile_py_func(a,b):
	return np.tile(a,(b.shape[0],1))

a=tf.placeholder(tf.float32,shape=[1,2])
b=tf.placeholder(tf.float32,shape=[None,2])

#tile_a=tf.tile(a,[b.get_shape()[0],1])           #使用tensorflow的接口会报错因为未知维度
tile_a= tf.py_func(tile_py_func,[a,b],tf.float32) #未知维度时

sess = tf.Session() 
array_a = np.array([[1., 2.]])  
array_b = np.array([[3., 4.],[5., 6.],[7., 8.]])
tile_a_value = sess.run(tile_a, feed_dict = {a:array_a,b:array_b})
print(tile_a_value)
