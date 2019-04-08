import tensorflow as tf
import math
import numpy as np

def sppnet(data,pyramid=[3,2,1]):
	result=[]
	shape= data.get_shape().as_list()
	w=shape[1]
	h=shape[2]
	dims=shape[3]
	index=0
	for i in pyramid:
		wi=w/i
		hi=h/i
		wksize=math.ceil(wi)
		wstrides=math.floor(wi)
		hksize=math.ceil(hi)
		hstrides=math.floor(hi)
		h_pool2=tf.nn.avg_pool(data, ksize=[1, wksize, hksize, 1],strides=[1, wstrides, hstrides, 1], padding='VALID')
		h_pool2=tf.reshape(h_pool2,[-1,dims*i*i])
		if index==0:
			result=h_pool2
		else:
			result=tf.concat([result,h_pool2],1)
		index=1
	return result


x = tf.placeholder("float", [None, 3,3,512])

print(sppnet(x))