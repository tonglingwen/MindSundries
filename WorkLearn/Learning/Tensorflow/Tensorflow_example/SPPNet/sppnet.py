import tensorflow as tf
import math
import numpy as np

def sppnet(data,pyramid=[4,2,1]):
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
		wstrides=(1 if wstrides==0 else wstrides)
		hksize=math.ceil(hi)
		hstrides=math.floor(hi)
		hstrides=(1 if hstrides==0 else hstrides)
		h_pool2=tf.nn.avg_pool(data, ksize=[1, wksize, hksize, 1],strides=[1, wstrides, hstrides, 1], padding='VALID')
		h_pool2=tf.reshape(h_pool2,[-1,dims*i*i])
		if index==0:
			result=h_pool2
		else:
			result=tf.concat([result,h_pool2],1)
		index=1
	return result


x = tf.placeholder("float", [None, 7,7,512])

print(sppnet(x,pyramid=[2,1]))