import tensorflow as tf

def sppnet(data,pyramid=[3,2,1]):
	result=[]
	w=tf.to_float(data.shape[1])
	h=tf.to_float(data.shape[2])
	dims=tf.to_int32(data.shape[3])
	for i in pyramid:
		wi=w/tf.to_float(i)
		hi=w/tf.to_float(i)
		wksize=tf.to_int32(tf.math.ceil(wi))
		wstrides=tf.math.floor(wi)
		hksize=tf.math.ceil(hi)
		hstrides=tf.math.floor(hi)
		h_pool2=tf.nn.avg_pool(data, ksize=[1, wksize, hksize, 1],strides=[1, wstrides, hstrides, 1], padding='VALID')
		h_pool2=tf.reshape(h_pool2,[-1,dims*i*i])
		tf.concat([result,h_pool2],0)
	return result



x = tf.placeholder("float", [None, 276,657,3])

print(sppnet(x))