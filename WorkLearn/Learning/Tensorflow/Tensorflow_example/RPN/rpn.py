import tensorflow as tf

def rpn(data):
	return ""

def conv_layer(data):
	result={}
	w_conv=tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.01)
	b_conv=tf.Variable(tf.constant(0.1, shape=[512]))
	h_conv=tf.nn.conv2d(data,w_conv,strides=[1, 1, 1, 1], padding='SAME')+b_conv
	h_conv=tf.nn.relu(h_conv)
	
	w_conv_cls=tf.Variable(tf.truncated_normal([1,1,512,2], stddev=0.01)
	b_conv_cls=tf.Variable(tf.constant(0.1, shape=[2]))
	h_conv_cls=tf.nn.conv2d(h_conv,w_conv_cls,strides=[1, 1, 1, 1], padding='SAME')+b_conv_cls
	h_conv_cls=tf.nn.relu(h_conv_cls)
	
	w_conv_bbox=tf.Variable(tf.truncated_normal([1,1,512,4], stddev=0.01)
	b_conv_bbox=tf.Variable(tf.constant(0.1, shape=[4]))
	h_conv_bbox=tf.nn.conv2d(h_conv,w_conv_bbox,strides=[1, 1, 1, 1], padding='SAME')+b_conv_bbox
	h_conv_bbox=tf.nn.relu(h_conv_bbox)
	
	result["cls"]=h_conv_cls
	result["bbox"]=h_conv_bbox
	
	return result


def def0():
	
	return ""












