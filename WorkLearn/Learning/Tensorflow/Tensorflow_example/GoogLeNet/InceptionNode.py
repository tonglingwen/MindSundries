import tensorflow as tf


def inception_v1(para,branch1=64,branch21=96,branch22=128,branch31=16,branch32=32,branch4=32):
	channel=tf.to_int32(para.shape[3])
	
	w_conv1=tf.Variable(tf.truncated_normal([1,1,channel,branch1], stddev=0.1))
	b_conv1=tf.Variable(tf.constant(0.1, shape=[branch1]))
	h_conv1=tf.nn.relu(tf.nn.conv2d(para, w_conv1, strides=[1, 1, 1, 1], padding='SAME')+b_conv1)
	print("inception:h_conv1",h_conv1)
	
	w_conv21=tf.Variable(tf.truncated_normal([1,1,channel,branch21],stddev=0.1))
	b_conv21=tf.Variable(tf.constant(0.1,shape=[branch21]))
	h_conv21=tf.nn.relu(tf.nn.conv2d(para, w_conv21, strides=[1,1,1,1], padding='SAME')+b_conv21)
	w_conv22=tf.Variable(tf.truncated_normal([3,3,branch21,branch22]))
	b_conv22=tf.Variable(tf.constant(0.1,shape=[branch22]))
	h_conv22=tf.nn.relu(tf.nn.conv2d(h_conv21,w_conv22,strides=[1,1,1,1],padding='SAME')+b_conv22)
	print("inception::h_conv21",h_conv21)
	print("inception::h_conv22",h_conv22)
	
	w_conv31=tf.Variable(tf.truncated_normal([1,1,channel,branch31],stddev=0.1))
	b_conv31=tf.Variable(tf.constant(0.1,shape=[branch31]))
	h_conv31=tf.nn.relu(tf.nn.conv2d(para,w_conv31,strides=[1,1,1,1],padding='SAME')+b_conv31)
	w_conv32=tf.Variable(tf.truncated_normal([5,5,branch31,branch32]))
	b_conv32=tf.Variable(tf.constant(0.1,shape=[branch32]))
	h_conv32=tf.nn.relu(tf.nn.conv2d(h_conv31,w_conv32,strides=[1,1,1,1],padding='SAME')+b_conv32)
	print("inception::h_conv31",h_conv31)
	print("inception::h_conv32",h_conv32)
	
	h_pool = tf.nn.max_pool(para, ksize=[1, 3, 3, 1],strides=[1, 1, 1, 1], padding='SAME')
	w_conv4=tf.Variable(tf.truncated_normal([1,1,channel,branch4]))
	b_conv4=tf.Variable(tf.constant(0.1,shape=[branch4]))
	h_conv4=tf.nn.relu(tf.nn.conv2d(h_pool,w_conv4,strides=[1,1,1,1],padding='SAME')+b_conv4)
	print("inception::h_pool",h_pool)
	print("inception::h_conv4",h_conv4)
	
	return tf.concat([h_conv1,h_conv22,h_conv32,h_conv4],3)