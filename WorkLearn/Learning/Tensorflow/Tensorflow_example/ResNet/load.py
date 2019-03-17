import tensorflow as tf


def residual_2(input,out_channel):
	channel =int(input.get_shape()[3])
	isChange=(channel!=out_channel[1])
	w_conv=tf.Variable(tf.truncated_normal([3,3,channel,out_channel[0]],stddev=0.0001))
	b_conv=tf.Variable(tf.constant(0.1,shape=[out_channel[0]]))
	if isChange:
		h_conv=tf.nn.relu(tf.nn.conv2d(input, w_conv, strides=[1, 2, 2, 1], padding='SAME')+b_conv)
	else:
		h_conv=tf.nn.relu(tf.nn.conv2d(input, w_conv, strides=[1, 1, 1, 1], padding='SAME')+b_conv)
	w_conv1=tf.Variable(tf.truncated_normal([3,3,out_channel[0],out_channel[1]],stddev=0.0001))
	b_conv1=tf.Variable(tf.constant(0.1,shape=[out_channel[1]]))
	h_conv1=tf.nn.relu(tf.nn.conv2d(h_conv, w_conv1, strides=[1, 1, 1, 1], padding='SAME')+b_conv1)
	
	if isChange:
		w_conv2=tf.Variable(tf.truncated_normal([1,1,channel,out_channel[1]],stddev=0.0001))
		#b_conv2=tf.Variable(tf.constant(0.1,shape=[out_channel[1]]))
		input=tf.nn.conv2d(input,w_conv2,strides=[1,2,2,1],padding='SAME')#+b_conv2
	return h_conv1+input
	

def residual_3(input,out_channel):
	channel=int(input.get_shape()[3])
	isChange=(channel!=out_channel[2])
	w_conv=tf.Variable(tf.truncated_normal([1,1,channel,out_channel[0]],stddev=0.0001))
	b_conv=tf.Variable(tf.constant(0.1,shape=[out_channel[0]]))
	if isChange:
		h_conv=tf.nn.relu(tf.nn.conv2d(input,w_conv,strides=[1,2,2,1],padding='SAME')+b_conv)
	else:
		h_conv=tf.nn.relu(tf.nn.conv2d(input,w_conv,strides=[1,1,1,1],padding='SAME')+b_conv)
	w_conv1=tf.Variable(tf.truncated_normal([3,3,out_channel[0],out_channel[1]],stddev=0.0001))
	b_conv1=tf.Variable(tf.constant(0.1,shape=[out_channel[1]]))
	h_conv1=tf.nn.relu(tf.nn.conv2d(h_conv,w_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)
	w_conv2=tf.Variable(tf.truncated_normal([1,1,out_channel[1],out_channel[2]],stddev=0.0001))
	b_conv2=tf.Variable(tf.constant(0.1,shape=[out_channel[2]]))
	h_conv2=tf.nn.relu(tf.nn.conv2d(h_conv1,w_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
	
	if isChange:
		w_conv3=tf.Variable(tf.truncated_normal([1,1,channel,out_channel[2]],stddev=0.0001))
		#b_conv3=tf.Variable(tf.constant(0.1,shape=[out_channel[1]]))
		input=tf.nn.conv2d(input,w_conv3,strides=[1,2,2,1],padding='SAME')#+b_conv3
	return h_conv2+input

input_d=tf.placeholder('float',[None,224,224,3])
w_conv=tf.Variable(tf.truncated_normal([7,7,3,64],stddev=0.0001))
b_conv=tf.constant(0.1,shape=[64])
h_conv=tf.nn.relu(tf.nn.conv2d(input_d,w_conv,strides=[1,2,2,1],padding='SAME')+b_conv)
h_pool1=tf.nn.max_pool(h_conv, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')

'''
h_residual1=residual_2(h_pool1,[64,64])
h_residual2=residual_2(h_residual1,[64,64])
h_residual3=residual_2(h_residual2,[64,64])

h_residual4=residual_2(h_residual3,[128,128])
h_residual5=residual_2(h_residual4,[128,128])
h_residual6=residual_2(h_residual5,[128,128])
h_residual7=residual_2(h_residual6,[128,128])

h_residual8=residual_2(h_residual7,[256,256])
h_residual9=residual_2(h_residual8,[256,256])
h_residual10=residual_2(h_residual9,[256,256])
h_residual11=residual_2(h_residual10,[256,256])
h_residual12=residual_2(h_residual11,[256,256])
h_residual13=residual_2(h_residual12,[256,256])

h_residual13=residual_2(h_residual12,[512,512])
h_residual14=residual_2(h_residual13,[512,512])
h_residual15=residual_2(h_residual14,[512,512])
'''

h_residual1=residual_3(h_pool1,[64,64,64])
h_residual2=residual_3(h_residual1,[64,64,64])
h_residual3=residual_3(h_residual2,[64,64,64])

h_residual4=residual_3(h_residual3,[128,128,128])
h_residual5=residual_3(h_residual4,[128,128,128])
h_residual6=residual_3(h_residual5,[128,128,128])
h_residual7=residual_3(h_residual6,[128,128,128])

h_residual8=residual_3(h_residual7,[256,256,256])
h_residual9=residual_3(h_residual8,[256,256,256])
h_residual10=residual_3(h_residual9,[256,256,256])
h_residual11=residual_3(h_residual10,[256,256,256])
h_residual12=residual_3(h_residual11,[256,256,256])
h_residual13=residual_3(h_residual12,[256,256,256])

h_residual13=residual_3(h_residual12,[512,512,512])
h_residual14=residual_3(h_residual13,[512,512,512])
h_residual15=residual_3(h_residual14,[512,512,512])

h_pool2=tf.nn.avg_pool(h_residual15, ksize=[1, 7, 7, 1],strides=[1, 1, 1, 1], padding='VALID')

h_fc=tf.reshape(h_pool2,[-1,512])

w_fc=tf.Variable(tf.truncated_normal([512,1000],stddev=0.01))
b_fc=tf.Variable(tf.constant(0.1,shape=[1000]))
h_fc=tf.matmul(h_fc,w_fc)+w_fc



print(h_fc)












	

