import readImageNet
import tensorflow as tf
import numpy as np  
import os
import matplotlib.pyplot as plt
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.set_printoptions(threshold=np.inf)  

dataset = readImageNet.ImageNetDataSet('F:/ILSVRC2012_dataset/image_train')#加载图片根目录
dataset.get_labels('train_label_origin.txt')
image_batch,label_batch = dataset.get_batch_data()



x = tf.placeholder("float", [None, 227*227*3])
y_ = tf.placeholder("float", [None,1000])

input_data=tf.reshape(x, [-1,227,227,3])

w_conv1=tf.Variable(tf.truncated_normal([11,11,3,96], stddev=0.1))
b_conv1=tf.Variable(tf.constant(0.1, shape=[96]))
h_conv1=tf.nn.relu(tf.nn.conv2d(input_data, w_conv1, strides=[1, 4, 4, 1], padding='VALID')+b_conv1)
h_pool1=tf.nn.local_response_normalization(tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID'))
print("h_conv1:",h_conv1.shape)
print("h_pool1:",h_pool1.shape)

w_conv2=tf.Variable(tf.truncated_normal([5,5,96,256], stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1, shape=[256]))
h_conv2=tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')+b_conv2)
h_pool2=tf.nn.local_response_normalization(tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID'))
print("h_conv2:",h_conv2.shape)
print("h_pool2:",h_pool2.shape)

w_conv3=tf.Variable(tf.truncated_normal([3,3,256,384], stddev=0.1))
b_conv3=tf.Variable(tf.constant(0.1, shape=[384]))
h_conv3=tf.nn.relu(tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME')+b_conv3)
print("h_conv3:",h_conv3.shape)

w_conv4=tf.Variable(tf.truncated_normal([3,3,384,384], stddev=0.1))
b_conv4=tf.Variable(tf.constant(0.1, shape=[384]))
h_conv4=tf.nn.relu(tf.nn.conv2d(h_conv3, w_conv4, strides=[1, 1, 1, 1], padding='SAME')+b_conv4)
print("h_conv4:",h_conv4.shape)

w_conv5=tf.Variable(tf.truncated_normal([3,3,384,256], stddev=0.1))
b_conv5=tf.Variable(tf.constant(0.1, shape=[256]))
h_conv5=tf.nn.relu(tf.nn.conv2d(h_conv4, w_conv5, strides=[1, 1, 1, 1], padding='SAME')+b_conv5)
h_pool5=tf.nn.max_pool(h_conv5, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID')
h_fullconv=tf.reshape(h_pool5,[-1,6*6*256])
print("h_conv5:",h_conv5.shape)
print("h_pool5:",h_pool5.shape)
print("h_fullconv:",h_fullconv.shape)

w_full6=tf.Variable(tf.truncated_normal([6*6*256,4096], stddev=0.1))
b_full6=tf.Variable(tf.constant(0.1, shape=[4096]))
h_full6=tf.nn.relu(tf.matmul(h_fullconv,w_full6)+b_full6)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_full6, keep_prob)
print("h_full6:",h_full6.shape)

w_full7=tf.Variable(tf.truncated_normal([4096,4096], stddev=0.1))
b_full7=tf.Variable(tf.constant(0.1, shape=[4096]))
h_full7=tf.nn.relu(tf.matmul(h_fc1_drop,w_full7)+b_full7)
h_fc2_drop = tf.nn.dropout(h_full7, keep_prob)
print("h_full7:",h_full7.shape)

w_softmax=tf.Variable(tf.truncated_normal([4096,1000], stddev=0.1))
b_softmax=tf.Variable(tf.constant(0.1, shape=[1000]))
y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, w_softmax) + b_softmax)
print("y_conv:",y_conv.shape)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


with tf.Session() as sess:
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
	sess.run(init_op)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	for i in range(26000):#训练过程
		try:
			image_v,label_v=sess.run([image_batch,label_batch])
			if i%100 == 0:
				train_accuracy = accuracy.eval(session=sess,feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
				print("step %d, training accuracy %g"%(i, train_accuracy))
			train_step.run(session=sess,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		except BaseException:
			pass
		else:
			pass
	coord.request_stop()
	coord.join(threads)






