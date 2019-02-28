import tensorflow as tf
import InceptionNode

x = tf.placeholder("float", [None, 224*224*3])
y_ = tf.placeholder("float", [None,1000])

input_data=tf.reshape(x, [-1,224,224,3])

w_conv1=tf.Variable(tf.truncated_normal([7,7,3,64], stddev=0.1))
b_conv1=tf.Variable(tf.constant(0.1,shape=[64]))
h_conv1=tf.nn.relu(tf.nn.conv2d(input_data,w_conv1,strides=[1,2,2,1],padding='VALID')+b_conv1)
h_pool1=tf.nn.relu(tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID'))
print("h_conv1",h_conv1)
print("h_pool1",h_pool1)

w_conv2=tf.Variable(tf.truncated_normal([3,3,64,192], stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1,shape=[192]))
h_conv2=tf.nn.relu(tf.nn.conv2d(h_pool1,w_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
h_pool2=tf.nn.relu(tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID'))
print("h_conv2",h_conv2)
print("h_pool2",h_pool2)

h_inception1=InceptionNode.inception_v1(h_pool2)
print("h_inception1",h_inception1)

h_inception2=InceptionNode.inception_v1(h_inception1,128,128,192,32,96,64)
print("h_inception2",h_inception2)
