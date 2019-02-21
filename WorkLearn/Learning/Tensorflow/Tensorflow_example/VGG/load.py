import tensorflow as tf

x = tf.placeholder("float", [None, 224*224*3])
y_ = tf.placeholder("float", [None,1000])

input_data=tf.reshape(x, [-1,224,224,3])

w_conv1=tf.Variable(tf.truncated_normal([3,3,3,64], stddev=0.1))
b_conv1=tf.Variable(tf.constant(0.1, shape=[64]))
h_conv1=tf.nn.relu(tf.nn.conv2d(input_data, w_conv1, strides=[1, 1, 1, 1], padding='SAME')+b_conv1)
w_conv1_2=tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.1))
b_conv1_2=tf.Variable(tf.constant(0.1, shape=[64]))
h_conv1_2=tf.nn.relu(tf.nn.conv2d(h_conv1, w_conv1_2, strides=[1, 1, 1, 1], padding='SAME')+b_conv1_2)
h_pool1=tf.nn.max_pool(h_conv1_2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')
print("h_conv1:",h_conv1.shape)
print("h_pool1:",h_pool1.shape)

w_conv2=tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1, shape=[128]))
h_conv2=tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')+b_conv2)
w_conv2_2=tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.1))
b_conv2_2=tf.Variable(tf.constant(0.1, shape=[128]))
h_conv2_2=tf.nn.relu(tf.nn.conv2d(h_conv2, w_conv2_2, strides=[1, 1, 1, 1], padding='SAME')+b_conv2_2)
h_pool2=tf.nn.max_pool(h_conv2_2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')
print("h_conv2:",h_conv2.shape)
print("h_pool2:",h_pool2.shape)

w_conv3=tf.Variable(tf.truncated_normal([3,3,128,256], stddev=0.1))
b_conv3=tf.Variable(tf.constant(0.1, shape=[256]))
h_conv3=tf.nn.relu(tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME')+b_conv3)
w_conv3_2=tf.Variable(tf.truncated_normal([3,3,256,256], stddev=0.1))
b_conv3_2=tf.Variable(tf.constant(0.1, shape=[256]))
h_conv3_2=tf.nn.relu(tf.nn.conv2d(h_conv3, w_conv3_2, strides=[1, 1, 1, 1], padding='SAME')+b_conv3_2)
w_conv3_3=tf.Variable(tf.truncated_normal([3,3,256,256], stddev=0.1))
b_conv3_3=tf.Variable(tf.constant(0.1, shape=[256]))
h_conv3_3=tf.nn.relu(tf.nn.conv2d(h_conv3_2, w_conv3_3, strides=[1, 1, 1, 1], padding='SAME')+b_conv3_3)
h_pool3=tf.nn.max_pool(h_conv3_3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')
print("h_conv3:",h_conv3.shape)
print("h_pool3:",h_pool3.shape)

w_conv4=tf.Variable(tf.truncated_normal([3,3,256,512], stddev=0.1))
b_conv4=tf.Variable(tf.constant(0.1, shape=[512]))
h_conv4=tf.nn.relu(tf.nn.conv2d(h_pool3, w_conv4, strides=[1, 1, 1, 1], padding='SAME')+b_conv4)
w_conv4_2=tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.1))
b_conv4_2=tf.Variable(tf.constant(0.1, shape=[512]))
h_conv4_2=tf.nn.relu(tf.nn.conv2d(h_conv4, w_conv4_2, strides=[1, 1, 1, 1], padding='SAME')+b_conv4_2)
w_conv4_3=tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.1))
b_conv4_3=tf.Variable(tf.constant(0.1, shape=[512]))
h_conv4_3=tf.nn.relu(tf.nn.conv2d(h_conv4_2, w_conv4_3, strides=[1, 1, 1, 1], padding='SAME')+b_conv4_3)
h_pool4=tf.nn.max_pool(h_conv4_3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')
print("h_conv4:",h_conv4.shape)
print("h_pool4:",h_pool4.shape)

w_conv5=tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.1))
b_conv5=tf.Variable(tf.constant(0.1, shape=[512]))
h_conv5=tf.nn.relu(tf.nn.conv2d(h_pool4, w_conv5, strides=[1, 1, 1, 1], padding='SAME')+b_conv5)
w_conv5_2=tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.1))
b_conv5_2=tf.Variable(tf.constant(0.1, shape=[512]))
h_conv5_2=tf.nn.relu(tf.nn.conv2d(h_conv5, w_conv5_2, strides=[1, 1, 1, 1], padding='SAME')+b_conv5_2)
w_conv5_3=tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.1))
b_conv5_3=tf.Variable(tf.constant(0.1, shape=[512]))
h_conv5_3=tf.nn.relu(tf.nn.conv2d(h_conv5_2, w_conv5_3, strides=[1, 1, 1, 1], padding='SAME')+b_conv5_3)
h_pool5=tf.nn.max_pool(h_conv5_3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')
h_fullconv=tf.reshape(h_pool5,[-1,7*7*512])
print("h_conv5:",h_conv5.shape)
print("h_pool5:",h_pool5.shape)
print("h_fullconv",h_fullconv.shape)

w_full6=tf.Variable(tf.truncated_normal([7*7*512,4096], stddev=0.1))
b_full6=tf.Variable(tf.constant(0.1, shape=[4096]))
h_full6=tf.nn.relu(tf.matmul(h_fullconv,w_full6)+b_full6)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_full6, keep_prob)
print("h_full6:",h_full6.shape)

w_full7=tf.Variable(tf.truncated_normal([4096,4096], stddev=0.1))
b_full7=tf.Variable(tf.constant(0.1, shape=[4096]))
h_full7=tf.nn.relu(tf.matmul(h_fc1_drop,w_full7)+b_full7)
print("h_full7:",h_full7.shape)

w_softmax=tf.Variable(tf.truncated_normal([4096,1000], stddev=0.1))
b_softmax=tf.Variable(tf.constant(0.1, shape=[1000]))
y_conv=tf.nn.softmax(tf.matmul(h_full7, w_softmax) + b_softmax)
print("y_conv:",y_conv.shape)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#for i in range(20000):#训练过程
#  batch = mnist.train.next_batch(50)
#  sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})







