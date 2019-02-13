import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,10])
sess = tf.Session()

#卷积操作
def conv2d(x, W):                                                             
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #卷积操作x为输入的图片矩阵，W为卷积核，strides为在每一个维度的步幅，padding为扫面图片的方式
#其中x的结构为一个四维向量[batch, in_height, in_width, in_channels]各个参数含义为[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
#其中W的结构为一个四维向量[filter_height, filter_width, in_channels, out_channels]各个参数含义为[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')#池化操作x为输入的图片矩阵，ksize为池化窗口的大小，strides为在每一个维度的步幅，padding为扫面图片的方式

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
						
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#第一层卷积操作
h_pool1 = max_pool_2x2(h_conv1)                         #第一层池化层

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#第二层卷积操作
h_pool2 = max_pool_2x2(h_conv2)                         #第二层池化层

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)#全连接层

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)              #dropout防止过拟合

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)#将全连接层载入分类器softmax
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))         #构建损失函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#设置损失函数的最小化方法
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())
for i in range(20000):#训练过程
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(session=sess,feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(session=sess,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))












































