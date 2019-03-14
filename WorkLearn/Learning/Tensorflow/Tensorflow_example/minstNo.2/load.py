import input_data
import tensorflow as tf
import numpy as np  
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.set_printoptions(threshold=np.inf) 

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder("float", [None, 784])
#with tf.variable_scope('input1'):
#W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,10])


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
tf.summary.histogram('histogram/',W_conv1)#绘制权重的直方图，用来统计各个权重的出现频率用正态分布拟合
tf.summary.histogram("x",x)
x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#第一层卷积操作
tf.summary.image('image/',h_conv1[:,:,:,0:3])#像素图，绘制张量的像素图最后一维必须为1，3或者4（对应灰度图，rgb图以及rgba图）
h_pool1 = max_pool_2x2(h_conv1)                         #第一层池化层
tf.summary.histogram("W_conv1",W_conv1)
tf.summary.histogram("h_conv1",h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)#第二层卷积操作
h_pool2 = max_pool_2x2(h_conv2)                         #第二层池化层
print(h_pool2.shape)
tf.summary.histogram("W_conv2",W_conv2)
tf.summary.histogram("h_conv2",h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)#全连接层
tf.summary.histogram("W_fc1",W_fc1)
tf.summary.histogram("h_fc1",h_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)              #dropout防止过拟合

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
h_fc2=tf.matmul(h_fc1_drop, W_fc2) + b_fc2
tf.summary.histogram("W_fc2",W_fc2)
tf.summary.histogram("h_fc2",h_fc2)

y_conv=tf.nn.softmax(h_fc2)#将全连接层载入分类器softmax


cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))         #构建损失函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#设置损失函数的最小化方法
#print("start")
#print(train_step)
#print("end")
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar("accuracy/",accuracy)#绘制标量图每次保存一个 比如每次训练后的正确率
writer = tf.summary.FileWriter("log/")#创建summary
summaries = tf.summary.merge_all()#整合所有要绘制的图形
sess = tf.Session()
sess.run(tf.global_variables_initializer())
getw_conv1= tf.reshape(W_conv1,[5,5*32])
asd=[]
for i in range(1):#训练过程
	batch = mnist.train.next_batch(1)
	print("batch:",batch[1])
	#if i%100 == 0:
	#	train_accuracy = accuracy.eval(session=sess,feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
	#	print("step %d, training accuracy %g"%(i, train_accuracy))
	summ,train,y_convyy= sess.run([summaries,train_step,h_fc1_drop],feed_dict={x: np.random.randint(-1,3,(1,784)), y_: batch[1], keep_prob: 0.5})
	print("max:",y_convyy.max()," min:",y_convyy.min())
	writer.add_summary(summ, global_step=i)#将本次的整合后的图形添加到summary中
#	if i==0:
#		np.savetxt('log/test.out', sess.run(getw_conv1), delimiter=',')
#	sdd=sess.run(W_fc2)
#	print(batch[0])
#	if len(asd)==0:
#		asd=sdd
#	if (asd==sdd).all():
#		print('==')
#	else:
#		print('!=')
#	asd=sdd
  #print(sess.run(h_conv2,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}).shape)
writer.close()
#tf.histogram_summary(layer_name + '/pre_activations', preactivate)
print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))












































