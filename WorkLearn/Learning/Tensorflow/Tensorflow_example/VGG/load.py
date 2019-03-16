import tensorflow as tf
import kaggleCatDogLoad
import numpy as np
import imagenet_classes
from scipy.misc import imread,imresize

ClassNum=1000
ImagePath='F:/kaggle_cat_dog_dataset/train'
LabelPath='train_label_origin.txt'
SavePath='./model/AlexNetModel.ckpt'
PathVGG='F:/vgg16_weights/vgg16_weights.npz'
BatchSize=1
#print(imagenet_classes.class_names)
dataset = kaggleCatDogLoad.ImageNetDataSet(ImagePath,BatchSize)#加载图片根目录
dataset.get_labels()
image_batch,label_batch = dataset.get_batch_data()

vgg_dcit=np.load(PathVGG)

x = tf.placeholder("float", [None, 224,224,3])
y_ = tf.placeholder("float", [None,ClassNum])

input_data=x#tf.reshape(x, [-1,224,224,3])

w_conv1=tf.Variable(vgg_dcit["conv1_1_W"])#tf.truncated_normal([3,3,3,64], stddev=0.1)
b_conv1=tf.Variable(vgg_dcit["conv1_1_b"])#tf.constant(0.1, shape=[64])
h_conv1=tf.nn.relu(tf.nn.conv2d(input_data, w_conv1, strides=[1, 1, 1, 1], padding='SAME')+b_conv1)
w_conv1_2=tf.Variable(vgg_dcit["conv1_2_W"])#tf.truncated_normal([3,3,64,64], stddev=0.1)
b_conv1_2=tf.Variable(vgg_dcit["conv1_2_b"])#tf.constant(0.1, shape=[64])
h_conv1_2=tf.nn.relu(tf.nn.conv2d(h_conv1, w_conv1_2, strides=[1, 1, 1, 1], padding='SAME')+b_conv1_2)
h_pool1=tf.nn.max_pool(h_conv1_2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
print("h_conv1:",h_conv1.shape)
print("h_pool1:",h_pool1.shape)

w_conv2=tf.Variable(vgg_dcit["conv2_1_W"])#tf.truncated_normal([3,3,64,128], stddev=0.1)
b_conv2=tf.Variable(vgg_dcit["conv2_1_b"])#tf.constant(0.1, shape=[128])
h_conv2=tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')+b_conv2)
w_conv2_2=tf.Variable(vgg_dcit["conv2_2_W"])#tf.truncated_normal([3,3,128,128], stddev=0.1)
b_conv2_2=tf.Variable(vgg_dcit["conv2_2_b"])#tf.constant(0.1, shape=[128])
h_conv2_2=tf.nn.relu(tf.nn.conv2d(h_conv2, w_conv2_2, strides=[1, 1, 1, 1], padding='SAME')+b_conv2_2)
h_pool2=tf.nn.max_pool(h_conv2_2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
print("h_conv2:",h_conv2.shape)
print("h_pool2:",h_pool2.shape)

w_conv3=tf.Variable(vgg_dcit["conv3_1_W"])#tf.truncated_normal([3,3,128,256], stddev=0.1)
b_conv3=tf.Variable(vgg_dcit["conv3_1_b"])#tf.constant(0.1, shape=[256])
h_conv3=tf.nn.relu(tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME')+b_conv3)
w_conv3_2=tf.Variable(vgg_dcit["conv3_2_W"])#tf.truncated_normal([3,3,256,256], stddev=0.1)
b_conv3_2=tf.Variable(vgg_dcit["conv3_2_b"])#tf.constant(0.1, shape=[256])
h_conv3_2=tf.nn.relu(tf.nn.conv2d(h_conv3, w_conv3_2, strides=[1, 1, 1, 1], padding='SAME')+b_conv3_2)
w_conv3_3=tf.Variable(vgg_dcit["conv3_3_W"])#tf.truncated_normal([3,3,256,256], stddev=0.1)
b_conv3_3=tf.Variable(vgg_dcit["conv3_3_b"])#tf.constant(0.1, shape=[256])
h_conv3_3=tf.nn.relu(tf.nn.conv2d(h_conv3_2, w_conv3_3, strides=[1, 1, 1, 1], padding='SAME')+b_conv3_3)
h_pool3=tf.nn.max_pool(h_conv3_3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
print("h_conv3:",h_conv3.shape)
print("h_pool3:",h_pool3.shape)

w_conv4=tf.Variable(vgg_dcit["conv4_1_W"])#tf.truncated_normal([3,3,256,512], stddev=0.1)
b_conv4=tf.Variable(vgg_dcit["conv4_1_b"])#tf.constant(0.1, shape=[512])
h_conv4=tf.nn.relu(tf.nn.conv2d(h_pool3, w_conv4, strides=[1, 1, 1, 1], padding='SAME')+b_conv4)
w_conv4_2=tf.Variable(vgg_dcit["conv4_2_W"])#tf.truncated_normal([3,3,512,512], stddev=0.1)
b_conv4_2=tf.Variable(vgg_dcit["conv4_2_b"])#tf.constant(0.1, shape=[512])
h_conv4_2=tf.nn.relu(tf.nn.conv2d(h_conv4, w_conv4_2, strides=[1, 1, 1, 1], padding='SAME')+b_conv4_2)
w_conv4_3=tf.Variable(vgg_dcit["conv4_3_W"])#tf.truncated_normal([3,3,512,512], stddev=0.1)
b_conv4_3=tf.Variable(vgg_dcit["conv4_3_b"])#tf.constant(0.1, shape=[512])
h_conv4_3=tf.nn.relu(tf.nn.conv2d(h_conv4_2, w_conv4_3, strides=[1, 1, 1, 1], padding='SAME')+b_conv4_3)
h_pool4=tf.nn.max_pool(h_conv4_3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
print("h_conv4:",h_conv4.shape)
print("h_pool4:",h_pool4.shape)

w_conv5=tf.Variable(vgg_dcit["conv5_1_W"])#tf.truncated_normal([3,3,512,512], stddev=0.1)
b_conv5=tf.Variable(vgg_dcit["conv5_1_b"])#tf.constant(0.1, shape=[512])
h_conv5=tf.nn.relu(tf.nn.conv2d(h_pool4, w_conv5, strides=[1, 1, 1, 1], padding='SAME')+b_conv5)
w_conv5_2=tf.Variable(vgg_dcit["conv5_2_W"])#tf.truncated_normal([3,3,512,512], stddev=0.1)
b_conv5_2=tf.Variable(vgg_dcit["conv5_2_b"])#tf.constant(0.1, shape=[512])
h_conv5_2=tf.nn.relu(tf.nn.conv2d(h_conv5, w_conv5_2, strides=[1, 1, 1, 1], padding='SAME')+b_conv5_2)
w_conv5_3=tf.Variable(vgg_dcit["conv5_3_W"])#tf.truncated_normal([3,3,512,512], stddev=0.1)
b_conv5_3=tf.Variable(vgg_dcit["conv5_3_b"])#tf.constant(0.1, shape=[512])
h_conv5_3=tf.nn.relu(tf.nn.conv2d(h_conv5_2, w_conv5_3, strides=[1, 1, 1, 1], padding='SAME')+b_conv5_3)
h_pool5=tf.nn.max_pool(h_conv5_3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
h_fullconv=tf.reshape(h_pool5,[-1,7*7*512])
print("h_conv5:",h_conv5.shape)
print("h_pool5:",h_pool5.shape)
print("h_fullconv",h_fullconv.shape)

w_full6=tf.Variable(vgg_dcit["fc6_W"])#tf.truncated_normal([7*7*512,4096], stddev=0.1)
b_full6=tf.Variable(vgg_dcit["fc6_b"])#tf.constant(0.1, shape=[4096])
h_full6=tf.nn.relu(tf.matmul(h_fullconv,w_full6)+b_full6)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_full6, keep_prob)
print("h_full6:",h_full6.shape)

w_full7=tf.Variable(vgg_dcit["fc7_W"])#tf.truncated_normal([4096,4096], stddev=0.1)
b_full7=tf.Variable(vgg_dcit["fc7_b"])#tf.constant(0.1, shape=[4096])
h_full7=tf.nn.relu(tf.matmul(h_fc1_drop,w_full7)+b_full7)
print("h_full7:",h_full7.shape)

w_softmax=tf.Variable(vgg_dcit["fc8_W"])#tf.truncated_normal([4096,ClassNum], stddev=0.1)
b_softmax=tf.Variable(vgg_dcit["fc8_b"])#tf.constant(0.1, shape=[ClassNum])
y_conv=tf.nn.softmax(tf.matmul(h_full7, w_softmax) + b_softmax)#tf.nn.relu(tf.matmul(h_full7, w_softmax) + b_softmax)#
print("y_conv:",y_conv.shape)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))#-tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())	
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	for i in range(9):
		img1=imread('dog.'+str(i)+'.jpg',mode='RGB')
		img1=imresize(img1,(224,224))
		image_v,label_v=sess.run([image_batch,label_batch])
		cal_result= sess.run(y_conv,feed_dict={x:[img1], keep_prob: 1.0})[0]
		preds=(np.argsort(cal_result)[::-1])[0:5]
		for p in preds:
			print(imagenet_classes.class_names[p],cal_result[p])
		print(label_v)
		#print(cal_result)
		#train_accuracy = accuracy.eval(session=sess,feed_dict={x:image_v, y_: label_v, keep_prob: 1.0})
		#print("step %d, training accuracy %g"%(i, train_accuracy))
	
	
	
	coord.request_stop()
	coord.join(threads)



	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	


