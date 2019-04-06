import tensorflow as tf
import readImageNet
from scipy.misc import imread,imresize
import kaggleCatDogLoad
import numpy as np

def batch_norm(inputs,is_training,is_conv_out=True,decay=0.999):
	scale=tf.Variable(tf.ones([inputs.get_shape()[-1]]))
	beta=tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
	pop_mean=tf.Variable(tf.zeros([inputs.get_shape()[-1]]),trainable=False)
	pop_var=tf.Variable(tf.ones([inputs.get_shape()[-1]]),trainable=False)
	if is_training:
		if is_conv_out:
			batch_mean,batch_var=tf.nn.moments(inputs,[0,1,2])
		else:
			batch_mean,batch_var=tf.nn.moments(inputs,[0])
		train_mean=tf.assign(pop_mean,pop_mean*decay+batch_mean*(1-decay))
		train_var=tf.assign(pop_var,pop_var*decay+batch_var*(1-decay))
		with tf.control_dependencies([train_mean,train_var]):
			return tf.nn.batch_normalization(inputs,batch_mean,batch_var,beta,scale,0.001)
	else:
		return tf.nn.batch_normalization(inputs,pop_mean,pop_var,beta,scale,0.001)

def residual_2(input,out_channel,i):
	attach=str(i)
	#with tf.variable_scope("residual_2_"+attach) as scope:
	#scope.reuse_variables()
	channel =int(input.get_shape()[3])
	isChange=(channel!=out_channel[1])
	w_conv=tf.get_variable("w_conv_"+str(i),initializer=tf.truncated_normal([3,3,channel,out_channel[0]],stddev=0.0001))
	b_conv=tf.get_variable("b_conv_"+str(i),initializer=tf.constant(0.1,shape=[out_channel[0]]))
	if isChange:
		h_conv=tf.nn.conv2d(input, w_conv, strides=[1, 2, 2, 1], padding='SAME')+b_conv
	else:
		h_conv=tf.nn.conv2d(input, w_conv, strides=[1, 1, 1, 1], padding='SAME')+b_conv
	h_conv=batch_norm(h_conv,True)
	h_conv= tf.nn.relu(h_conv)
	w_conv1=tf.get_variable("w_conv1_"+str(i),initializer=tf.truncated_normal([3,3,out_channel[0],out_channel[1]],stddev=0.0001))
	b_conv1=tf.get_variable("b_conv1_"+str(i),initializer=tf.constant(0.1,shape=[out_channel[1]]))
	h_conv1=tf.nn.conv2d(h_conv, w_conv1, strides=[1, 1, 1, 1], padding='SAME')+b_conv1
	h_conv1= batch_norm(h_conv1,True)
	h_conv1= tf.nn.relu(h_conv1)
	if isChange:
		w_conv2=tf.get_variable("w_conv2_"+str(i),initializer=tf.truncated_normal([1,1,channel,out_channel[1]],stddev=0.0001))
		b_conv2=tf.get_variable("b_conv2_"+str(i),initializer=tf.constant(0.1,shape=[out_channel[1]]))
		input=tf.nn.conv2d(input,w_conv2,strides=[1,2,2,1],padding='SAME')+b_conv2
		input=batch_norm(input,True)
		input=tf.nn.relu(input)
	return h_conv1+input
	

def residual_3(input,out_channel,i):
	channel=int(input.get_shape()[3])
	isChange=(channel!=out_channel[2])
	w_conv=tf.get_variable("w_conv_"+str(i),initializer=tf.truncated_normal([1,1,channel,out_channel[0]],stddev=0.0001))
	b_conv=tf.get_variable("b_conv_"+str(i),initializer=tf.constant(0.1,shape=[out_channel[0]]))
	if isChange:
		h_conv=tf.nn.conv2d(input,w_conv,strides=[1,2,2,1],padding='SAME')+b_conv
	else:
		h_conv=tf.nn.conv2d(input,w_conv,strides=[1,1,1,1],padding='SAME')+b_conv
	h_conv=batch_norm(h_conv,True)
	h_conv=tf.nn.relu(h_conv)
	w_conv1=tf.get_variable("w_conv1_"+str(i),initializer=tf.truncated_normal([3,3,out_channel[0],out_channel[1]],stddev=0.0001))
	b_conv1=tf.get_variable("b_conv1_"+str(i),initializer=tf.constant(0.1,shape=[out_channel[1]]))
	h_conv1=tf.nn.conv2d(h_conv,w_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1
	h_conv1=batch_norm(h_conv1,True)
	h_conv1=tf.nn.relu(h_conv1)
	w_conv2=tf.get_variable("w_conv2_"+str(i),initializer=tf.truncated_normal([1,1,out_channel[1],out_channel[2]],stddev=0.0001))
	b_conv2=tf.get_variable("b_conv2_"+str(i),initializer=tf.constant(0.1,shape=[out_channel[2]]))
	h_conv2=tf.nn.conv2d(h_conv1,w_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2
	h_conv2=batch_norm(h_conv2,True)
	h_conv2=tf.nn.relu(h_conv2)
	
	if isChange:
		input=tf.nn.max_pool(input,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
		dims=tf.to_int32((out_channel[2]-input.shape[3]))/2
		input=tf.pad(input,[[0,0],[0,0],[0,0],[dims,dims]])
		'''
		print('start',inp.shape)
		w_conv3=tf.get_variable("w_conv3_"+str(i),initializer=tf.truncated_normal([3,3,channel,out_channel[2]],stddev=0.0001))
		b_conv3=tf.get_variable("b_conv3_"+str(i),initializer=tf.constant(0.1,shape=[out_channel[1]]))
		input=tf.nn.conv2d(input,w_conv3,strides=[1,2,2,1],padding='SAME')+b_conv3
		input=batch_norm(input,True)
		input=tf.nn.relu(input)
		print('end',input.shape)
		'''
	return h_conv2+input


ClassNum=2
ImagePath='F:/kaggle_cat_dog_dataset/train'
#ImagePath='F:/kaggle_cat_dog_dataset/test1'
LabelPath='train_label.txt'
SavePath='./model/AlexNetModel.ckpt'
BatchSize=64

dataset = kaggleCatDogLoad.ImageNetDataSet(ImagePath,BatchSize)#加载图片根目录
dataset.get_labels()

#dataset = readImageNet.ImageNetDataSet(ImagePath,ClassNum,BatchSize)#加载图片根目录
#dataset.get_labels(LabelPath)
image_batch,label_batch = dataset.get_batch_data()

	
x = tf.placeholder("float", [None, 224*224*3])
y_ = tf.placeholder("float", [None,ClassNum])

input_d=tf.reshape(x,[-1,224,224,3])
tf.summary.image("input_data",input_d)
w_conv=tf.Variable(tf.truncated_normal([7,7,3,64],stddev=0.0001))
b_conv=tf.constant(0.1,shape=[64])
h_conv=tf.nn.relu(tf.nn.conv2d(input_d,w_conv,strides=[1,2,2,1],padding='SAME')+b_conv)
h_pool1=tf.nn.max_pool(h_conv, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')



'''
with tf.variable_scope("residual_2_0"):
	tf.summary.histogram("residual_2_0_w_conv",tf.get_variable('w_conv'))
	tf.summary.histogram("residual_2_0_h_conv",tf.get_variable('h_conv'))
	tf.summary.histogram("residual_2_0_w_conv1",tf.get_variable('w_conv1'))
	tf.summary.histogram("residual_2_0_h_conv1",tf.get_variable('h_conv1'))
'''

'''
h_residual1=residual_2(h_pool1,[64,64],0)
h_residual2=residual_2(h_residual1,[64,64],1)
h_residual3=residual_2(h_residual2,[64,64],2)

h_residual4=residual_2(h_residual3,[128,128],3)
h_residual5=residual_2(h_residual4,[128,128],4)
h_residual6=residual_2(h_residual5,[128,128],5)
h_residual7=residual_2(h_residual6,[128,128],6)

h_residual8=residual_2(h_residual7,[256,256],7)
h_residual9=residual_2(h_residual8,[256,256],8)
h_residual10=residual_2(h_residual9,[256,256],9)
h_residual11=residual_2(h_residual10,[256,256],10)
h_residual12=residual_2(h_residual11,[256,256],11)
h_residual13=residual_2(h_residual12,[256,256],12)

h_residual14=residual_2(h_residual13,[512,512],13)
h_residual15=residual_2(h_residual14,[512,512],14)
h_residual16=residual_2(h_residual15,[512,512],15)
'''

h_residual1=residual_3(h_pool1,[64,64,64],0)
h_residual2=residual_3(h_residual1,[64,64,64],1)
h_residual3=residual_3(h_residual2,[64,64,64],2)

h_residual4=residual_3(h_residual3,[128,128,128],3)
h_residual5=residual_3(h_residual4,[128,128,128],4)
h_residual6=residual_3(h_residual5,[128,128,128],5)
h_residual7=residual_3(h_residual6,[128,128,128],6)

h_residual8=residual_3(h_residual7,[256,256,256],7)
h_residual9=residual_3(h_residual8,[256,256,256],8)
h_residual10=residual_3(h_residual9,[256,256,256],9)
h_residual11=residual_3(h_residual10,[256,256,256],10)
h_residual12=residual_3(h_residual11,[256,256,256],11)
h_residual13=residual_3(h_residual12,[256,256,256],12)

h_residual14=residual_3(h_residual13,[512,512,512],13)
h_residual15=residual_3(h_residual14,[512,512,512],14)
h_residual16=residual_3(h_residual15,[512,512,512],15)


h_pool2=tf.nn.avg_pool(h_residual16, ksize=[1, 7, 7, 1],strides=[1, 1, 1, 1], padding='VALID')

h_fc=tf.reshape(h_pool2,[-1,1*1*512])

w_fc=tf.Variable(tf.truncated_normal([1*1*512,ClassNum],stddev=0.01))
b_fc=tf.Variable(tf.constant(0.1,shape=[ClassNum]))
h_fc=tf.matmul(h_fc,w_fc)+b_fc

cross_entropy =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_fc,labels=y_))# -tf.reduce_sum(y_*tf.log(y_conv_clip))#
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)#tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)#

h_fc=tf.nn.softmax(h_fc)
correct_prediction = tf.equal(tf.argmax(h_fc,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

writer=tf.summary.FileWriter("log/")
summaries=tf.summary.merge_all()

saver=tf.train.Saver()#保存模型
images_single=np.array([1])
labels_single=np.array([1])
sess=tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
#save_model=tf.train.latest_checkpoint('.//model')
#saver.restore(sess,save_model)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for i in range(4000):
	image_v,label_v=sess.run([image_batch,label_batch])
	
	'''
	if images_single.size==1:
		#images_single=image_v
		#labels_single=label_v
		#np.save("images.npy",images_single)
		#np.save("labels.npy",labels_single)
		images_single=np.load("images.npy")
		labels_single=np.load("labels.npy")
		#print(images_single)
		#print(images_single.shape)
		#print(labels_single.shape)
	'''
	train_step_out,cross_entropy_out,summaries_out=sess.run([train_step,cross_entropy,summaries],feed_dict={x:image_v,y_:label_v})
	#accuracy_out=sess.run(accuracy,feed_dict={x:image_v,y_:label_v})
	#writer.add_summary(summaries_out)
	print("cross_entropy:"+str(i),cross_entropy_out)
#saver=tf.train.Saver()
saver.save(sess,SavePath)
coord.request_stop()
coord.join(threads)









	

