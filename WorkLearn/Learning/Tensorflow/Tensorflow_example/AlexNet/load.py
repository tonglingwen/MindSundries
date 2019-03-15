import readImageNet
import kaggleCatDogLoad
import tensorflow as tf
import numpy as np  
import os
import matplotlib.pyplot as plt
import time

data=True
cov1=True
cov2=True
cov3=True
cov4=True
cov5=True
full6=True
full7=True
full8=True
cross=True
isBatchNormal=True

ClassNum=2
ImagePath='F:/kaggle_cat_dog_dataset/train'
LabelPath='train_label_origin.txt'
SavePath='./model/AlexNetModel.ckpt'
BatchSize=50

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


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.set_printoptions(threshold=np.inf)  

dataset = kaggleCatDogLoad.ImageNetDataSet(ImagePath,BatchSize)#加载图片根目录
dataset.get_labels()
image_batch,label_batch = dataset.get_batch_data()



x = tf.placeholder("float", [None, 227*227*3])
y_ = tf.placeholder("float", [None,ClassNum])
x_y_=y_
input_data=tf.reshape(x, [-1,227,227,3])

if data:
	tf.summary.image("input_data",input_data)
	tf.summary.histogram("x",x)

w_conv1=tf.Variable(tf.truncated_normal([11,11,3,96], stddev=0.0001))
b_conv1=tf.Variable(tf.constant(0.1, shape=[96]))
h_conv1=tf.nn.conv2d(input_data, w_conv1, strides=[1, 4, 4, 1], padding='VALID')+b_conv1
if isBatchNormal:
	h_conv1=batch_norm(h_conv1,True)
h_conv1=tf.nn.relu(h_conv1)
h_pool1=tf.nn.local_response_normalization(tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID'),alpha=0.001/9.0,beta=0.75)
print("h_conv1:",h_conv1.shape)
print("h_pool1:",h_pool1.shape)
if cov1:
	tf.summary.histogram("w_conv1",w_conv1)
	tf.summary.histogram("h_conv1",h_conv1)

w_conv2=tf.Variable(tf.truncated_normal([5,5,96,256], stddev=0.01))
b_conv2=tf.Variable(tf.constant(0.1, shape=[256]))
h_conv2=tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')+b_conv2
if isBatchNormal:
	h_conv2=batch_norm(h_conv2,True)
h_conv2=tf.nn.relu(h_conv2)
h_pool2=tf.nn.local_response_normalization(tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID'),alpha=0.001/9.0,beta=0.75)
print("h_conv2:",h_conv2.shape)
print("h_pool2:",h_pool2.shape)
if cov2:
	tf.summary.histogram("w_conv2",w_conv2)
	tf.summary.histogram("h_conv2",h_conv2)

w_conv3=tf.Variable(tf.truncated_normal([3,3,256,384], stddev=0.01))
b_conv3=tf.Variable(tf.constant(0.1, shape=[384]))
h_conv3=tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME')+b_conv3
if isBatchNormal:
	h_conv3=batch_norm(h_conv3,True)
h_conv3=tf.nn.relu(h_conv3)
print("h_conv3:",h_conv3.shape)
if cov3:
	tf.summary.histogram("h_conv3",h_conv3)

w_conv4=tf.Variable(tf.truncated_normal([3,3,384,384], stddev=0.01))
b_conv4=tf.Variable(tf.constant(0.1, shape=[384]))
h_conv4=tf.nn.conv2d(h_conv3, w_conv4, strides=[1, 1, 1, 1], padding='SAME')+b_conv4
if isBatchNormal:
	h_conv4=batch_norm(h_conv4,True)
h_conv4=tf.nn.relu(h_conv4)
print("h_conv4:",h_conv4.shape)
if cov4:
	tf.summary.histogram("h_conv4",h_conv4)

w_conv5=tf.Variable(tf.truncated_normal([3,3,384,256], stddev=0.01))
b_conv5=tf.Variable(tf.constant(0.1, shape=[256]))
h_conv5=tf.nn.conv2d(h_conv4, w_conv5, strides=[1, 1, 1, 1], padding='SAME')+b_conv5
if isBatchNormal:
	h_conv5=batch_norm(h_conv5,True)
h_conv5=tf.nn.relu(h_conv5)
h_pool5=tf.nn.max_pool(h_conv5, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID')
h_fullconv=tf.reshape(h_pool5,[-1,6*6*256])
print("h_conv5:",h_conv5.shape)
print("h_pool5:",h_pool5.shape)
print("h_fullconv:",h_fullconv.shape)
if cov5:
	tf.summary.histogram("w_conv5",w_conv5)
	tf.summary.histogram("h_conv5",h_conv5)

w_full6=tf.Variable(tf.truncated_normal([6*6*256,4096], stddev=0.1))
b_full6=tf.Variable(tf.constant(0.1, shape=[4096]))
h_full6=tf.matmul(h_fullconv,w_full6)+b_full6
if isBatchNormal:
	h_full6=batch_norm(h_full6,True,False)
h_full6=tf.nn.relu(h_full6)#h_fullconv
keep_prob = tf.placeholder("float")
#h_fc1_drop = tf.nn.dropout(h_full6, keep_prob)
print("h_full6:",h_full6.shape)
if full6:
	tf.summary.histogram("w_full6",w_full6)
	tf.summary.histogram("h_full6",h_full6)

w_full7=tf.Variable(tf.truncated_normal([4096,4096], stddev=0.1))
b_full7=tf.Variable(tf.constant(0.1, shape=[4096]))
h_full7=tf.matmul(h_full6,w_full7)+b_full7
if isBatchNormal:
	h_full7=batch_norm(h_full7,True,False)
h_full7=tf.nn.relu(h_full7)
h_fc2_drop = tf.nn.dropout(h_full7, keep_prob)
print("h_full7:",h_full7.shape)
if full7:
	tf.summary.histogram("w_full7",w_full7)
	tf.summary.histogram("h_full7",h_full7)

w_softmax=tf.Variable(tf.truncated_normal([4096,2], stddev=0.1))
tf.summary.histogram("w_softmax",w_softmax)
b_softmax=tf.Variable(tf.constant(0.1, shape=[ClassNum]))
y_conv=tf.matmul(h_fc2_drop, w_softmax) + b_softmax
#y_conv_mat=tf.nn.relu(y_conv_mat0)
#y_conv_mat=y_conv_mat-y_conv_mat[tf.argmax(y_conv_mat)]
#y_conv=tf.nn.softmax(y_conv_mat)
print("y_conv:",y_conv.shape)
if full8:
	tf.summary.histogram("y_conv",y_conv)
	
y_conv_clip=tf.clip_by_value(y_conv,1e-10,1)
cross_entropy =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))# -tf.reduce_sum(y_*tf.log(y_conv_clip))#
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)#
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
if cross:
	tf.summary.scalar("cross_entropy",cross_entropy)
	tf.summary.histogram("y_conv",y_conv)
	tf.summary.histogram("y_conv_clip",y_conv_clip)
	tf.summary.histogram("y_",y_)
	
writer = tf.summary.FileWriter("log/")#创建summary
summaries = tf.summary.merge_all()#整合所有要绘制的图形
saver=tf.train.Saver()#保存模型
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	save_model=tf.train.latest_checkpoint('.//model')
	saver.restore(sess,save_model)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	asd=[]
	for i in range(1601):#训练过程
		try:
			image_v,label_v=sess.run([image_batch,label_batch])
			#print(label_v)
			'''
			#print(image_v)
			for i in tf.reshape(image_v,[BatchSize,227,227,3]).eval(session=sess):
				plt.imshow(i)
				plt.draw()
				plt.pause(1)
			'''
			
			if i%100 == 0:
				train_accuracy = accuracy.eval(session=sess,feed_dict={x:image_v, y_: label_v, keep_prob: 1.0})
				print("step %d, training accuracy %g"%(i, train_accuracy))
			#train_step.run(session=sess,feed_dict={x: image_v, y_: label_v, keep_prob: 0.5})
			#np.random.randint(0.0,1.0,(1,154587))
			train,summ,y_conv_out,cross_entropy_out =sess.run([train_step,summaries,y_conv,cross_entropy],feed_dict={x:image_v , y_: label_v, keep_prob: 0.5})
			print("cross_entropy:",cross_entropy_out)
			#print("label:",label_v)
			#print("y_conv_out:",y_conv_out)
			#print("y_conv_mat_out:",y_conv_mat_out)
			#print("y_conv_mat0_out:",y_conv_mat0_out)
			writer.add_summary(summ, global_step=i)
			'''
			sdd=sess.run(w_full7)
			np.savetxt("image_v"+str(i)+".txt",image_v,fmt="%s",delimiter=",")							
			np.savetxt("label_v"+str(i)+".txt",label_v,fmt="%s",delimiter=",")
			if len(asd)==0:
				asd=sdd
			if (asd==sdd).all():
				print('==')
				if i!=0:
					#print(image_v)
					#print(label_v)
					break
			else:
				print('!=')	
			asd=sdd
			
			print(sess.run(cross_entropy,feed_dict={x: image_v, y_: label_v, keep_prob: 1}))
			'''
		except BaseException as e:
			print(str(e))
			break
		else:
			pass
	saver=tf.train.Saver()
	saver.save(sess,SavePath)
	coord.request_stop()
	coord.join(threads)







	
	