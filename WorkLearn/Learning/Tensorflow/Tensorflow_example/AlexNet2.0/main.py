import tensorflow as tf
import data_align
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

isBatchNormal=True

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
 
with tf.device('/cpu:0'):
    #参数值设置
    learning_rate=1e-4
    training_iters=200
    batch_size=50
    display_step=5
    n_classes=2
    n_fc1=4096
    n_fc2=2048
	
    #构建模型
    x= tf.placeholder(tf.float32,[None,227,227,3])
    y=tf.placeholder(tf.float32,[None,n_classes])
    tf.summary.histogram("x",x)
    W_conv={
        'conv1':tf.Variable(tf.truncated_normal([11,11,3,96],
                                                stddev=0.0001)),
        'conv2':tf.Variable(tf.truncated_normal([5,5,96,256],
                                                stddev=0.01)),
        'conv3':tf.Variable(tf.truncated_normal([3,3,256,384],
                                                stddev=0.01)),
        'conv4':tf.Variable(tf.truncated_normal([3,3,384,384],
                                                stddev=0.01)),
        'conv5':tf.Variable(tf.truncated_normal([3,3,384,256],
                                                stddev=0.01)),
        'fc1':tf.Variable(tf.truncated_normal([6*6*256,n_fc1],
                                              stddev=0.1)),
        'fc2':tf.Variable(tf.truncated_normal([n_fc1,n_fc2],stddev=0.1)),
        'fc3':tf.Variable(tf.truncated_normal([n_fc2,n_classes],stddev=0.1))
    }
    b_conv={    #必须初始化 否则可能导致不收敛
        'conv1':tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[96])),
        'conv2':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[256])),
        'conv3':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[384])),
        'conv4':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[384])),
        'conv5':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[256])),
        'fc1':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[n_fc1])),
        'fc2':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[n_fc2])),
        'fc3':tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[n_classes]))
    } 
 
    #第1层卷积层
    conv1=tf.nn.conv2d(x,W_conv['conv1'],strides=[1,4,4,1],padding='VALID')
    conv1=tf.nn.bias_add(conv1,b_conv['conv1'])
    if isBatchNormal:
        conv1=batch_norm(conv1,True)
    conv1=tf.nn.relu(conv1)
    #第1层池化层
    pool1=tf.nn.avg_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
    #LRN层
    norml=tf.nn.lrn(pool1,5,bias=1.0,alpha=0.001/9.0,beta=0.75)
 
    #第2层卷积层
    conv2=tf.nn.conv2d(norml,W_conv['conv2'],strides=[1,1,1,1],padding='SAME')
    conv2=tf.nn.bias_add(conv2,b_conv['conv2'])
    if isBatchNormal:
        conv2=batch_norm(conv2,True)
    conv2=tf.nn.relu(conv2)
    #第2层池化层
    pool2=tf.nn.avg_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
    #LRN层
    norm2=tf.nn.lrn(pool2,5,bias=1.0,alpha=0.001/9.0,beta=0.75)
 
    #第3层卷积层
    conv3=tf.nn.conv2d(norm2,W_conv['conv3'],strides=[1,1,1,1],padding='SAME')
    conv3=tf.nn.bias_add(conv3,b_conv['conv3'])
    if isBatchNormal:
        conv3=batch_norm(conv3,True)
    conv3=tf.nn.relu(conv3)
 
    #第4层卷积层
    conv4=tf.nn.conv2d(conv3,W_conv['conv4'],strides=[1,1,1,1],padding='SAME')
    conv4=tf.nn.bias_add(conv4,b_conv['conv4'])
    if isBatchNormal:
        conv4=batch_norm(conv4,True)
    conv4=tf.nn.relu(conv4)
 
    #第5层卷积层
    conv5=tf.nn.conv2d(conv4,W_conv['conv5'],strides=[1,1,1,1],padding='SAME')
    conv5=tf.nn.bias_add(conv5,b_conv['conv5'])
    if isBatchNormal:
        conv5=batch_norm(conv5,True)
    conv5=tf.nn.relu(conv5)
    #第5层池化层
    pool5=tf.nn.avg_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
 
    #第6层全连接层
    # print(pool5.shape)
    reshape=tf.reshape(pool5,[-1,6*6*256])
    fc1=tf.add(tf.matmul(reshape,W_conv['fc1']),b_conv['fc1'])
    if isBatchNormal:
        fc1=batch_norm(fc1,True,False)
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, 0.5)
 
    #第7层全连接层
    fc2=tf.add(tf.matmul(fc1,W_conv['fc2']),b_conv['fc2'])
    if isBatchNormal:
        fc2=batch_norm(fc2,True,False)
    fc2=tf.nn.relu(fc2)
    fc2=tf.nn.dropout(fc2,0.5)
 
    #第8层全连接层
    fc3=tf.add(tf.matmul(fc2,W_conv['fc3']),b_conv['fc3'])
 
    #定义损失
    '''
    对fc3进行 exp/+exp归一化、求log值，求相反数，最终得到正实数，(最初的时候)对y进行one_hot编码，然后对位相乘，reduce_mean求得是平均数
    '''
    # labels=tf.argmax(y,axis=1)
    # loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=fc3))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=fc3, logits=y))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc3))#-tf.reduce_sum(y*tf.log(tf.clip_by_value(fc3,1e-10,1)))#
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)#tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    #评估模型
    correct_pred=tf.equal(tf.argmax(fc3,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
 
init=tf.global_variables_initializer()
save_model = r".//model\AlexNet.ckpt"
def train(opech):
    with tf.Session() as sess:
        sess.run(init)
 
        train_writer=tf.summary.FileWriter(r'.//log',sess.graph)  # 输出日志的地方
        train_writer_merge=tf.summary.merge_all()
        saver = tf.train.Saver()
 
        c=[]
        start_time=time.time()
 
        coord=tf.train.Coordinator()  #实例化队列协调器
        threads=tf.train.start_queue_runners(coord=coord)
        step=0
        for i in range(opech):
            step=i
            image,label=sess.run([image_batch, label_batch])
 
            # image,label=data_align.read_and_decode(tfrecords_file,batch_size)#
            labels=data_align.onehot(label)    #对标签进行one_hot
            optimizer_out,train_writer_merge_out,fc3_out= sess.run([optimizer,train_writer_merge,fc3],feed_dict={x:image,y:labels})
            train_writer.add_summary(train_writer_merge_out, global_step=i)
            #print(fc3_out)
            loss_record=sess.run(loss,feed_dict={x:image,y:labels})
            print('now the loss is %f'%loss_record)
 
            c.append(loss_record)
            end_time=time.time()
            print('time: ',end_time-start_time)
            start_time=end_time
            print('------------------%d onpech is finished------------------'%i)
        print('Optimization Finished!')
        saver.save(sess,save_model)
        print('Model Save Finished!')
 
        coord.request_stop()
        coord.join(threads)
        plt.plot(c)
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.title('lr=%f, ti=%d, bs=%d' % (learning_rate, training_iters, batch_size))
        plt.tight_layout()
        plt.savefig(r'cnn-tf-AlexNet.png',dpi=200)
        plt.show()
 
def per_class(imagefile):
 
    image = Image.open(imagefile)
    image = image.resize([227, 227])
    image_array = np.array(image)
 
    image = tf.cast(image_array,tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [1, 227, 227, 3])
 
    saver = tf.train.Saver()
    with tf.Session() as sess:
 
        save_model =  tf.train.latest_checkpoint('.//model') #
        saver.restore(sess, save_model)
        image = tf.reshape(image, [1, 227, 227, 3])
        image = sess.run(image)
        prediction = sess.run(fc3, feed_dict={x: image})
 
        max_index = np.argmax(prediction)
        if max_index==0:
            return "cat"
        else:
            return "dog"
images_single=np.array([1])
labels_single=np.array([1])
if __name__=='__main__':
    model='train'
    if model=='train':
        get_images=r'F:/kaggle_cat_dog_dataset/train'
 
        X_train, y_train = data_align.get_file(get_images)
        image_batch, label_batch = data_align.get_batch(X_train, y_train, 227, 227, 1, 900000)
        if images_single.size==1:
            images_single=image_batch
            labels_single=label_batch
            #np.save("image.npy",images_single)
            #np.save("label.npy",labels_single)
        train(90)
    elif model=='test':
        imagefile = r'.//9.jpg'
        r = per_class(imagefile)
        print(r)
