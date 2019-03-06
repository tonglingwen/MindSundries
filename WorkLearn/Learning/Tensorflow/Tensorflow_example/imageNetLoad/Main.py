import readImageNet
import tensorflow as tf
import numpy as np  
np.set_printoptions(threshold=np.inf)  

dataset = readImageNet.ImageNetDataSet()
#image_raw_data_jpg = tf.gfile.FastGFile('n01491361_25.JPEG', 'rb').read()
#image_raw_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
dataset.get_labels('train_label_test.txt')
image_batch,label_batch = dataset.get_batch_data()
#print(dataset.trainLabel)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#print(sess.run(dataset.input_queue))
#print(sess.run(dataset.trainLabel))
#for i in range(10):
print(sess.run(image_batch))
#print(sess.run(image_raw_data_jpg).shape)
coord.request_stop()
coord.join(threads)