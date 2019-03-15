import kaggleCatDogLoad
import tensorflow as tf
import numpy as np  
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.set_printoptions(threshold=np.inf)  

reader = kaggleCatDogLoad.ImageNetDataSet("C:/Users/25285/Desktop/testdataset/train",1)
reader.get_labels()
image_v,label_v=reader.get_batch_data()
sess=tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
image_v_out,label_v_out=sess.run([image_v,label_v])
print(image_v_out)
print(label_v_out)
coord.request_stop()
coord.join(threads)