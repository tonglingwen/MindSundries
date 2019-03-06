import os
import tensorflow as tf

class ImageNetDataSet:
	#trainLabel=[]
	#trainLabel_value=[]

	def __init__(self):
		self.trainLabel = []
		self.trainLabel_value=[]
	
	def get_labels(self,path):
		for line in open(path):
			strlist=line.split(' ')
			self.trainLabel.append(strlist[0].replace(' ',''))
			self.trainLabel_value.append(int(strlist[1]))
		self.trainLabel = tf.convert_to_tensor(self.trainLabel)
		self.trainLabel_value=tf.convert_to_tensor(self.trainLabel_value)
		self.input_image,self.input_label = tf.train.slice_input_producer([self.trainLabel, self.trainLabel_value], shuffle=True,num_epochs=None)
		self.images=tf.image.decode_jpeg(tf.read_file(self.input_image))		
		self.images = tf.image.resize_images(self.images, size=[227, 227])
		self.images=tf.reshape(self.images,[227*227*3])

	def get_batch_data(self):
		image_batch, label_batch = tf.train.batch([self.images,self.input_label], batch_size=5, num_threads=2, capacity=2048,allow_smaller_final_batch=False)
		label_batch = tf.one_hot(label_batch,1000,1,0)
		return image_batch,label_batch