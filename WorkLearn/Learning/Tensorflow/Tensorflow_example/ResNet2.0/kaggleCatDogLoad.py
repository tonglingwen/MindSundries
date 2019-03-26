import os
import tensorflow as tf

class ImageNetDataSet:
	#trainLabel=[]
	#trainLabel_value=[]

	def __init__(self,rootpath,batchsize):
		self.trainLabel = []
		self.trainLabel_value=[]
		self.rootpath=rootpath
		self.batchsize=batchsize
	
	def get_labels(self):
		dirs = os.listdir(self.rootpath)
		for letter in dirs:
			for pathdir in os.listdir(self.rootpath+"/"+letter):
				if letter=="cat":
					self.trainLabel.append("cat/"+pathdir)
					self.trainLabel_value.append(0)
				else:
					self.trainLabel.append("dog/"+pathdir)
					self.trainLabel_value.append(1)
		self.trainLabel = tf.convert_to_tensor(self.trainLabel)
		self.trainLabel_value=tf.convert_to_tensor(self.trainLabel_value)
		self.input_image,self.input_label = tf.train.slice_input_producer([self.trainLabel, self.trainLabel_value], shuffle=True,num_epochs=None)
		self.images=tf.image.decode_jpeg(tf.read_file(self.rootpath+'/'+self.input_image),channels=3)
		#self.images=tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(self.rootpath+'/'+self.input_image),channels=3),tf.float32)
		self.images=tf.image.resize_image_with_crop_or_pad(self.images,224,224)
		#self.images =tf.image.resize_images(self.images, size=[227, 227])
		self.images=tf.image.per_image_standardization(self.images)
		self.images=tf.reshape(self.images,[224*224*3])

	def get_batch_data(self):
		image_batch, label_batch = tf.train.batch([self.images,self.input_label], batch_size=self.batchsize, num_threads=64, capacity=2048,allow_smaller_final_batch=True)
		label_batch = tf.one_hot(label_batch,2,1.0,0.0,dtype=tf.float32)
		return image_batch,label_batch