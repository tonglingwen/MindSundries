import os
import tensorflow as tf
import bbox

class ImageNetDataSet:
	#trainLabel=[]
	#trainLabel_value=[]

	def __init__(self,rootpath,bbboxrootpath):
		self.trainLabel = []
		self.trainLabel_value=[]
		self.bboxs_value=[]
		self.rootpath=rootpath
	
	def get_labels(self,labelpath,type=None,):
		self.types={}
		for i in range(len(type)):
			self.types[type[i]]=i
		for line in open(path):
			strlist=line.split(' ')
			if type==None or int(strlist[1]) in type:
				filepath=strlist[0].replace(' ','')
				self.trainLabel.append(filepath)
				self.trainLabel_value.append(self.types(int(strlist[1])))
				try:
					self.bboxs_value.append(bbox.GetBBoxByPath(bbboxrootpath+'/'+filepath))
				except BaseException as e:
					print("加载权重时出现异常")
				else:
					pass
		self.trainLabel = tf.convert_to_tensor(self.trainLabel)
		self.trainLabel_value=tf.convert_to_tensor(self.trainLabel_value)
		self.input_image,self.input_label,self.input_bboxs= tf.train.slice_input_producer([self.trainLabel, self.trainLabel_value,self.bboxs_value], shuffle=True,num_epochs=None)
		self.images=tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(self.rootpath+'/'+self.input_image),channels=3),tf.float32)
		self.images =tf.image.resize_images(self.images, size=[227, 227])
		self.images=tf.reshape(self.images,[227*227*3])


	def get_batch_data(self):
		image_batch, label_batch,bboxs_batch= tf.train.batch([self.images,self.input_label,self.input_bboxs], batch_size=50, num_threads=2, capacity=2048,allow_smaller_final_batch=True)
		label_batch = tf.one_hot(label_batch,1000,1,0)
		return image_batch,label_batch,bboxs_batch