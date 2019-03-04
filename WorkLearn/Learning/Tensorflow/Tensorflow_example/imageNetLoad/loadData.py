import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

image_raw=tf.gfile.FastGFile('n01440764_37.JPEG','rb').read()
img=tf.image.decode_jpeg(image_raw)
resizeimg=tf.image.resize_images(img,[224,224],method=0)
croppad=tf.image.resize_image_with_crop_or_pad(img,224,224)

#tf.image.resize_images(img,[224,224],method=0)
#tf.image.resize_image_with_crop_or_pad(img,224,224)
with tf.Session() as sess:
	print(type(image_raw))
	print(type(img))
	print(type(img.eval()))
	print(img.eval().shape)
	print('img',img)
	print('resizeimg',resizeimg)
	print('croppad',croppad)
	plt.figure(1)
	plt.imshow(np.asarray(resizeimg.eval(),dtype='uint8'))
	plt.show()