import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

original_images_ex=tf.image.decode_jpeg(tf.read_file('F:/ILSVRC2012_dataset/image_train/n01440764/n01440764_18.JPEG'),channels=3)
resize_images_ex =tf.image.resize_images(original_images_ex, size=[227, 227])
resize_images_ex=tf.image.per_image_standardization(resize_images_ex)
resize_image_with_crop_or_pad_ex=tf.image.resize_image_with_crop_or_pad(original_images_ex,227,227)
resize_image_with_crop_or_pad_ex=tf.image.per_image_standardization(resize_image_with_crop_or_pad_ex)

sess=tf.Session()
original_images_ex_out,resize_images_ex_out,resize_image_with_crop_or_pad_ex_out=sess.run([original_images_ex,resize_images_ex,resize_image_with_crop_or_pad_ex])
plt.figure("Image")
#plt.imshow(original_images_ex_out)
plt.imshow(resize_images_ex_out)
#plt.imshow(resize_image_with_crop_or_pad_ex_out)
plt.axis('on')
plt.title('image')
plt.show()