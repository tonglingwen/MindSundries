import tensorflow as tf

cons= tf.constant([10.0,1.0,1.0,2.0,3.0,3.0],shape=[2,3])
mean=tf.reduce_mean(cons)#张量所有元素的平均值
sum=tf.reduce_sum(cons)
max=tf.reduce_max(cons)
min=tf.reduce_min(cons)
sess=tf.Session()
print("mean",":",sess.run(mean))
print("sum",":",sess.run(sum))
print("max",":",sess.run(max))
print("min",":",sess.run(min))