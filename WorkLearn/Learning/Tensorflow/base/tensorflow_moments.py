import tensorflow as tf

con0=tf.constant([12,34,56,43],shape=[2,2],dtype=tf.float32)
axis=list(range(len(con0.get_shape())-1))
mean,variance=tf.nn.moments(con0,[0])

sess=tf.Session()
mean_out,variance_out=sess.run([mean,variance])
print("mean_out:",mean_out)
print("variance_out:",variance_out)