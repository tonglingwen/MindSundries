import tensorflow as tf

c=tf.constant([1,2.3,4,2,13,3])
rec=tf.reshape(c,[1,6])


sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(rec))


