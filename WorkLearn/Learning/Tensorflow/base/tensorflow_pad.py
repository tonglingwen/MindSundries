import tensorflow as tf

cons=tf.constant([1,2,3,4,5,6,7,8,9],shape=[2,2,2,2])

pad=tf.pad(cons,[[0,0],[1,1],[1,1],[1,1]])
print(cons.shape)
print(pad.shape)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(cons))
print(sess.run(pad))