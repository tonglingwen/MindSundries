import tensorflow as tf

con= tf.constant([0,-22,-19],shape=[1, 3],dtype=tf.float32)
con1=tf.truncated_normal([10], stddev=0.1)
var=tf.Variable(con)

softmax=tf.nn.softmax(var)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
#print(sess.run(softmax))
print(sess.run(con1))
