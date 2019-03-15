import tensorflow as tf

con= tf.constant([1000000,1000000,-999999],shape=[1, 3],dtype=tf.float32)
con1=tf.truncated_normal([10], stddev=0.0001)
var=tf.Variable(con)

softmax=tf.nn.softmax(var)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(con1))
#print(sess.run(con1))
