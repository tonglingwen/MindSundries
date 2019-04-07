import tensorflow as tf

y = tf.placeholder("float", [1,3])
result=tf.reduce_sum(y)
train=tf.train.AdamOptimizer(1e-2).minimize(result)

sess=tf.Session()
var=tf.Variable(tf.constant([12,41,78],stddev=0.01))
sess.run(var.initializer)
for i in range(10):
	sess.run([train],feed_dict={y:var})
