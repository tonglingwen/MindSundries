import tensorflow as tf


con=tf.constant([1.2,4.4,6.4],shape=[1,3])
var0=tf.Variable(con)
var1=tf.Variable(con)

var_result=tf.equal(var0,var1)
var_result= tf.cast(var_result,'float')

sess=tf.Session()
sess.run(tf.global_variables_initializer())
var_result_out=sess.run(var_result)

print(var_result_out)