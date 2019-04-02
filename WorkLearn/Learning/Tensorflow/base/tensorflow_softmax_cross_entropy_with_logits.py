import tensorflow as tf

y_logit=tf.constant([1,2,3,4],shape=[2,2],dtype=tf.float32)
y_label=tf.constant([1,0.1,1,2],shape=[2,2],dtype=tf.float32)

scewl= tf.nn.softmax_cross_entropy_with_logits(logits=y_logit,labels=y_label)

soft= tf.nn.softmax(y_logit)
logy = -tf.reduce_sum(y_label*tf.log(soft),axis=1)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
print("soft_logy",sess.run(logy))
print("scewl:",sess.run(scewl))