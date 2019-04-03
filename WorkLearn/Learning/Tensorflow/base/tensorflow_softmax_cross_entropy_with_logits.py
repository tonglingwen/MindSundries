import tensorflow as tf

y_logit=tf.constant([0,1000000000000000],shape=[1,2],dtype=tf.float32)
y_label=tf.constant([0,1000000000000000],shape=[1,2],dtype=tf.float32)

scewl= tf.nn.softmax_cross_entropy_with_logits(logits=y_logit,labels=y_label)

soft= tf.nn.softmax(y_logit)
logy = -tf.reduce_sum(y_label*tf.log(tf.clip_by_value(soft,1e-8,1)),axis=1)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
print("soft_logy",sess.run(logy))
print("scewl:",sess.run(scewl))