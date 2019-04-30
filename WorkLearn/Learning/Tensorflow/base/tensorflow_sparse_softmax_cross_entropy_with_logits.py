import tensorflow as tf

y_logit=tf.constant([0,1,0.1,0.9],shape=[2,2],dtype=tf.float32)
y_label=tf.constant([0,1],shape=[2],dtype=tf.int32)#shape是一个数组不是一个形状类似于[2,0]或者[2,1]的矩阵

scewl= tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logit,labels=y_label)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
print("scewl:",sess.run(scewl))