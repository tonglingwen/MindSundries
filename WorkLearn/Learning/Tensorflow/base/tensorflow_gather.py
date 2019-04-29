import tensorflow as tf

vars=tf.constant([12.1,52.3,78.1,45.1,2.356])
indices=tf.constant([0,2,1,2,0,1,2],shape=[1,7])
res = tf.gather(vars,indices)#根据变量vars的索引indices集合构建新的tensor
sess=tf.Session()
res_out=sess.run(res)
print("res_out:\n",res_out)