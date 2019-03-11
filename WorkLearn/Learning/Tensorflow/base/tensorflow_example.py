import tensorflow as tf

input1=tf.constant(
	2,#值
    dtype=tf.float32,#类型
    shape=None,#tensor形状
    name='Const',#名字
    verify_shape=False
	)#创建常量
input2=tf.Variable(1.0,tf.float32)#创建变量
input3=tf.placeholder(tf.float32)#占位符
input2=input1
sess=tf.Session()
print(sess.run(input1))