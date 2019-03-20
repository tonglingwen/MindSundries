import tensorflow as tf

def foo(i):
	with tf.variable_scope("foo"+str(i), reuse=tf.AUTO_REUSE):
		v = tf.get_variable("v", [1])
	return v

def foov():
	return tf.Variable([12])
	

v1 = foo(0)  # Creates v.
v2 = foo(0)  # Gets the same, existing v.
print("get_variable:",v1 == v2)

with tf.variable_scope("foo0",reuse=tf.AUTO_REUSE):#在foo0这个variable_scope下进行操作配合foo(0)
	v3=tf.get_variable("v")

print("ex_get_variable:",v1 == v3)

vv1=foov()
vv2=foov()

print("Variable:",vv1==vv2)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
#print(sess.run(v2))
