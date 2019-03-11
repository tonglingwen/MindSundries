import tensorflow as tf

q=tf.FIFOQueue(4,'float')
init=q.enqueue_many(([1.2,1.5,4.5],))
init2=q.enqueue(21)        #如果队列已满将阻塞进程
init3=q.enqueue(1.3)
deque=q.dequeue()         #队列弹出如果队列中没有元素则阻塞进程直到有数据压入

qr=tf.train.QueueRunner(q,enqueue_ops=[init2,init3]) #队列入队进程 init2,init3入队操作 enqueue_ops操作列表
sess=tf.Session()
coord=tf.train.Coordinator()  #线程协调器 通知其他线程结束
sess.run(tf.global_variables_initializer())
enqueue_threads= qr.create_threads(sess,coord=coord,start=True) #启动入队线程
#sess.run(init)
#sess.run(init2)
for i in range(100):
	print(sess.run(deque))
print(sess.run(deque))
print(sess.run(deque))
print(sess.run(deque))
print(sess.run(deque))   #非QueueRunner下阻塞进程
print(sess.run(q.size()))
print('结束')
coord.request_stop()    #线程协调器通知程序结束，通知其他线程 不通知的话会导致线程挂起使程序无法退出
coord.join(enqueue_threads)#等待指定线程终止