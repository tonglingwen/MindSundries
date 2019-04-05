#coding = utf-8

import collections
import kaggleCatDogLoad
import tensorflow as tf
from datetime import datetime
import math
import time
import numpy as np
import os



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.set_printoptions(threshold=np.inf)  

slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    '''A named tuple describing a ResNet block.'''

def subsample(inputs, factor, scope=None):
    '''降采样方法：
    factor:采样因子 1：不做修改直接返回 不为1：使用slim.max_pool2d降采样'''
    if factor ==1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    '''创建卷积层'''
    if stride == 1:
        '''stride为1，使用slim.conv2d，padding为SAME'''
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           padding='SAME', scope=scope)

    else:
        '''显示地pad zero：
        pad zero总数为kernel size-1，pad_beg:pad//2, pad_end:余下部分'''
        pad_total = kernel_size-1
        pad_beg = pad_total//2
        pad_end = pad_total - pad_beg
        '''tf.pad对inputs进行补零操作'''
        inputs = tf.pad(inputs, [[0,0], [pad_beg, pad_end],
                                 [pad_beg, pad_end], [0, 0]])

        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                            padding='VALID', scope=scope)

@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):
    '''net:input
       blocks:Block的class的列表
       outputs_collections:收集各个end_points的collections'''
    for block in blocks:
        '''双层for循环，逐个Block，逐个Residual Unit堆叠'''
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            '''两个tf.variable将残差学习单元命名为block_1/unit_1形式'''

            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' %(i+1), values=[net]):

                    '''利用第二层for循环拿到前面定义Blocks Residual Unit中args，
                    将其展开为depth、depth_bottleneck、stride'''
                    unit_depth, unit_depth_bottleneck, unit_stride = unit

                    '''使用unit_fn函数（残差学习单元的生成函数）
                    顺序地创建并连接所有的残差学习单元'''
                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)

            '''slim.utils.collect_named_outputs将输出net添加到collection中'''
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

        '''所有的Residual Unit都堆叠完后，最后返回net作为stack_blocks_dense的结果'''
        return net


def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.097,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    '''创建ResNet通用的arg_scope(作用：定义某些函数的参数默认值)'''

    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,#默认为0.0001，BN的衰减速率默认为：0.997
        'epsilon': batch_norm_epsilon,#默认为1e-5
        'scale': batch_norm_scale,#BN的scale默认为True
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):

        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:

                return arg_sc

@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
    '''bottleneck残差学习单元
    inputs：输入
    depth、depth_bottleneck、stride是Blocks类中的args
    outputs_collections：收集end_points的collection
    scope：unit的名称'''
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:

        '''slim.utils.last_dimension获取输入的最后一个维度,输出通道数,min_rank=4限定最少为4个维度'''
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)

        '''slim.batch_norm对输入进行Batch Normalization，接着用relu进行预激活的Preactivate'''
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu,
                                 scope='preact')
        '''定义shortcut(直连的x)'''
        if depth == depth_in:
            '''如果残差单元输入通道数和输出通道数一样
            使用subsample按步长对inputs进行空间上的降采样'''
            shortcut = subsample(inputs, stride, 'shortcut')

        else:
            '''如果残差单元输入通道数和输出通道数不一样，
            使用stride步长的1x1卷积改变其通道数，使得输入通道数和输出通道数一致'''
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')
        '''定义残差：
        第一步：1x1尺寸、步长为1、输出通道数为depth_bottleneck的卷积
        第二步：3x3尺寸、步长为stride、输出通道数为depth_bottleneck的卷积
        第三步：1x1尺寸、步长为1、输出通道数为depth的卷积'''
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')

        residual = slim.conv2d(residual, depth_bottleneck, 3, stride,
                               scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = shortcut + residual

        '''slim.utils.collect_named_ouputs将结果添加到outputs_collections并返回output作为函数结果'''
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              include_root_block=True,
              reuse=None,
              scope=None):
    '''定义生成ResNet V2的主函数
       inputs:输入
       blocks：定义好的Blocks类的的列表
       num_classes:最后输出的类数
       global_pool:是否加上最后的一层全局平均池化的标志
       include_root_blocks:是否加上ResNet网络最前面通常使用的7x7卷积核最大池化的标志
       reuse：是否重用的标志
       scope：整个网络名称'''

    with tf.variable_scope(scope, 'resent_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        '''slim.arg_scope将slim.conv2d, bottleneck,stack_blocks_dense 3个函数的参数
        outputs_collections默认设置为end_points_collection'''
        with slim.arg_scope([slim.conv2d, bottleneck,
                            stack_blocks_dense],
                            outputs_collections=end_points_collection):

            net = inputs

            if include_root_block:

                with slim.arg_scope([slim.conv2d], activation_fn=None,
                                    normalizer_fn=None):
                    '''根据include_root_block标记，创建ResNet
                    最前面的64输出通道的步长为2的7x7卷积'''
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')

                    '''步长为2的3x3最大池化，经过2次步长为2的层后，图片尺寸已经缩小为1/4'''
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            '''利用stack_blocks_dens将残差学习模块完成'''
            net = stack_blocks_dense(net, blocks)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')

            if global_pool:
                '''根据标记添加平均池化层，这里用tf.reduce_mean比avg_pool高'''
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

            if num_classes is not None:
                '''根据是否有分类数，添加一个输出通道为num_classes的1x1卷积'''
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')

            '''slim.utils.convert_collection_to_dict将collection转化为dict'''
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            if num_classes is not None:
                '''添加一个softmax层输出网络结果'''
                end_points['prediction'] = slim.softmax(net, scope='predictions')

            return net, end_points


def resnet_v2_50(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_50'):
    '''设计50层的ResNet
    四个blocks的units数量为3、4、6、3，总层数为(3+4+6+3)*3+2=50
    前3个blocks包含步长为2的层，总尺寸224/(4*2*2*2)=7 输出通道变为2048'''
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)]*2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]

    return resnet_v2(inputs, blocks, num_classes, global_pool,
                      include_root_block=True, reuse=reuse, scope=scope)

def resnet_v2_101(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_101'):
    '''设计101层的ResNet
    四个blocks的units数量为3、4、23、3，总层数为(3+4+23+3)*3+2=101
    前3个blocks包含步长为2的层，总尺寸224/(4*2*2*2)=7 输出通道变为2048'''
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]

    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)

def resnet_v2_152(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_152'):
    '''设计152层的ResNet
    四个blocks的units数量为3、8、36、3，总层数为(3+8+36+3)*3+2=152
    前3个blocks包含步长为2的层，总尺寸224/(4*2*2*2)=7 输出通道变为2048'''
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]

    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)

def resnet_v2_200(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_200'):
    '''设计200层的ResNet
    四个blocks的units数量为3、8、36、3，总层数为(3+24+36+3)*3+2=200
    前3个blocks包含步长为2的层，总尺寸224/(4*2*2*2)=7 输出通道变为2048'''
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]

    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)

def time_tensorflow_run(session, target, info_string):

    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches+num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time()-start_time

        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %(datetime.now(), i-num_steps_burn_in, duration))
                total_duration += duration
                total_duration_squared += duration*duration

    mn = total_duration/num_batches
    vr = total_duration_squared/num_batches-mn*mn
    sd = math.sqrt(vr)

    print('%s: %s across %d steps, %.3f +/- %3.3f sec/batch' %(datetime.now(), info_string, num_batches, mn, sd))

batch_size = 1
height, width = 224, 224

reader=kaggleCatDogLoad.ImageNetDataSet("F:/kaggle_cat_dog_dataset/train",batch_size)
reader.get_labels()
images,labels=reader.get_batch_data()
x = tf.placeholder("float", [None, 224*224*3])
y_ = tf.placeholder("float", [None,2])
tf.summary.histogram("x",x)
input_data=tf.reshape(x, [-1,224,224,3])
tf.summary.image("input_data",input_data)
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(resnet_arg_scope(is_training=True)):
    net, end_points = resnet_v2_152(input_data, 2)
#end_points['prediction']
y_conv=tf.reshape(net,[-1,2])
#tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))# 
cross_entropy =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))#tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-8,1))))#
train_step = tf.train.AdamOptimizer(1).minimize(cross_entropy)#tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)#
print("prediction:",y_conv.shape)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
saver=tf.train.Saver()

writer = tf.summary.FileWriter("log/")#创建summary
summaries = tf.summary.merge_all()#整合所有要绘制的图形

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
save_model=tf.train.latest_checkpoint('.//model')
#saver.restore(sess,save_model)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
num_batches = 100
images_single=np.array([1])
labels_single=np.array([1])
for i in range(20):
	images_out,labels_out=sess.run([images,labels])
	
	if images_single.size==1:
		#images_single=images_out
		#labels_single=labels_out
		#np.save("images.npy",images_single)
		#np.save("labels.npy",labels_single)
		images_single=np.load("images.npy")
		labels_single=np.load("labels.npy")
		#print(images_single)
		#print(images_single.shape)
		#print(labels_single.shape)
	'''
	if i%100==0:
		accuracy_out=sess.run(accuracy,feed_dict={x:images_single,y_:labels_single})
		print("accuracy:",accuracy_out)
	'''
	train_step_out,cross_entropy_out,summaries_out,y_conv_out=sess.run([train_step,cross_entropy,summaries,y_conv],feed_dict={x:images_single,y_:labels_single})
	print(y_conv_out)
	print(cross_entropy_out)
	#y_conv_out=sess.run(y_conv,feed_dict={x:images_single,y_:labels_single})
	#writer.add_summary(summaries_out, global_step=i)
	#print("loss"+str(i)+",",cross_entropy_out)
	#print(np.sum(y_conv_out-labels_single))
#time_tensorflow_run(sess, net, 'Forward')
saver=tf.train.Saver()
saver.save(sess,'./model/AlexNetModel.ckpt')
coord.request_stop()
coord.join(threads)








