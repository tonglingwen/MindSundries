import cv2
import os
import numpy as np
import tensorflow as tf
from skimage import io

def rebuild(dir):
    for root, dirs, files in os.walk(dir):
        print(root,dirs,files)
        for file in files:
            filepath=os.path.join(root,file)
            try:
                image=cv2.imread(filepath)
                dim=(227,227)
                resized=cv2.resize(image,dim)
                path=r'E:\TensorFlow\AlexNet_raw\kaggledogscats\\'+file
                cv2.imwrite(path,resized)
            except:
                print(filepath)
                os.remove(filepath)
        cv2.waitKey(0)
 
def get_file(file_dir):
    images=[]
    temp=[]
    for root,sub_folders,files in os.walk(file_dir):
        # print(root,sub_folders,files)
        #image directories
        for name in files:
            images.append(os.path.join(root,name))
        #get 10 sub-folder names
        for name in sub_folders:
            temp.append(os.path.join(root,name))
 
        # print(files)
    #assign 10 labels based on the folder names
    labels=[]
    for one_folder in temp:
        n_img=len(os.listdir(one_folder))
        letter=one_folder.split('\\')[-1]
 
        if letter=='cat':
            labels=np.append(labels,n_img*[0])
        else:
            labels=np.append(labels,n_img*[1])
 
    #shuffle
    temp=np.array([images,labels])
    # print(temp)
    temp=temp.transpose()
    np.random.shuffle(temp)
    print(temp.shape)
    image_list=list(temp[:,0])
    label_list=list(temp[:,1])
    label_list=[int(float(i)) for i in label_list]
    return image_list,label_list
 
def int64_feature(value):                                                 #[]输入为list
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
 
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  #方括号表示输入为列表 转化为二进制形式
 
def convert_to_tfrecord(images_list,labels_list,save_dir,name):
    filename=os.path.join(save_dir,name+'.tfrecords')
    n_samples=len(labels_list)
    writer=tf.python_io.TFRecordWriter(filename)  #实例化并传入保存文件路径 写入到文件中
    print('\nTransform start......')
    for i in np.arange(0,n_samples):
        try:
            image=io.imread(images_list[i])
            image_raw=image.tostring()
            label=int(labels_list[i])
            example=tf.train.Example(features=tf.train.Features(feature={    #协议内存块
                'label':int64_feature(label),
                'image_raw':bytes_feature(image_raw),
            }))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:',images_list[i])
    writer.close()
    print('Transform done!')
 
def read_and_decode(tfrecords_file,batch_size):
    # 返回输出队列，QueueRunner加入到当前图中的QUEUE_RUNNER收集器
    filename_queue=tf.train.string_input_producer([tfrecords_file])
 
    reader=tf.TFRecordReader()        #实例化读取器
    _,serialized_example=reader.read(filename_queue) #返回队列当中的下一个键值对tensor
 
    # 输入标量字符串张量,输出字典映射向量tensor和稀疏向量值
    img_features=tf.parse_single_example(serialized_example,
                                         features={
                                             'label':tf.FixedLenFeature([],
                                                                        tf.int64),
                                             'image_raw':tf.FixedLenFeature([],
                                                                            tf.string),
                                         })
    image=tf.decode_raw(img_features['image_raw'],tf.uint8) #解析字符向量tensor为实数，需要有相同长度
    image=tf.reshape(image,[227,227,3])
    label=tf.cast(img_features['label'],tf.int32)
 
    #从TFRecords中读取数据，保证内容和标签同步，
    '''
    Args:
    tensors: 入队列表向量或字典向量The list or dictionary of tensors to enqueue.
    batch_size: 每次入队出队的数量The new batch size pulled from the queue.
    capacity: 队列中最大的元素数量An integer. The maximum number of elements in the queue.
    min_after_dequeue: 在一次出队以后对列中最小元素数量Minimum number elements in the queue after a dequeue, used to ensure a level of mixing of elements.
    num_threads: 向量列表入队的线程数The number of threads enqueuing tensor_list.
    seed: 队列中shuffle的种子Seed for the random shuffling within the queue.
    enqueue_many: 向量列表中的每个向量是否是单个实例Whether each tensor in tensor_list is a single example.
    shapes: (Optional) The shapes for each example. Defaults to the inferred shapes for tensor_list.
    allow_smaller_final_batch: (Optional) Boolean. If True, allow the final batch to be smaller if there are insufficient items left in the queue.
    shared_name: (Optional) If set, this queue will be shared under the given name across multiple sessions.
    name: (Optional) A name for the operations.
    '''
    image_batch,label_batch=tf.train.shuffle_batch([image,label],
                                                   batch_size=batch_size,
                                                   min_after_dequeue=100,
                                                   num_threads=64,
                                                   capacity=200)
    return image_batch,tf.reshape(label_batch,[batch_size])
 
def onehot(labels):
    n_sample=len(labels)
    n_class=max(labels)+1
    onehot_labels=np.zeros((n_sample,n_class))
    onehot_labels[np.arange(n_sample),labels]=1
    return onehot_labels
 
def get_batch(image_list,label_list,img_width,img_height,batch_size,capacity):
    image=tf.cast(image_list,tf.string)
    label=tf.cast(label_list,tf.int32)
 
    input_queue=tf.train.slice_input_producer([image,label],num_epochs=None)
    label=input_queue[1]
    image_contents=tf.read_file(input_queue[0])
    image=tf.image.decode_jpeg(image_contents,channels=3)
 
    image=tf.image.resize_image_with_crop_or_pad(image,img_width,img_height)
    image=tf.image.per_image_standardization(image)
    image_batch,label_batch=tf.train.batch([image,label],batch_size=batch_size,num_threads=64,capacity=capacity)
    label_batch=tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch
