import tensorflow as tf
import numpy as np
import numpy.random as npr

np.set_printoptions(threshold=np.inf)  

def rpn(data):
	return ""

def conv_layer(data):
	result={}
	w_conv=tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.01))
	b_conv=tf.Variable(tf.constant(0.1, shape=[512]))
	h_conv=tf.nn.conv2d(data,w_conv,strides=[1, 1, 1, 1], padding='SAME')+b_conv
	h_conv=tf.nn.relu(h_conv)
	
	w_conv_cls=tf.Variable(tf.truncated_normal([1,1,512,2], stddev=0.01))
	b_conv_cls=tf.Variable(tf.constant(0.1, shape=[2]))
	h_conv_cls=tf.nn.conv2d(h_conv,w_conv_cls,strides=[1, 1, 1, 1], padding='SAME')+b_conv_cls
	h_conv_cls=tf.nn.relu(h_conv_cls)
	
	w_conv_bbox=tf.Variable(tf.truncated_normal([1,1,512,4], stddev=0.01))
	b_conv_bbox=tf.Variable(tf.constant(0.1, shape=[4]))
	h_conv_bbox=tf.nn.conv2d(h_conv,w_conv_bbox,strides=[1, 1, 1, 1], padding='SAME')+b_conv_bbox
	h_conv_bbox=tf.nn.relu(h_conv_bbox)
	
	result["cls"]=h_conv_cls
	result["bbox"]=h_conv_bbox
	
	return result


def _reshape_layer(bottom, num_dim, name): #bottom.shape=[1,60,40,255]
	input_shape = tf.shape(bottom)
	with tf.variable_scope(name) as scope:
		# change the channel to the caffe format
		to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
		# then force it to have channel 2
		reshaped = tf.reshape(to_caffe,
		tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
		print(np.concatenate(([1, 2, -1], [40]),axis=0))
		# then swap the channel back
		to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
		return to_tf#to_tf.shape=[1,7650,40,2]
	
	
	
	
	
	
	
def cls_bbox():
	conv_result = conv_layer()
	cls = conv_result["cls"]
	bbox=conv_result["bbox"]
	gt_boxs=[]
	im_info=[]
	feat_stride=[]
	anchors=[]
	num_anchors=[]
	res_anchor_targets=_anchor_target_layer(cls,gt_boxs,im_info,feat_stride,anchors,num_anchors)
	
	
	rpn_cls_score_reshape = self._reshape_layer(cls, 2, 'rpn_cls_score_reshape')
	rpn_cls_score=tf.reshape(rpn_cls_score_reshape,[-1,2])
	rpn_label=tf.reshape(res_anchor_targets["rpn_labels"],[-1])
	rpn_select = tf.where(tf.not_equal(rpn_label, -1))
	
	rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
	rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
	rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))#RPN类别判断及其损失函数
	
	
	return ""


def _anchor_target_layer(rpn_cls_score,gt_boxs,im_info,feat_stride,anchors,num_anchors):
	#with tf.variable_scope(name) as scope:
	rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
	anchor_target,
	[rpn_cls_score, gt_boxes, im_info, feat_stride, anchors, num_anchors],
	[tf.float32, tf.float32, tf.float32, tf.float32],
	name="anchor_target")

	rpn_labels.set_shape([1, 1, None, None])
	rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
	rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
	rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

	rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
	_anchor_targets['rpn_labels'] = rpn_labels
	_anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
	_anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
	_anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

	#self._score_summaries.update(self._anchor_targets)

	return _anchor_targets


	
def get_All_Anchor(width=60,height=40,stride=[224/60.0,224/40.0],ratios=[0.5,1,2],scale=[8,16,32]):#获取所有anchors
	anchors =get_Anchor(ratios=ratios,scale=scale)
	A = anchors.shape[0]
	shift_x = np.arange(0, width)*stride[0]
	shift_y = np.arange(0, height)*stride[1]
	shift_x,shift_y=np.meshgrid(shift_x,shift_y)
	shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
	K = shifts.shape[0]
	anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
	anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
	length = np.int32(anchors.shape[0])
	return anchors,length
	
	
	
	
	
def get_Anchor(base_anchor=[16,16],ratios=[0.5,1,2],scale=[8,16,32]):
	base_anchor_axis=np.array(base_anchor)-1
	base_anchor_axis=base_anchor_axis/2
	ratios=np.array(ratios)
	area=base_anchor[0]*base_anchor[1]
	area_ratios=area/ratios
	ws=np.round(np.sqrt(area_ratios))
	hs=np.round(ws*ratios)
	base_anchor=np.vstack([ws,hs])
	zeros=np.zeros((1,base_anchor.shape[1]))
	base_anchor= np.vstack([zeros,zeros,base_anchor])
	base_anchor=np.transpose(base_anchor)
	base_anchor_axis=np.hstack([base_anchor_axis,base_anchor_axis])
	base_anchor_axis=np.tile(base_anchor_axis,base_anchor.shape[0])
	base_anchor_axis=np.reshape(base_anchor_axis,base_anchor.shape)
	result=np.zeros([0,4])
	for i in scale:
		result=np.vstack([result,center_to_axis(base_anchor*i)+base_anchor_axis])
	return result

def center_to_axis(val):
	result=np.zeros(val.shape)
	result[:,0]=val[:,0]-(val[:,2]-1)*0.5
	result[:,1]=val[:,1]-(val[:,3]-1)*0.5
	result[:,2]=val[:,0]+(val[:,2]-1)*0.5
	result[:,3]=val[:,1]+(val[:,3]-1)*0.5
	return result


def iou(boxes,query_boxes):
	N = boxes.shape[0]
	K = query_boxes.shape[0]
	overlaps = np.zeros((N, K))
	for k in range(K):
		box_area = (
			(query_boxes[k, 2] - query_boxes[k, 0] + 1) *
			(query_boxes[k, 3] - query_boxes[k, 1] + 1)
		)
		for n in range(N):
			iw = (
				min(boxes[n, 2], query_boxes[k, 2]) -
				max(boxes[n, 0], query_boxes[k, 0]) + 1
			)
			if iw > 0:
				ih = (
					min(boxes[n, 3], query_boxes[k, 3]) -
					max(boxes[n, 1], query_boxes[k, 1]) + 1
				)
				if ih > 0:
					ua = float(
						(boxes[n, 2] - boxes[n, 0] + 1) *
						(boxes[n, 3] - boxes[n, 1] + 1) +
						box_area - iw * ih
					)
					overlaps[n, k] = iw * ih / ua
	return overlaps


def anchor_target(rpn_cls_score,all_anchor,gt_boxs,num_anchors=9,batchsize=256,limit=[0,0,224,224]):
	A=num_anchors
	height, width=rpn_cls_score.shape[1:3]
	#去除掉图片外的边框
	inds_inside = np.where((all_anchor[:,0]>=limit[0])&
							(all_anchor[:,1]>=limit[1])&
							(all_anchor[:,2]<limit[2])&
							(all_anchor[:,3]<limit[3]))[0]
	
	#根据iou对标签进行评分
	labels = np.empty((len(inds_inside),), dtype=np.float32)
	labels.fill(-1)
	anchors=all_anchor[inds_inside,:]
	res_iou=iou(anchors,gt_boxs)
	argmax_res_iou=res_iou.argmax(axis=1)
	max_res_iou=res_iou[np.arange(len(inds_inside)),argmax_res_iou]      #所有框最近的真实框的iou
	gt_argmax_res_iou=res_iou.argmax(axis=0)
	gt_max_res_iou=res_iou[gt_argmax_res_iou,np.arange(res_iou.shape[1])]#最接近真实框的iou
	labels[max_res_iou <= 0.3] = 0
	labels[max_res_iou >= 0.7] = 1
	
	#选取适当的batch
	positive_label=np.where(labels==1)[0]
	if len(positive_label)>batchsize*0.5:
		positive_label = npr.choice(positive_label,size=int(len(positive_label)-batchsize*0.5),replace=False)
		labels[positive_label]=-1
	
	negative_label=np.where(labels==0)[0]
	if len(negative_label)>batchsize*0.5:
		negative_label = npr.choice(negative_label,size=int(len(negative_label)-batchsize*0.5),replace=False)
		labels[negative_label]=-1
	
	#计算bbox_targets(与真实框的差距)
	bbox_targets= bbox_transform(anchors,gt_boxs[argmax_res_iou,:])

	#计算smooth_l1_loss所需要的权重
	bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
	bbox_inside_weights[labels == 1, :] = np.array((1.0,1.0,1.0,1.0))
	bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
	num_examples = np.sum(labels >= 0)
	positive_weights = np.ones((1, 4)) * 1.0 / num_examples
	negative_weights = np.ones((1, 4)) * 1.0 / num_examples
	bbox_outside_weights[labels == 1, :] = positive_weights
	bbox_outside_weights[labels == 0, :] = negative_weights
	
	#替换到原all_anchor中
	total_anchors=all_anchor.shape[0]
	labels = replace_all_anchors(labels, total_anchors, inds_inside, fill=-1)
	bbox_targets = replace_all_anchors(bbox_targets, total_anchors, inds_inside, fill=0)
	bbox_inside_weights = replace_all_anchors(bbox_inside_weights, total_anchors, inds_inside, fill=0)
	bbox_outside_weights = replace_all_anchors(bbox_outside_weights, total_anchors, inds_inside, fill=0)
	
	#设置形状
	labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
	labels = labels.reshape((1, 1, A * height, width))                            #shape=(1,1,540,60)
	bbox_targets = bbox_targets.reshape((1, height, width, A * 4))                #shape=(1,60,40,36)
	bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4))  #shape=(1,60,40,36)
	bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4))#shape=(1,60,40,36)
	
	return labels,bbox_targets,bbox_inside_weights,bbox_outside_weights

def replace_all_anchors(data, count, inds, fill=0):
	if len(data.shape) == 1:
		ret = np.empty((count,), dtype=np.float32)
		ret.fill(fill)
		ret[inds] = data
	else:
		ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
		ret.fill(fill)
		ret[inds, :] = data
	return ret

def bbox_transform(ex_rois, gt_rois):
	ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
	ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
	ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
	ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

	gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
	gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
	gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
	gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

	targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
	targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
	targets_dw = np.log(gt_widths / ex_widths)
	targets_dh = np.log(gt_heights / ex_heights)

	targets = np.vstack(
	(targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
	return targets


	
con=tf.Variable(tf.truncated_normal([1,60,40,255],stddev=0.0001))
print(_reshape_layer(con,2,"da"))



	
	
	
	
'''
anchors,lenn = get_All_Anchor()
labels,bbox_targets,bbox_inside_weights,bbox_outside_weights= anchor_target(np.zeros((1,60,40,9)),anchors,np.array([[0,0,150,150],[50,50,200,200]]))
print(labels.shape)
'''

#box=np.array([[2,3,4,5],[1,3,5,7]])
#query=np.array([[2,3,4,5],[1,3,5,7]])

#print(iou(box,query))













