import tensorflow as tf
import numpy as np

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


def anchor_target(weight,height,all_anchor,gt_boxs,limit=[0,0,224,244]):
	#去除掉图片外的边框
	inds_inside = np.where((all_anchor[:,0]>=limit[0])&
							(all_anchor[:,1]>=limit[1])&
							(all_anchor[:,2]<limit[2])&
							(all_anchor[:,3]<limit[3]))[0]
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
	print(labels)
	return anchors



anchors,lenn = get_All_Anchor()
anchor_target(60,40,anchors,np.array([[50,50,200,200],[0,0,150,150]]))
	
	
	
	
#box=np.array([[2,3,4,5],[1,3,5,7]])
#query=np.array([[2,3,4,5],[1,3,5,7]])

#print(iou(box,query))
