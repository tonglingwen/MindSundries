import os
import tensorflow as tf

def get_labels(path):
	dic={}
	for line in open(path):
		strlist=line.split(' ')
		dic[strlist[0].replace(' ','')]=int(strlist[1])
	return dic

print(get_labels('train_label.txt'))