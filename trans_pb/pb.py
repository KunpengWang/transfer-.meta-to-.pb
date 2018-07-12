#coding:utf-8
import tensorflow as tf
from mtcnn_model import P_Net,R_Net, O_Net

#1.input placeholder
#2.定义网络结构，输入是placeholder;
#  定义网络结构时把“if training”部分去掉了，只留下inference部分
#3.restore 旧的ckpt文件（名称类似于xxx.ckpt.data-xxx）的参数值（如，weight）；
#  如果直接freeze旧的ckpt文件和meta文件为pb文件，会发现print出的节点名字会对不上，直接拿来用也会报错，是不是可以认为训练图和inference的图中节点对不上
#4.save->新的meta文件和ckpt文件 
#5.freeze.py  freeze graph 生成pb文件
#6.测试是不是可以用。

with tf.Graph().as_default():
	with tf.Session() as sess:
		#inputs_placeholder = tf.placeholder(tf.float32, shape=[None,None,None,3], name='input') #1 ；pnet是全卷积网络，输入大小任意
		#inputs_placeholder = tf.placeholder(tf.float32, shape=[None,24,24,3], name='input') #1 ；rnet有全连接层，所以输入是固定大小，onet同理
		inputs_placeholder = tf.placeholder(tf.float32, shape=[None,48,48,3], name='input') #1
		
		#cls_pro,bbox_pred,landmark_pred = P_Net(inputs_placeholder) #2
		#cls_pro,bbox_pred,landmark_pred = R_Net(inputs_placeholder) #2
		cls_pro,bbox_pred,landmark_pred = O_Net(inputs_placeholder) #2
		
		saver = tf.train.Saver() #2
		
		#saver.restore(sess, './old_ckpt/PNet_landmark/PNets/PNet-30') #3
		#saver.restore(sess, './old_ckpt/RNet_landmark/RNets/RNet-22') #3
		saver.restore(sess, './old_ckpt/ONet_landmark/ONets/ONet-22') #3
		
		for op in tf.get_default_graph().get_operations():  #打印网络中的节点
			for t in op.values():
				print(t.name)

		#saver.save(sess,'./new_ckpt_pb/pnet/pnet.ckpt')  #4
		#saver.save(sess,'./new_ckpt_pb/rnet/rnet.ckpt')  #4
		saver.save(sess,'./new_ckpt_pb/onet/onet.ckpt')  #4