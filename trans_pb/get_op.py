#coding:utf-8
import tensorflow as tf
from facenet.MTCNN.mtcnn_model import P_Net, R_Net, O_Net
with tf.Graph().as_default():
	image_op = tf.placeholder(tf.float32, shape=[None, 24, 24, 3], name='input_image')
	
	cls_prob, bbox_pred, landmark_pred = R_Net(image_op, training=False)
	sess = tf.Session(
		config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
	saver = tf.train.Saver()
	
	model_path = r'C:\Users\Administrator\Desktop\face\facenet\MTCNN\MTCNN_model\RNet_landmark\RNet-20.ckpt'
	model_dict = '/'.join(model_path.split('/')[:-1])
	print(model_path)
	print(model_dict)
	ckpt = tf.train.get_checkpoint_state(model_dict)
	print(ckpt)
	print('ckpt.model_checkpoint_path',ckpt.model_checkpoint_path)
	readstate=ckpt and ckpt.model_checkpoint_path
	assert  readstate, "the params dictionary is not valid"

	saver.restore(sess, ckpt.model_checkpoint_path)
	print(tf.Graph().as_graph_def())
	print(tf.Graph().get_operations())
	print(tf.get_default_graph().get_operation_by_name(cls_loss))
	for op in tf.get_default_graph().get_operations():
		for t in op.values():
			print(t.name)
	#tensor_names=[t.name for op in tf.get_default_graph().get_operations() for t in op.values()]
	#print(tensor_names)