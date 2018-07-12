#coding:utf-8
import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.python.platform import gfile
import time
import cv2


def load_model(model):   #copied from facent.py  load_model
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    print('loading facenet model...')
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        #print('Model filename: %s' % model_exp)
        t = time.time()
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        print('time cost : ', time.time()-t)
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
        t = time.time() 
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        t1 = time.time()
        print (t1 - t)
        t = time.time()
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
        print (time.time()-t)
        #saver.save(tf.get_default_session(), './model_facenet/model.ckpt')

def test(image):
    sess = tf.Session()
    with sess.as_default():
        #load_model('./new_ckpt_pb/pnet/pnet.pb')
        load_model('./frozen_model.pb')
        #load_model('./new_ckpt_pb/rnet/rnet.pb')
        #load_model('./new_ckpt_pb/onet/onet.pb')
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0") #input
	
    #cls_pro = tf.get_default_graph().get_tensor_by_name("Squeeze:0")   #output  pnet
    #bbox_pred = tf.get_default_graph().get_tensor_by_name("Squeeze_1:0")  #output
    #landmark_pred = tf.get_default_graph().get_tensor_by_name("Squeeze_2:0")  #output
    
    #cls_pro = tf.get_default_graph().get_tensor_by_name("cls_fc/Softmax:0")   #output  rnet onet
    #bbox_pred = tf.get_default_graph().get_tensor_by_name("bbox_fc/BiasAdd:0")  #output
    #landmark_pred = tf.get_default_graph().get_tensor_by_name("landmark_fc/BiasAdd:0")  #output

    cls_pro = tf.get_default_graph().get_tensor_by_name("cls_loss:0")   #output  rnet onet
    bbox_pred = tf.get_default_graph().get_tensor_by_name("bbox_loss:0")  #output
    landmark_pred = tf.get_default_graph().get_tensor_by_name("landmark_loss:0")  #output
    
    feed_dict = {images_placeholder: [image]}
    output1, output2,output3= sess.run([cls_pro,bbox_pred,landmark_pred], feed_dict=feed_dict)
    #print("jieguo",output1,'\n',output2,'\n',output3)
    return output1, output2,output3
def main():
    image=cv2.imread('./img/two.jpg')
    image=cv2.resize(image,(24,24))
    print(test(image))
main()