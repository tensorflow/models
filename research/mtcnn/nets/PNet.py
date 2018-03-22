from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from nets.AbstractFaceDetector import AbstractFaceDetector
from utils.prelu import prelu

class PNet(AbstractFaceDetector):

	def __init__(self):		
		self.network_size = 12
		self.network_name = 'PNet'

	def setup_network(self, inputs):
    		with slim.arg_scope([slim.conv2d],
                        	activation_fn = prelu,
                        	weights_initializer = slim.xavier_initializer(),
                        	biases_initializer = tf.zeros_initializer(),
                        	weights_regularizer = slim.l2_regularizer(0.0005), 
                        	padding='valid'):

        		print( inputs.get_shape() )
        		net = slim.conv2d(inputs, 10, 3, stride=1,scope='conv1')
        		print( net.get_shape() )
        		net = slim.max_pool2d(net, kernel_size=[2,2], stride=2, scope='pool1', padding='SAME')
        		print( net.get_shape() )
        		net = slim.conv2d(net,num_outputs=16,kernel_size=[3,3],stride=1,scope='conv2')
        		print( net.get_shape() )
        		net = slim.conv2d(net,num_outputs=32,kernel_size=[3,3],stride=1,scope='conv3')
        		print( net.get_shape() )
        		#batch*H*W*2
        		conv4_1 = slim.conv2d(net,num_outputs=2,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)
        		#conv4_1 = slim.conv2d(net,num_outputs=1,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.sigmoid)
        
        		print( conv4_1.get_shape() )
        		#batch*H*W*4
        		bbox_pred = slim.conv2d(net,num_outputs=4,kernel_size=[1,1],stride=1,scope='conv4_2',activation_fn=None)
        		print( bbox_pred.get_shape() )
        		#batch*H*W*10
        		landmark_pred = slim.conv2d(net,num_outputs=10,kernel_size=[1,1],stride=1,scope='conv4_3',activation_fn=None)
        		print( landmark_pred.get_shape() )
        		#cls_prob_original = conv4_1 
        		#bbox_pred_original = bbox_pred
        		if(self.is_training):
            			#batch*2
            			cls_prob = tf.squeeze(conv4_1,[1,2],name='cls_prob')
            			cls_loss = cls_ohem(cls_prob,label)
            			#batch
            			bbox_pred = tf.squeeze(bbox_pred,[1,2],name='bbox_pred')
            			bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            			#batch*10
            			landmark_pred = tf.squeeze(landmark_pred,[1,2],name="landmark_pred")
            			landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)

            			accuracy = cal_accuracy(cls_prob,label)
            			L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            			return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy 
        		#test
        		else:
            			#when test,batch_size = 1
            			cls_pro_test = tf.squeeze(conv4_1, axis=0)
            			bbox_pred_test = tf.squeeze(bbox_pred,axis=0)
            			landmark_pred_test = tf.squeeze(landmark_pred,axis=0)
            			return( cls_pro_test,bbox_pred_test,landmark_pred_test )

	def load_model(self, checkpoint_path):
		self.is_training = False
		self.model_path = checkpoint_path

        	graph = tf.Graph()
        	with graph.as_default():
            		self.input_image = tf.placeholder(tf.float32, name='input_image')
            		self.image_width = tf.placeholder(tf.int32, name='image_width')
            		self.image_height = tf.placeholder(tf.int32, name='image_height')
            		image_reshape = tf.reshape(self.input_image, [1, self.image_height, self.image_width, 3])
			self.probability, self.bounding_box, _ = self.setup_network(image_reshape)
			self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
			saver = tf.train.Saver()
            		model_dictionary = '/'.join(checkpoint_path.split('/')[:-1])
            		check_point = tf.train.get_checkpoint_state(model_dictionary)
            		read_state = ( check_point and check_point.model_checkpoint_path )
            		assert  read_state, "Invalid parameter dictionary."
            		saver.restore(self.session, checkpoint_path)

	def detect(self, data_batch):
        	height, width, _ = data_batch.shape
        	probabilities, bounding_boxes = self.session.run([self.probability, self.bounding_box],
                                                           	 feed_dict={self.input_image: data_batch, self.image_width: width, self.image_height: height})
        	return( probabilities, bounding_boxes )
		
