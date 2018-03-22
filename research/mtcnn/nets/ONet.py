from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from nets.RNet import RNet
from utils.prelu import prelu

class ONet(RNet):

	def __init__(self, batch_size = 1):
		self.network_size = 48
		self.batch_size = batch_size
		self.network_name = 'ONet'

	def setup_network(self, inputs):
    		with slim.arg_scope([slim.conv2d],
                        		activation_fn = prelu,
                        		weights_initializer=slim.xavier_initializer(),
                        		biases_initializer=tf.zeros_initializer(),
                        		weights_regularizer=slim.l2_regularizer(0.0005),                        
                        		padding='valid'):
        		print( inputs.get_shape() )
        		net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
        		print( net.get_shape() )
        		net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        		print( net.get_shape() )
        		net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv2")
        		print( net.get_shape() )
        		net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        		print( net.get_shape() )
        		net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
        		print( net.get_shape() )
        		net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
        		print( net.get_shape() )
        		net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope="conv4")
        		print( net.get_shape() )
        		fc_flatten = slim.flatten(net)
        		print( fc_flatten.get_shape() )
        		fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1", activation_fn=prelu)
        		print( fc1.get_shape() )
        		#batch*2
        		cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        		print( cls_prob.get_shape() )
        		#batch*4
        		bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        		print( bbox_pred.get_shape() )
        		#batch*10
        		landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
        		print( landmark_pred.get_shape() )
        		#train
        		if(self.is_training):
            			cls_loss = cls_ohem(cls_prob,label)
            			bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            			accuracy = cal_accuracy(cls_prob,label)
            			landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            			L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            			return( cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy )
        		else:
            			return( cls_prob,bbox_pred,landmark_pred )

