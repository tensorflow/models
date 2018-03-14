from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from nets.AbstractFaceDetector import AbstractFaceDetector
from losses.prelu import prelu

class RNet(AbstractFaceDetector):

	def __init__(self, batch_size = 1):
		print('RNet')
		self.network_size = 24
		self.batch_size = batch_size

	def setup_network(self, inputs):
		with slim.arg_scope([slim.conv2d],
                        	activation_fn = prelu,
                        	weights_initializer=slim.xavier_initializer(),
                        	biases_initializer=tf.zeros_initializer(),
                        	weights_regularizer=slim.l2_regularizer(0.0005),                        
                        	padding='valid'):
        		print( inputs.get_shape() )
        		net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3,3], stride=1, scope="conv1")
        		print( net.get_shape() )
        		net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        		print( net.get_shape() )
        		net = slim.conv2d(net,num_outputs=48,kernel_size=[3,3],stride=1,scope="conv2")
        		print( net.get_shape() )
        		net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="pool2")
        		print( net.get_shape() )
        		net = slim.conv2d(net,num_outputs=64,kernel_size=[2,2],stride=1,scope="conv3")
        		print( net.get_shape() )
        		fc_flatten = slim.flatten(net)
        		print( fc_flatten.get_shape() )
        		fc1 = slim.fully_connected(fc_flatten, num_outputs=128,scope="fc1", activation_fn=prelu)
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
            			landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)
            			L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            			return( cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy )
        		else:
            			return( cls_prob, bbox_pred, landmark_pred )

	def load_model(self, checkpoint_path):
		self.is_training = False
		self.model_path = checkpoint_path

        	graph = tf.Graph()
        	with graph.as_default():
            		self.image_op = tf.placeholder(tf.float32, shape=[self.batch_size, self.network_size, self.network_size, 3], name='input_image')
            		#figure out landmark            
            		self.probability, self.bounding_box, self.landmark = self.setup_network(self.image_op)
            		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))

            		saver = tf.train.Saver()            		
            		model_dictionary = '/'.join(self.model_path.split('/')[:-1])
            		check_point = tf.train.get_checkpoint_state(model_dictionary)            		
            		read_state = check_point and check_point.model_checkpoint_path
            		assert  read_state, "Invalid parameter dictionary."
            		saver.restore(self.sess, self.model_path)      

	def detect(self, data_batch):
        	scores = []
        	batch_size = self.batch_size

        	minibatch = []
        	cur = 0

	        n = data_batch.shape[0]
	        while cur < n:
	            minibatch.append(data_batch[cur:min(cur + batch_size, n), :, :, :])
	            cur += batch_size

	        cls_prob_list = []
	        bbox_pred_list = []
	        landmark_pred_list = []
	        for idx, data in enumerate(minibatch):
	            m = data.shape[0]
	            real_size = self.batch_size

	            if m < batch_size:
	                keep_inds = np.arange(m)

	                gap = self.batch_size - m
	                while gap >= len(keep_inds):
	                    gap -= len(keep_inds)
	                    keep_inds = np.concatenate((keep_inds, keep_inds))
	                if gap != 0:
	                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
	                data = data[keep_inds]
	                real_size = m

	            cls_prob, bbox_pred,landmark_pred = self.sess.run([self.probability, self.bounding_box, self.landmark], feed_dict={self.image_op: data})
	            cls_prob_list.append(cls_prob[:real_size])
	            bbox_pred_list.append(bbox_pred[:real_size])
	            landmark_pred_list.append(landmark_pred[:real_size])

	        return( np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(landmark_pred_list, axis=0) )
	
