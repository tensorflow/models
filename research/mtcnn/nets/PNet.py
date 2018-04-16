# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
		AbstractFaceDetector.__init__(self)	
		self._network_size = 12
		self._network_name = 'PNet'		

	def setup_network(self, inputs):	
		self._end_points = {}
	
    		with slim.arg_scope([slim.conv2d],
                        	activation_fn = prelu,
                        	weights_initializer = slim.xavier_initializer(),
                        	biases_initializer = tf.zeros_initializer(),
                        	weights_regularizer = slim.l2_regularizer(0.0005), 
                        	padding='valid'):

			end_point = 'conv1'
        		net = slim.conv2d(inputs, 10, 3, stride=1, scope=end_point)
			self._end_points[end_point] = net

			end_point = 'pool1'
        		net = slim.max_pool2d(net, kernel_size=[2,2], stride=2, scope=end_point, padding='SAME')
			self._end_points[end_point] = net

			end_point = 'conv2'
        		net = slim.conv2d(net, num_outputs=16, kernel_size=[3,3], stride=1, scope=end_point)
			self._end_points[end_point] = net

			end_point = 'conv3'
        		net = slim.conv2d(net, num_outputs=32, kernel_size=[3,3], stride=1, scope=end_point)
			self._end_points[end_point] = net

        		#batch*H*W*2
			end_point = 'conv4_1'
        		conv4_1 = slim.conv2d(net, num_outputs=2, kernel_size=[1,1], stride=1, scope=end_point, activation_fn=tf.nn.softmax)
        		#conv4_1 = slim.conv2d(net, num_outputs=1, kernel_size=[1,1], stride=1, scope=end_point, activation_fn=tf.nn.sigmoid)
			self._end_points[end_point] = conv4_1        

        		#batch*H*W*4
			end_point = 'conv4_2'
        		bounding_box_predictions = slim.conv2d(net, num_outputs=4, kernel_size=[1,1], stride=1, scope=end_point, activation_fn=None)
			self._end_points[end_point] = bounding_box_predictions

        		#batch*H*W*10
			end_point = 'conv4_3'
        		landmark_predictions = slim.conv2d(net, num_outputs=10, kernel_size=[1,1], stride=1, scope=end_point, activation_fn=None)
			self._end_points[end_point] = landmark_predictions

        		#cls_prob_original = conv4_1 
        		#bbox_pred_original = bbox_pred

        		if(self.is_training):
            			#batch*2
            			class_probability = tf.squeeze(conv4_1, [1,2], name='class_probability')
            			class_loss = cls_ohem(class_probability, label)

            			#batch
            			bounding_box_predictions = tf.squeeze(bounding_box_predictions, [1,2], name='bounding_box_predictions')
            			bounding_box_loss = bbox_ohem(bounding_box_predictions, bounding_box_targets, label)

            			#batch*10
            			landmark_predictions = tf.squeeze(landmark_predictions, [1,2], name="landmark_predictions")
            			landmark_loss = landmark_ohem(landmark_predictions, landmark_targets, label)

            			accuracy = cal_accuracy(class_probability, label)
            			L2_loss = tf.add_n(slim.losses.get_regularization_losses())

            			return(class_loss, bounding_box_loss, landmark_loss, L2_loss, accuracy) 
        		else:
            			output_class_probability = tf.squeeze(conv4_1, axis=0)
            			output_bounding_box = tf.squeeze(bounding_box_predictions, axis=0)
            			output_landmarks = tf.squeeze(landmark_predictions, axis=0)

            			return(output_class_probability, output_bounding_box, output_landmarks)

	def load_model(self, checkpoint_path):
		self.is_training = False

        	graph = tf.Graph()
        	with graph.as_default():
            		self.input_batch = tf.placeholder(tf.float32, name='input_batch')
            		self.image_width = tf.placeholder(tf.int32, name='image_width')
            		self.image_height = tf.placeholder(tf.int32, name='image_height')
            		image_reshape = tf.reshape(self.input_batch, [1, self.image_height, self.image_width, 3])

			self.output_class_probability, self.output_bounding_box, self.output_landmarks = self.setup_network(image_reshape)

			self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))			
			self.load_model_from(checkpoint_path)

	def detect(self, input_batch):
        	image_height, image_width, _ = input_batch.shape
        	class_probabilities, bounding_boxes = self.session.run([self.output_class_probability, self.output_bounding_box],
                                                           	 feed_dict={self.input_batch: input_batch, self.image_width: image_width, self.image_height: image_height})
        	return( class_probabilities, bounding_boxes )
		
