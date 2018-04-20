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

	def setup_basic_network(self, inputs):	
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

			return(conv4_1, bounding_box_predictions, landmark_predictions)

	def setup_training_network(self, inputs):
		convolution_output, bounding_box_predictions, landmark_predictions = self.setup_basic_network(inputs)

		output_class_probability = tf.squeeze(convolution_output, [1,2], name='class_probability')
		output_bounding_box = tf.squeeze(bounding_box_predictions, [1,2], name='bounding_box_predictions')
		output_landmarks = tf.squeeze(landmark_predictions, [1,2], name="landmark_predictions")
		
		return(output_class_probability, output_bounding_box, output_landmarks)


	def load_model(self, checkpoint_path):
        	graph = tf.Graph()
        	with graph.as_default():
            		self._input_batch = tf.placeholder(tf.float32, name='input_batch')
            		self._image_width = tf.placeholder(tf.int32, name='image_width')
            		self._image_height = tf.placeholder(tf.int32, name='image_height')
            		image_reshape = tf.reshape(self._input_batch, [1, self._image_height, self._image_width, 3])

			convolution_output, bounding_box_predictions, landmark_predictions = self.setup_basic_network(image_reshape)

       			self._output_class_probability = tf.squeeze(convolution_output, axis=0)
       			self._output_bounding_box = tf.squeeze(bounding_box_predictions, axis=0)
       			self._output_landmarks = tf.squeeze(landmark_predictions, axis=0)

			self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))			
			self._load_model_from(checkpoint_path)

	def detect(self, input_batch):
        	image_height, image_width, _ = input_batch.shape
        	class_probabilities, bounding_boxes = self._session.run([self._output_class_probability, self._output_bounding_box],
                                                           	 feed_dict={self._input_batch: input_batch, self._image_width: image_width, self._image_height: image_height})
        	return( class_probabilities, bounding_boxes )
		
