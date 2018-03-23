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

from nets.RNet import RNet
from utils.prelu import prelu

class ONet(RNet):

	def __init__(self, batch_size = 1):
		RNet.__init__(self, batch_size)
		self.network_size = 48
		self.network_name = 'ONet'

	def setup_network(self, inputs):
		self.end_points = {}

    		with slim.arg_scope([slim.conv2d],
                        		activation_fn = prelu,
                        		weights_initializer=slim.xavier_initializer(),
                        		biases_initializer=tf.zeros_initializer(),
                        		weights_regularizer=slim.l2_regularizer(0.0005),                        
                        		padding='valid'):

			end_point = 'conv1'
        		net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope=end_point)
			self.end_points[end_point] = net

			end_point = 'pool1'
        		net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope=end_point, padding='SAME')
			self.end_points[end_point] = net

			end_point = 'conv2'
        		net = slim.conv2d(net, num_outputs=64, kernel_size=[3,3], stride=1, scope=end_point)
			self.end_points[end_point] = net
		
			end_point = 'pool2'
        		net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope=end_point)
			self.end_points[end_point] = net

			end_point = 'conv3'
        		net = slim.conv2d(net, num_outputs=64, kernel_size=[3,3], stride=1, scope=end_point)
			self.end_points[end_point] = net

			end_point = 'pool3'
        		net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope=end_point, padding='SAME')
			self.end_points[end_point] = net

			end_point = 'conv4'
        		net = slim.conv2d(net, num_outputs=128, kernel_size=[2,2], stride=1, scope=end_point)
			self.end_points[end_point] = net

        		fc_flatten = slim.flatten(net)

			end_point = 'fc1'
        		fc1 = slim.fully_connected(fc_flatten, num_outputs=256, scope=end_point, activation_fn=prelu)
			self.end_points[end_point] = fc1

        		#batch*2
			end_point = 'cls_fc'
        		class_probability = slim.fully_connected(fc1, num_outputs=2, scope=end_point, activation_fn=tf.nn.softmax)
			self.end_points[end_point] = class_probability

        		#batch*4
			end_point = 'bbox_fc'
        		bounding_box_predictions = slim.fully_connected(fc1, num_outputs=4, scope=end_point, activation_fn=None)
			self.end_points[end_point] = bounding_box_predictions

        		#batch*10
			end_point = 'landmark_fc'
        		landmark_predictions = slim.fully_connected(fc1, num_outputs=10, scope=end_point, activation_fn=None)
			self.end_points[end_point] = landmark_predictions

        		if(self.is_training):
            			class_loss = cls_ohem(class_probability, label)
            			bounding_box_loss = bbox_ohem(bounding_box_predictions, bounding_box_targets, label)
            			landmark_loss = landmark_ohem(landmark_predictions, landmark_targets,label)

            			accuracy = cal_accuracy(class_probability, label)
            			L2_loss = tf.add_n(slim.losses.get_regularization_losses())

            			return(class_loss, bounding_box_loss, landmark_loss, L2_loss, accuracy )
        		else:
            			return(class_probability, bounding_box_predictions, landmark_predictions)

