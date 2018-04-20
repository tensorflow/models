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

from nets.PNet import PNet
from nets.RNet import RNet
from nets.ONet import ONet

class NetworkFactory(object):

	def __init__(self):	
		pass

	@classmethod
	def network(cls, network_name='PNet', is_training=False):
		if (network_name == 'PNet'): 
			network_object = PNet(is_training)
			return(network_object)
		elif (network_name == 'RNet'): 
			network_object = RNet(is_training)
			return(network_object)
		elif (network_name == 'ONet'): 
			network_object = ONet(is_training)
			return(network_object)
		else:
			network_object = PNet(is_training)
			return(network_object)
	@classmethod
	def model_deploy_dir(cls):
        	model_root_dir, _ = os.path.split(os.path.realpath(__file__))
        	model_root_dir = os.path.join(model_root_dir, '../data/mtcnn/deploy/')
		return(model_root_dir)

	@classmethod
	def model_train_dir(cls):
        	model_root_dir, _ = os.path.split(os.path.realpath(__file__))
        	model_root_dir = os.path.join(model_root_dir, '../data/mtcnn/train/')
		return(model_root_dir)

	@classmethod
	def loss_ratio(cls, network_name):
		if (network_name == 'PNet'): 
			class_loss_ratio = 1.0
			bbox_loss_ratio = 0.5
			landmark_loss_ratio = 0.5
		elif (network_name == 'RNet'): 
			class_loss_ratio = 1.0
			bbox_loss_ratio = 0.5
			landmark_loss_ratio = 0.5
		elif (network_name == 'ONet'): 
			class_loss_ratio = 1.0
			bbox_loss_ratio = 0.5
			landmark_loss_ratio = 1.0
		else: # PNet
			class_loss_ratio = 1.0
			bbox_loss_ratio = 0.5
			landmark_loss_ratio = 0.5

		return(class_loss_ratio, bbox_loss_ratio, landmark_loss_ratio)



