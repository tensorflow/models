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

import os
from easydict import EasyDict as edict

from nets.NetworkFactory import NetworkFactory
from datasets.TensorFlowDataset import TensorFlowDataset

class AbstractNetworkTrainer(object):

	def __init__(self, network_name):
		self._network = NetworkFactory.network(network_name)
		self._number_of_samples = 0
		self._config = edict()

		self._batch_size = 384
		self._config.CLS_OHEM = True
		self._config.CLS_OHEM_RATIO = 0.7
		self._config.BBOX_OHEM = False
		self._config.BBOX_OHEM_RATIO = 0.7

		self._config.EPS = 1e-14
		self._config.LR_EPOCH = [6,16,20]

	def network_name(self):
		return(self._network.network_name())

	def network_size(self):
		return(self._network.network_size())
		
	def dataset_dir(self, dataset_root_dir):
		dataset_dir = os.path.join(dataset_root_dir, self.network_name())
		tensorflow_dir = os.path.join(dataset_dir, 'tensorflow')
		return(tensorflow_dir)

	def network_train_dir(self, train_root_dir):
		network_train_dir = os.path.join(train_root_dir, self.network_name())
		return(network_train_dir)

	def _positive_file_name(self, dataset_dir):
		positive_file_name = TensorFlowDataset.tensorflow_file_name(dataset_dir, 'positive')
		return(positive_file_name)

	def _part_file_name(self, dataset_dir):
		part_file_name = TensorFlowDataset.tensorflow_file_name(dataset_dir, 'part')
		return(part_file_name)

	def _negative_file_name(self, dataset_dir):
		negative_file_name = TensorFlowDataset.tensorflow_file_name(dataset_dir, 'negative')
		return(negative_file_name)

	def _image_list_file_name(self, dataset_dir):
		image_list_file_name = TensorFlowDataset.tensorflow_file_name(dataset_dir, 'image_list')
		return(image_list_file_name)

	def train(self, network_name, dataset_root_dir, train_root_dir, base_learning_rate, max_number_of_epoch, log_every_n_steps):
		raise NotImplementedError('Must be implemented by the subclass.')

