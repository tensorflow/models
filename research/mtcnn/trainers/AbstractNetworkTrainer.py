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

from nets.PNet import PNet
from nets.RNet import RNet
from nets.ONet import ONet

class AbstractNetworkTrainer(object):

	def __init__(self, network_name):
		if(network_name == 'PNet'):
			self._network = PNet(True)
		elif (network_name == 'RNet'):
			self._network = RNet(True)
		elif (network_name == 'ONet'):
			self._network = ONet(True)

		self._config = edict()

		self._config.BATCH_SIZE = 384
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
		
	def dataset_dir(self, dataset_dir):
		dataset_dir = os.path.join(dataset_dir, self.network_name())
		tensorflow_dir = os.path.join(dataset_dir, 'tensorflow')
		return(tensorflow_dir)

	def network_train_dir(self, train_root_dir):
		network_train_dir = os.path.join(train_root_dir, self.network_name())
		return(network_train_dir)

	def train(self):
		raise NotImplementedError('Must be implemented by the subclass.')

