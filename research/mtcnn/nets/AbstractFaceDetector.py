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

import tensorflow as tf

class AbstractFaceDetector(object):

	def __init__(self):
		self._network_size = 0
		self._network_name = ''
		self._end_points = {}
		self._session = None

	def network_size(self):
		return(self._network_size)

	def network_name(self):
		return(self._network_name)

	def _setup_basic_network(self, inputs):
		raise NotImplementedError('Must be implemented by the subclass.')

	def setup_training_network(self, inputs):
		raise NotImplementedError('Must be implemented by the subclass.')

	def setup_inference_network(self, checkpoint_path):
		raise NotImplementedError('Must be implemented by the subclass.')

	def load_model(self, session, checkpoint_path):

  		if( tf.gfile.IsDirectory(checkpoint_path) ):
    			self.model_path = tf.train.latest_checkpoint(checkpoint_path)
  		else:
    			self.model_path = checkpoint_path

		if(not self.model_path):
			return(False)
		else:
			saver = tf.train.Saver()
      			saver.restore(session, self.model_path)
			return(True)

	def detect(self, data_batch):	
		raise NotImplementedError('Must be implemented by the subclass.')

