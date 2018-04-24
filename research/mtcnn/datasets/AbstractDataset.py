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

class AbstractDataset(object):

	def __init__(self, network_name):
		self._network_name = network_name

	def network_name(self):
		return(self._network_name)

	def _image_list_file_name(self, target_root_dir):
		image_list_file_name = os.path.join(target_root_dir, 'image_list.txt')
		return(image_list_file_name)

	def generate_dataset(self, target_root_dir):
		raise NotImplementedError('Must be implemented by the subclass.')

