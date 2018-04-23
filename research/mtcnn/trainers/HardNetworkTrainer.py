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

from trainers.SimpleNetworkTrainer import SimpleNetworkTrainer
from datasets.TensorFlowDataset import TensorFlowDataset


class HardNetworkTrainer(SimpleNetworkTrainer):

	def __init__(self, network_name='RNet'):	
		SimpleNetworkTrainer.__init__(self, network_name)	

	def _read_data(self, dataset_root_dir):

		dataset_dir = self.dataset_dir(dataset_root_dir)

        	positive_file_name = self._positive_file_name(dataset_dir)
	     	part_file_name = self._part_file_name(dataset_dir)
        	negative_file_name = self._negative_file_name(dataset_dir)
	       	image_list_file_name = self._image_list_file_name(dataset_dir)

        	tensorflow_file_names = [positive_file_name, part_file_name, negative_file_name, image_list_file_name]

        	positive_ratio = 1.0/6
		part_ratio = 1.0/6
		landmark_ratio = 1.0/6
		negative_ratio = 3.0/6

        	positive_batch_size = int(np.ceil(self._batch_size*positive_ratio))
        	part_batch_size = int(np.ceil(self._batch_size*part_ratio))
        	negative_batch_size = int(np.ceil(self._batch_size*negative_ratio))
        	landmark_batch_size = int(np.ceil(self._batch_size*landmark_ratio))

        	batch_sizes = [positive_batch_size, part_batch_size, negative_batch_size, landmark_batch_size]
		
        	for d in tensorflow_file_names:
            		self._number_of_samples += sum(1 for _ in tf.python_io.tf_record_iterator(d))

		image_size = self.network_size()
		tensorflow_dataset = TensorFlowDataset()
		return(tensorflow_dataset.read_tensorflow_files(tensorflow_file_names, batch_sizes, image_size))

