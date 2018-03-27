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
import numpy as np
import tensorflow as tf
import numpy as np
import cv2
import numpy.random as np_random

from tensorflow.contrib import slim

from datasets.AbstractDataset import AbstractDataset
from utils.IoU import IoU

class BasicDataset(AbstractDataset):

	def __init__(self, name='PNet'):	
		AbstractDataset.__init__(self, name)	

	def generate_samples(self, annotation_file, input_image_dir, target_root_dir):
		print('BasicDataset-generate_samples')
		return(True)

	def generate_dataset(self, target_root_dir):
		print('BasicDataset-generate_dataset')
		return(True)

	def generate(self, annotation_file, input_image_dir, target_root_dir):

		if(not os.path.isfile(annotation_file)):
			return(False)

		if(not os.path.exists(input_image_dir)):
			return(False)

		target_dir = os.path.expanduser(target_root_dir)
		if(not os.path.exists(target_dir)):
			os.makedirs(target_dir)
		
		if(not self.generate_samples(annotation_file, input_image_dir, target_root_dir)):
			return(False)

		if(not self.generate_dataset(target_root_dir)):
			return(False)

		return(True)


