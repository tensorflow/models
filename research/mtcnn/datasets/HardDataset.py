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
import cv2

from datasets.SimpleDataset import SimpleDataset
from datasets.WIDERFaceDataset import WIDERFaceDataset
from datasets.HardWIDERFaceDataset import HardWIDERFaceDataset
from datasets.LandmarkDataset import LandmarkDataset
from datasets.TensorFlowDataset import TensorFlowDataset

from nets.FaceDetector import FaceDetector
from nets.NetworkFactory import NetworkFactory

class HardDataset(SimpleDataset):

	def __init__(self, name):	
		SimpleDataset.__init__(self, name)	

	def _generate_image_samples(self, annotation_file_name, annotation_image_dir, model_train_dir, minimum_face, target_root_dir):
		wider_dataset = HardWIDERFaceDataset()
		return(wider_dataset.generate_samples(annotation_image_dir, annotation_file_name, model_train_dir, self.network_name(), minimum_face, target_root_dir))

	def _generate_image_list(self, target_root_dir):
		positive_file = open(WIDERFaceDataset.positive_file_name(target_root_dir), 'r')
		positive_data = positive_file.readlines()

		part_file = open(WIDERFaceDataset.part_file_name(target_root_dir), 'r')
		part_data = part_file.readlines()

		negative_file = open(WIDERFaceDataset.negative_file_name(target_root_dir), 'r')
		negative_data = negative_file.readlines()

		landmark_file = open(LandmarkDataset.landmark_file_name(target_root_dir), 'r')
		landmark_data = landmark_file.readlines()

		image_list_file = open(self._image_list_file_name(target_root_dir), 'w')

    		for i in np.arange(len(positive_data)):
        		image_list_file.write(positive_data[i])

    		for i in np.arange(len(negative_data)):
        		image_list_file.write(negative_data[i])

    		for i in np.arange(len(part_data)):
        		image_list_file.write(part_data[i])

    		for i in np.arange(len(landmark_data)):
        		image_list_file.write(landmark_data[i])

		return(True)

	def _generate_dataset(self, target_root_dir):
		tensorflow_dataset = TensorFlowDataset()

		print('Generating TensorFlow dataset for positive images.')
		if(not tensorflow_dataset.generate(WIDERFaceDataset.positive_file_name(target_root_dir), target_root_dir, 'positive')):
			print('Error generating TensorFlow dataset for positive images.')
			return(False) 
		print('Generated TensorFlow dataset for positive images.')

		print('Generating TensorFlow dataset for partial images.')
		if(not tensorflow_dataset.generate(WIDERFaceDataset.part_file_name(target_root_dir), target_root_dir, 'part')):
			print('Error generating TensorFlow dataset for partial images.')		
			return(False) 
		print('Generated TensorFlow dataset for partial images.')

		print('Generating TensorFlow dataset for negative images.')
		if(not tensorflow_dataset.generate(WIDERFaceDataset.negative_file_name(target_root_dir), target_root_dir, 'negative')):
			print('Error generating TensorFlow dataset for negative images.')
			return(False) 
		print('Generated TensorFlow dataset for negative images.')

		print('Generating TensorFlow dataset for landmark images.')
		if(not tensorflow_dataset.generate(self._image_list_file_name(target_root_dir), target_root_dir, 'image_list')):
			print('Error generating TensorFlow dataset for landmark images.')
			return(False) 
		print('Generated TensorFlow dataset for landmark images.')

		return(True)

	def generate(self, annotation_image_dir, annotation_file_name, landmark_image_dir, landmark_file_name, model_train_dir, minimum_face, target_root_dir):

		if(not os.path.isfile(annotation_file_name)):
			return(False)

		if(not os.path.exists(annotation_image_dir)):
			return(False)

		if(not os.path.isfile(landmark_file_name)):
			return(False)

		if(not os.path.exists(landmark_image_dir)):
			return(False)

		target_root_dir = os.path.expanduser(target_root_dir)
		target_root_dir = os.path.join(target_root_dir, self.network_name())
		if(not os.path.exists(target_root_dir)):
			os.makedirs(target_root_dir)

		image_size = NetworkFactory.network_size(self.network_name())

		print('Generating landmark samples.')
		if(not super(HardDataset, self)._generate_landmark_samples(landmark_image_dir, landmark_file_name, image_size, target_root_dir)):
			print('Error generating landmark samples.')
			return(False)
		print('Generated landmark samples.')

		print('Generating image samples.')
		if(not self._generate_image_samples(annotation_file_name, annotation_image_dir, model_train_dir, minimum_face, target_root_dir)):
			print('Error generating image samples.')
			return(False)
		print('Generated image samples.')

		if(not self._generate_image_list(target_root_dir)):
			return(False)

		print('Generating TensorFlow dataset.')
		if(not self._generate_dataset(target_root_dir)):
			print('Error generating TensorFlow dataset.')
			return(False)
		print('Generated TensorFlow dataset.')

		return(True)

