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
import numpy.random as npr

from datasets.AbstractDataset import AbstractDataset
from datasets.LandmarkDataset import LandmarkDataset
from datasets.WIDERFaceDataset import WIDERFaceDataset

class BasicDataset(AbstractDataset):

	def __init__(self, name='PNet'):	
		AbstractDataset.__init__(self, name)	
	
	def _generate_landmark_samples(self, landmark_image_dir, landmark_file_name, minimum_face, target_root_dir):
		landmark_dataset = LandmarkDataset()		
		return(landmark_dataset.generate(landmark_image_dir, landmark_file_name, minimum_face, target_root_dir))
		
	def _generate_image_samples(self, annotation_image_dir, annotation_file_name, minimum_face, target_root_dir):
		wider_dataset = WIDERFaceDataset()		
		return(wider_dataset.generate(annotation_image_dir, annotation_file_name, minimum_face, target_root_dir))

	def _generate_image_list(self, target_root_dir):
		positive_file = open(os.path.join(target_root_dir, 'positive.txt'), 'r')
		positive_data = positive_file.readlines()

		part_file = open(os.path.join(target_root_dir, 'part.txt'), 'r')
		part_data = part_file.readlines()

		negative_file = open(os.path.join(target_root_dir, 'negative.txt'), 'r')
		negative_data = negative_file.readlines()

		landmark_file = open(os.path.join(target_root_dir, 'landmark.txt'), 'r')
		landmark_data = landmark_file.readlines()

		image_list_file = open(os.path.join(target_root_dir, 'image_list.txt'), 'w')

    		nums = [len(negative_data), len(positive_data), len(part_data)]
    		ratio = [3, 1, 1]
    		base_number_of_images = 25000
    		print(len(negative_data), len(positive_data), len(part_data), base_number_of_images)

    		if len(negative_data) > base_number_of_images * 3:
        		neg_keep = npr.choice(len(negative_data), size=base_number_of_images * 3, replace=True)
    		else:
        		neg_keep = npr.choice(len(negative_data), size=len(negative_data), replace=True)

    		pos_keep = npr.choice(len(positive_data), size=base_number_of_images, replace=True)
    		part_keep = npr.choice(len(part_data), size=base_number_of_images, replace=True)
    		print(len(neg_keep), len(pos_keep), len(part_keep))

    		for i in pos_keep:
        		image_list_file.write(positive_data[i])
    		for i in neg_keep:
        		image_list_file.write(negative_data[i])
    		for i in part_keep:
        		image_list_file.write(part_data[i])

    		for item in landmark_data:
        		image_list_file.write(item)

		return(True)

	def _generate_dataset(self, target_root_dir):
		print('BasicDataset-generate_dataset')
		return(True)

	def generate(self, annotation_image_dir, annotation_file_name, landmark_image_dir, landmark_file_name, minimum_face, target_root_dir):

		if(not os.path.isfile(annotation_file_name)):
			return(False)

		if(not os.path.exists(annotation_image_dir)):
			return(False)

		if(not os.path.isfile(landmark_file_name)):
			return(False)

		if(not os.path.exists(landmark_image_dir)):
			return(False)

		target_root_dir = os.path.expanduser(target_root_dir)
		target_root_dir = os.path.join(target_root_dir, self.name())
		if(not os.path.exists(target_root_dir)):
			os.makedirs(target_root_dir)

		if(not self._generate_landmark_samples(landmark_image_dir, landmark_file_name, minimum_face, target_root_dir)):
			return(False)
		
		if(not self._generate_image_samples(annotation_image_dir, annotation_file_name, minimum_face, target_root_dir)):
			return(False)

		if(not self._generate_image_list(target_root_dir)):
			return(False)

		if(not self._generate_dataset(target_root_dir)):
			return(False)

		return(True)


