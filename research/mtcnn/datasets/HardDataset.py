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
import cPickle as pickle
import cv2

from datasets.AbstractDataset import AbstractDataset
from datasets.WIDERFaceDataset import WIDERFaceDataset
from datasets.InferenceBatch import InferenceBatch

from nets.FaceDetector import FaceDetector

class HardDataset(AbstractDataset):

	def __init__(self, name):	
		AbstractDataset.__init__(self, name)	
		self._pickle_file_name = 'detections.pkl'

	def pickle_file_name(self):
		return(self._pickle_file_name)

	def generate_samples(self, network_name, annotation_file, input_image_dir, minimum_face, target_root_dir):

		wider_dataset = WIDERFaceDataset()
		if(wider_dataset.read_annotation(input_image_dir, annotation_file)):
			wider_data = wider_dataset.data()
		else:
			return(False)

		target_root_dir = os.path.expanduser(target_root_dir)
		target_root_dir = os.path.join(target_root_dir, self.name())

		positive_dir = os.path.join(target_root_dir, 'positive')
		part_dir = os.path.join(target_root_dir, 'part')
		negative_dir = os.path.join(target_root_dir, 'negative')

		if(not os.path.exists(target_root_dir)):
			os.makedirs(target_root_dir)
		if(not os.path.exists(positive_dir)):
    			os.makedirs(positive_dir)
		if(not os.path.exists(part_dir)):
    			os.makedirs(part_dir)
		if(not os.path.exists(negative_dir)):
    			os.makedirs(negative_dir)

		test_data = InferenceBatch(wider_data['images'])
		face_detector = FaceDetector()
		detections, landmarks = face_detector.detect_face(test_data)

    		pickle_file_path = os.path.join(target_root_dir, self.pickle_file_name())
    		with open(pickle_file_path, 'wb') as f:
        		pickle.dump(detections, f, 1)
		
		return(True)

	def generate_dataset(self, target_root_dir):
		print('HardDataset-generate_dataset')

	def generate(self, network_name, annotation_file, input_image_dir, minimum_face, target_root_dir):

		if(not os.path.isfile(annotation_file)):
			return(False)

		if(not os.path.exists(input_image_dir)):
			return(False)

		target_root_dir = os.path.expanduser(target_root_dir)
		if(not os.path.exists(target_root_dir)):
			os.makedirs(target_root_dir)

		if(not self.generate_samples(network_name, annotation_file, input_image_dir, minimum_face, target_root_dir)):
			return(False)

		return(True)

