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
import cv2

class WIDERFaceDataset(object):

	def __init__(self, name='WIDERFace'):
		self._name = name
		self._is_valid = False
		self._data = dict()

	def is_valid(self):
		return(self._is_valid)

	def data(self):
		return(self._data)

	def read_annotation(self, base_dir, annotation_file_path):
		
		if(not os.path.isfile(annotation_file_path)):
			return(False)

		self._data = dict()
		self._is_valid = False

		images = []
		bounding_boxes = []
		annotation_file = open(annotation_file_path, 'r')
		while( True ):
       			image_path = annotation_file.readline().strip('\n')
       			if( not image_path ):
       				break			

       			image_path = os.path.join(base_dir, image_path)
			image = cv2.imread(image_path)
			#if(image is None):
			#	continue

       			images.append(image_path)

       			nums = annotation_file.readline().strip('\n')
       			one_image_boxes = []
       			for face_index in range(int(nums)):
       				bounding_box_info = annotation_file.readline().strip('\n').split(' ')

       				face_box = [float(bounding_box_info[i]) for i in range(4)]

       				xmin = face_box[0]
       				ymin = face_box[1]
       				xmax = xmin + face_box[2]
       				ymax = ymin + face_box[3]

       				one_image_boxes.append([xmin, ymin, xmax, ymax])

       			bounding_boxes.append(one_image_boxes)

		if(len(images)):			
			self._data['images'] = images
			self._data['bboxes'] = bounding_boxes
			self._is_valid = True
		else:
			self._is_valid = False

		return(self.is_valid())

