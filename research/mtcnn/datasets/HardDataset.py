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
import cPickle as pickle
import cv2

from datasets.AbstractDataset import AbstractDataset
from datasets.WIDERFaceDataset import WIDERFaceDataset
from datasets.InferenceBatch import InferenceBatch

from nets.FaceDetector import FaceDetector

from utils.convert_to_square import convert_to_square
from utils.IoU import IoU

class HardDataset(AbstractDataset):

	def __init__(self, name):	
		AbstractDataset.__init__(self, name)	
		self._pickle_file_name = self.name() + '.pkl'

	def pickle_file_name(self):
		return(self._pickle_file_name)

	def save_hard_samples(self, wider_data, minimum_face, target_root_dir):

		pickle_file_path = os.path.join(target_root_dir, self.pickle_file_name())
		detected_boxes = pickle.load(open(os.path.join(pickle_file_path), 'rb'))

		image_file_names = wider_data['images']
		generated_boxes = wider_data['bboxes']
    		number_of_images = len(image_file_names)

		if(not (len(detected_boxes) == number_of_images)):
			return(False)

		if(self.name() == 'ONet'):
			image_size = 48
		else:
			image_size = 24

		positive_dir = os.path.join(target_root_dir, 'positive')
		part_dir = os.path.join(target_root_dir, 'part')
		negative_dir = os.path.join(target_root_dir, 'negative')

		if(not os.path.exists(positive_dir)):
    			os.makedirs(positive_dir)
		if(not os.path.exists(part_dir)):
    			os.makedirs(part_dir)
		if(not os.path.exists(negative_dir)):
    			os.makedirs(negative_dir)

		positive_file = open(os.path.join(target_root_dir, 'positive.txt'), 'w')
		part_file = open(os.path.join(target_root_dir, 'part.txt'), 'w')
		negative_file = open(os.path.join(target_root_dir, 'negative.txt'), 'w')

    		negative_images = 0
    		positive_images = 0
    		part_images = 0

    		for image_file_path, detected_box, generated_box in zip(image_file_names, detected_boxes, generated_boxes):
        		generated_box = np.array(generated_box, dtype=np.float32).reshape(-1, 4)

        		if( detected_box.shape[0] == 0 ):
            			continue

        		detected_box = convert_to_square(detected_box)
        		detected_box[:, 0:4] = np.round(detected_box[:, 0:4])

        		current_image = cv2.imread(image_file_path)

        		neg_num = 0
        		for box in detected_box:
            			x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            			width = x_right - x_left + 1
            			height = y_bottom - y_top + 1

            			if( (width < minimum_face) or (x_left < 0) or (y_top < 0) or (x_right > current_image.shape[1] - 1) or (y_bottom > current_image.shape[0] - 1) ):
                			continue

            			current_IoU = IoU(box, generated_box)
            			cropped_image = current_image[y_top:y_bottom + 1, x_left:x_right + 1, :]
            			resized_image = cv2.resize(cropped_image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

            			if( (np.max(current_IoU) < 0.3) and (neg_num < 60) ):
                			file_path = os.path.join(negative_dir, "%s.jpg" % negative_images)
                			negative_file.write(file_path + ' 0\n')
                			cv2.imwrite(file_path, resized_image)
                			negative_images += 1
                			neg_num += 1
            			else:
                			idx = np.argmax(current_IoU)
                			assigned_gt = generated_box[idx]
                			x1, y1, x2, y2 = assigned_gt

                			offset_x1 = (x1 - x_left) / float(width)
                			offset_y1 = (y1 - y_top) / float(height)
                			offset_x2 = (x2 - x_right) / float(width)
                			offset_y2 = (y2 - y_bottom) / float(height)

                			if( np.max(current_IoU) >= 0.65 ):
                    				file_path = os.path.join(positive_dir, "%s.jpg" % positive_images)
                    				positive_file.write(file_path + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    				cv2.imwrite(file_path, resized_image)
                    				positive_images += 1

                			elif( np.max(current_IoU) >= 0.4 ):
                    				file_path = os.path.join(part_dir, "%s.jpg" % part_images)
                    				part_file.write(file_path + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    				cv2.imwrite(file_path, resized_image)
                    				part_images += 1
    		negative_file.close()
    		part_file.close()
    		positive_file.close()

		return(True)

	def generate_samples(self, annotation_file, input_image_dir, model_train_dir, minimum_face, target_root_dir):

		wider_dataset = WIDERFaceDataset()
		if(wider_dataset.read_annotation(input_image_dir, annotation_file)):
			wider_data = wider_dataset.data()
		else:
			return(False)

		target_root_dir = os.path.expanduser(target_root_dir)
		target_root_dir = os.path.join(target_root_dir, self.name())

		if(not os.path.exists(target_root_dir)):
			os.makedirs(target_root_dir)

		test_data = InferenceBatch(wider_data['images'])

		if(not model_train_dir):
			model_train_dir = FaceDetector.model_train_dir()			
		face_detector = FaceDetector(model_train_dir)
		detections, landmarks = face_detector.detect_face(test_data)

    		pickle_file_path = os.path.join(target_root_dir, self.pickle_file_name())
    		with open(pickle_file_path, 'wb') as f:
        		pickle.dump(detections, f, 1)
		
		return(self.save_hard_samples(wider_data, minimum_face, target_root_dir))

	def generate_dataset(self, target_root_dir):
		print('HardDataset-generate_dataset')

	def generate(self, annotation_file, input_image_dir, model_train_dir, minimum_face, target_root_dir):

		if(not os.path.isfile(annotation_file)):
			return(False)

		if(not os.path.exists(input_image_dir)):
			return(False)

		target_root_dir = os.path.expanduser(target_root_dir)
		if(not os.path.exists(target_root_dir)):
			os.makedirs(target_root_dir)

		if(not self.generate_samples(annotation_file, input_image_dir, model_train_dir, minimum_face, target_root_dir)):
			return(False)

		return(True)

