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
from os.path import join, exists
import random
import cv2
import numpy as np
import numpy.random as npr

from utils.BBox import BBox
from utils.IoU import IoU

from datasets.Landmark import rotate
from datasets.Landmark import flip
from datasets.Landmark import randomShift
from datasets.Landmark import randomShiftWithArgument

class LandmarkDataset(object):

	def __init__(self, name='Landmark'):
		self._name = name
		self._is_valid = False
		self._landmark_data = []

	@classmethod
	def landmark_file_name(cls, target_root_dir):
		landmark_file_name = os.path.join(target_root_dir, 'landmark.txt')
		return(landmark_file_name)

	def is_valid(self):
		return(self._is_valid)

	def data(self):
		return(self._landmark_data)

	def _read(self, landmark_image_dir, landmark_file_name, with_landmark=True):

		self._is_valid = False
    		self._landmark_data = []

		with open(landmark_file_name, 'r') as landmark_file:
        		lines = landmark_file.readlines()

    		for line in lines:
        			line = line.strip()
        			components = line.split(' ')
        			image_path = os.path.join(landmark_image_dir, components[0])

        			bounding_box = (components[1], components[3], components[2], components[4])        
        			bounding_box = [float(_) for _ in bounding_box]
        			bounding_box = map(int,bounding_box)

        			if( not with_landmark ):
            				result.append((image_path, BBox(bounding_box)))
            				continue

        			landmark = np.zeros((5, 2))
        			for index in range(0, 5):
            				rv = (float(components[5+2*index]), float(components[5+2*index+1]))
            				landmark[index] = rv

        			self._landmark_data.append((image_path, BBox(bounding_box), landmark))

		if(len(self._landmark_data)):
			self._is_valid = True

    		return(self._is_valid)

	def generate(self, landmark_image_dir, landmark_file_name, minimum_face, target_root_dir):

		if(not self._read(landmark_image_dir, landmark_file_name)):
			return(False)
		
		landmark_dir = os.path.join(target_root_dir, 'landmark')
		if(not os.path.exists(landmark_dir)):
    			os.makedirs(landmark_dir)

		landmark_file = open(LandmarkDataset.landmark_file_name(target_root_dir), 'w')

		size = minimum_face
		argument = True

		number_of_images = 0
    		number_of_input_images = 0
    		for (image_path, bounding_box, landmarkGt) in self._landmark_data:
        		F_imgs = []
        		F_landmarks = []  

			image_path = image_path.replace("\\", '/')
        		image = cv2.imread(image_path)
			if( image is None):
				continue     
   		
        		image_height, image_width, image_channels = image.shape
        		gt_box = np.array([bounding_box.left,bounding_box.top,bounding_box.right,bounding_box.bottom])
        		f_face = image[bounding_box.top:bounding_box.bottom+1,bounding_box.left:bounding_box.right+1]
        		f_face = cv2.resize(f_face,(size,size))
        		landmark = np.zeros((5, 2))

        		for index, one in enumerate(landmarkGt):
            			rv = ((one[0]-gt_box[0])/(gt_box[2]-gt_box[0]), (one[1]-gt_box[1])/(gt_box[3]-gt_box[1]))
            			landmark[index] = rv

        		F_imgs.append(f_face)
        		F_landmarks.append(landmark.reshape(10))
        		landmark = np.zeros((5, 2))  

			if argument:
            			number_of_input_images = number_of_input_images + 1
            			if( number_of_input_images % 1000 == 0 ):
                			print( number_of_input_images, " number of input images are done.")

            			x1, y1, x2, y2 = gt_box
            			gt_w = x2 - x1 + 1
            			gt_h = y2 - y1 + 1        
            			if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                			continue

				for i in range(10):

                			bounding_box_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                			delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                			delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                			nx1 = int(max(x1+gt_w/2-bounding_box_size/2+delta_x,0))
                			ny1 = int(max(y1+gt_h/2-bounding_box_size/2+delta_y,0))
                
                			nx2 = nx1 + int(bounding_box_size)
                			ny2 = ny1 + int(bounding_box_size)
                			if nx2 > image_width or ny2 > image_height:
                    				continue

                			crop_box = np.array([nx1,ny1,nx2,ny2])
                			cropped_im = image[ny1:ny2+1,nx1:nx2+1,:]
                			resized_im = cv2.resize(cropped_im, (size, size))

                			iou = IoU(crop_box, np.expand_dims(gt_box,0))

					if iou > 0.65:
                    				F_imgs.append(resized_im)

                    				for index, one in enumerate(landmarkGt):
                        				rv = ((one[0]-nx1)/bounding_box_size, (one[1]-ny1)/bounding_box_size)
                        				landmark[index] = rv
                    				F_landmarks.append(landmark.reshape(10))
                    				landmark = np.zeros((5, 2))
                    				landmark_ = F_landmarks[-1].reshape(-1,2)
                    				bounding_box = BBox([nx1,ny1,nx2,ny2])   

                    				#mirror                    
                    				if random.choice([0,1]) > 0:
                        				face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        				face_flipped = cv2.resize(face_flipped, (size, size))
                        				#c*h*w
                        				F_imgs.append(face_flipped)
                        				F_landmarks.append(landmark_flipped.reshape(10))
                    				#rotate
                    				if random.choice([0,1]) > 0:
                        				face_rotated_by_alpha, landmark_rotated = rotate(image, bounding_box,bounding_box.reprojectLandmark(landmark_), 5)
                        				#landmark_offset
                        				landmark_rotated = bounding_box.projectLandmark(landmark_rotated)
                        				face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        				F_imgs.append(face_rotated_by_alpha)
                        				F_landmarks.append(landmark_rotated.reshape(10))
                
                        				#flip
                        				face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        				face_flipped = cv2.resize(face_flipped, (size, size))
                        				F_imgs.append(face_flipped)
                       					F_landmarks.append(landmark_flipped.reshape(10))                
                    
                    				#inverse clockwise rotation
                    				if random.choice([0,1]) > 0: 
                        				face_rotated_by_alpha, landmark_rotated = rotate(image, bounding_box, bounding_box.reprojectLandmark(landmark_), -5)
                        				landmark_rotated = bounding_box.projectLandmark(landmark_rotated)
                        				face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        				F_imgs.append(face_rotated_by_alpha)
                        				F_landmarks.append(landmark_rotated.reshape(10))
                
                        				face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        				face_flipped = cv2.resize(face_flipped, (size, size))
                        				F_imgs.append(face_flipped)
                        				F_landmarks.append(landmark_flipped.reshape(10)) 

				F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)

            			for i in range(len(F_imgs)):
                			if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                    				continue

                			if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                    				continue

                			cv2.imwrite(join(landmark_dir,"%d.jpg" %(number_of_images)), F_imgs[i])
                			landmarks = map(str,list(F_landmarks[i]))
                			landmark_file.write(join(landmark_dir,"%d.jpg" %(number_of_images))+" -2 "+" ".join(landmarks)+"\n")
                			number_of_images = number_of_images + 1

    		landmark_file.close()
		return(True)



