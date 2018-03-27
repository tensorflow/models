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
import numpy.random as npr

from tensorflow.contrib import slim

from datasets.AbstractDataset import AbstractDataset
from utils.IoU import IoU

class BasicDataset(AbstractDataset):

	def __init__(self, name='PNet'):	
		AbstractDataset.__init__(self, name)	
	
	def generate_samples(self, annotation_file, input_image_dir, target_root_dir):
		target_root_dir = os.path.expanduser(target_root_dir)
		positive_dir = os.path.join(target_root_dir, self.name, 'positive')
		part_dir = os.path.join(target_root_dir, self.name, 'part')
		negative_dir = os.path.join(target_root_dir, self.name, 'negative')

		if(not os.path.exists(target_root_dir)):
			os.makedirs(target_root_dir)
		if(not os.path.exists(positive_dir)):
    			os.makedirs(positive_dir)
		if(not os.path.exists(part_dir)):
    			os.makedirs(part_dir)
		if(not os.path.exists(negative_dir)):
    			os.makedirs(negative_dir)

		f1 = open(os.path.join(target_root_dir, self.name, 'positive.txt'), 'w')
		f2 = open(os.path.join(target_root_dir, self.name, 'negative.txt'), 'w')
		f3 = open(os.path.join(target_root_dir, self.name, 'part.txt'), 'w')
		with open(annotation_file, 'r') as f:
			annotations = f.readlines()

		num = len(annotations)
		print('Total number of images are - %d.' % num)

		p_idx = 0 # positive
		n_idx = 0 # negative
		d_idx = 0 # dont care
		idx = 0
		box_idx = 0

		for annotation in annotations:
    			annotation = annotation.strip().split(' ')
			
			im_path = annotation[0]

			bbox = map(float, annotation[1:])
			boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)

			img = cv2.imread(os.path.join(input_image_dir, im_path + '.jpg'))

		    	idx += 1
    			if( idx % 100 == 0 ):
        			print('%d number of images are done.' % idx)
        
    			height, width, channel = img.shape

			neg_num = 0
			while(neg_num < 50):
				size = npr.randint(12, min(width, height) / 2)
			        nx = npr.randint(0, width - size)
        			ny = npr.randint(0, height - size)
        
        			crop_box = np.array([nx, ny, nx + size, ny + size])
        
        			sample_IoU = IoU(crop_box, boxes)
        
        			cropped_im = img[ny : ny + size, nx : nx + size, :]
        			resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
				if( np.max(sample_IoU) < 0.3 ):
					save_file = os.path.join(negative_dir, "%s.jpg"%n_idx)
					f2.write("PNet/negative/%s.jpg"%n_idx + ' 0\n')
					cv2.imwrite(save_file, resized_im)
            				n_idx += 1
            				neg_num += 1

			for box in boxes:
				x1, y1, x2, y2 = box
				w = x2 - x1 + 1
				h = y2 - y1 + 1

				if( max(w, h) < 40 or x1 < 0 or y1 < 0 ):
            				continue

				for i in range(5):
			            	size = npr.randint(12, min(width, height) / 2)
            				delta_x = npr.randint(max(-size, -x1), w)
            				delta_y = npr.randint(max(-size, -y1), h)
            				nx1 = int(max(0, x1 + delta_x))
            				ny1 = int(max(0, y1 + delta_y))
            				if nx1 + size > width or ny1 + size > height:
                				continue
            				crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            				sample_IoU = IoU(crop_box, boxes)
    
            				cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
            				resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
    
            				if np.max(sample_IoU) < 0.3:
                				save_file = os.path.join(negative_dir, "%s.jpg" % n_idx)
                				f2.write("PNet/negative/%s.jpg" % n_idx + ' 0\n')
                				cv2.imwrite(save_file, resized_im)
                				n_idx += 1 

				for i in range(20):
            				size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            				delta_x = npr.randint(-w * 0.2, w * 0.2)
            				delta_y = npr.randint(-h * 0.2, h * 0.2)

            				nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
            				ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
            				nx2 = nx1 + size
            				ny2 = ny1 + size

            				if nx2 > width or ny2 > height:
                				continue 
            				crop_box = np.array([nx1, ny1, nx2, ny2])
            				offset_x1 = (x1 - nx1) / float(size)
					offset_y1 = (y1 - ny1) / float(size)
            				offset_x2 = (x2 - nx2) / float(size)
            				offset_y2 = (y2 - ny2) / float(size)

            				cropped_im = img[ny1 : ny2, nx1 : nx2, :]
            				resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            				box_ = box.reshape(1, -1)
            				if IoU(crop_box, box_) >= 0.65:
                				save_file = os.path.join(positive_dir, "%s.jpg"%p_idx)
                				f1.write("PNet/positive/%s.jpg"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                				cv2.imwrite(save_file, resized_im)
                				p_idx += 1
            				elif IoU(crop_box, box_) >= 0.4:
                				save_file = os.path.join(part_dir, "%s.jpg"%d_idx)
                				f3.write("PNet/part/%s.jpg"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                				cv2.imwrite(save_file, resized_im)
                				d_idx += 1

				box_idx += 1
				print( "%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx))
		f3.close()
		f2.close()
		f1.close()

		return(True)

	def generate_dataset(self, target_root_dir):
		print('BasicDataset-generate_dataset')
		return(True)

	def generate(self, annotation_file, input_image_dir, target_root_dir):

		if(not os.path.isfile(annotation_file)):
			return(False)

		if(not os.path.exists(input_image_dir)):
			return(False)

		target_root_dir = os.path.expanduser(target_root_dir)
		if(not os.path.exists(target_root_dir)):
			os.makedirs(target_root_dir)
		
		if(not self.generate_samples(annotation_file, input_image_dir, target_root_dir)):
			return(False)

		if(not self.generate_dataset(target_root_dir)):
			return(False)

		return(True)


