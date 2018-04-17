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
import sys
import random
import cv2
import tensorflow as tf

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _convert_to_example_simple(image_example, image_buffer):
    class_label = image_example['label']
    bbox = image_example['bbox']
    roi = [bbox['xmin'],bbox['ymin'],bbox['xmax'],bbox['ymax']]
    landmark = [bbox['xlefteye'],bbox['ylefteye'],bbox['xrighteye'],bbox['yrighteye'],bbox['xnose'],bbox['ynose'],
                bbox['xleftmouth'],bbox['yleftmouth'],bbox['xrightmouth'],bbox['yrightmouth']]
                
      
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_buffer),
        'image/label': _int64_feature(class_label),
        'image/roi': _float_feature(roi),
        'image/landmark': _float_feature(landmark)
    }))
    return example

def _process_image_withoutcoder(filename):
    image = cv2.imread(filename)
    image_data = image.tostring()
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width

class TensorFlowDataset(object):

	def __init__(self):
		self._is_valid = False
		self._dataset = []

	def is_valid(self):
		return(self._is_valid)

	def data(self):
		return(self._dataset)

	def _get_output_filename(self, target_dir, target_name):
		target_name = target_name + '.tfrecord'
		file_name = os.path.join(target_dir, target_name)
		return(file_name)

	def _read_dataset(self, input_file_name): 
   	
		self._is_valid = False
    		self._dataset = []

		imagelist = open(input_file_name, 'r')

    		for line in imagelist.readlines():
        		info = line.strip().split(' ')
        		data_example = dict()
        		bbox = dict()
        		data_example['filename'] = info[0]
        		data_example['label'] = int(info[1])
        		bbox['xmin'] = 0
        		bbox['ymin'] = 0
        		bbox['xmax'] = 0
        		bbox['ymax'] = 0
        		bbox['xlefteye'] = 0
        		bbox['ylefteye'] = 0
        		bbox['xrighteye'] = 0
        		bbox['yrighteye'] = 0
        		bbox['xnose'] = 0
        		bbox['ynose'] = 0
        		bbox['xleftmouth'] = 0
        		bbox['yleftmouth'] = 0
        		bbox['xrightmouth'] = 0
        		bbox['yrightmouth'] = 0        
        		if len(info) == 6:
            			bbox['xmin'] = float(info[2])
            			bbox['ymin'] = float(info[3])
            			bbox['xmax'] = float(info[4])
            			bbox['ymax'] = float(info[5])
        		if len(info) == 12:
            			bbox['xlefteye'] = float(info[2])
            			bbox['ylefteye'] = float(info[3])
            			bbox['xrighteye'] = float(info[4])
            			bbox['yrighteye'] = float(info[5])
            			bbox['xnose'] = float(info[6])
            			bbox['ynose'] = float(info[7])
            			bbox['xleftmouth'] = float(info[8])
			        bbox['yleftmouth'] = float(info[9])
            			bbox['xrightmouth'] = float(info[10])
            			bbox['yrightmouth'] = float(info[11])
            
        		data_example['bbox'] = bbox
        		self._dataset.append(data_example)

		if(len(self._dataset)):
			random.shuffle(self._dataset)
			self._is_valid = True

    		return(self._is_valid)

	def _add_to_tfrecord(self, filename, image_example, tfrecord_writer): 
    		image_data, height, width = _process_image_withoutcoder(filename)
    		example = _convert_to_example_simple(image_example, image_data)
    		tfrecord_writer.write(example.SerializeToString())

	def generate(self, input_file_name, target_dir, target_name):

    		tensorflow_filename = self._get_output_filename(target_dir, target_name)
    		if( tf.gfile.Exists(tensorflow_filename) ):
        		print('Dataset files already exist. Exiting without re-creating them.')
	        	return(True)

		if(not self._read_dataset(input_file_name)):
			return(False)

    		with tf.python_io.TFRecordWriter(tensorflow_filename) as tfrecord_writer:
        		for i, image_example in enumerate(self._dataset):
            			filename = image_example['filename']
            			self._add_to_tfrecord(filename, image_example, tfrecord_writer)
    		tfrecord_writer.close()

		return(True)



