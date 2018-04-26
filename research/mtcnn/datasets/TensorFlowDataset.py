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

	@classmethod
	def tensorflow_file_name(cls, target_dir, target_name):
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

	def read_tensorflow_file(self, tensorflow_file_name, batch_size, image_size):
    		filename_queue = tf.train.string_input_producer([tensorflow_file_name], shuffle=True)

    		reader = tf.TFRecordReader()
    		_, serialized_example = reader.read(filename_queue)
    		image_features = tf.parse_single_example(
        			serialized_example,
        				features={
            					'image/encoded': tf.FixedLenFeature([], tf.string),#one image  one record
            					'image/label': tf.FixedLenFeature([], tf.int64),
            					'image/roi': tf.FixedLenFeature([4], tf.float32),
            					'image/landmark': tf.FixedLenFeature([10],tf.float32)
        					}
    				)

    		image = tf.decode_raw(image_features['image/encoded'], tf.uint8)
    		image = tf.reshape(image, [image_size, image_size, 3])
    		image = (tf.cast(image, tf.float32)-127.5) / 128
    
    		# image = tf.image.per_image_standardization(image)
    		label = tf.cast(image_features['image/label'], tf.float32)
    		roi = tf.cast(image_features['image/roi'],tf.float32)
    		landmark = tf.cast(image_features['image/landmark'],tf.float32)
    		image, label, roi, landmark = tf.train.batch(
        						[image, label,roi,landmark],
        						batch_size=batch_size,
        						num_threads=2,
        						capacity=1 * batch_size
    							)
    		label = tf.reshape(label, [batch_size])
    		roi = tf.reshape(roi,[batch_size,4])
    		landmark = tf.reshape(landmark,[batch_size,10])
    		return( image, label, roi, landmark )

	def read_tensorflow_files(self, tensorflow_file_names, batch_sizes, image_size):
    		tensorflow_positive_file_name, tensorflow_part_file_name, tensorflow_negative_file_name, tensorflow_landmark_file_name = tensorflow_file_names
    		positive_batch_size, part_batch_size, negative_batch_size, landmark_batch_size = batch_sizes

    		positive_images, pos_label, pos_roi, pos_landmark = self.read_tensorflow_file(tensorflow_positive_file_name, positive_batch_size, image_size)
    		part_images, part_label, part_roi, part_landmark = self.read_tensorflow_file(tensorflow_part_file_name, part_batch_size, image_size)
    		neg_image,neg_label,neg_roi,neg_landmark = self.read_tensorflow_file(tensorflow_negative_file_name, negative_batch_size, image_size)
    		landmark_image,landmark_label,landmark_roi,landmark_landmark = self.read_tensorflow_file(tensorflow_landmark_file_name, landmark_batch_size, image_size)
    
    		images = tf.concat([positive_images, part_images, neg_image, landmark_image], 0, name="concat/image")
    		labels = tf.concat([pos_label,part_label,neg_label,landmark_label],0,name="concat/label")
    		rois = tf.concat([pos_roi,part_roi,neg_roi,landmark_roi],0,name="concat/roi")
    		landmarks = tf.concat([pos_landmark,part_landmark,neg_landmark,landmark_landmark],0,name="concat/landmark")

    		return( images, labels, rois, landmarks )

	def generate(self, input_file_name, target_root_dir, target_name):
		tensorflow_dir = os.path.join(target_root_dir, 'tensorflow')
		if(not os.path.exists(tensorflow_dir)):
			os.makedirs(tensorflow_dir)

    		tensorflow_filename = TensorFlowDataset.tensorflow_file_name(tensorflow_dir, target_name)
		if(not self._read_dataset(input_file_name)):
			return(False)

		total_number_of_samples = len(self._dataset)
		number_of_samples = 0
    		with tf.python_io.TFRecordWriter(tensorflow_filename) as tfrecord_writer:
        		for i, image_example in enumerate(self._dataset):
            			filename = image_example['filename']
            			self._add_to_tfrecord(filename, image_example, tfrecord_writer)
				number_of_samples = number_of_samples + 1
				if( number_of_samples % 1000 == 0):
					print('Processed ( %s / %s ) image samples.' % ( number_of_samples, total_number_of_samples ) )

		return(True)



