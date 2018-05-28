# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Converts the generic dataset to TFRecords of TF-Example protos.

This module reads the files that make up the generic dataset and creates two 
TFRecord datasets: one for train and one for test. Each TFRecord dataset is 
comprised of a set of TF-Example protocol buffers, each of which contain 
a single image and label.

The script can take less or more time depending upon size of the generic dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

# The number of images in the validation set.
_NUM_VALIDATION = 0

# The percentage of images in the validation set.
_VALIDATION_PERCENTAGE = 20.0

# Seed for repeatability.
_RANDOM_SEED = 7

# The number of samples per shard for a given dataset split.
_MAX_NUM_PER_SHARD = 40960

# Minimum number of shards per dataset split.
_MIN_NUM_SHARDS = 16

# The number of shards per dataset split.
_NUM_SHARDS = 0

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return( image )


def _get_filenames_and_classes(source_dir, dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    source_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.
    dataset_dir: The dataset directory where the dataset is stored.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  directories = []
  class_names = []
  for filename in os.listdir(source_dir):
    path = os.path.join(source_dir, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return( photo_filenames, sorted(class_names) )


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'generic_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return( os.path.join(dataset_dir, output_filename) )


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  number_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
  if( number_per_shard%4 != 0):
    number_per_shard = number_per_shard + (4 - number_per_shard%4)

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * number_per_shard
          end_ndx = min((shard_id+1) * number_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            if( i % 1000 == 0 ):
            	sys.stdout.write('\rConverting image %d/%d from shard %d' % (i, len(filenames), shard_id))
            	sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return( False )
  return( True )

def create_metadata_file(photo_filenames, class_names, dataset_dir):
  global _NUM_VALIDATION

  total_images = len(photo_filenames)
  _NUM_VALIDATION = int((total_images*_VALIDATION_PERCENTAGE)/100.0)

  no_of_validation_images = _NUM_VALIDATION  
  no_of_training_images = total_images - no_of_validation_images
  no_of_classes = len(class_names)
  if( (no_of_classes <=0) or  ( (no_of_validation_images <= 0) and (no_of_training_images <= 0) ) ):
      return(False) 
 
  dataset_metadata = {}
  dataset_metadata['classes'] = no_of_classes
  dataset_metadata['validation'] = no_of_validation_images
  dataset_metadata['train'] = no_of_training_images
  dataset_utils.write_metadata_file(dataset_metadata, dataset_dir)

  return(True)

def run(source_dir, dataset_dir, validation_percentage):
  """Runs the conversion operation.

  Args:
    source_dir: The directory from where input data is read for processing.
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(source_dir):
    print('The source directory is missing. Error create the dataset without the source directory.')
    return

  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  photo_filenames, class_names = _get_filenames_and_classes(source_dir, dataset_dir)

  global _NUM_SHARDS  
  _NUM_SHARDS = int(math.ceil(len(photo_filenames) / float(_MAX_NUM_PER_SHARD)))
  _NUM_SHARDS = max(_NUM_SHARDS, _MIN_NUM_SHARDS)
  if( _NUM_SHARDS%2 != 0):
    _NUM_SHARDS = _NUM_SHARDS + (2 - _NUM_SHARDS%2)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  global _VALIDATION_PERCENTAGE
  if(not ( (validation_percentage < 0.0) or (validation_percentage > 100.0) ) ):
    _VALIDATION_PERCENTAGE = validation_percentage

  if(not create_metadata_file(photo_filenames, class_names, dataset_dir)):
    print('Error creating the metadata file.')
    return

  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  validation_filenames = photo_filenames[:_NUM_VALIDATION]
  training_filenames = photo_filenames[_NUM_VALIDATION:]

  # First, convert the training and validation sets.
  if(len(validation_filenames) > 0):
    _convert_dataset('validation', validation_filenames, class_names_to_ids, dataset_dir)

  if(len(training_filenames) > 0):
    _convert_dataset('train', training_filenames, class_names_to_ids, dataset_dir)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  print('\nFinished converting the generic dataset.')

