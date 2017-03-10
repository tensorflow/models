# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the Cifar10 dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/data/create_cifar10_dataset.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'lipread_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 2, 'validation': 2}

_NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading cifar10.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN

  # # The format is as below
  # file_pattern = 'flowers_%s_*.tfrecord'
  # file_pattern % split_name = file_pattern = 'flowers_train_*.tfrecord'
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'pair/speech': tf.FixedLenFeature((), tf.string, default_value=''),
      'pair/mouth': tf.FixedLenFeature((), tf.string, default_value=''),
      'speech/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      'mouth/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      'pair/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'pair/channel_speech': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'pair/feature_speech': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'pair/frame_speech': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'pair/channel_mouth': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'pair/height_mouth': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'pair/width_mouth': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      # 'image': slim.tfexample_decoder.Image(shape=[1,2000,1]),
      # 'lip': slim.tfexample_decoder.Image(shape=[1, 2000, 1]),
      'speech': slim.tfexample_decoder.Image(format_key='speech/format', image_key='pair/speech', shape=[13,15,1],  channels=1),
      'mouth': slim.tfexample_decoder.Image(format_key='mouth/format', image_key='pair/mouth', shape=[47,73,9], channels=9),
      # 'speech': slim.tfexample_decoder.Tensor(tensor_key='pair/speech', shape=[13,15,1]),
      # 'mouth': slim.tfexample_decoder.Tensor(tensor_key='mouth/format', shape=[47,73,9]),
      'speech_format': slim.tfexample_decoder.Tensor('speech/format'),
      'mouth_format': slim.tfexample_decoder.Tensor('mouth/format'),
      'label': slim.tfexample_decoder.Tensor('pair/class/label'),
      'channel_speech': slim.tfexample_decoder.Tensor('pair/channel_speech'),
      'feature_speech': slim.tfexample_decoder.Tensor('pair/feature_speech'),
      'frame_speech': slim.tfexample_decoder.Tensor('pair/frame_speech'),
      'channel_mouth': slim.tfexample_decoder.Tensor('pair/channel_mouth'),
      'height_mouth': slim.tfexample_decoder.Tensor('pair/height_mouth'),
      'width_mouth': slim.tfexample_decoder.Tensor('pair/width_mouth'),

  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)


  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
