# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Provides dataset dictionaries as used in our network models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.slim.python.slim.data import dataset
from tensorflow.contrib.slim.python.slim.data import dataset_data_provider
from tensorflow.contrib.slim.python.slim.data import tfexample_decoder

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'Images',
    'mask': 'Masks',
    'vox': 'Voxels'
}


def _get_split(file_pattern, num_samples, num_views, image_size, vox_size):
  """Get dataset.Dataset for the given dataset file pattern and properties."""

  # A dictionary from TF-Example keys to tf.FixedLenFeature instance.
  keys_to_features = {
      'image': tf.FixedLenFeature(
          shape=[num_views, image_size, image_size, 3],
          dtype=tf.float32, default_value=None),
      'mask': tf.FixedLenFeature(
          shape=[num_views, image_size, image_size, 1],
          dtype=tf.float32, default_value=None),
      'vox': tf.FixedLenFeature(
          shape=[vox_size, vox_size, vox_size, 1],
          dtype=tf.float32, default_value=None),
  }

  items_to_handler = {
      'image': tfexample_decoder.Tensor(
          'image', shape=[num_views, image_size, image_size, 3]),
      'mask': tfexample_decoder.Tensor(
          'mask', shape=[num_views, image_size, image_size, 1]),
      'vox': tfexample_decoder.Tensor(
          'vox', shape=[vox_size, vox_size, vox_size, 1])
  }

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handler)

  return dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=num_samples,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)


def get(dataset_dir,
        dataset_name,
        split_name,
        shuffle=True,
        num_readers=1,
        common_queue_capacity=64,
        common_queue_min=50):
  """Provides input data for a specified dataset and split."""

  dataset_to_kwargs = {
      'shapenet_chair': {
          'file_pattern': '03001627_%s.tfrecords' % split_name,
          'num_views': 24,
          'image_size': 64,
          'vox_size': 32,
      }, 'shapenet_all': {
          'file_pattern': '*_%s.tfrecords' % split_name,
          'num_views': 24,
          'image_size': 64,
          'vox_size': 32,
      },
  }

  split_sizes = {
      'shapenet_chair': {
          'train': 4744,
          'val': 678,
          'test': 1356,
      },
      'shapenet_all': {
          'train': 30643,
          'val': 4378,
          'test': 8762,
      }
  }

  kwargs = dataset_to_kwargs[dataset_name]
  kwargs['file_pattern'] = os.path.join(dataset_dir, kwargs['file_pattern'])
  kwargs['num_samples'] = split_sizes[dataset_name][split_name]

  dataset_split = _get_split(**kwargs)
  data_provider = dataset_data_provider.DatasetDataProvider(
      dataset_split,
      num_readers=num_readers,
      common_queue_capacity=common_queue_capacity,
      common_queue_min=common_queue_min,
      shuffle=shuffle)

  inputs = {
      'num_samples': dataset_split.num_samples,
  }

  [image, mask, vox] = data_provider.get(['image', 'mask', 'vox'])
  inputs['image'] = image
  inputs['mask'] = mask
  inputs['voxel'] = vox

  return inputs
