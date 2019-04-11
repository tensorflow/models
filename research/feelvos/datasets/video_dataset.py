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

"""Provides data from video object segmentation datasets.

This file provides both images and annotations (instance segmentations) for
TensorFlow. Currently, we support the following datasets:

1. DAVIS 2017 (https://davischallenge.org/davis2017/code.html).

2. DAVIS 2016 (https://davischallenge.org/davis2016/code.html).

3. YouTube-VOS (https://youtube-vos.org/dataset/download).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path
import tensorflow as tf
from feelvos.datasets import tfsequence_example_decoder

slim = tf.contrib.slim
dataset = slim.dataset
tfexample_decoder = slim.tfexample_decoder


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    ['splits_to_sizes',   # Splits of the dataset into training, val, and test.
     'num_classes',   # Number of semantic classes.
     'ignore_label',  # Ignore label value.
    ]
)

_DAVIS_2016_INFORMATION = DatasetDescriptor(
    splits_to_sizes={'train': [30, 1830],
                     'val': [20, 1376]},
    num_classes=2,
    ignore_label=255,
)

_DAVIS_2017_INFORMATION = DatasetDescriptor(
    splits_to_sizes={'train': [60, 4219],
                     'val': [30, 2023],
                     'test-dev': [30, 2037]},
    num_classes=None,  # Number of instances per videos differ.
    ignore_label=255,
)

_YOUTUBE_VOS_2018_INFORMATION = DatasetDescriptor(
    # Leave these sizes as None to allow for different splits into
    # training and validation sets.
    splits_to_sizes={'train': [None, None],
                     'val': [None, None]},
    num_classes=None,  # Number of instances per video differs.
    ignore_label=255,
)

_DATASETS_INFORMATION = {
    'davis_2016': _DAVIS_2016_INFORMATION,
    'davis_2017': _DAVIS_2017_INFORMATION,
    'youtube_vos_2018': _YOUTUBE_VOS_2018_INFORMATION,
}

# Default file pattern of SSTable. Note we include '-' to avoid the confusion
# between `train-` and `trainval-` sets.
_FILE_PATTERN = '%s-*'


def get_dataset(dataset_name,
                split_name,
                dataset_dir,
                file_pattern=None,
                data_type='tf_sequence_example',
                decode_video_frames=False):
  """Gets an instance of slim Dataset.

  Args:
    dataset_name: String, dataset name.
    split_name: String, the train/val Split name.
    dataset_dir: String, the directory of the dataset sources.
    file_pattern: String, file pattern of SSTable.
    data_type: String, data type. Currently supports 'tf_example' and
      'annotated_image'.
    decode_video_frames: Boolean, decode the images or not. Not decoding it here
        is useful if we subsample later

  Returns:
    An instance of slim Dataset.

  Raises:
    ValueError: If the dataset_name or split_name is not recognized, or if
      the dataset_type is not supported.
  """
  if dataset_name not in _DATASETS_INFORMATION:
    raise ValueError('The specified dataset is not supported yet.')

  splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes

  if split_name not in splits_to_sizes:
    raise ValueError('data split name %s not recognized' % split_name)

  # Prepare the variables for different datasets.
  num_classes = _DATASETS_INFORMATION[dataset_name].num_classes
  ignore_label = _DATASETS_INFORMATION[dataset_name].ignore_label

  if file_pattern is None:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
  if data_type == 'tf_sequence_example':
    keys_to_context_features = {
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/width': tf.FixedLenFeature((), tf.int64, default_value=0),
        'segmentation/object/format': tf.FixedLenFeature(
            (), tf.string, default_value='png'),
        'video_id': tf.FixedLenFeature((), tf.string, default_value='unknown')
    }
    label_name = 'class' if dataset_name == 'davis_2016' else 'object'
    keys_to_sequence_features = {
        'image/encoded': tf.FixedLenSequenceFeature((), dtype=tf.string),
        'segmentation/{}/encoded'.format(label_name):
            tf.FixedLenSequenceFeature((), tf.string),
        'segmentation/{}/encoded'.format(label_name):
            tf.FixedLenSequenceFeature((), tf.string),
    }
    items_to_handlers = {
        'height': tfexample_decoder.Tensor('image/height'),
        'width': tfexample_decoder.Tensor('image/width'),
        'video_id': tfexample_decoder.Tensor('video_id')
    }
    if decode_video_frames:
      decode_image_handler = tfexample_decoder.Image(
          image_key='image/encoded',
          format_key='image/format',
          channels=3,
          repeated=True)
      items_to_handlers['image'] = decode_image_handler
      decode_label_handler = tfexample_decoder.Image(
          image_key='segmentation/{}/encoded'.format(label_name),
          format_key='segmentation/{}/format'.format(label_name),
          channels=1,
          repeated=True)
      items_to_handlers['labels_class'] = decode_label_handler
    else:
      items_to_handlers['image/encoded'] = tfexample_decoder.Tensor(
          'image/encoded')
      items_to_handlers[
          'segmentation/object/encoded'] = tfexample_decoder.Tensor(
              'segmentation/{}/encoded'.format(label_name))
    decoder = tfsequence_example_decoder.TFSequenceExampleDecoder(
        keys_to_context_features, keys_to_sequence_features, items_to_handlers)
  else:
    raise ValueError('Unknown data type.')

  size = splits_to_sizes[split_name]
  if isinstance(size, collections.Sequence):
    num_videos = size[0]
    num_samples = size[1]
  else:
    num_videos = 0
    num_samples = size

  return dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=num_samples,
      num_videos=num_videos,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      ignore_label=ignore_label,
      num_classes=num_classes,
      name=dataset_name,
      multi_label=True)
