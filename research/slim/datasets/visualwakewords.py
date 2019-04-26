# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Provides data for Visual WakeWords Dataset with images+labels.

Visual WakeWords Dataset derives from the COCO dataset to design tiny models
classifying two classes, such as person/not-person. The COCO annotations
are filtered to two classes: person and not-person (or another user-defined
category). Bounding boxes for small objects with area less than 5% of the image
area are filtered out.
See build_visualwakewords_data.py which generates the Visual WakeWords dataset
annotations from the raw COCO dataset and converts them to TFRecord.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils


slim = tf.contrib.slim

_FILE_PATTERN = '%s.record-*'

_SPLITS_TO_SIZES = {
    'train': 82783,
    'validation': 40504,
}


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, an integer in {0, 1}',
    'object/bbox': 'A list of bounding boxes.',
    'object/label': 'A list of labels, all objects belong to the same class.',
}

_NUM_CLASSES = 2

# labels file
LABELS_FILENAME = 'labels.txt'


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading ImageNet.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources. It
      is assumed that the pattern contains a '%s' string so that the split name
      can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
          tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/class/label':
          tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/object/bbox/xmin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/class/label':
          tf.VarLenFeature(dtype=tf.int64),
  }

  items_to_handlers = {
      'image':
          slim.tfexample_decoder.Image('image/encoded', 'image/format'),
      'label':
          slim.tfexample_decoder.Tensor('image/class/label'),
      'object/bbox':
          slim.tfexample_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'],
                                             'image/object/bbox/'),
      'object/label':
          slim.tfexample_decoder.Tensor('image/object/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                    items_to_handlers)

  labels_to_names = None
  labels_file = os.path.join(dataset_dir, LABELS_FILENAME)
  if tf.gfile.Exists(labels_file):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
