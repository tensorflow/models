# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applsicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cityscale data loader for RNGDet."""

import tensorflow as tf
import numpy as np
from official.vision.dataloaders import decoder
from official.vision.dataloaders import parser

class Decoder(decoder.Decoder):
  """A tf.Example decoder for RNGDet."""

  def __init__(self):

    self._keys_to_features = {
    "sat_roi": tf.io.VarLenFeature(tf.int64),
    "label_masks_roi": tf.io.VarLenFeature(tf.int64),
    "historical_roi": tf.io.VarLenFeature(tf.int64),
    "gt_probs": tf.io.VarLenFeature(tf.float32),
    "gt_coords": tf.io.VarLenFeature(tf.float32),
    "list_len": tf.io.FixedLenFeature((), tf.int64),
    "gt_masks": tf.io.VarLenFeature(tf.int64),
    }

  def decode(self, serialized_example):
    parsed_tensors = tf.io.parse_single_example(
        serialized=serialized_example, features=self._keys_to_features)
    for k in parsed_tensors:
      if isinstance(parsed_tensors[k], tf.SparseTensor):
        if parsed_tensors[k].dtype == tf.string:
          parsed_tensors[k] = tf.sparse.to_dense(
              parsed_tensors[k], default_value='')
        else:
          parsed_tensors[k] = tf.sparse.to_dense(
              parsed_tensors[k], default_value=0)
    decoded_tensors = {
        'sat_roi': parsed_tensors['sat_roi'],
        'label_masks_roi': parsed_tensors['label_masks_roi'],
        'historical_roi': parsed_tensors['historical_roi'],
        'gt_probs': tf.cast(parsed_tensors['gt_probs'], tf.int64),
        'gt_coords': parsed_tensors['gt_coords'],
        'list_len': parsed_tensors['list_len'],
        'gt_masks': parsed_tensors['gt_masks']
    }

    return decoded_tensors


class Parser(parser.Parser):
  """Parse an image and its annotations into a dictionary of tensors."""

  def __init__(
      self,
      roi_size: int = 128,
      num_queries: int = 10,
      dtype='float32'
  ):
    self._roi_size = roi_size
    self._num_queries = num_queries
    self._dtype = dtype

  def parse_fn(self, is_training):
    """Returns a parse fn that reads and parses raw tensors from the decoder.
    Args:
      is_training: a `bool` to indicate whether it is in training mode.
    Returns:
      parse: a `callable` that takes the serialized example and generate the
        images, labels tuple where labels is a dict of Tensors that contains
        labels.
    """
    def parse(decoded_tensors):
      """Parses the serialized example data."""
      if is_training:
        return self._parse_train_data(decoded_tensors)
      else:
        return self._parse_eval_data(decoded_tensors)

    return parse

  def _parse_train_data(self, data):
    """Parses data for training and evaluation."""
    sat_roi = tf.reshape(data['sat_roi'], [self._roi_size, self._roi_size, 3])
    label_masks_roi = tf.reshape(
        data['label_masks_roi'], [self._roi_size, self._roi_size, 2])
    historical_roi = tf.reshape(
        data['historical_roi'], [self._roi_size, self._roi_size, 1])
    gt_coords = tf.reshape(data['gt_coords'], [self._num_queries, 2])
    gt_probs = tf.reshape(data['gt_probs'], [self._num_queries])
    gt_masks = tf.reshape( data['gt_masks'], [self._roi_size, self._roi_size, self._num_queries])

    sat_roi = tf.cast(sat_roi, tf.float32)/255
    sat_roi = sat_roi * (
        0.7 + 0.3 * tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32))

    rot_index = np.random.randint(0, 4)
    cos_theta = 0 if rot_index%2 is 1 else (1 if rot_index is 0 else -1)
    sin_theta = 0 if rot_index%2 is 0 else (1 if rot_index is 1 else -1) 
    R = tf.constant([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=tf.float32) 

    gt_coords = tf.reverse(gt_coords, axis=[1])
    gt_coords = tf.transpose(tf.linalg.matmul(R, gt_coords, transpose_b=True) ) 
    gt_coords = tf.reverse(gt_coords, axis=[1])

    label_masks_roi = tf.image.rot90(label_masks_roi, rot_index)/255 #counter clock wise
    historical_roi = tf.image.rot90(historical_roi, rot_index)/255 #counter clock wise 
    sat_roi = tf.image.rot90(sat_roi, rot_index)
    gt_masks = tf.image.rot90(gt_masks, rot_index)/255
    sat_roi = tf.cast(sat_roi, dtype=self._dtype)
    historical_roi = tf.cast(historical_roi, dtype=self._dtype)
    images = {
        'sat_roi': sat_roi,
        'historical_roi': historical_roi,
    }
    labels = {
        'label_masks_roi': label_masks_roi,
        'gt_probs': gt_probs,
        'gt_coords': gt_coords,
        'list_len': data['list_len'],
        'gt_masks': gt_masks,
    }

    return images, labels

  def _parse_eval_data(self, data):
    """Parses data for training and evaluation."""
    sat_roi = tf.reshape(data['sat_roi'], [self._roi_size, self._roi_size, 3])
    sat_roi = tf.cast(sat_roi, tf.float32)/255
    label_masks_roi = tf.reshape(
        data['label_masks_roi'], [self._roi_size, self._roi_size, 2])/255
    historical_roi = tf.reshape(
        data['historical_roi'], [self._roi_size, self._roi_size, 1])/255
    gt_coords = tf.reshape(data['gt_coords'], [self._num_queries, 2])
    gt_probs = tf.reshape(data['gt_probs'], [self._num_queries])
    gt_masks = tf.reshape(
        data['gt_masks'], [self._roi_size, self._roi_size, self._num_queries])/255

    sat_roi = sat_roi * (
        0.7 + 0.3 * tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32))
    rot_index = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)

    # Define the rotation matrix
    cos_theta = 0 if rot_index%2 is 1 else (1 if rot_index is 0 else -1)
    sin_theta = 0 if rot_index%2 is 0 else (1 if rot_index is 1 else -1)
    R = tf.constant([[cos_theta, -sin_theta], [sin_theta, cos_theta]],
                    dtype=tf.float32)

    gt_coords = tf.transpose(tf.linalg.matmul(R, gt_coords, transpose_b=True)) 
    label_masks_roi = tf.image.rot90(label_masks_roi, rot_index)
    historical_roi = tf.image.rot90(historical_roi, rot_index)
    sat_roi = tf.image.rot90(sat_roi, rot_index)
    gt_masks = tf.image.rot90(gt_masks, rot_index)

    images = {
        'sat_roi': sat_roi,
        'historical_roi': historical_roi,
    }
    labels = {
        'label_masks_roi': label_masks_roi,
        'gt_probs': gt_probs,
        'gt_coords': gt_coords,
        'list_len': data['list_len'],
        'gt_masks': gt_masks,
    }

    return images, labels
