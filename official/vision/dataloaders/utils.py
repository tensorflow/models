# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Data loader utils."""
from typing import Dict

import tensorflow as tf, tf_keras

from official.vision.ops import preprocess_ops


def process_source_id(source_id: tf.Tensor) -> tf.Tensor:
  """Processes source_id to the right format.

  Args:
    source_id: A `tf.Tensor` that contains the source ID. It can be empty.

  Returns:
    A formatted source ID.
  """
  if source_id.dtype == tf.string:
    source_id = tf.strings.to_number(source_id, tf.int64)
  with tf.control_dependencies([source_id]):
    source_id = tf.cond(
        pred=tf.equal(tf.size(input=source_id), 0),
        true_fn=lambda: tf.cast(tf.constant(-1), tf.int64),
        false_fn=lambda: tf.identity(source_id))
  return source_id


def pad_groundtruths_to_fixed_size(groundtruths: Dict[str, tf.Tensor],
                                   size: int) -> Dict[str, tf.Tensor]:
  """Pads the first dimension of groundtruths labels to the fixed size.

  Args:
    groundtruths: A dictionary of {`str`: `tf.Tensor`} that contains groundtruth
      annotations of `boxes`, `is_crowds`, `areas` and `classes`.
    size: An `int` that specifies the expected size of the first dimension of
      padded tensors.

  Returns:
    A dictionary of the same keys as input and padded tensors as values.

  """
  groundtruths['boxes'] = preprocess_ops.clip_or_pad_to_fixed_size(
      groundtruths['boxes'], size, -1)
  groundtruths['is_crowds'] = preprocess_ops.clip_or_pad_to_fixed_size(
      groundtruths['is_crowds'], size, 0)
  groundtruths['areas'] = preprocess_ops.clip_or_pad_to_fixed_size(
      groundtruths['areas'], size, -1)
  groundtruths['classes'] = preprocess_ops.clip_or_pad_to_fixed_size(
      groundtruths['classes'], size, -1)
  if 'attributes' in groundtruths:
    for k, v in groundtruths['attributes'].items():
      groundtruths['attributes'][k] = preprocess_ops.clip_or_pad_to_fixed_size(
          v, size, -1)
  return groundtruths


def binarize_matting_map(matting_map: tf.Tensor,
                         threshold: float = 0.5) -> tf.Tensor:
  """Binarizes a matting map.

  If the matting_map value is above a threshold, set it as 1 otherwise 0. The
  binarization is done for every element in the matting_map.

  Args:
    matting_map: The groundtruth in the matting map format.
    threshold: The threshold used to binarize the matting map.

  Returns:
    The binarized labels (0 for BG, 1 for FG) as tf.float32.
  """
  return tf.cast(tf.greater(matting_map, threshold), tf.float32)
