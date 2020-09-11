# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Data loader utils."""

# Import libraries
import tensorflow as tf

from official.vision.beta.ops import preprocess_ops


def process_source_id(source_id):
  """Processes source_id to the right format."""
  if source_id.dtype == tf.string:
    source_id = tf.cast(tf.strings.to_number(source_id), tf.int64)
  with tf.control_dependencies([source_id]):
    source_id = tf.cond(
        pred=tf.equal(tf.size(input=source_id), 0),
        true_fn=lambda: tf.cast(tf.constant(-1), tf.int64),
        false_fn=lambda: tf.identity(source_id))
  return source_id


def pad_groundtruths_to_fixed_size(groundtruths, size):
  """Pads the first dimension of groundtruths labels to the fixed size."""
  groundtruths['boxes'] = preprocess_ops.clip_or_pad_to_fixed_size(
      groundtruths['boxes'], size, -1)
  groundtruths['is_crowds'] = preprocess_ops.clip_or_pad_to_fixed_size(
      groundtruths['is_crowds'], size, 0)
  groundtruths['areas'] = preprocess_ops.clip_or_pad_to_fixed_size(
      groundtruths['areas'], size, -1)
  groundtruths['classes'] = preprocess_ops.clip_or_pad_to_fixed_size(
      groundtruths['classes'], size, -1)
  return groundtruths
