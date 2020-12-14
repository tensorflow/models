# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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
# Lint as: python3
"""A module for miscelaneous utils."""
import tensorflow as tf


def random_substr(str_tensor, max_words):
  """Select random substring if the input has more than max_words."""
  word_batch_r = tf.strings.split(str_tensor)
  row_splits = word_batch_r.row_splits
  words = word_batch_r.values
  start_idx = row_splits[:-1]
  end_idx = row_splits[1:]
  words_per_example = end_idx - start_idx
  ones = tf.ones_like(end_idx)
  max_val = tf.maximum(ones, words_per_example - max_words)
  max_words_batch = tf.reduce_max(words_per_example)
  rnd = tf.random.uniform(
      tf.shape(start_idx), minval=0, maxval=max_words_batch, dtype=tf.int64)
  off_start_idx = tf.math.floormod(rnd, max_val)
  new_words_per_example = tf.where(
      tf.equal(max_val, 1), words_per_example, ones * max_words)
  new_start_idx = start_idx + off_start_idx
  new_end_idx = new_start_idx + new_words_per_example
  indices = tf.expand_dims(tf.range(tf.size(words), dtype=tf.int64), axis=0)
  within_limit = tf.logical_and(
      tf.greater_equal(indices, tf.expand_dims(new_start_idx, axis=1)),
      tf.less(indices, tf.expand_dims(new_end_idx, axis=1)))
  keep_indices = tf.reduce_any(within_limit, axis=0)
  keep_indices = tf.cast(keep_indices, dtype=tf.int32)
  _, selected_words = tf.dynamic_partition(words, keep_indices, 2)
  row_splits = tf.math.cumsum(new_words_per_example)
  row_splits = tf.concat([[0], row_splits], axis=0)
  new_tensor = tf.RaggedTensor.from_row_splits(
      values=selected_words, row_splits=row_splits)
  return tf.strings.reduce_join(new_tensor, axis=1, separator=" ")
