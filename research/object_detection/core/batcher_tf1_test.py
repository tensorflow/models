# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for object_detection.core.batcher."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from object_detection.core import batcher

slim = contrib_slim


class BatcherTest(tf.test.TestCase):

  def test_batch_and_unpad_2d_tensors_of_different_sizes_in_1st_dimension(self):
    with self.test_session() as sess:
      batch_size = 3
      num_batches = 2
      examples = tf.Variable(tf.constant(2, dtype=tf.int32))
      counter = examples.count_up_to(num_batches * batch_size + 2)
      boxes = tf.tile(
          tf.reshape(tf.range(4), [1, 4]), tf.stack([counter, tf.constant(1)]))
      batch_queue = batcher.BatchQueue(
          tensor_dict={'boxes': boxes},
          batch_size=batch_size,
          batch_queue_capacity=100,
          num_batch_queue_threads=1,
          prefetch_queue_capacity=100)
      batch = batch_queue.dequeue()

      for tensor_dict in batch:
        for tensor in tensor_dict.values():
          self.assertAllEqual([None, 4], tensor.get_shape().as_list())

      tf.initialize_all_variables().run()
      with slim.queues.QueueRunners(sess):
        i = 2
        for _ in range(num_batches):
          batch_np = sess.run(batch)
          for tensor_dict in batch_np:
            for tensor in tensor_dict.values():
              self.assertAllEqual(tensor, np.tile(np.arange(4), (i, 1)))
              i += 1
        with self.assertRaises(tf.errors.OutOfRangeError):
          sess.run(batch)

  def test_batch_and_unpad_2d_tensors_of_different_sizes_in_all_dimensions(
      self):
    with self.test_session() as sess:
      batch_size = 3
      num_batches = 2
      examples = tf.Variable(tf.constant(2, dtype=tf.int32))
      counter = examples.count_up_to(num_batches * batch_size + 2)
      image = tf.reshape(
          tf.range(counter * counter), tf.stack([counter, counter]))
      batch_queue = batcher.BatchQueue(
          tensor_dict={'image': image},
          batch_size=batch_size,
          batch_queue_capacity=100,
          num_batch_queue_threads=1,
          prefetch_queue_capacity=100)
      batch = batch_queue.dequeue()

      for tensor_dict in batch:
        for tensor in tensor_dict.values():
          self.assertAllEqual([None, None], tensor.get_shape().as_list())

      tf.initialize_all_variables().run()
      with slim.queues.QueueRunners(sess):
        i = 2
        for _ in range(num_batches):
          batch_np = sess.run(batch)
          for tensor_dict in batch_np:
            for tensor in tensor_dict.values():
              self.assertAllEqual(tensor, np.arange(i * i).reshape((i, i)))
              i += 1
        with self.assertRaises(tf.errors.OutOfRangeError):
          sess.run(batch)

  def test_batch_and_unpad_2d_tensors_of_same_size_in_all_dimensions(self):
    with self.test_session() as sess:
      batch_size = 3
      num_batches = 2
      examples = tf.Variable(tf.constant(1, dtype=tf.int32))
      counter = examples.count_up_to(num_batches * batch_size + 1)
      image = tf.reshape(tf.range(1, 13), [4, 3]) * counter
      batch_queue = batcher.BatchQueue(
          tensor_dict={'image': image},
          batch_size=batch_size,
          batch_queue_capacity=100,
          num_batch_queue_threads=1,
          prefetch_queue_capacity=100)
      batch = batch_queue.dequeue()

      for tensor_dict in batch:
        for tensor in tensor_dict.values():
          self.assertAllEqual([4, 3], tensor.get_shape().as_list())

      tf.initialize_all_variables().run()
      with slim.queues.QueueRunners(sess):
        i = 1
        for _ in range(num_batches):
          batch_np = sess.run(batch)
          for tensor_dict in batch_np:
            for tensor in tensor_dict.values():
              self.assertAllEqual(tensor, np.arange(1, 13).reshape((4, 3)) * i)
              i += 1
        with self.assertRaises(tf.errors.OutOfRangeError):
          sess.run(batch)

  def test_batcher_when_batch_size_is_one(self):
    with self.test_session() as sess:
      batch_size = 1
      num_batches = 2
      examples = tf.Variable(tf.constant(2, dtype=tf.int32))
      counter = examples.count_up_to(num_batches * batch_size + 2)
      image = tf.reshape(
          tf.range(counter * counter), tf.stack([counter, counter]))
      batch_queue = batcher.BatchQueue(
          tensor_dict={'image': image},
          batch_size=batch_size,
          batch_queue_capacity=100,
          num_batch_queue_threads=1,
          prefetch_queue_capacity=100)
      batch = batch_queue.dequeue()

      for tensor_dict in batch:
        for tensor in tensor_dict.values():
          self.assertAllEqual([None, None], tensor.get_shape().as_list())

      tf.initialize_all_variables().run()
      with slim.queues.QueueRunners(sess):
        i = 2
        for _ in range(num_batches):
          batch_np = sess.run(batch)
          for tensor_dict in batch_np:
            for tensor in tensor_dict.values():
              self.assertAllEqual(tensor, np.arange(i * i).reshape((i, i)))
              i += 1
        with self.assertRaises(tf.errors.OutOfRangeError):
          sess.run(batch)


if __name__ == '__main__':
  tf.test.main()
