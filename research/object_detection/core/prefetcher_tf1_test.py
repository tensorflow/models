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

"""Tests for object_detection.core.prefetcher."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow as tf

# pylint: disable=g-bad-import-order,
from object_detection.core import prefetcher
from tensorflow.contrib import slim as contrib_slim
slim = contrib_slim
# pylint: disable=g-bad-import-order


class PrefetcherTest(tf.test.TestCase):
  """Test class for prefetcher."""

  def test_prefetch_tensors_with_fully_defined_shapes(self):
    with self.test_session() as sess:
      batch_size = 10
      image_size = 32
      num_batches = 5
      examples = tf.Variable(tf.constant(0, dtype=tf.int64))
      counter = examples.count_up_to(num_batches)
      image = tf.random_normal([batch_size, image_size,
                                image_size, 3],
                               dtype=tf.float32,
                               name='images')
      label = tf.random_uniform([batch_size, 1], 0, 10,
                                dtype=tf.int32, name='labels')

      prefetch_queue = prefetcher.prefetch(tensor_dict={'counter': counter,
                                                        'image': image,
                                                        'label': label},
                                           capacity=100)
      tensor_dict = prefetch_queue.dequeue()

      self.assertAllEqual(tensor_dict['image'].get_shape().as_list(),
                          [batch_size, image_size, image_size, 3])
      self.assertAllEqual(tensor_dict['label'].get_shape().as_list(),
                          [batch_size, 1])

      tf.initialize_all_variables().run()
      with slim.queues.QueueRunners(sess):
        for _ in range(num_batches):
          results = sess.run(tensor_dict)
          self.assertEquals(results['image'].shape,
                            (batch_size, image_size, image_size, 3))
          self.assertEquals(results['label'].shape, (batch_size, 1))
        with self.assertRaises(tf.errors.OutOfRangeError):
          sess.run(tensor_dict)

  def test_prefetch_tensors_with_partially_defined_shapes(self):
    with self.test_session() as sess:
      batch_size = 10
      image_size = 32
      num_batches = 5
      examples = tf.Variable(tf.constant(0, dtype=tf.int64))
      counter = examples.count_up_to(num_batches)
      image = tf.random_normal([batch_size,
                                tf.Variable(image_size),
                                tf.Variable(image_size), 3],
                               dtype=tf.float32,
                               name='image')
      image.set_shape([batch_size, None, None, 3])
      label = tf.random_uniform([batch_size, tf.Variable(1)], 0,
                                10, dtype=tf.int32, name='label')
      label.set_shape([batch_size, None])

      prefetch_queue = prefetcher.prefetch(tensor_dict={'counter': counter,
                                                        'image': image,
                                                        'label': label},
                                           capacity=100)
      tensor_dict = prefetch_queue.dequeue()

      self.assertAllEqual(tensor_dict['image'].get_shape().as_list(),
                          [batch_size, None, None, 3])
      self.assertAllEqual(tensor_dict['label'].get_shape().as_list(),
                          [batch_size, None])

      tf.initialize_all_variables().run()
      with slim.queues.QueueRunners(sess):
        for _ in range(num_batches):
          results = sess.run(tensor_dict)
          self.assertEquals(results['image'].shape,
                            (batch_size, image_size, image_size, 3))
          self.assertEquals(results['label'].shape, (batch_size, 1))
        with self.assertRaises(tf.errors.OutOfRangeError):
          sess.run(tensor_dict)


if __name__ == '__main__':
  tf.test.main()
