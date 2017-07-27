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

"""Tests for object_detection.utils.shape_utils."""

import tensorflow as tf

from object_detection.utils import shape_utils


class UtilTest(tf.test.TestCase):

  def test_pad_tensor_using_integer_input(self):
    t1 = tf.constant([1], dtype=tf.int32)
    pad_t1 = shape_utils.pad_tensor(t1, 2)
    t2 = tf.constant([[0.1, 0.2]], dtype=tf.float32)
    pad_t2 = shape_utils.pad_tensor(t2, 2)

    self.assertEqual(2, pad_t1.get_shape()[0])
    self.assertEqual(2, pad_t2.get_shape()[0])

    with self.test_session() as sess:
      pad_t1_result, pad_t2_result = sess.run([pad_t1, pad_t2])
      self.assertAllEqual([1, 0], pad_t1_result)
      self.assertAllClose([[0.1, 0.2], [0, 0]], pad_t2_result)

  def test_pad_tensor_using_tensor_input(self):
    t1 = tf.constant([1], dtype=tf.int32)
    pad_t1 = shape_utils.pad_tensor(t1, tf.constant(2))
    t2 = tf.constant([[0.1, 0.2]], dtype=tf.float32)
    pad_t2 = shape_utils.pad_tensor(t2, tf.constant(2))

    with self.test_session() as sess:
      pad_t1_result, pad_t2_result = sess.run([pad_t1, pad_t2])
      self.assertAllEqual([1, 0], pad_t1_result)
      self.assertAllClose([[0.1, 0.2], [0, 0]], pad_t2_result)

  def test_clip_tensor_using_integer_input(self):
    t1 = tf.constant([1, 2, 3], dtype=tf.int32)
    clip_t1 = shape_utils.clip_tensor(t1, 2)
    t2 = tf.constant([[0.1, 0.2], [0.2, 0.4], [0.5, 0.8]], dtype=tf.float32)
    clip_t2 = shape_utils.clip_tensor(t2, 2)

    self.assertEqual(2, clip_t1.get_shape()[0])
    self.assertEqual(2, clip_t2.get_shape()[0])

    with self.test_session() as sess:
      clip_t1_result, clip_t2_result = sess.run([clip_t1, clip_t2])
      self.assertAllEqual([1, 2], clip_t1_result)
      self.assertAllClose([[0.1, 0.2], [0.2, 0.4]], clip_t2_result)

  def test_clip_tensor_using_tensor_input(self):
    t1 = tf.constant([1, 2, 3], dtype=tf.int32)
    clip_t1 = shape_utils.clip_tensor(t1, tf.constant(2))
    t2 = tf.constant([[0.1, 0.2], [0.2, 0.4], [0.5, 0.8]], dtype=tf.float32)
    clip_t2 = shape_utils.clip_tensor(t2, tf.constant(2))

    with self.test_session() as sess:
      clip_t1_result, clip_t2_result = sess.run([clip_t1, clip_t2])
      self.assertAllEqual([1, 2], clip_t1_result)
      self.assertAllClose([[0.1, 0.2], [0.2, 0.4]], clip_t2_result)

  def test_pad_or_clip_tensor_using_integer_input(self):
    t1 = tf.constant([1], dtype=tf.int32)
    tt1 = shape_utils.pad_or_clip_tensor(t1, 2)
    t2 = tf.constant([[0.1, 0.2]], dtype=tf.float32)
    tt2 = shape_utils.pad_or_clip_tensor(t2, 2)

    t3 = tf.constant([1, 2, 3], dtype=tf.int32)
    tt3 = shape_utils.clip_tensor(t3, 2)
    t4 = tf.constant([[0.1, 0.2], [0.2, 0.4], [0.5, 0.8]], dtype=tf.float32)
    tt4 = shape_utils.clip_tensor(t4, 2)

    self.assertEqual(2, tt1.get_shape()[0])
    self.assertEqual(2, tt2.get_shape()[0])
    self.assertEqual(2, tt3.get_shape()[0])
    self.assertEqual(2, tt4.get_shape()[0])

    with self.test_session() as sess:
      tt1_result, tt2_result, tt3_result, tt4_result = sess.run(
          [tt1, tt2, tt3, tt4])
      self.assertAllEqual([1, 0], tt1_result)
      self.assertAllClose([[0.1, 0.2], [0, 0]], tt2_result)
      self.assertAllEqual([1, 2], tt3_result)
      self.assertAllClose([[0.1, 0.2], [0.2, 0.4]], tt4_result)

  def test_pad_or_clip_tensor_using_tensor_input(self):
    t1 = tf.constant([1], dtype=tf.int32)
    tt1 = shape_utils.pad_or_clip_tensor(t1, tf.constant(2))
    t2 = tf.constant([[0.1, 0.2]], dtype=tf.float32)
    tt2 = shape_utils.pad_or_clip_tensor(t2, tf.constant(2))

    t3 = tf.constant([1, 2, 3], dtype=tf.int32)
    tt3 = shape_utils.clip_tensor(t3, tf.constant(2))
    t4 = tf.constant([[0.1, 0.2], [0.2, 0.4], [0.5, 0.8]], dtype=tf.float32)
    tt4 = shape_utils.clip_tensor(t4, tf.constant(2))

    with self.test_session() as sess:
      tt1_result, tt2_result, tt3_result, tt4_result = sess.run(
          [tt1, tt2, tt3, tt4])
      self.assertAllEqual([1, 0], tt1_result)
      self.assertAllClose([[0.1, 0.2], [0, 0]], tt2_result)
      self.assertAllEqual([1, 2], tt3_result)
      self.assertAllClose([[0.1, 0.2], [0.2, 0.4]], tt4_result)


if __name__ == '__main__':
  tf.test.main()
