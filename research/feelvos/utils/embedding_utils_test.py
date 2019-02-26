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

"""Tests for embedding utils."""

import unittest
import numpy as np
import tensorflow as tf
from feelvos.utils import embedding_utils

if embedding_utils.USE_CORRELATION_COST:
  # pylint: disable=g-import-not-at-top
  from correlation_cost.python.ops import correlation_cost_op


class EmbeddingUtilsTest(tf.test.TestCase):

  def test_pairwise_distances(self):
    x = np.arange(100, dtype=np.float32).reshape(20, 5)
    y = np.arange(100, 200, dtype=np.float32).reshape(20, 5)
    g = tf.Graph()
    with g.as_default():
      with self.test_session(graph=g) as sess:
        x = tf.constant(x)
        y = tf.constant(y)
        d1 = embedding_utils.pairwise_distances(x, y)
        d2 = embedding_utils.pairwise_distances2(x, y)
        d1_val, d2_val = sess.run([d1, d2])
        self.assertAllClose(d1_val, d2_val)

  @unittest.skipIf(not embedding_utils.USE_CORRELATION_COST,
                   'depends on correlation_cost')
  def test_correlation_cost_one_dimensional(self):
    a = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    b = np.array([[[[2.0], [1.0]], [[4.0], [3.0]]]])
    g = tf.Graph()
    with g.as_default():
      with self.test_session(graph=g) as sess:
        c = correlation_cost_op.correlation_cost(
            a, b, kernel_size=1, max_displacement=1, stride_1=1, stride_2=1,
            pad=1)
        c = tf.squeeze(c, axis=0)
        c_val = sess.run(c)
        self.assertAllEqual(c_val.shape, (2, 2, 9))
        for y in range(2):
          for x in range(2):
            for dy in range(-1, 2):
              for dx in range(-1, 2):
                a_slice = a[0, y, x, 0]
                if y + dy < 0 or y + dy > 1 or x + dx < 0 or x + dx > 1:
                  b_slice = 0
                else:
                  b_slice = b[0, y + dy, x + dx, 0]
                expected = a_slice * b_slice
                dy0 = dy + 1
                dx0 = dx + 1
                self.assertAlmostEqual(c_val[y, x, 3 * dy0 + dx0], expected)

  @unittest.skipIf(not embedding_utils.USE_CORRELATION_COST,
                   'depends on correlation_cost')
  def test_correlation_cost_two_dimensional(self):
    a = np.array([[[[1.0, -5.0], [7.0, 2.0]], [[1.0, 3.0], [3.0, 4.0]]]])
    b = np.array([[[[2.0, 1.0], [0.0, -9.0]], [[4.0, 3.0], [3.0, 1.0]]]])
    g = tf.Graph()
    with g.as_default():
      with self.test_session(graph=g) as sess:
        c = correlation_cost_op.correlation_cost(
            a, b, kernel_size=1, max_displacement=1, stride_1=1, stride_2=1,
            pad=1)
        c = tf.squeeze(c, axis=0)
        c_val = sess.run(c)
        self.assertAllEqual(c_val.shape, (2, 2, 9))
        for y in range(2):
          for x in range(2):
            for dy in range(-1, 2):
              for dx in range(-1, 2):
                a_slice = a[0, y, x, :]
                if y + dy < 0 or y + dy > 1 or x + dx < 0 or x + dx > 1:
                  b_slice = 0
                else:
                  b_slice = b[0, y + dy, x + dx, :]
                expected = (a_slice * b_slice).mean()
                dy0 = dy + 1
                dx0 = dx + 1
                self.assertAlmostEqual(c_val[y, x, 3 * dy0 + dx0], expected)

  @unittest.skipIf(not embedding_utils.USE_CORRELATION_COST,
                   'depends on correlation_cost')
  def test_local_pairwise_distances_one_dimensional(self):
    a = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
    b = np.array([[[2.0], [1.0]], [[4.0], [3.0]]])
    g = tf.Graph()
    with g.as_default():
      with self.test_session(graph=g) as sess:
        a_tf = tf.constant(a, dtype=tf.float32)
        b_tf = tf.constant(b, dtype=tf.float32)
        d = embedding_utils.local_pairwise_distances(a_tf, b_tf,
                                                     max_distance=1)
        d_val = sess.run(d)
        for y in range(2):
          for x in range(2):
            for dy in range(-1, 2):
              for dx in range(-1, 2):
                a_slice = a[y, x, 0]
                if y + dy < 0 or y + dy > 1 or x + dx < 0 or x + dx > 1:
                  expected = np.float('inf')
                else:
                  b_slice = b[y + dy, x + dx, 0]
                  expected = (a_slice - b_slice) ** 2
                dy0 = dy + 1
                dx0 = dx + 1
                self.assertAlmostEqual(d_val[y, x, 3 * dy0 + dx0], expected)

  @unittest.skipIf(not embedding_utils.USE_CORRELATION_COST,
                   'depends on correlation_cost')
  def test_local_pairwise_distances_shape(self):
    a = np.zeros((4, 5, 2))
    b = np.zeros((4, 5, 2))
    g = tf.Graph()
    with g.as_default():
      with self.test_session(graph=g) as sess:
        a_tf = tf.constant(a, dtype=tf.float32)
        b_tf = tf.constant(b, dtype=tf.float32)
        d = embedding_utils.local_pairwise_distances(a_tf, b_tf, max_distance=4)
        d_val = sess.run(d)
        self.assertAllEqual(d_val.shape, (4, 5, 81))

  @unittest.skipIf(not embedding_utils.USE_CORRELATION_COST,
                   'depends on correlation_cost')
  def test_local_pairwise_distances_two_dimensional(self):
    a = np.array([[[1.0, -5.0], [7.0, 2.0]], [[1.0, 3.0], [3.0, 4.0]]])
    b = np.array([[[2.0, 1.0], [0.0, -9.0]], [[4.0, 3.0], [3.0, 1.0]]])
    g = tf.Graph()
    with g.as_default():
      with self.test_session(graph=g) as sess:
        a_tf = tf.constant(a, dtype=tf.float32)
        b_tf = tf.constant(b, dtype=tf.float32)
        d = embedding_utils.local_pairwise_distances(a_tf, b_tf,
                                                     max_distance=1)
        d_val = sess.run(d)
        for y in range(2):
          for x in range(2):
            for dy in range(-1, 2):
              for dx in range(-1, 2):
                a_slice = a[y, x, :]
                if y + dy < 0 or y + dy > 1 or x + dx < 0 or x + dx > 1:
                  expected = np.float('inf')
                else:
                  b_slice = b[y + dy, x + dx, :]
                  expected = ((a_slice - b_slice) ** 2).sum()
                dy0 = dy + 1
                dx0 = dx + 1
                self.assertAlmostEqual(d_val[y, x, 3 * dy0 + dx0], expected)

  @unittest.skipIf(not embedding_utils.USE_CORRELATION_COST,
                   'depends on correlation_cost')
  def test_local_previous_frame_nearest_neighbor_features_per_object(self):
    prev_frame_embedding = np.array([[[1.0, -5.0], [7.0, 2.0]],
                                     [[1.0, 3.0], [3.0, 4.0]]]) / 10
    query_embedding = np.array([[[2.0, 1.0], [0.0, -9.0]],
                                [[4.0, 3.0], [3.0, 1.0]]]) / 10
    prev_frame_labels = np.array([[[0], [1]], [[1], [0]]])
    gt_ids = np.array([0, 1])
    g = tf.Graph()
    with g.as_default():
      with self.test_session(graph=g) as sess:
        prev_frame_embedding_tf = tf.constant(prev_frame_embedding,
                                              dtype=tf.float32)
        query_embedding_tf = tf.constant(query_embedding, dtype=tf.float32)
        embu = embedding_utils
        dists = (
            embu.local_previous_frame_nearest_neighbor_features_per_object(
                prev_frame_embedding_tf, query_embedding_tf,
                prev_frame_labels, gt_ids, max_distance=1))
        dists = tf.squeeze(dists, axis=4)
        dists = tf.squeeze(dists, axis=0)
        dists_val = sess.run(dists)
        for obj_id in gt_ids:
          for y in range(2):
            for x in range(2):
              curr_min = 1.0
              for dy in range(-1, 2):
                for dx in range(-1, 2):
                  # Attention: here we shift the prev frame embedding,
                  # not the query.
                  if y + dy < 0 or y + dy > 1 or x + dx < 0 or x + dx > 1:
                    continue
                  if prev_frame_labels[y + dy, x + dx, 0] != obj_id:
                    continue
                  prev_frame_slice = prev_frame_embedding[y + dy, x + dx, :]
                  query_frame_slice = query_embedding[y, x, :]
                  v_unnorm = ((prev_frame_slice - query_frame_slice) ** 2).sum()
                  v = ((1.0 / (1.0 + np.exp(-v_unnorm))) - 0.5) * 2
                  curr_min = min(curr_min, v)
              expected = curr_min
              self.assertAlmostEqual(dists_val[y, x, obj_id], expected,
                                     places=5)


if __name__ == '__main__':
  tf.test.main()
