# Copyright 2018 Google Inc. All Rights Reserved.
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

"""Tests for maximum spanning tree ops."""

import math

import numpy as np
import tensorflow as tf

from dragnn.python import mst_ops


class MstOpsTest(tf.test.TestCase):
  """Testing rig."""

  def testMaximumSpanningTree(self):
    """Tests that the MST op can recover a simple tree."""
    with self.test_session() as session:
      # The first batch element prefers 3 as root, then 3->0->1->2, for a total
      # score of 4+2+1=7.  The second batch element is smaller and has reversed
      # scores, so 0 is root and 0->2->1.
      num_nodes = tf.constant([4, 3], tf.int32)
      scores = tf.constant([[[0, 0, 0, 0],
                             [1, 0, 0, 0],
                             [1, 2, 0, 0],
                             [1, 2, 3, 4]],
                            [[4, 3, 2, 9],
                             [0, 0, 2, 9],
                             [0, 0, 0, 9],
                             [9, 9, 9, 9]]], tf.int32)  # pyformat: disable

      mst_outputs = mst_ops.maximum_spanning_tree(
          num_nodes, scores, forest=False)
      max_scores, argmax_sources = session.run(mst_outputs)
      tf.logging.info('\nmax_scores=%s\nargmax_sources=\n%s', max_scores,
                      argmax_sources)

      self.assertAllEqual(max_scores, [7, 6])
      self.assertAllEqual(argmax_sources, [[3, 0, 1, 3],
                                           [0, 2, 0, -1]])  # pyformat: disable

  def testMaximumSpanningTreeGradient(self):
    """Tests the MST max score gradient."""
    with self.test_session() as session:
      num_nodes = tf.constant([4, 3], tf.int32)
      scores = tf.constant([[[0, 0, 0, 0],
                             [1, 0, 0, 0],
                             [1, 2, 0, 0],
                             [1, 2, 3, 4]],
                            [[4, 3, 2, 9],
                             [0, 0, 2, 9],
                             [0, 0, 0, 9],
                             [9, 9, 9, 9]]], tf.int32)  # pyformat: disable

      mst_ops.maximum_spanning_tree(num_nodes, scores, forest=False, name='MST')
      mst_op = session.graph.get_operation_by_name('MST')

      d_loss_d_max_scores = tf.constant([3, 7], tf.float32)
      d_loss_d_num_nodes, d_loss_d_scores = (
          mst_ops.maximum_spanning_tree_gradient(mst_op, d_loss_d_max_scores))

      # The num_nodes input is non-differentiable.
      self.assertTrue(d_loss_d_num_nodes is None)
      tf.logging.info('\nd_loss_d_scores=\n%s', d_loss_d_scores.eval())

      self.assertAllEqual(d_loss_d_scores.eval(),
                          [[[0, 0, 0, 3],
                            [3, 0, 0, 0],
                            [0, 3, 0, 0],
                            [0, 0, 0, 3]],
                           [[7, 0, 0, 0],
                            [0, 0, 7, 0],
                            [7, 0, 0, 0],
                            [0, 0, 0, 0]]])  # pyformat: disable

  def testMaximumSpanningTreeGradientError(self):
    """Numerically validates the max score gradient."""
    with self.test_session():
      # The maximum-spanning-tree-score function, as a max of linear functions,
      # is piecewise-linear (i.e., faceted).  The numerical gradient estimate
      # may be inaccurate if the epsilon ball used for the estimate crosses an
      # edge from one facet to another.  To avoid spurious errors, we manually
      # set the sample point so the epsilon ball fits in a facet.  Or in other
      # words, we set the scores so there is a non-trivial margin between the
      # best and second-best trees.
      scores_raw = [[[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [1, 2, 0, 0],
                     [1, 2, 3, 4]],
                    [[4, 3, 2, 9],
                     [0, 0, 2, 9],
                     [0, 0, 0, 9],
                     [9, 9, 9, 9]]]  # pyformat: disable

      # Use 64-bit floats to reduce numerical error.
      scores = tf.constant(scores_raw, tf.float64)
      init_scores = np.array(scores_raw)

      num_nodes = tf.constant([4, 3], tf.int32)
      max_scores = mst_ops.maximum_spanning_tree(
          num_nodes, scores, forest=False)[0]

      gradient_error = tf.test.compute_gradient_error(
          scores, [2, 4, 4], max_scores, [2], init_scores)
      tf.logging.info('gradient_error=%s', gradient_error)

  def testLogPartitionFunctionOneTree(self):
    """Tests the log partition function with one feasible tree with score 1."""
    with self.test_session():
      for forest in [False, True]:

        # Each score matrix supports exactly one tree with score=1*1*1, and
        # the rest with score=0.  Thus the log partition function will be 1.0
        # in each case.
        pad = 12345.6
        scores = tf.constant([[[  1, pad, pad],
                               [pad, pad, pad],
                               [pad, pad, pad]],
                              [[  1,   0, pad],
                               [  1,   0, pad],
                               [pad, pad, pad]],
                              [[  1,   0,   0],
                               [  1,   0,   0],
                               [  0,   1,   0]]],
                             tf.float64)  # pyformat: disable
        scores = tf.log(scores)
        num_nodes = tf.constant([1, 2, 3], tf.int32)

        log_partition_functions = mst_ops.log_partition_function(
            num_nodes, scores, forest=forest)

        self.assertAlmostEqual(tf.exp(log_partition_functions[0]).eval(), 1.0)
        self.assertAlmostEqual(tf.exp(log_partition_functions[1]).eval(), 1.0)
        self.assertAlmostEqual(tf.exp(log_partition_functions[2]).eval(), 1.0)

  def testLogPartitionFunctionOneTreeScaled(self):
    """Tests the log partition function with one feasible tree."""
    with self.test_session():
      for forest in [False, True]:

        # Each score matrix supports exactly one tree with varying score, and
        # the rest with score=0.  Thus the log partition function will equal
        # the score of that single tree in each case.
        pad = 12345.6
        scores = tf.constant([[[  2, pad, pad],
                               [pad, pad, pad],
                               [pad, pad, pad]],
                              [[  3,   0, pad],
                               [  5,   0, pad],
                               [pad, pad, pad]],
                              [[  7,   0,   0],
                               [ 11,   0,   0],
                               [  0,  13,   0]]],
                             tf.float64)  # pyformat: disable
        scores = tf.log(scores)
        num_nodes = tf.constant([1, 2, 3], tf.int32)

        log_partition_functions = mst_ops.log_partition_function(
            num_nodes, scores, forest=forest)

        self.assertAlmostEqual(tf.exp(log_partition_functions[0]).eval(), 2.0)
        self.assertAlmostEqual(
            tf.exp(log_partition_functions[1]).eval(), 3.0 * 5.0)
        self.assertAlmostEqual(
            tf.exp(log_partition_functions[2]).eval(), 7.0 * 11.0 * 13.0)

  def testLogPartitionFunctionTwoTreesScaled(self):
    """Tests the log partition function with two feasible trees."""
    with self.test_session():
      for forest in [False, True]:

        # Each score matrix supports exactly two trees with varying score, and
        # the rest with score=0.  Thus the log partition function will equal
        # the sum of scores of those two trees in each case.
        pad = 12345.6
        scores = tf.constant([[[  2,   0,   0, pad],
                               [  3,   0,   0, pad],
                               [  5,   7,   0, pad],
                               [pad, pad, pad, pad]],
                              [[  0,  11,   0,  13],
                               [  0,  17,   0,   0],
                               [  0,  19,   0,   0],
                               [  0,  23,   0,   0]]],
                             tf.float64)  # pyformat: disable
        scores = tf.log(scores)
        num_nodes = tf.constant([3, 4], tf.int32)

        log_partition_functions = mst_ops.log_partition_function(
            num_nodes, scores, forest=forest)

        self.assertAlmostEqual(
            tf.exp(log_partition_functions[0]).eval(),
            2.0 * 3.0 * 5.0 + 2.0 * 3.0 * 7.0)
        self.assertAlmostEqual(
            tf.exp(log_partition_functions[1]).eval(),
            11.0 * 17.0 * 19.0 * 23.0 + 13.0 * 17.0 * 19.0 * 23.0)

  def testLogPartitionFunctionInfeasible(self):
    """Tests the log partition function on infeasible scores."""
    with self.test_session():
      for forest in [False, True]:

        # The scores form cycles of various sizes.  Note that one can compute
        # the partition function for infeasible scores---it's the gradient that
        # may be impacted by numerical error.
        pad = 12345.6
        scores = tf.constant([[[  0,   1, pad, pad],
                               [  1,   0, pad, pad],
                               [pad, pad, pad, pad],
                               [pad, pad, pad, pad]],
                              [[  0,   1,   0, pad],
                               [  0,   0,   1, pad],
                               [  1,   0,   0, pad],
                               [pad, pad, pad, pad]],
                              [[  0,   1,   0,   0],
                               [  0,   0,   1,   0],
                               [  0,   0,   0,   1],
                               [  1,   0,   0,   0]]],
                             tf.float64)  # pyformat: disable
        scores = tf.log(scores)
        num_nodes = tf.constant([2, 3, 4], tf.int32)

        log_partition_functions = mst_ops.log_partition_function(
            num_nodes, scores, forest=forest)

        self.assertAlmostEqual(tf.exp(log_partition_functions[0]).eval(), 0.0)
        self.assertAlmostEqual(tf.exp(log_partition_functions[1]).eval(), 0.0)
        self.assertAlmostEqual(tf.exp(log_partition_functions[2]).eval(), 0.0)

  def testLogPartitionFunctionAllTrees(self):
    """Tests the log partition function with all trees feasible."""
    with self.test_session():
      for forest in [False, True]:
        # The scores allow all trees.  Using Cayley's formula, the
        # number of directed spanning trees and forests in a complete
        # digraph of n nodes is n^{n-1} and (n+1)^{n-1}, respectively.
        # https://en.wikipedia.org/wiki/Cayley%27s_formula
        scores = tf.zeros([10, 10, 10], tf.float64)  # = 1 in log domain
        num_nodes = tf.range(1, 11, dtype=tf.int32)

        log_partition_functions = mst_ops.log_partition_function(
            num_nodes, scores, forest=forest)

        base_offset = 1 if forest else 0  # n+1 for forest, n for tree
        for size in range(1, 11):
          self.assertAlmostEqual(log_partition_functions[size - 1].eval(),
                                 (size - 1) * math.log(size + base_offset))

  def testLogPartitionFunctionWithVeryHighValues(self):
    """Tests the overflow protection in the log partition function."""
    with self.test_session():
      for forest in [False, True]:
        # Set the scores to very high values to test overflow protection.
        scores = 1000 * tf.ones([10, 10, 10], tf.float64)
        num_nodes = tf.range(1, 11, dtype=tf.int32)

        log_partition_functions = mst_ops.log_partition_function(
            num_nodes, scores, forest=forest)

        base_offset = 1 if forest else 0  # n+1 for forest, n for tree
        for size in range(1, 11):
          self.assertAlmostEqual(
              log_partition_functions[size - 1].eval(),
              (size - 1) * math.log(size + base_offset) + size * 1000)

  def testLogPartitionFunctionWithVeryLowValues(self):
    """Tests the underflow protection in the log partition function."""
    with self.test_session():
      for forest in [False, True]:
        # Set the scores to very low values to test underflow protection.
        scores = -1000 * tf.ones([10, 10, 10], tf.float64)
        num_nodes = tf.range(1, 11, dtype=tf.int32)

        log_partition_functions = mst_ops.log_partition_function(
            num_nodes, scores, forest=forest)

        base_offset = 1 if forest else 0  # n+1 for forest, n for tree
        for size in range(1, 11):
          self.assertAlmostEqual(
              log_partition_functions[size - 1].eval(),
              (size - 1) * math.log(size + base_offset) - size * 1000)

  def testLogPartitionFunctionGradientError(self):
    """Validates the log partition function gradient."""
    with self.test_session():
      for forest in [False, True]:
        # To avoid numerical issues, provide score matrices that are weighted
        # towards feasible trees or forests.
        scores_raw = [[[0, 0, 0, 0],
                       [1, 0, 0, 0],
                       [1, 2, 0, 0],
                       [1, 2, 3, 4]],
                      [[4, 3, 2, 9],
                       [0, 0, 2, 9],
                       [0, 0, 0, 9],
                       [9, 9, 9, 9]]]  # pyformat: disable

        scores = tf.constant(scores_raw, tf.float64)
        init_scores = np.array(scores_raw)

        num_nodes = tf.constant([4, 3], tf.int32)
        log_partition_functions = mst_ops.log_partition_function(
            num_nodes, scores, forest=forest)

        gradient_error = tf.test.compute_gradient_error(
            scores, [2, 4, 4], log_partition_functions, [2], init_scores)
        tf.logging.info('forest=%s gradient_error=%s', forest, gradient_error)

        self.assertLessEqual(gradient_error, 1e-7)

  def testLogPartitionFunctionGradientErrorFailsIfInfeasible(self):
    """Tests that the partition function gradient fails on infeasible scores."""
    with self.test_session():
      for forest in [False, True]:

        # The scores form cycles of various sizes.
        pad = 12345.6
        scores_raw = [[[  0,   1, pad, pad],
                       [  1,   0, pad, pad],
                       [pad, pad, pad, pad],
                       [pad, pad, pad, pad]],
                      [[  0,   1,   0, pad],
                       [  0,   0,   1, pad],
                       [  1,   0,   0, pad],
                       [pad, pad, pad, pad]],
                      [[  0,   1,   0,   0],
                       [  0,   0,   1,   0],
                       [  0,   0,   0,   1],
                       [  1,   0,   0,   0]]]  # pyformat: disable

        scores = tf.log(scores_raw)
        init_scores = np.log(np.array(scores_raw))
        num_nodes = tf.constant([2, 3, 4], tf.int32)

        log_partition_functions = mst_ops.log_partition_function(
            num_nodes, scores, forest=forest)

        with self.assertRaises(Exception):
          tf.test.compute_gradient_error(
              scores, [3, 4, 4], log_partition_functions, [3], init_scores)

  def testLogPartitionFunctionGradientErrorOkIfInfeasibleWithClipping(self):
    """Tests that the log partition function gradient is OK after clipping."""
    with self.test_session():
      for forest in [False, True]:

        # The scores form cycles of various sizes.
        pad = 12345.6
        scores_raw = [[[  0,   1, pad, pad],
                       [  1,   0, pad, pad],
                       [pad, pad, pad, pad],
                       [pad, pad, pad, pad]],
                      [[  0,   1,   0, pad],
                       [  0,   0,   1, pad],
                       [  1,   0,   0, pad],
                       [pad, pad, pad, pad]],
                      [[  0,   1,   0,   0],
                       [  0,   0,   1,   0],
                       [  0,   0,   0,   1],
                       [  1,   0,   0,   0]]]  # pyformat: disable

        scores = tf.log(scores_raw)
        init_scores = np.log(np.array(scores_raw))
        num_nodes = tf.constant([2, 3, 4], tf.int32)

        log_partition_functions = mst_ops.log_partition_function(
            num_nodes, scores, forest=forest, max_dynamic_range=10)

        gradient_error = tf.test.compute_gradient_error(
            scores, [3, 4, 4], log_partition_functions, [3], init_scores)
        tf.logging.info('forest=%s gradient_error=%s', forest, gradient_error)

        # There's still a lot of error.
        self.assertLessEqual(gradient_error, 1e-3)


if __name__ == '__main__':
  tf.test.main()
