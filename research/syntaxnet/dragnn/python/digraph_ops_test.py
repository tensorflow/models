# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Tests for digraph ops."""

import tensorflow as tf

from dragnn.python import digraph_ops


class DigraphOpsTest(tf.test.TestCase):
  """Testing rig."""

  def testArcPotentialsFromTokens(self):
    with self.test_session():
      # Batch of two, where the second batch item is the reverse of the first.
      source_tokens = tf.constant([[[1, 2],
                                    [2, 3],
                                    [3, 4]],
                                   [[3, 4],
                                    [2, 3],
                                    [1, 2]]],
                                  tf.float32)  # pyformat: disable
      target_tokens = tf.constant([[[4, 5, 6],
                                    [5, 6, 7],
                                    [6, 7, 8]],
                                   [[6, 7, 8],
                                    [5, 6, 7],
                                    [4, 5, 6]]],
                                  tf.float32)  # pyformat: disable
      weights = tf.constant([[2, 3, 5],
                             [7, 11, 13]],
                            tf.float32)  # pyformat: disable

      arcs = digraph_ops.ArcPotentialsFromTokens(source_tokens, target_tokens,
                                                 weights)

      # For example,
      # ((1 * 2 * 4 + 1 * 3  * 5 + 1 *  5 * 6) +
      #  (2 * 7 * 4 + 2 * 11 * 5 + 2 * 13 * 6)) = 375
      self.assertAllEqual(arcs.eval(),
                          [[[375, 447, 519],
                            [589, 702, 815],
                            [803, 957, 1111]],
                           [[1111, 957, 803],  # reflected through the center
                            [815, 702, 589],
                            [519, 447, 375]]])  # pyformat: disable

  def testArcSourcePotentialsFromTokens(self):
    with self.test_session():
      tokens = tf.constant([[[4, 5, 6],
                             [5, 6, 7],
                             [6, 7, 8]],
                            [[6, 7, 8],
                             [5, 6, 7],
                             [4, 5, 6]]], tf.float32)  # pyformat: disable
      weights = tf.constant([2, 3, 5], tf.float32)

      arcs = digraph_ops.ArcSourcePotentialsFromTokens(tokens, weights)

      self.assertAllEqual(arcs.eval(), [[[53, 53, 53],
                                         [63, 63, 63],
                                         [73, 73, 73]],
                                        [[73, 73, 73],
                                         [63, 63, 63],
                                         [53, 53, 53]]])  # pyformat: disable

  def testRootPotentialsFromTokens(self):
    with self.test_session():
      root = tf.constant([1, 2], tf.float32)
      tokens = tf.constant([[[4, 5, 6],
                             [5, 6, 7],
                             [6, 7, 8]],
                            [[6, 7, 8],
                             [5, 6, 7],
                             [4, 5, 6]]], tf.float32)  # pyformat: disable
      weights_arc = tf.constant([[2, 3, 5],
                                 [7, 11, 13]],
                                tf.float32)  # pyformat: disable
      weights_source = tf.constant([11, 10], tf.float32)

      roots = digraph_ops.RootPotentialsFromTokens(root, tokens, weights_arc,
                                                   weights_source)

      self.assertAllEqual(roots.eval(), [[406, 478, 550],
                                         [550, 478, 406]])  # pyformat: disable

  def testCombineArcAndRootPotentials(self):
    with self.test_session():
      arcs = tf.constant([[[1, 2, 3],
                           [2, 3, 4],
                           [3, 4, 5]],
                          [[3, 4, 5],
                           [2, 3, 4],
                           [1, 2, 3]]], tf.float32)  # pyformat: disable
      roots = tf.constant([[6, 7, 8],
                           [8, 7, 6]], tf.float32)  # pyformat: disable

      potentials = digraph_ops.CombineArcAndRootPotentials(arcs, roots)

      self.assertAllEqual(potentials.eval(), [[[6, 2, 3],
                                               [2, 7, 4],
                                               [3, 4, 8]],
                                              [[8, 4, 5],
                                               [2, 7, 4],
                                               [1, 2, 6]]])  # pyformat: disable

  def testLabelPotentialsFromTokens(self):
    with self.test_session():
      tokens = tf.constant([[[1, 2],
                             [3, 4],
                             [5, 6]],
                            [[6, 5],
                             [4, 3],
                             [2, 1]]], tf.float32)  # pyformat: disable


      weights = tf.constant([[ 2,  3],
                             [ 5,  7],
                             [11, 13]], tf.float32)  # pyformat: disable

      labels = digraph_ops.LabelPotentialsFromTokens(tokens, weights)

      self.assertAllEqual(labels.eval(),

                          [[[  8,  19,  37],
                            [ 18,  43,  85],
                            [ 28,  67, 133]],
                           [[ 27,  65, 131],
                            [ 17,  41,  83],
                            [  7,  17,  35]]])  # pyformat: disable

  def testLabelPotentialsFromTokenPairs(self):
    with self.test_session():
      sources = tf.constant([[[1, 2],
                              [3, 4],
                              [5, 6]],
                             [[6, 5],
                              [4, 3],
                              [2, 1]]], tf.float32)  # pyformat: disable
      targets = tf.constant([[[3, 4],
                              [5, 6],
                              [7, 8]],
                             [[8, 7],
                              [6, 5],
                              [4, 3]]], tf.float32)  # pyformat: disable


      weights = tf.constant([[[ 2,  3],
                              [ 5,  7]],
                             [[11, 13],
                              [17, 19]],
                             [[23, 29],
                              [31, 37]]], tf.float32)  # pyformat: disable

      labels = digraph_ops.LabelPotentialsFromTokenPairs(sources, targets,
                                                         weights)

      self.assertAllEqual(labels.eval(),

                          [[[ 104,  339,  667],
                            [ 352, 1195, 2375],
                            [ 736, 2531, 5043]],
                           [[ 667, 2419, 4857],
                            [ 303, 1115, 2245],
                            [  75,  291,  593]]])  # pyformat: disable

  def testValidArcAndTokenMasks(self):
    with self.test_session():

      lengths = tf.constant([1, 2, 3], tf.int64)
      max_length = 4
      valid_arcs, valid_tokens = digraph_ops.ValidArcAndTokenMasks(
          lengths, max_length)
      self.assertAllEqual(valid_arcs.eval(),
                          [[[1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]],
                           [[1, 1, 0, 0],
                            [1, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]],
                           [[1, 1, 1, 0],
                            [1, 1, 1, 0],
                            [1, 1, 1, 0],
                            [0, 0, 0, 0]]])  # pyformat: disable
      self.assertAllEqual(valid_tokens.eval(),
                          [[1, 0, 0, 0],
                           [1, 1, 0, 0],
                           [1, 1, 1, 0]])  # pyformat: disable

  def testLaplacianMatrixTree(self):
    with self.test_session():

      pad = 12345.6
      arcs = tf.constant([[[  2, pad, pad, pad],
                           [pad, pad, pad, pad],
                           [pad, pad, pad, pad],
                           [pad, pad, pad, pad]],
                          [[  2,   3, pad, pad],
                           [  5,   7, pad, pad],
                           [pad, pad, pad, pad],
                           [pad, pad, pad, pad]],
                          [[  2,   3,   5, pad],
                           [  7,  11,  13, pad],
                           [ 17,  19,  23, pad],
                           [pad, pad, pad, pad]],
                          [[  2,   3,   5,   7],
                           [ 11,  13,  17,  19],
                           [ 23,  29,  31,  37],
                           [ 41,  43,  47,  53]]],
                         tf.float32)  # pyformat: disable
      lengths = tf.constant([1, 2, 3, 4], tf.int64)
      laplacian = digraph_ops.LaplacianMatrix(lengths, arcs)
      self.assertAllEqual(laplacian.eval(),
                          [[[  2,   0,   0,   0],
                            [  0,   1,   0,   0],
                            [  0,   0,   1,   0],
                            [  0,   0,   0,   1]],
                           [[  2,  -3,   0,   0],
                            [  7,   5,   0,   0],
                            [  0,   0,   1,   0],
                            [  0,   0,   0,   1]],
                           [[  2,  -3,  -5,   0],
                            [ 11,  20, -13,   0],
                            [ 23, -19,  36,   0],
                            [  0,   0,   0,   1]],
                           [[  2,  -3,  -5,  -7],
                            [ 13,  47, -17, -19],
                            [ 31, -29,  89, -37],
                            [ 53, -43, -47, 131]]])  # pyformat: disable

  def testLaplacianMatrixForest(self):
    with self.test_session():

      pad = 12345.6
      arcs = tf.constant([[[  2, pad, pad, pad],
                           [pad, pad, pad, pad],
                           [pad, pad, pad, pad],
                           [pad, pad, pad, pad]],
                          [[  2,   3, pad, pad],
                           [  5,   7, pad, pad],
                           [pad, pad, pad, pad],
                           [pad, pad, pad, pad]],
                          [[  2,   3,   5, pad],
                           [  7,  11,  13, pad],
                           [ 17,  19,  23, pad],
                           [pad, pad, pad, pad]],
                          [[  2,   3,   5,   7],
                           [ 11,  13,  17,  19],
                           [ 23,  29,  31,  37],
                           [ 41,  43,  47,  53]]],
                         tf.float32)  # pyformat: disable
      lengths = tf.constant([1, 2, 3, 4], tf.int64)
      laplacian = digraph_ops.LaplacianMatrix(lengths, arcs, forest=True)
      self.assertAllEqual(laplacian.eval(),
                          [[[  2,   0,   0,   0],
                            [  0,   1,   0,   0],
                            [  0,   0,   1,   0],
                            [  0,   0,   0,   1]],
                           [[  5,  -3,   0,   0],
                            [ -5,  12,   0,   0],
                            [  0,   0,   1,   0],
                            [  0,   0,   0,   1]],
                           [[ 10,  -3,  -5,   0],
                            [ -7,  31, -13,   0],
                            [-17, -19,  59,   0],
                            [  0,   0,   0,   1]],
                           [[ 17,  -3,  -5,  -7],
                            [-11,  60, -17, -19],
                            [-23, -29, 120, -37],
                            [-41, -43, -47, 184]]])  # pyformat: disable


if __name__ == "__main__":
  tf.test.main()
