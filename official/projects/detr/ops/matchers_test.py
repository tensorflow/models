# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for tensorflow_models.official.projects.detr.ops.matchers."""

import numpy as np
from scipy import optimize
import tensorflow as tf, tf_keras

from official.projects.detr.ops import matchers


class MatchersOpsTest(tf.test.TestCase):

  def testLinearSumAssignment(self):
    """Check a simple 2D test case of the Linear Sum Assignment problem.

    Ensures that the implementation of the matching algorithm is correct
    and functional on TPUs.
    """
    cost_matrix = np.array([[[4, 1, 3], [2, 0, 5], [3, 2, 2]]],
                           dtype=np.float32)
    _, adjacency_matrix = matchers.hungarian_matching(tf.constant(cost_matrix))
    adjacency_output = adjacency_matrix.numpy()

    correct_output = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=bool)
    self.assertAllEqual(adjacency_output[0], correct_output)

  def testBatchedLinearSumAssignment(self):
    """Check a batched case of the Linear Sum Assignment Problem.

    Ensures that a correct solution is found for all inputted problems within
    a batch.
    """
    cost_matrix = np.array([
        [[4, 1, 3], [2, 0, 5], [3, 2, 2]],
        [[1, 4, 3], [0, 2, 5], [2, 3, 2]],
        [[1, 3, 4], [0, 5, 2], [2, 2, 3]],
    ],
                           dtype=np.float32)
    _, adjacency_matrix = matchers.hungarian_matching(tf.constant(cost_matrix))
    adjacency_output = adjacency_matrix.numpy()

    # Hand solved correct output for the linear sum assignment problem
    correct_output = np.array([
        [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
    ],
                              dtype=bool)
    self.assertAllClose(adjacency_output, correct_output)

  def testMaximumBipartiteMatching(self):
    """Check that the maximum bipartite match assigns the correct numbers."""
    adj_matrix = tf.cast([[
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ]], tf.bool)
    _, assignment = matchers._maximum_bipartite_matching(adj_matrix)
    self.assertEqual(np.sum(assignment.numpy()), 5)

  def testAssignmentMatchesScipy(self):
    """Check that the Linear Sum Assignment matches the Scipy implementation."""
    batch_size, num_elems = 2, 25
    weights = tf.random.uniform((batch_size, num_elems, num_elems),
                                minval=0.,
                                maxval=1.)
    weights, assignment = matchers.hungarian_matching(weights)

    for idx in range(batch_size):
      _, scipy_assignment = optimize.linear_sum_assignment(weights.numpy()[idx])
      hungarian_assignment = np.where(assignment.numpy()[idx])[1]

      self.assertAllEqual(hungarian_assignment, scipy_assignment)

if __name__ == '__main__':
  tf.test.main()
