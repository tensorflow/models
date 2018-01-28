# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Tests for tcn.labeled_eval."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import labeled_eval
import tensorflow as tf


class LabeledEvalTest(tf.test.TestCase):

  def testNearestCrossSequenceNeighbors(self):
    # Generate embeddings.
    num_data = 64
    embedding_size = 4
    num_tasks = 8
    n_neighbors = 2
    data = np.random.randn(num_data, embedding_size)
    tasks = np.repeat(range(num_tasks), num_data // num_tasks)

    # Get nearest cross-sequence indices.
    indices = labeled_eval.nearest_cross_sequence_neighbors(
        data, tasks, n_neighbors=n_neighbors)

    # Assert that no nearest neighbor indices come from the same task.
    repeated_tasks = np.tile(np.reshape(tasks, (num_data, 1)), n_neighbors)
    self.assertTrue(np.all(np.not_equal(repeated_tasks, tasks[indices])))

  def testPerfectCrossSequenceRecall(self):
    # Make sure cross-sequence recall@k returns 1.0 for near-duplicate features.
    embeddings = np.random.randn(10, 2)
    embeddings[5:, :] = 0.00001 + embeddings[:5, :]
    tasks = np.repeat([0, 1], 5)
    labels = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    # find k=1, k=2 nearest neighbors.
    k_list = [1, 2]

    # Compute knn indices.
    indices = labeled_eval.nearest_cross_sequence_neighbors(
        embeddings, tasks, n_neighbors=max(k_list))
    retrieved_labels = labels[indices]
    recall_list = labeled_eval.compute_cross_sequence_recall_at_k(
        retrieved_labels=retrieved_labels,
        labels=labels,
        k_list=k_list)
    self.assertTrue(np.allclose(
        np.array(recall_list), np.array([1.0, 1.0])))

  def testRelativeRecall(self):
    # Make sure cross-sequence recall@k is strictly non-decreasing over k.
    num_data = 100
    num_tasks = 10
    embeddings = np.random.randn(100, 5)
    tasks = np.repeat(range(num_tasks), num_data // num_tasks)
    labels = np.random.randint(0, 5, 100)

    k_list = [1, 2, 4, 8, 16, 32, 64]
    indices = labeled_eval.nearest_cross_sequence_neighbors(
        embeddings, tasks, n_neighbors=max(k_list))
    retrieved_labels = labels[indices]
    recall_list = labeled_eval.compute_cross_sequence_recall_at_k(
        retrieved_labels=retrieved_labels,
        labels=labels,
        k_list=k_list)
    recall_list_sorted = sorted(recall_list)
    self.assertTrue(np.allclose(
        np.array(recall_list), np.array(recall_list_sorted)))

if __name__ == "__main__":
  tf.test.main()
