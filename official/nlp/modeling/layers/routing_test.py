# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for routing."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.nlp.modeling.layers import routing


class TokenImportanceTest(tf.test.TestCase, parameterized.TestCase):

  def test_token_importance(self):
    token_importance_embed = routing.TokenImportanceWithMovingAvg(
        vocab_size=4,
        init_importance=10.0,
        moving_average_beta=0.995)
    importance = token_importance_embed(np.array([[0, 1], [2, 3]]))
    self.assertAllClose(importance, np.array([[10.0, 10.0], [10.0, 10.0]]))
    token_importance_embed.update_token_importance(
        token_ids=np.array([[0, 1]]),
        importance=np.array([[0.0, 0.0]]))
    importance = token_importance_embed(np.array([[0, 1], [2, 3]]))
    self.assertAllClose(importance, np.array([[9.95, 9.95], [10.0, 10.0]]))


class TopKSelectionTest(tf.test.TestCase, parameterized.TestCase):

  def test_top_k_selection(self):
    token_selection = routing.SelectTopK(top_k=2)
    selected, _ = token_selection(np.array([[0, 1, 2, 3], [4, 3, 2, 1]]))
    self.assertAllClose(selected, np.array([[3, 2], [0, 1]]))

  def test_random_k_selection(self):
    token_selection = routing.SelectTopK(random_k=2)
    selected, _ = token_selection(np.array([[0, 1, 2, 3], [4, 3, 2, 1]]))
    self.assertAllClose(selected.shape, (2, 2))

  def test_top_k_random_k(self):
    token_selection = routing.SelectTopK(top_k=1, random_k=1)
    selected, _ = token_selection(np.array([[0, 1, 2, 3], [4, 3, 2, 1]]))
    self.assertAllClose(selected.shape, (2, 2))


if __name__ == "__main__":
  tf.test.main()
