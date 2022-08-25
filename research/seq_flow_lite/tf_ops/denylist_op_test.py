# Copyright 2022 The TensorFlow Authors All Rights Reserved.
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
"""Test denylist op and show example usage from python wrapper."""
import tensorflow as tf

from tf_ops import denylist_op # import seq_flow_lite module


class SkipgramDenylistTest(tf.test.TestCase):

  def test_correct(self):
    result = denylist_op.skipgram_denylist(
        input=["q a q b q c q", "q a b q q c"],
        max_skip_size=1,
        denylist=["a b c"],
        denylist_category=[1],
        categories=2,
        negative_categories=1)
    self.assertAllEqual(result, [[0.0, 1.0], [1.0, 0.0]])


class SubsequenceDenylistTest(tf.test.TestCase):

  def test_correct(self):
    result = denylist_op.subsequence_denylist(
        input=["qaqbqcq", "qabqqc"],
        max_skip_size=1,
        denylist=["a b c"],
        denylist_category=[1],
        categories=2,
        negative_categories=1)
    self.assertAllEqual(result, [[0.0, 1.0], [1.0, 0.0]])


class TokenizedDenylistTest(tf.test.TestCase):

  def test_correct(self):
    result = denylist_op.tokenized_denylist(
        input=[["q", "a", "q", "b", "q", "c", "q"],
               ["q", "a", "b", "q", "q", "c", ""]],
        token_count=[7, 6],
        max_skip_size=1,
        denylist=["a b c"],
        denylist_category=[1],
        categories=2,
        negative_categories=1)
    self.assertAllEqual(result, [[0.0, 1.0], [1.0, 0.0]])


if __name__ == "__main__":
  tf.test.main()
