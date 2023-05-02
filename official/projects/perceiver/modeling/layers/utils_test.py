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

"""Tests for utils."""

import tensorflow as tf

from official.projects.perceiver.modeling.layers import utils


class PerceiverUtilsSelfAttentionBlockArgsTest(tf.test.TestCase):

  def test_output_last_dim_is_same_as_input_last_dim(self):
    q_seq_len = 10
    input_last_dim = 30
    some_num_heads = 2

    some_input_shape = ((2, q_seq_len, input_last_dim),)
    args = utils.build_self_attention_block_args(
        some_input_shape,
        num_heads=some_num_heads)
    self.assertEqual(args['output_last_dim'], input_last_dim)

  def test_value_dim_is_same_as_input_last_dim_div_num_heads(self):
    q_seq_len = 10
    input_last_dim = 30
    some_num_heads = 2

    some_input_shape = ((2, q_seq_len, input_last_dim),)
    args = utils.build_self_attention_block_args(
        some_input_shape,
        num_heads=some_num_heads)
    self.assertEqual(args['value_dim'], input_last_dim // some_num_heads)

  # TODO(b/222634115) Add tests for `build_self_attention_block_args` for
  # better coverage


class PerceiverUtilsCrossAttentionBlockArgsTest(tf.test.TestCase):

  def test_1(self):
    some_batch_size = 2
    q_seq_len = 10
    q_input_last_dim = 30
    kv_seq_len = 6
    kv_input_last_dim = 60
    some_num_heads = 2

    some_input_shape = (
        (some_batch_size, q_seq_len, q_input_last_dim),
        (some_batch_size, kv_seq_len, kv_input_last_dim))
    args = utils.build_cross_attention_block_args(
        some_input_shape,
        num_heads=some_num_heads)
    self.assertEqual(args['output_last_dim'], q_input_last_dim)

  # TODO(b/222634115) Add tests for `build_cross_attention_block_args` for
  # better coverage


if __name__ == '__main__':
  tf.test.main()
