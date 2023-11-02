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

"""Tests for optimized_multiheadattention_layer."""

import numpy as np
import tensorflow as tf, tf_keras

from official.projects.edgetpu.vision.modeling import optimized_multiheadattention_layer

_BATCH_SIZE = 32
_SEQ_LEN = 4
_EMBEDDING_SIZE = 8
_NUM_HEADS = 2
_KEY_DIM = 2


class OptimizedMultiheadattentionLayerTest(tf.test.TestCase):

  def test_same_output(self):
    """Tests that OptimizedMultiHeadAttention returns the expected outputs."""

    input_tensor_1 = tf.random.uniform((_BATCH_SIZE, _SEQ_LEN, _EMBEDDING_SIZE))
    input_tensor_2 = tf.random.uniform((_BATCH_SIZE, _SEQ_LEN, _EMBEDDING_SIZE))

    # Instantiate layer and call with inputs to build.
    orig_layer = tf_keras.layers.MultiHeadAttention(
        num_heads=_NUM_HEADS, key_dim=_KEY_DIM)
    _ = orig_layer(input_tensor_1, input_tensor_2)
    opt_layer = optimized_multiheadattention_layer.OptimizedMultiHeadAttention(
        num_heads=_NUM_HEADS, key_dim=_KEY_DIM)
    _ = opt_layer(input_tensor_1, input_tensor_2)

    # Set the weights of the two layers to be the same.
    query_dense_weights = np.random.uniform(
        size=(_EMBEDDING_SIZE, _NUM_HEADS, _KEY_DIM))
    query_dense_bias = np.random.uniform(size=(_NUM_HEADS, _KEY_DIM))
    key_dense_weights = np.random.uniform(
        size=(_EMBEDDING_SIZE, _NUM_HEADS, _KEY_DIM))
    key_dense_bias = np.random.uniform(size=(_NUM_HEADS, _KEY_DIM))
    value_dense_weights = np.random.uniform(
        size=(_EMBEDDING_SIZE, _NUM_HEADS, _KEY_DIM))
    value_dense_bias = np.random.uniform(size=(_NUM_HEADS, _KEY_DIM))
    attention_output_dense_weights = np.random.uniform(
        size=(_NUM_HEADS, _KEY_DIM, _EMBEDDING_SIZE))
    attention_output_dense_bias = np.random.uniform(size=(_EMBEDDING_SIZE,))

    orig_layer._query_dense.set_weights([query_dense_weights, query_dense_bias])
    orig_layer._key_dense.set_weights([key_dense_weights, key_dense_bias])
    orig_layer._value_dense.set_weights([value_dense_weights, value_dense_bias])
    orig_layer._output_dense.set_weights(
        [attention_output_dense_weights, attention_output_dense_bias])

    opt_layer._query_dense.set_weights([query_dense_weights, query_dense_bias])
    opt_layer._key_dense.set_weights([key_dense_weights, key_dense_bias])
    opt_layer._value_dense.set_weights([value_dense_weights, value_dense_bias])
    opt_layer._output_dense.set_weights(
        [attention_output_dense_weights, attention_output_dense_bias])

    # Calculate two sets of attention outputs and scores and compare.
    orig_attn_output, orig_attn_score = orig_layer(
        input_tensor_1, input_tensor_2, return_attention_scores=True)
    opt_attn_output, opt_attn_score = opt_layer(
        input_tensor_1, input_tensor_2, return_attention_scores=True)
    self.assertAllClose(orig_attn_output, opt_attn_output)
    self.assertAllClose(orig_attn_score, opt_attn_score)


if __name__ == '__main__':
  tf.test.main()
