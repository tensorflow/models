# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for t5."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.nlp.modeling.models import t5


def _create_cache(batch_size,
                  init_decode_length,
                  num_heads,
                  head_size,
                  dtype=tf.float32):
  if num_heads is None:
    kv_shape = [batch_size, init_decode_length, head_size]
  else:
    kv_shape = [batch_size, init_decode_length, num_heads, head_size]

  return {
      "key": tf.zeros(kv_shape, dtype=dtype),
      "value": tf.zeros(kv_shape, dtype=dtype)
  }


class ModulesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("bfloat16", tf.bfloat16),
                                  ("float32", tf.float32))
  def test_embed(self, dtype):
    l = t5.Embed(vocab_size=5, features=4, compute_dtype=dtype, name="foo")
    inputs = np.array([[2, 3], [1, 2]], dtype=np.int32)
    inputs = tf.convert_to_tensor(inputs)
    one_hot_outputs = l(inputs, one_hot=True)
    gather_outputs = l(inputs, one_hot=False)
    self.assertEqual(one_hot_outputs.shape, (2, 2, 4))
    self.assertLen(l.trainable_variables, 1)
    self.assertAllClose(one_hot_outputs, gather_outputs)

    outputs = l.attend(query=tf.zeros((2, 2, 4), dtype))
    self.assertEqual(outputs.shape, (2, 2, 5))

    # Test initializers.
    l = t5.Embed(
        vocab_size=5,
        features=4,
        compute_dtype=dtype,
        name="foo",
        embeddings_initializer=tf.keras.initializers.Zeros())
    self.assertAllClose(l(inputs), tf.zeros((2, 2, 4), dtype))

  @parameterized.named_parameters(("bfloat16", tf.bfloat16),
                                  ("float32", tf.float32))
  def test_rms_norm(self, dtype):
    l = t5.RMSNorm(hidden_size=4, epsilon=0.0, name="foo")
    inputs = tf.ones((2, 4), dtype=dtype)
    outputs = l(inputs)
    self.assertAllEqual(l(inputs), inputs)
    self.assertEqual(outputs.dtype, dtype)
    self.assertLen(l.trainable_variables, 1)
    self.assertIn("foo/scale", l.trainable_variables[0].name)

  @parameterized.named_parameters(("bfloat16", tf.bfloat16),
                                  ("float32", tf.float32))
  def test_linear(self, dtype):
    l = t5.Linear(
        in_features=4,
        out_features=4,
        w_init=tf.keras.initializers.Ones(),
        name="foo")
    inputs = tf.ones((2, 4), dtype=dtype)
    outputs = l(inputs)
    self.assertEqual(outputs.shape, inputs.shape)
    self.assertEqual(outputs.dtype, dtype)
    self.assertLen(l.trainable_variables, 2)

  def test_linear3d(self):
    batch_size = 2
    l = t5.Linear3D(
        in_features=4,
        out_features=4,
        num_heads=2,
        to_3d=True,
        w_init=tf.keras.initializers.Ones(),
        name="foo")
    inputs = np.ones((batch_size, 2, 4), dtype=np.float32)
    self.assertEqual(l(inputs).shape, (batch_size, 2, 2, 4))

    l = t5.Linear3D(
        in_features=2,
        out_features=4,
        num_heads=2,
        to_3d=False,
        w_init=tf.keras.initializers.Ones(),
        name="foo")
    inputs = np.ones((batch_size, 2, 2, 2), dtype=np.float32)
    self.assertEqual(l(inputs).shape, (batch_size, 2, 4))

  def test_ffn(self):
    inputs = np.ones((2, 4), dtype=np.float32)
    for activation in ["relu", "linear", "gelu", "swish"]:
      l = t5.FFN(
          d_model=4,
          d_ff=8,
          use_bias=True,
          dropout_rate=0.1,
          activations=[activation],
          name="foo")
      self.assertEqual(l(inputs).shape, inputs.shape)
      self.assertLen(l.trainable_variables, 4)

    l = t5.FFN(
        d_model=4,
        d_ff=8,
        dropout_rate=0.1,
        activations=["linear", "gelu"],
        name="bar")
    self.assertLen(l.trainable_variables, 3)
    self.assertEqual(l(inputs).shape, inputs.shape)

  @parameterized.named_parameters(("bfloat16", tf.bfloat16),
                                  ("float32", tf.float32))
  def test_relative_position(self, dtype):
    l = t5.RelativePositionEmbedding(
        num_heads=4,
        bidirectional=False,
        embeddings_initializer=tf.keras.initializers.Ones(),
        compute_dtype=dtype,
        name="foo")
    self.assertEqual(l(4, 2).shape, (1, 4, 4, 2))
    l = t5.RelativePositionEmbedding(
        num_heads=4,
        bidirectional=True,
        embeddings_initializer=tf.keras.initializers.Ones(),
        compute_dtype=dtype,
        name="bar")
    outputs = l(4, 2)
    self.assertEqual(outputs.shape, (1, 4, 4, 2))
    self.assertEqual(outputs.dtype, dtype)

  def test_masks(self):
    causal_mask = t5.make_causal_mask(np.zeros((2, 5)))
    self.assertEqual(causal_mask.shape, (2, 1, 5, 5))

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.default_strategy,
              strategy_combinations.cloud_tpu_strategy,
          ],
          mode="eager"))
  def test_attention(self, distribution):
    num_heads, head_size = 2, 4
    from_seq_length, to_seq_length = 4, 6
    batch_size = 2
    pos_embed = t5.RelativePositionEmbedding(
        num_heads=4,
        bidirectional=False,
        embeddings_initializer=tf.keras.initializers.Ones(),
        name="pos_embed")
    position_bias = pos_embed(from_seq_length, from_seq_length)
    l = t5.MultiHeadAttention(d_model=4, d_kv=2, num_heads=4, dropout_rate=0.1)
    query = tf.convert_to_tensor(
        np.ones((batch_size, from_seq_length, 4), dtype=np.float32))
    self.assertEqual(
        l(query, position_bias=position_bias)["context"].shape, query.shape)
    kv = tf.convert_to_tensor(
        np.ones((batch_size, to_seq_length, 4), dtype=np.float32))
    position_bias = pos_embed(from_seq_length, to_seq_length)
    outputs = l(query, kv=kv, position_bias=position_bias)
    self.assertEqual(outputs["context"].shape, query.shape)

    with distribution.scope():
      l = t5.MultiHeadAttention(
          d_model=4, d_kv=head_size, num_heads=num_heads, dropout_rate=0.1)

      @tf.function
      def step(inputs):

        def _step_fn(inputs):
          cache = _create_cache(batch_size, from_seq_length, num_heads,
                                head_size)
          mask = t5.make_causal_mask(tf.ones((batch_size, 1)))
          return l(
              query=inputs,
              mask=mask,
              cache=cache,
              decode_position=decode_position)

        outputs = distribution.run(_step_fn, args=(inputs,))
        return tf.nest.map_structure(distribution.experimental_local_results,
                                     outputs)

      decode_position = 2
      query = tf.convert_to_tensor(np.ones((2, 1, 4), dtype=np.float32))
      local_outputs = step(query)
      self.assertEqual(local_outputs["context"][0].shape, (2, 1, 4))
      self.assertNotEqual(
          np.sum(local_outputs["cache"]["key"][0][:, decode_position,
                                                  ...].numpy()), 0.0)


class T5Test(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.default_strategy,
              strategy_combinations.cloud_tpu_strategy,
          ],
          mode="eager"))
  def test_attention_layers(self, distribution):
    num_heads, head_size = 2, 2
    from_seq_length = 4
    # TPU decoding should pre-allocate the entire sequence.
    batch_size = 2
    with distribution.scope():
      pos_embed = t5.RelativePositionEmbedding(
          num_heads=head_size,
          bidirectional=False,
          embeddings_initializer=tf.keras.initializers.Ones(),
          name="pos_embed")
      l = t5.SelfAttention(
          d_model=4, d_kv=head_size, num_heads=num_heads, dropout_rate=0.1)
      decode_position = 2

      @tf.function
      def step(inputs):

        def _step_fn(inputs):
          cache = _create_cache(batch_size, from_seq_length, num_heads,
                                head_size)
          mask = t5.make_causal_mask(tf.ones((batch_size, 1)))
          position_bias = pos_embed(from_seq_length, from_seq_length)
          return l(
              hidden_states=inputs,
              cache=cache,
              attention_mask=mask,
              decode_position=decode_position,
              position_bias=position_bias)

        outputs = distribution.run(_step_fn, args=(inputs,))
        return tf.nest.map_structure(distribution.experimental_local_results,
                                     outputs)

      query = tf.convert_to_tensor(np.ones((2, 1, 4), dtype=np.float32))
      local_outputs = step(query)
      self.assertEqual(local_outputs["layer_output"][0].shape, (2, 1, 4))
      self.assertNotEqual(
          np.sum(
              local_outputs["cache"]["key"][0][:,
                                               decode_position, :, :].numpy()),
          0.0)

      l = t5.CrossAttention(
          d_model=4, d_kv=head_size, num_heads=num_heads, dropout_rate=0.1)
      to_seq_length = 6
      query = tf.convert_to_tensor(
          np.ones((2, from_seq_length, 4), dtype=np.float32))
      kv = tf.convert_to_tensor(
          np.ones((2, to_seq_length, 4), dtype=np.float32))

      @tf.function
      def step_cross_attn(inputs):

        def _step_fn(inputs):
          query, kv = inputs
          mask = t5.make_attention_mask(
              tf.ones((batch_size, from_seq_length)),
              tf.ones((batch_size, to_seq_length)))
          return l(hidden_states=query, kv=kv, attention_mask=mask)

        outputs = distribution.run(_step_fn, args=(inputs,))
        return tf.nest.map_structure(distribution.experimental_local_results,
                                     outputs)

      local_outputs = step_cross_attn((query, kv))
      self.assertEqual(local_outputs["layer_output"][0].shape,
                       (2, from_seq_length, 4))

  def test_encoder_block(self):
    batch_size = 2
    from_seq_length = 5
    d_model = 4
    l = t5.EncoderBlock(d_model=4, d_kv=3, num_heads=2, d_ff=8, name="foo")
    pos_embed = t5.RelativePositionEmbedding(
        num_heads=2,
        bidirectional=True,
        embeddings_initializer=tf.keras.initializers.Ones(),
        name="bar")
    attention_mask = t5.make_attention_mask(
        tf.ones((batch_size, from_seq_length)),
        tf.ones((batch_size, from_seq_length)))
    position_bias = pos_embed(from_seq_length, from_seq_length)
    inputs = tf.ones((batch_size, from_seq_length, d_model), dtype=tf.float32)
    outputs = l(
        inputs, attention_mask=attention_mask, position_bias=position_bias)
    self.assertEqual(outputs.shape, (batch_size, from_seq_length, d_model))

  def test_encdec_block(self):
    batch_size = 2
    from_seq_length = 5
    to_seq_length = 3
    d_model = 4
    l = t5.EncDecoderBlock(d_model=4, d_kv=3, num_heads=2, d_ff=8, name="foo")
    pos_embed = t5.RelativePositionEmbedding(
        num_heads=2,
        bidirectional=True,
        embeddings_initializer=tf.keras.initializers.Ones(),
        name="bar")
    encoder_decoder_mask = t5.make_attention_mask(
        tf.ones((batch_size, from_seq_length)),
        tf.ones((batch_size, to_seq_length)))
    position_bias = pos_embed(from_seq_length, from_seq_length)
    inputs = tf.ones((batch_size, from_seq_length, d_model), dtype=tf.float32)
    encoder_hidden_states = tf.ones((batch_size, to_seq_length, d_model),
                                    dtype=tf.float32)
    outputs = l(
        inputs,
        encoder_hidden_states,
        encoder_decoder_mask=encoder_decoder_mask,
        position_bias=position_bias)
    self.assertEqual(outputs[0].shape, (batch_size, from_seq_length, d_model))

  @parameterized.named_parameters(("bfloat16", tf.bfloat16),
                                  ("float32", tf.float32))
  def test_encoder(self, dtype):
    config = t5.T5TransformerParams(
        num_layers=2,
        d_model=4,
        d_kv=3,
        num_heads=4,
        d_ff=16,
        vocab_size=10,
        vocab_embeddings_initializer=tf.keras.initializers.Ones(),
        relative_embeddings_initializer=tf.keras.initializers.Ones())
    encoder = t5.Encoder(config, compute_dtype=dtype)
    encoded = encoder(tf.zeros((4, 8), dtype=tf.int32))
    self.assertEqual(encoded.shape, (4, 8, config.d_model))

  @parameterized.named_parameters(("bfloat16", tf.bfloat16),
                                  ("float32", tf.float32))
  def test_encoder_with_dense(self, dtype):
    config = t5.T5TransformerParams(
        num_layers=2,
        d_model=4,
        d_kv=3,
        num_heads=4,
        d_ff=16,
        vocab_size=10,
        vocab_embeddings_initializer=tf.keras.initializers.Ones(),
        relative_embeddings_initializer=tf.keras.initializers.Ones())
    encoder = t5.Encoder(config, compute_dtype=dtype)
    encoded = encoder(
        tf.zeros((4, 8), dtype=tf.int32),
        dense_inputs=tf.ones((4, 2, 4), dtype=dtype))
    self.assertEqual(encoded.shape, (4, 10, config.d_model))

  @parameterized.named_parameters(("bfloat16", tf.bfloat16),
                                  ("float32", tf.float32))
  def test_encoder_only_dense(self, dtype):
    config = t5.T5TransformerParams(
        num_layers=2,
        d_model=4,
        d_kv=3,
        num_heads=4,
        d_ff=16,
        vocab_size=10,
        vocab_embeddings_initializer=tf.keras.initializers.Ones(),
        relative_embeddings_initializer=tf.keras.initializers.Ones())
    encoder = t5.Encoder(config, compute_dtype=dtype)
    encoded = encoder(dense_inputs=tf.ones((4, 2, 4), dtype=dtype))
    self.assertEqual(encoded.shape, (4, 2, config.d_model))

  def test_decoder(self):
    max_decode_len = 10
    config = t5.T5TransformerParams(
        num_layers=2,
        d_model=4,
        d_kv=3,
        num_heads=4,
        d_ff=16,
        vocab_size=10,
        vocab_embeddings_initializer=tf.keras.initializers.Ones(),
        relative_embeddings_initializer=tf.keras.initializers.Ones())
    decoder = t5.Decoder(config)
    batch_size = 4
    targets = tf.zeros((4, 8), dtype=tf.int32)
    encoded = tf.zeros((4, 8, config.d_model), dtype=tf.float32)
    outputs = decoder(targets, encoded)
    logits = outputs["logits"]
    cache = outputs["cache"]
    self.assertEqual(logits.shape, (4, 8, config.vocab_size))

    cache = {}
    cache[0] = _create_cache(batch_size, max_decode_len, config.num_heads,
                             config.d_kv)
    cache[1] = _create_cache(batch_size, max_decode_len, config.num_heads,
                             config.d_kv)
    targets = tf.zeros((4, 1), dtype=tf.int32)
    outputs = decoder(
        targets,
        encoded,
        decode_position=2,
        cache=cache,
        decode=True,
        max_decode_len=max_decode_len)
    logits = outputs["logits"]
    cache = outputs["cache"]
    self.assertEqual(logits.shape, (batch_size, 1, config.vocab_size))
    for entry in cache.values():
      for tensor in entry.values():
        self.assertNotAllEqual(tensor.numpy()[:, 2, :, :], 0.0)

  @parameterized.named_parameters(
      ("t5_10", ("relu",), True, 26, False, tf.float32),
      ("t5_11", ("gelu", "linear"), False, 29, False, tf.float32),
      ("t5_10_bfloat16", ("relu",), True, 26, False, tf.bfloat16),
      ("t5_11_bfloat16", ("gelu", "linear"), False, 29, False, tf.bfloat16),
      ("t5_10_layer_sharing", ("relu",), True, 26, True, tf.float32),
      ("t5_11_layer_sharing", ("gelu", "linear"), False, 29, True, tf.float32),
      ("t5_10_bfloat16_layer_sharing", ("relu",), True, 26, True, tf.bfloat16),
      ("t5_11_bfloat16_layer_sharing",
       ("gelu", "linear"), False, 29, True, tf.bfloat16))
  def test_transformer(self, ffn_activations, logits_via_embedding,
                       expect_num_variables, layer_sharing, dtype):
    max_decode_len = 10
    config = t5.T5TransformerParams(
        num_layers=1,
        d_model=8,
        d_kv=4,
        num_heads=4,
        d_ff=32,
        vocab_size=10,
        shared_embedding=True,
        layer_sharing=layer_sharing,
        ffn_activations=ffn_activations,
        logits_via_embedding=logits_via_embedding)
    transformer = t5.T5Transformer(config, compute_dtype=dtype)
    self.assertLen(transformer.trainable_variables, expect_num_variables)
    inputs = tf.convert_to_tensor(
        np.array([[2, 2, 1, 3, 1, 0], [3, 3, 1, 2, 2, 1]]))
    segments = tf.convert_to_tensor(
        np.array([[1, 1, 1, 2, 2, 0], [1, 1, 1, 2, 2, 2]]))

    outputs = transformer(
        encoder_input_tokens=inputs,
        decoder_input_tokens=inputs,
        decoder_target_tokens=inputs,
        encoder_segment_ids=segments,
        decoder_segment_ids=segments)
    cache = {}
    batch_size = 2
    cache[0] = _create_cache(
        batch_size, max_decode_len, config.num_heads, config.d_kv, dtype=dtype)
    outputs = transformer.decode(
        encoder_input_tokens=inputs,
        encoded=outputs["encoded"],
        decoder_target_tokens=tf.ones((batch_size, 1), dtype=tf.int32),
        decode_position=1,
        decode=True,
        max_decode_len=max_decode_len,
        cache=cache)
    self.assertEqual(outputs["logits"].shape,
                     (batch_size, 1, config.vocab_size))
    for v in transformer.trainable_variables:
      print(v.name, v.shape)
      self.assertEqual(v.dtype, tf.float32)

  @parameterized.named_parameters(
      ("t5_10_dense", ("relu",), True, 26, False, tf.float32),)
  def test_transformer_with_dense(self, ffn_activations, logits_via_embedding,
                                  expect_num_variables, layer_sharing, dtype):
    max_decode_len = 10
    config = t5.T5TransformerParams(
        num_layers=1,
        d_model=8,
        d_kv=4,
        num_heads=4,
        d_ff=32,
        vocab_size=10,
        shared_embedding=True,
        layer_sharing=layer_sharing,
        ffn_activations=ffn_activations,
        logits_via_embedding=logits_via_embedding)
    transformer = t5.T5Transformer(config, compute_dtype=dtype)

    self.assertLen(transformer.trainable_variables, expect_num_variables)
    inputs = tf.convert_to_tensor(
        np.array([[2, 2, 1, 3, 1, 0], [3, 3, 1, 2, 2, 1]]))
    segments = tf.convert_to_tensor(
        np.array([[1, 1, 1, 2, 2, 0], [1, 1, 1, 2, 2, 2]]))

    dense_inputs = tf.convert_to_tensor(np.random.randn(2, 2, 8), dtype=dtype)
    dense_segments = tf.convert_to_tensor(np.array([[1, 2], [1, 2]]))
    outputs = transformer(
        encoder_input_tokens=inputs,
        encoder_dense_inputs=dense_inputs,
        decoder_input_tokens=inputs,
        decoder_target_tokens=inputs,
        encoder_segment_ids=segments,
        encoder_dense_segment_ids=dense_segments,
        decoder_segment_ids=segments)
    cache = {}
    batch_size = 2
    cache[0] = _create_cache(
        batch_size, max_decode_len, config.num_heads, config.d_kv, dtype=dtype)
    outputs = transformer.decode(
        encoder_input_tokens=inputs,
        encoder_dense_inputs=dense_inputs,
        encoded=outputs["encoded"],
        decoder_target_tokens=tf.ones((batch_size, 1), dtype=tf.int32),
        decode_position=1,
        decode=True,
        max_decode_len=max_decode_len,
        cache=cache)
    self.assertEqual(outputs["logits"].shape,
                     (batch_size, 1, config.vocab_size))
    for v in transformer.trainable_variables:
      print(v.name, v.shape)
      self.assertEqual(v.dtype, tf.float32)

  @parameterized.named_parameters(
      ("t5_10_dense_layerwise_relpos",
       ("relu",), True, 26, False, tf.float32, False, 1),
      ("t5_10_dense_shared_relpos_d2",
       ("relu",), True, 39, False, tf.float32, True, 2),
      ("t5_10_dense_layerwise_relpos_d2",
       ("relu",), True, 40, False, tf.float32, False, 2),
  )
  def test_transformer_with_lw_relpos(self, ffn_activations,
                                      logits_via_embedding,
                                      expect_num_variables, layer_sharing,
                                      dtype, use_shared_relpos,
                                      num_decoder_layers):
    max_decode_len = 10
    config = t5.T5TransformerParams(
        num_layers=1,
        num_decoder_layers=num_decoder_layers,
        d_model=8,
        d_kv=4,
        num_heads=4,
        d_ff=32,
        vocab_size=10,
        shared_embedding=True,
        layer_sharing=layer_sharing,
        ffn_activations=ffn_activations,
        logits_via_embedding=logits_via_embedding,
        use_shared_relative_position_bias=use_shared_relpos)
    transformer = t5.T5Transformer(config, compute_dtype=dtype)

    self.assertLen(transformer.trainable_variables, expect_num_variables)
    inputs = tf.convert_to_tensor(
        np.array([[2, 2, 1, 3, 1, 0], [3, 3, 1, 2, 2, 1]]))
    segments = tf.convert_to_tensor(
        np.array([[1, 1, 1, 2, 2, 0], [1, 1, 1, 2, 2, 2]]))

    dense_inputs = tf.convert_to_tensor(np.random.randn(2, 2, 8), dtype=dtype)
    dense_segments = tf.convert_to_tensor(np.array([[1, 2], [1, 2]]))
    outputs = transformer(
        encoder_input_tokens=inputs,
        encoder_dense_inputs=dense_inputs,
        decoder_input_tokens=inputs,
        decoder_target_tokens=inputs,
        encoder_segment_ids=segments,
        encoder_dense_segment_ids=dense_segments,
        decoder_segment_ids=segments)
    cache = {}
    batch_size = 2
    for i in range(num_decoder_layers):
      cache[i] = _create_cache(
          batch_size,
          max_decode_len,
          config.num_heads,
          config.d_kv,
          dtype=dtype)
    outputs = transformer.decode(
        encoder_input_tokens=inputs,
        encoder_dense_inputs=dense_inputs,
        encoded=outputs["encoded"],
        decoder_target_tokens=tf.ones((batch_size, 1), dtype=tf.int32),
        decode_position=1,
        decode=True,
        max_decode_len=max_decode_len,
        cache=cache)
    self.assertEqual(outputs["logits"].shape,
                     (batch_size, 1, config.vocab_size))
    for v in transformer.trainable_variables:
      print(v.name, v.shape)
      self.assertEqual(v.dtype, tf.float32)

  @parameterized.named_parameters(
      ("t5_10", ("relu",), True, 26, False, tf.float32),)
  def test_transformer_with_dense_only(self, ffn_activations,
                                       logits_via_embedding,
                                       expect_num_variables, layer_sharing,
                                       dtype):
    max_decode_len = 10
    config = t5.T5TransformerParams(
        num_layers=1,
        d_model=8,
        d_kv=4,
        num_heads=4,
        d_ff=32,
        vocab_size=10,
        shared_embedding=True,
        layer_sharing=layer_sharing,
        ffn_activations=ffn_activations,
        logits_via_embedding=logits_via_embedding)
    transformer = t5.T5Transformer(config, compute_dtype=dtype)
    self.assertLen(transformer.trainable_variables, expect_num_variables)

    decoder_inputs = tf.convert_to_tensor(
        np.array([[2, 2, 1, 3, 1, 0], [3, 3, 1, 2, 2, 1]]))
    decoder_segments = tf.convert_to_tensor(
        np.array([[1, 1, 1, 2, 2, 0], [1, 1, 1, 2, 2, 2]]))

    dense_inputs = tf.convert_to_tensor(np.random.randn(2, 2, 8), dtype=dtype)
    dense_segments = tf.convert_to_tensor(np.array([[1, 2], [1, 2]]))
    outputs = transformer(
        encoder_dense_inputs=dense_inputs,
        encoder_dense_segment_ids=dense_segments,
        decoder_input_tokens=decoder_inputs,
        decoder_target_tokens=decoder_inputs,
        decoder_segment_ids=decoder_segments)
    cache = {}
    batch_size = 2
    cache[0] = _create_cache(
        batch_size, max_decode_len, config.num_heads, config.d_kv, dtype=dtype)
    outputs = transformer.decode(
        encoder_dense_inputs=dense_inputs,
        encoded=outputs["encoded"],
        decoder_target_tokens=tf.ones((batch_size, 1), dtype=tf.int32),
        decode_position=1,
        decode=True,
        max_decode_len=max_decode_len,
        cache=cache)
    self.assertEqual(outputs["logits"].shape,
                     (batch_size, 1, config.vocab_size))
    for v in transformer.trainable_variables:
      print(v.name, v.shape)
      self.assertEqual(v.dtype, tf.float32)

  @parameterized.named_parameters(
      ("t5_10", ("relu",), True, 39, tf.float32, 2),
      ("t5_10_bfloat16", ("relu",), True, 39, tf.bfloat16, 2))
  def test_transformer_different_num_decoder_layers(self, ffn_activations,
                                                    logits_via_embedding,
                                                    expect_num_variables, dtype,
                                                    num_decoder_layers):
    max_decode_len = 10
    config = t5.T5TransformerParams(
        num_decoder_layers=num_decoder_layers,
        num_layers=1,
        d_model=8,
        d_kv=4,
        num_heads=4,
        d_ff=32,
        vocab_size=10,
        shared_embedding=True,
        ffn_activations=ffn_activations,
        logits_via_embedding=logits_via_embedding)
    transformer = t5.T5Transformer(config, compute_dtype=dtype)
    self.assertLen(transformer.trainable_variables, expect_num_variables)
    inputs = tf.convert_to_tensor(
        np.array([[2, 2, 1, 3, 1, 0], [3, 3, 1, 2, 2, 1]]))
    segments = tf.convert_to_tensor(
        np.array([[1, 1, 1, 2, 2, 0], [1, 1, 1, 2, 2, 2]]))

    outputs = transformer(
        encoder_input_tokens=inputs,
        decoder_input_tokens=inputs,
        decoder_target_tokens=inputs,
        encoder_segment_ids=segments,
        decoder_segment_ids=segments)
    cache = {}
    batch_size = 2
    for i in range(num_decoder_layers):
      cache[i] = _create_cache(
          batch_size,
          max_decode_len,
          config.num_heads,
          config.d_kv,
          dtype=dtype)
    outputs = transformer.decode(
        encoder_input_tokens=inputs,
        encoded=outputs["encoded"],
        decoder_target_tokens=tf.ones((batch_size, 1), dtype=tf.int32),
        decode_position=1,
        decode=True,
        max_decode_len=max_decode_len,
        cache=cache)
    self.assertEqual(outputs["logits"].shape,
                     (batch_size, 1, config.vocab_size))
    for v in transformer.trainable_variables:
      print(v.name, v.shape)
      self.assertEqual(v.dtype, tf.float32)


if __name__ == "__main__":
  tf.test.main()
