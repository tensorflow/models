# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Test Transformer model."""

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.nlp.modeling.layers import attention
from official.nlp.modeling.models import seq2seq_transformer


class Seq2SeqTransformerTest(tf.test.TestCase, parameterized.TestCase):

  def _build_model(
      self,
      padded_decode,
      decode_max_length,
      embedding_width,
      self_attention_cls=None,
      cross_attention_cls=None,
  ):
    num_layers = 1
    num_attention_heads = 2
    intermediate_size = 32
    vocab_size = 100
    encdec_kwargs = dict(
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        activation="relu",
        dropout_rate=0.01,
        attention_dropout_rate=0.01,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        intermediate_dropout=0.01)
    encoder_layer = seq2seq_transformer.TransformerEncoder(**encdec_kwargs)
    decoder_layer = seq2seq_transformer.TransformerDecoder(
        **encdec_kwargs,
        self_attention_cls=self_attention_cls,
        cross_attention_cls=cross_attention_cls
    )

    return seq2seq_transformer.Seq2SeqTransformer(
        vocab_size=vocab_size,
        embedding_width=embedding_width,
        dropout_rate=0.01,
        padded_decode=padded_decode,
        decode_max_length=decode_max_length,
        beam_size=4,
        alpha=0.6,
        encoder_layer=encoder_layer,
        decoder_layer=decoder_layer)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.default_strategy,
              strategy_combinations.cloud_tpu_strategy,
          ],
          embed=[True, False],
          is_training=[True, False],
          custom_self_attention=[False, True],
          custom_cross_attention=[False, True],
          mode="eager"))
  def test_create_model_with_ds(
      self,
      distribution,
      embed,
      is_training,
      custom_self_attention,
      custom_cross_attention,
  ):
    self_attention_called = False
    cross_attention_called = False

    class SelfAttention(attention.CachedAttention):
      """Dummy implementation of custom attention."""

      def __call__(
          self, *args, **kwargs
      ):
        nonlocal self_attention_called
        self_attention_called = True
        return super().__call__(*args, **kwargs)

    class CrossAttention:
      """Dummy implementation of custom attention."""

      def __init__(self, *args, **kwargs):
        pass

      def __call__(self, query, value, attention_mask, **kwargs):
        nonlocal cross_attention_called
        cross_attention_called = True
        return query

    with distribution.scope():
      padded_decode = isinstance(
          distribution,
          (tf.distribute.TPUStrategy, tf.distribute.experimental.TPUStrategy))
      decode_max_length = 10
      batch_size = 4
      embedding_width = 16
      model = self._build_model(
          padded_decode,
          decode_max_length,
          embedding_width,
          self_attention_cls=SelfAttention if custom_self_attention else None,
          cross_attention_cls=CrossAttention
          if custom_cross_attention
          else None,
      )

      @tf.function
      def step(inputs):

        def _step_fn(inputs):
          return model(inputs)

        outputs = distribution.run(_step_fn, args=(inputs,))
        return tf.nest.map_structure(distribution.experimental_local_results,
                                     outputs)

      if embed:
        fake_inputs = dict(
            embedded_inputs=np.zeros(
                (batch_size, decode_max_length, embedding_width),
                dtype=np.float32),
            input_masks=np.ones((batch_size, decode_max_length), dtype=bool))
      else:
        fake_inputs = dict(
            inputs=np.zeros((batch_size, decode_max_length), dtype=np.int32))

      if is_training:
        fake_inputs["targets"] = np.zeros((batch_size, 8), dtype=np.int32)
        local_outputs = step(fake_inputs)
        logging.info("local_outputs=%s", local_outputs)
        self.assertEqual(local_outputs[0].shape, (4, 8, 100))
      else:
        local_outputs = step(fake_inputs)
        logging.info("local_outputs=%s", local_outputs)
        self.assertEqual(local_outputs["outputs"][0].shape, (4, 10))
    self.assertEqual(self_attention_called, custom_self_attention)
    self.assertEqual(cross_attention_called, custom_cross_attention)

  @parameterized.parameters(True, False)
  def test_create_savedmodel(self, padded_decode):
    decode_max_length = 10
    embedding_width = 16
    model = self._build_model(
        padded_decode, decode_max_length, embedding_width)

    class SaveModule(tf.Module):

      def __init__(self, model):
        super(SaveModule, self).__init__()
        self.model = model

      @tf.function
      def serve(self, inputs):
        return self.model.call(dict(inputs=inputs))

      @tf.function
      def embedded_serve(self, embedded_inputs, input_masks):
        return self.model.call(
            dict(embedded_inputs=embedded_inputs, input_masks=input_masks))

    save_module = SaveModule(model)
    if padded_decode:
      tensor_shape = (4, decode_max_length)
      embedded_tensor_shape = (4, decode_max_length, embedding_width)
    else:
      tensor_shape = (None, None)
      embedded_tensor_shape = (None, None, embedding_width)
    signatures = dict(
        serving_default=save_module.serve.get_concrete_function(
            tf.TensorSpec(shape=tensor_shape, dtype=tf.int32, name="inputs")),
        embedded_serving=save_module.embedded_serve.get_concrete_function(
            tf.TensorSpec(
                shape=embedded_tensor_shape, dtype=tf.float32,
                name="embedded_inputs"),
            tf.TensorSpec(
                shape=tensor_shape, dtype=tf.bool, name="input_masks"),
            ))
    tf.saved_model.save(save_module, self.get_temp_dir(), signatures=signatures)


if __name__ == "__main__":
  tf.test.main()
