# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test Transformer model."""

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.nlp.modeling.models import seq2seq_transformer
from official.nlp.transformer import model_params


class Seq2SeqTransformerTest(tf.test.TestCase, parameterized.TestCase):

  def test_create_model(self):
    self.params = model_params.TINY_PARAMS
    self.params["batch_size"] = 16
    self.params["hidden_size"] = 12
    self.params["num_hidden_layers"] = 2
    self.params["filter_size"] = 14
    self.params["num_heads"] = 2
    self.params["vocab_size"] = 41
    self.params["extra_decode_length"] = 2
    self.params["beam_size"] = 3
    self.params["dtype"] = tf.float32
    model = seq2seq_transformer.create_model(self.params, is_train=True)
    inputs, outputs = model.inputs, model.outputs
    self.assertLen(inputs, 2)
    self.assertLen(outputs, 1)
    self.assertEqual(inputs[0].shape.as_list(), [None, None])
    self.assertEqual(inputs[0].dtype, tf.int64)
    self.assertEqual(inputs[1].shape.as_list(), [None, None])
    self.assertEqual(inputs[1].dtype, tf.int64)
    self.assertEqual(outputs[0].shape.as_list(), [None, None, 41])
    self.assertEqual(outputs[0].dtype, tf.float32)

    model = seq2seq_transformer.create_model(self.params, is_train=False)
    inputs, outputs = model.inputs, model.outputs
    self.assertLen(inputs, 1)
    self.assertLen(outputs, 2)
    self.assertEqual(inputs[0].shape.as_list(), [None, None])
    self.assertEqual(inputs[0].dtype, tf.int64)
    self.assertEqual(outputs[0].shape.as_list(), [None, None])
    self.assertEqual(outputs[0].dtype, tf.int32)
    self.assertEqual(outputs[1].shape.as_list(), [None])
    self.assertEqual(outputs[1].dtype, tf.float32)

  def _build_model(self, padded_decode, decode_max_length):
    num_layers = 1
    num_attention_heads = 2
    intermediate_size = 32
    vocab_size = 100
    embedding_width = 16
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
    decoder_layer = seq2seq_transformer.TransformerDecoder(**encdec_kwargs)

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
              strategy_combinations.tpu_strategy,
          ],
          mode="eager"))
  def test_create_model_with_ds(self, distribution):
    with distribution.scope():
      padded_decode = isinstance(distribution,
                                 tf.distribute.experimental.TPUStrategy)
      decode_max_length = 10
      batch_size = 4
      model = self._build_model(padded_decode, decode_max_length)

      @tf.function
      def step(inputs):

        def _step_fn(inputs):
          return model(inputs)

        outputs = distribution.run(_step_fn, args=(inputs,))
        return tf.nest.map_structure(distribution.experimental_local_results,
                                     outputs)

      fake_inputs = [np.zeros((batch_size, decode_max_length), dtype=np.int32)]
      local_outputs = step(fake_inputs)
      logging.info("local_outputs=%s", local_outputs)
      self.assertEqual(local_outputs["outputs"][0].shape, (4, 10))

      fake_inputs = [
          np.zeros((batch_size, decode_max_length), dtype=np.int32),
          np.zeros((batch_size, 8), dtype=np.int32)
      ]
      local_outputs = step(fake_inputs)
      logging.info("local_outputs=%s", local_outputs)
      self.assertEqual(local_outputs[0].shape, (4, 8, 100))

  @parameterized.parameters(True, False)
  def test_create_savedmodel(self, padded_decode):
    decode_max_length = 10
    model = self._build_model(padded_decode, decode_max_length)

    class SaveModule(tf.Module):

      def __init__(self, model):
        super(SaveModule, self).__init__()
        self.model = model

      @tf.function
      def serve(self, inputs):
        return self.model.call([inputs])

    save_module = SaveModule(model)
    if padded_decode:
      tensor_shape = (4, 10)
    else:
      tensor_shape = (None, None)
    signatures = dict(
        serving_default=save_module.serve.get_concrete_function(
            tf.TensorSpec(shape=tensor_shape, dtype=tf.int32, name="inputs")))
    tf.saved_model.save(save_module, self.get_temp_dir(), signatures=signatures)


if __name__ == "__main__":
  tf.test.main()
