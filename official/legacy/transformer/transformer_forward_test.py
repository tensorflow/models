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

"""Forward pass test for Transformer model refactoring."""

import numpy as np
import tensorflow as tf

from official.legacy.transformer import metrics
from official.legacy.transformer import model_params
from official.legacy.transformer import transformer
from official.nlp.modeling import models


def _count_params(layer, trainable_only=True):
  """Returns the count of all model parameters, or just trainable ones."""
  if not trainable_only:
    return layer.count_params()
  else:
    return int(
        np.sum([
            tf.keras.backend.count_params(p) for p in layer.trainable_weights
        ]))


def _create_model(params, is_train):
  """Creates transformer model."""

  encdec_kwargs = dict(
      num_layers=params["num_hidden_layers"],
      num_attention_heads=params["num_heads"],
      intermediate_size=params["filter_size"],
      activation="relu",
      dropout_rate=params["relu_dropout"],
      attention_dropout_rate=params["attention_dropout"],
      use_bias=False,
      norm_first=True,
      norm_epsilon=1e-6,
      intermediate_dropout=params["relu_dropout"])
  encoder_layer = models.TransformerEncoder(**encdec_kwargs)
  decoder_layer = models.TransformerDecoder(**encdec_kwargs)

  model_kwargs = dict(
      vocab_size=params["vocab_size"],
      embedding_width=params["hidden_size"],
      dropout_rate=params["layer_postprocess_dropout"],
      padded_decode=params["padded_decode"],
      decode_max_length=params["decode_max_length"],
      dtype=params["dtype"],
      extra_decode_length=params["extra_decode_length"],
      beam_size=params["beam_size"],
      alpha=params["alpha"],
      encoder_layer=encoder_layer,
      decoder_layer=decoder_layer,
      name="transformer_v2")

  if is_train:
    inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
    targets = tf.keras.layers.Input((None,), dtype="int64", name="targets")
    internal_model = models.Seq2SeqTransformer(**model_kwargs)
    logits = internal_model(
        dict(inputs=inputs, targets=targets), training=is_train)
    vocab_size = params["vocab_size"]
    label_smoothing = params["label_smoothing"]
    if params["enable_metrics_in_training"]:
      logits = metrics.MetricLayer(vocab_size)([logits, targets])
    logits = tf.keras.layers.Lambda(
        lambda x: x, name="logits", dtype=tf.float32)(
            logits)
    model = tf.keras.Model([inputs, targets], logits)
    loss = metrics.transformer_loss(logits, targets, label_smoothing,
                                    vocab_size)
    model.add_loss(loss)
    return model

  batch_size = params["decode_batch_size"] if params["padded_decode"] else None
  inputs = tf.keras.layers.Input((None,),
                                 batch_size=batch_size,
                                 dtype="int64",
                                 name="inputs")
  internal_model = models.Seq2SeqTransformer(**model_kwargs)
  ret = internal_model(dict(inputs=inputs), training=is_train)
  outputs, scores = ret["outputs"], ret["scores"]
  return tf.keras.Model(inputs, [outputs, scores])


class TransformerForwardTest(tf.test.TestCase):

  def setUp(self):
    super(TransformerForwardTest, self).setUp()
    self.params = params = model_params.TINY_PARAMS
    params["batch_size"] = params["default_batch_size"] = 16
    params["hidden_size"] = 12
    params["num_hidden_layers"] = 3
    params["filter_size"] = 14
    params["num_heads"] = 2
    params["vocab_size"] = 41
    params["extra_decode_length"] = 0
    params["beam_size"] = 3
    params["dtype"] = tf.float32
    params["layer_postprocess_dropout"] = 0.0
    params["attention_dropout"] = 0.0
    params["relu_dropout"] = 0.0

  def test_forward_pass_train(self):
    # Set input_len different from target_len
    inputs = np.asarray([[5, 2, 1], [7, 5, 0], [1, 4, 0], [7, 5, 11]])
    targets = np.asarray([[4, 3, 4, 0], [13, 19, 17, 8], [20, 14, 1, 2],
                          [5, 7, 3, 0]])

    # src_model is the original model before refactored.
    src_model = transformer.create_model(self.params, True)
    src_num_weights = _count_params(src_model)
    src_weights = src_model.get_weights()
    src_model_output = src_model([inputs, targets], training=True)

    # dest_model is the refactored model.
    dest_model = _create_model(self.params, True)
    dest_num_weights = _count_params(dest_model)
    self.assertEqual(src_num_weights, dest_num_weights)
    dest_model.set_weights(src_weights)
    dest_model_output = dest_model([inputs, targets], training=True)
    self.assertAllEqual(src_model_output, dest_model_output)

  def test_forward_pass_not_train(self):
    inputs = np.asarray([[5, 2, 1], [7, 5, 0], [1, 4, 0], [7, 5, 11]])

    # src_model is the original model before refactored.
    src_model = transformer.create_model(self.params, False)
    src_num_weights = _count_params(src_model)
    src_weights = src_model.get_weights()
    src_model_output = src_model([inputs], training=False)

    # dest_model is the refactored model.
    dest_model = _create_model(self.params, False)
    dest_num_weights = _count_params(dest_model)
    self.assertEqual(src_num_weights, dest_num_weights)
    dest_model.set_weights(src_weights)
    dest_model_output = dest_model([inputs], training=False)
    self.assertAllEqual(src_model_output[0], dest_model_output[0])
    self.assertAllEqual(src_model_output[1], dest_model_output[1])


if __name__ == "__main__":
  tf.test.main()
