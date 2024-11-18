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

"""Tests LaBSE's export_tfhub."""

import os

import numpy as np
import tensorflow as tf, tf_keras
import tensorflow_hub as hub
from official.legacy.bert import configs
from official.projects.labse import export_tfhub


class ExportModelTest(tf.test.TestCase):

  def test_export_model(self):
    # Exports a savedmodel for TF-Hub
    hidden_size = 16
    bert_config = configs.BertConfig(
        vocab_size=100,
        hidden_size=hidden_size,
        intermediate_size=32,
        max_position_embeddings=128,
        num_attention_heads=2,
        num_hidden_layers=1)
    labse_model, encoder = export_tfhub.create_labse_model(
        None, bert_config, normalize=True)
    model_checkpoint_dir = os.path.join(self.get_temp_dir(), "checkpoint")
    checkpoint = tf.train.Checkpoint(encoder=encoder)
    checkpoint.save(os.path.join(model_checkpoint_dir, "test"))
    model_checkpoint_path = tf.train.latest_checkpoint(model_checkpoint_dir)

    vocab_file = os.path.join(self.get_temp_dir(), "uncased_vocab.txt")
    with tf.io.gfile.GFile(vocab_file, "w") as f:
      f.write("dummy content")

    hub_destination = os.path.join(self.get_temp_dir(), "hub")
    export_tfhub.export_labse_model(
        None,  # bert_tfhub_module
        bert_config,
        model_checkpoint_path,
        hub_destination,
        vocab_file,
        do_lower_case=True,
        normalize=True)

    # Restores a hub KerasLayer.
    hub_layer = hub.KerasLayer(hub_destination, trainable=True)

    if hasattr(hub_layer, "resolved_object"):
      # Checks meta attributes.
      self.assertTrue(hub_layer.resolved_object.do_lower_case.numpy())
      with tf.io.gfile.GFile(
          hub_layer.resolved_object.vocab_file.asset_path.numpy()) as f:
        self.assertEqual("dummy content", f.read())
    # Checks the hub KerasLayer.
    for source_weight, hub_weight in zip(labse_model.trainable_weights,
                                         hub_layer.trainable_weights):
      self.assertAllClose(source_weight.numpy(), hub_weight.numpy())

    seq_length = 10
    dummy_ids = np.zeros((2, seq_length), dtype=np.int32)
    hub_outputs = hub_layer([dummy_ids, dummy_ids, dummy_ids])
    source_outputs = labse_model([dummy_ids, dummy_ids, dummy_ids])

    self.assertEqual(hub_outputs["pooled_output"].shape, (2, hidden_size))
    self.assertEqual(hub_outputs["sequence_output"].shape,
                     (2, seq_length, hidden_size))
    for output_name in source_outputs:
      self.assertAllClose(hub_outputs[output_name].numpy(),
                          hub_outputs[output_name].numpy())

    # Test that training=True makes a difference (activates dropout).
    def _dropout_mean_stddev(training, num_runs=20):
      input_ids = np.array([[14, 12, 42, 95, 99]], np.int32)
      inputs = [input_ids, np.ones_like(input_ids), np.zeros_like(input_ids)]
      outputs = np.concatenate([
          hub_layer(inputs, training=training)["pooled_output"]
          for _ in range(num_runs)
      ])
      return np.mean(np.std(outputs, axis=0))

    self.assertLess(_dropout_mean_stddev(training=False), 1e-6)
    self.assertGreater(_dropout_mean_stddev(training=True), 1e-3)

    # Test propagation of seq_length in shape inference.
    input_word_ids = tf_keras.layers.Input(shape=(seq_length,), dtype=tf.int32)
    input_mask = tf_keras.layers.Input(shape=(seq_length,), dtype=tf.int32)
    input_type_ids = tf_keras.layers.Input(shape=(seq_length,), dtype=tf.int32)
    outputs = hub_layer([input_word_ids, input_mask, input_type_ids])
    self.assertEqual(outputs["pooled_output"].shape.as_list(),
                     [None, hidden_size])
    self.assertEqual(outputs["sequence_output"].shape.as_list(),
                     [None, seq_length, hidden_size])


if __name__ == "__main__":
  tf.test.main()
