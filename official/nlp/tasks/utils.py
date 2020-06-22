# Lint as: python3
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
"""Common utils for tasks."""
import tensorflow as tf
import tensorflow_hub as hub


def get_encoder_from_hub(hub_module: str) -> tf.keras.Model:
  """Gets an encoder from hub."""
  input_word_ids = tf.keras.layers.Input(
      shape=(None,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.layers.Input(
      shape=(None,), dtype=tf.int32, name='input_mask')
  input_type_ids = tf.keras.layers.Input(
      shape=(None,), dtype=tf.int32, name='input_type_ids')
  hub_layer = hub.KerasLayer(hub_module, trainable=True)
  pooled_output, sequence_output = hub_layer(
      [input_word_ids, input_mask, input_type_ids])
  return tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=[sequence_output, pooled_output])
