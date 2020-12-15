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
from typing import Any, Callable

import orbit
import tensorflow as tf
import tensorflow_hub as hub


def get_encoder_from_hub(hub_model_path: str) -> tf.keras.Model:
  """Gets an encoder from hub.

  Args:
    hub_model_path: The path to the tfhub model.

  Returns:
    A tf.keras.Model.
  """
  input_word_ids = tf.keras.layers.Input(
      shape=(None,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.layers.Input(
      shape=(None,), dtype=tf.int32, name='input_mask')
  input_type_ids = tf.keras.layers.Input(
      shape=(None,), dtype=tf.int32, name='input_type_ids')
  hub_layer = hub.KerasLayer(hub_model_path, trainable=True)
  output_dict = {}
  dict_input = dict(
      input_word_ids=input_word_ids,
      input_mask=input_mask,
      input_type_ids=input_type_ids)
  output_dict = hub_layer(dict_input)

  return tf.keras.Model(inputs=dict_input, outputs=output_dict)


def predict(predict_step_fn: Callable[[Any], Any],
            aggregate_fn: Callable[[Any, Any], Any], dataset: tf.data.Dataset):
  """Runs prediction.

  Args:
    predict_step_fn: A callable such as `def predict_step(inputs)`, where
      `inputs` are input tensors.
    aggregate_fn: A callable such as `def aggregate_fn(state, value)`, where
      `value` is the outputs from `predict_step_fn`.
    dataset: A `tf.data.Dataset` object.

  Returns:
    The aggregated predictions.
  """

  @tf.function
  def predict_step(iterator):
    """Predicts on distributed devices."""
    outputs = tf.distribute.get_strategy().run(
        predict_step_fn, args=(next(iterator),))
    return tf.nest.map_structure(
        tf.distribute.get_strategy().experimental_local_results, outputs)

  loop_fn = orbit.utils.create_loop_fn(predict_step)
  # Set `num_steps` to -1 to exhaust the dataset.
  outputs = loop_fn(
      iter(dataset), num_steps=-1, state=None, reduce_fn=aggregate_fn)  # pytype: disable=wrong-arg-types
  return outputs
