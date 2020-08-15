# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Span labeling network."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Text')
class SpanLabeling(tf.keras.Model):
  """Span labeling network head for BERT modeling.

  This network implements a simple single-span labeler based on a dense layer.
  *Note* that the network is constructed by
  [Keras Functional API](https://keras.io/guides/functional_api/).

  Arguments:
    input_width: The innermost dimension of the input tensor to this network.
    activation: The activation, if any, for the dense layer in this network.
    initializer: The initializer for the dense layer in this network. Defaults
      to a Glorot uniform initializer.
    output: The output style for this network. Can be either 'logits' or
      'predictions'.
  """

  def __init__(self,
               input_width,
               activation=None,
               initializer='glorot_uniform',
               output='logits',
               **kwargs):
    self._self_setattr_tracking = False
    self._config = {
        'input_width': input_width,
        'activation': activation,
        'initializer': initializer,
        'output': output,
    }

    sequence_data = tf.keras.layers.Input(
        shape=(None, input_width), name='sequence_data', dtype=tf.float32)

    intermediate_logits = tf.keras.layers.Dense(
        2,  # This layer predicts start location and end location.
        activation=activation,
        kernel_initializer=initializer,
        name='predictions/transform/logits')(
            sequence_data)
    self.start_logits, self.end_logits = (
        tf.keras.layers.Lambda(self._split_output_tensor)(intermediate_logits))

    start_predictions = tf.keras.layers.Activation(tf.nn.log_softmax)(
        self.start_logits)
    end_predictions = tf.keras.layers.Activation(tf.nn.log_softmax)(
        self.end_logits)

    if output == 'logits':
      output_tensors = [self.start_logits, self.end_logits]
    elif output == 'predictions':
      output_tensors = [start_predictions, end_predictions]
    else:
      raise ValueError(
          ('Unknown `output` value "%s". `output` can be either "logits" or '
           '"predictions"') % output)

    super(SpanLabeling, self).__init__(
        inputs=[sequence_data], outputs=output_tensors, **kwargs)

  def _split_output_tensor(self, tensor):
    transposed_tensor = tf.transpose(tensor, [2, 0, 1])
    return tf.unstack(transposed_tensor)

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
