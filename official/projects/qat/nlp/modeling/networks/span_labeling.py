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

"""Span labeling network."""
# pylint: disable=g-classes-have-attributes
import collections
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from official.projects.qat.nlp.quantization import configs


def _apply_paragraph_mask(logits, paragraph_mask):
  """Applies a position mask to calculated logits."""
  masked_logits = logits * (paragraph_mask) - 1e30 * (1 - paragraph_mask)
  return tf.nn.log_softmax(masked_logits, -1), masked_logits


@tf.keras.utils.register_keras_serializable(package='Text')
class SpanLabelingQuantized(tf.keras.Model):
  """Span labeling network head for BERT modeling.

  This network implements a simple single-span labeler based on a dense layer.
  *Note* that the network is constructed by
  [Keras Functional API](https://keras.io/guides/functional_api/).

  Args:
    input_width: The innermost dimension of the input tensor to this network.
    activation: The activation, if any, for the dense layer in this network.
    initializer: The initializer for the dense layer in this network. Defaults
      to a Glorot uniform initializer.
    output: The output style for this network. Can be either `logits` or
      `predictions`.
  """

  def __init__(self,
               input_width,
               activation=None,
               initializer='glorot_uniform',
               output='logits',
               **kwargs):

    sequence_data = tf.keras.layers.Input(
        shape=(None, input_width), name='sequence_data', dtype=tf.float32)

    logits_layer = tf.keras.layers.Dense(
        2,  # This layer predicts start location and end location.
        activation=activation,
        kernel_initializer=initializer,
        name='predictions/transform/logits')
    logits_layer = tfmot.quantization.keras.QuantizeWrapperV2(
        logits_layer,
        configs.Default8BitQuantizeConfig(['kernel'], ['activation'], False))
    intermediate_logits = logits_layer(sequence_data)
    start_logits, end_logits = self._split_output_tensor(intermediate_logits)

    start_predictions = tf.keras.layers.Activation(tf.nn.log_softmax)(
        start_logits)
    end_predictions = tf.keras.layers.Activation(tf.nn.log_softmax)(end_logits)

    if output == 'logits':
      output_tensors = [start_logits, end_logits]
    elif output == 'predictions':
      output_tensors = [start_predictions, end_predictions]
    else:
      raise ValueError(
          ('Unknown `output` value "%s". `output` can be either "logits" or '
           '"predictions"') % output)

    # b/164516224
    # Once we've created the network using the Functional API, we call
    # super().__init__ as though we were invoking the Functional API Model
    # constructor, resulting in this object having all the properties of a model
    # created using the Functional API. Once super().__init__ is called, we
    # can assign attributes to `self` - note that all `self` assignments are
    # below this line.
    super().__init__(
        inputs=[sequence_data], outputs=output_tensors, **kwargs)
    config_dict = {
        'input_width': input_width,
        'activation': activation,
        'initializer': initializer,
        'output': output,
    }
    # We are storing the config dict as a namedtuple here to ensure checkpoint
    # compatibility with an earlier version of this model which did not track
    # the config dict attribute. TF does not track immutable attrs which
    # do not contain Trackables, so by creating a config namedtuple instead of
    # a dict we avoid tracking it.
    config_cls = collections.namedtuple('Config', config_dict.keys())
    self._config = config_cls(**config_dict)
    self.start_logits = start_logits
    self.end_logits = end_logits

  def _split_output_tensor(self, tensor):
    transposed_tensor = tf.transpose(tensor, [2, 0, 1])
    return tf.unstack(transposed_tensor)

  def get_config(self):
    return dict(self._config._asdict())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

