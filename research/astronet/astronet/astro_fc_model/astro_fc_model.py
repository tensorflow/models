# Copyright 2018 The TensorFlow Authors.
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

"""A model for classifying light curves using (locally) fully connected layers.

Note that the first layer of each fully connected stack is optionally
implemented as a convolution with a wide kernel followed by pooling. This causes
invariance to small translations.

See the base class (in astro_model.py) for a description of the general
framework of AstroModel and its subclasses.

The architecture of this model is:


                                     predictions
                                          ^
                                          |
                                       logits
                                          ^
                                          |
                                (fully connected layers)
                                          ^
                                          |
                                   pre_logits_concat
                                          ^
                                          |
                                    (concatenate)

              ^                           ^                          ^
              |                           |                          |
 (locally fully connected 1)  (locally fully connected 2)  ...       |
              ^                           ^                          |
              |                           |                          |
     time_series_feature_1      time_series_feature_2      ...  aux_features
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from astronet.astro_model import astro_model


class AstroFCModel(astro_model.AstroModel):
  """A model for classifying light curves using fully connected layers."""

  def _build_local_fc_layers(self, inputs, hparams, scope):
    """Builds locally fully connected layers.

    Note that the first layer of the fully connected stack is optionally
    implemented as a convolution with a wide kernel followed by pooling. This
    makes the fully connected stack invariant to small translations of its
    input.

    Args:
      inputs: A Tensor of shape [batch_size, length].
      hparams: Object containing hyperparameters.
      scope: Name of the variable scope.

    Returns:
      A Tensor of shape [batch_size, hparams.local_layer_size].

    Raises:
      ValueError: If hparams.pooling_type is unrecognized.
    """
    if hparams.num_local_layers == 0:
      return inputs

    net = inputs
    with tf.variable_scope(scope):
      # First layer is optionally implemented as a wide convolution for
      # invariance to small translations.
      if hparams.translation_delta > 0:
        kernel_size = inputs.shape.as_list()[1] - 2 * hparams.translation_delta
        net = tf.expand_dims(net, -1)  # [batch, length, channels=1]
        net = tf.layers.conv1d(
            inputs=net,
            filters=hparams.local_layer_size,
            kernel_size=kernel_size,
            padding="valid",
            activation=tf.nn.relu,
            name="conv1d")

        # net is [batch, length, num_filters], where length = 1 +
        # 2 * translation_delta. Pool along the length dimension.
        if hparams.pooling_type == "max":
          net = tf.reduce_max(net, axis=1, name="max_pool")
        elif hparams.pooling_type == "avg":
          net = tf.reduce_mean(net, axis=1, name="avg_pool")
        else:
          raise ValueError("Unrecognized pooling_type: {}".format(
              hparams.pooling_type))

        remaining_layers = hparams.num_local_layers - 1
      else:
        remaining_layers = hparams.num_local_layers

      # Remaining fully connected layers.
      for i in range(remaining_layers):
        net = tf.contrib.layers.fully_connected(
            inputs=net,
            num_outputs=hparams.local_layer_size,
            activation_fn=tf.nn.relu,
            scope="fully_connected_{}".format(i + 1))

        if hparams.dropout_rate > 0:
          net = tf.layers.dropout(
              net, hparams.dropout_rate, training=self.is_training)

    return net

  def build_time_series_hidden_layers(self):
    """Builds hidden layers for the time series features.

    Inputs:
      self.time_series_features

    Outputs:
      self.time_series_hidden_layers
    """
    time_series_hidden_layers = {}
    for name, time_series in self.time_series_features.items():
      time_series_hidden_layers[name] = self._build_local_fc_layers(
          inputs=time_series,
          hparams=self.hparams.time_series_hidden[name],
          scope=name + "_hidden")

    self.time_series_hidden_layers = time_series_hidden_layers
