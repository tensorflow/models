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

"""Dbof model definitions."""

import functools
from typing import Any, Optional

import tensorflow as tf, tf_keras

from official.modeling import hyperparams
from official.modeling import tf_utils
from official.projects.yt8m.configs import yt8m as yt8m_cfg
from official.projects.yt8m.modeling import nn_layers
from official.projects.yt8m.modeling import yt8m_model_utils
from official.vision.configs import common
from official.vision.modeling.backbones import factory


layers = tf_keras.layers


class Dbof(layers.Layer):
  """A YT8M model class builder.

  Creates a Deep Bag of Frames model.
  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.
  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.
  """

  def __init__(
      self,
      input_specs: layers.InputSpec = layers.InputSpec(
          shape=[None, None, 1152]
      ),
      params: yt8m_cfg.DbofModel = yt8m_cfg.DbofModel(),
      norm_activation: common.NormActivation = common.NormActivation(),
      l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs,
  ):
    """YT8M initialization function.

    Args:
      input_specs: `tf_keras.layers.InputSpec` specs of the input tensor.
        [batch_size x num_frames x num_features].
      params: model configuration parameters.
      norm_activation: Model normalization and activation configs.
      l2_regularizer: An optional kernel weight regularizer.
      **kwargs: keyword arguments to be passed.
    """
    super().__init__(**kwargs)
    self._input_specs = input_specs
    self._params = params
    self._norm_activation = norm_activation
    self._l2_regularizer = l2_regularizer
    self._act_fn = tf_utils.get_activation(self._norm_activation.activation)
    self._norm = functools.partial(
        layers.BatchNormalization,
        momentum=self._norm_activation.norm_momentum,
        epsilon=self._norm_activation.norm_epsilon,
        synchronized=self._norm_activation.use_sync_bn,
    )
    feature_size = input_specs.shape[-1]

    # Configure model batch norm layer.
    if self._params.add_batch_norm:
      self._input_bn = self._norm(name="input_bn")
      self._cluster_bn = self._norm(name="cluster_bn")
      self._hidden_bn = self._norm(name="hidden_bn")
    else:
      self._hidden_biases = self.add_weight(
          name="hidden_biases",
          shape=[self._params.hidden_size],
          initializer=tf.random_normal_initializer(stddev=0.01),
      )
      self._cluster_biases = self.add_weight(
          name="cluster_biases",
          shape=[self._params.cluster_size],
          initializer=tf.random_normal_initializer(
              stddev=1.0 / tf.math.sqrt(feature_size)
          ),
      )

    if self._params.use_context_gate_cluster_layer:
      self._context_gate = nn_layers.ContextGate(
          normalizer_fn=self._norm,
          pooling_method=None,
          hidden_layer_size=self._params.context_gate_cluster_bottleneck_size,
          kernel_regularizer=self._l2_regularizer,
          name="context_gate_cluster",
      )

    self._hidden_dense = layers.Dense(
        self._params.hidden_size,
        kernel_regularizer=self._l2_regularizer,
        kernel_initializer=tf.random_normal_initializer(
            stddev=1.0 / tf.sqrt(tf.cast(self._params.cluster_size, tf.float32))
        ),
        name="hidden_dense",
    )

    if self._params.cluster_size > 0:
      self._cluster_dense = layers.Dense(
          self._params.cluster_size,
          kernel_regularizer=self._l2_regularizer,
          kernel_initializer=tf.random_normal_initializer(
              stddev=1.0 / tf.sqrt(tf.cast(feature_size, tf.float32))
          ),
          name="cluster_dense",
      )

  def call(
      self, inputs: tf.Tensor, num_frames: Any = None,
  ) -> tf.Tensor:
    # L2 normalize input features
    activation = tf.nn.l2_normalize(inputs, -1)

    if self._params.add_batch_norm:
      activation = self._input_bn(activation)

    if self._params.cluster_size > 0:
      activation = self._cluster_dense(activation)
      if self._params.add_batch_norm:
        activation = self._cluster_bn(activation)
    if not self._params.add_batch_norm:
      activation += self._cluster_biases

    activation = self._act_fn(activation)

    if self._params.use_context_gate_cluster_layer:
      activation = self._context_gate(activation)

    activation = yt8m_model_utils.frame_pooling(
        activation,
        method=self._params.pooling_method,
        num_frames=num_frames,
    )

    activation = self._hidden_dense(activation)
    if self._params.add_batch_norm:
      activation = self._hidden_bn(activation)
    else:
      activation += self._hidden_biases

    activation = self._act_fn(activation)
    return activation


@factory.register_backbone_builder("dbof")
def build_dbof(
    input_specs: tf_keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
    **kwargs,
) -> tf_keras.Model:
  """Builds a dbof backbone from a config."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == "dbof", f"Inconsistent backbone type {backbone_type}"

  dbof = Dbof(
      input_specs=input_specs,
      params=backbone_cfg,
      norm_activation=norm_activation_config,
      l2_regularizer=l2_regularizer,
      **kwargs,
  )

  # Warmup calls to build model variables.
  dbof(tf_keras.Input(input_specs.shape[1:]))
  return dbof
