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

"""Dbof model definitions."""

import functools
from typing import Optional

import tensorflow as tf

from official.modeling import hyperparams
from official.modeling import tf_utils
from official.projects.yt8m.configs import yt8m as yt8m_cfg
from official.projects.yt8m.modeling import yt8m_model_utils as utils
from official.vision.configs import common
from official.vision.modeling.backbones import factory


layers = tf.keras.layers


class Dbof(tf.keras.Model):
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
      l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs,
  ):
    """YT8M initialization function.

    Args:
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
        [batch_size x num_frames x num_features].
      params: model configuration parameters.
      norm_activation: Model normalization and activation configs.
      l2_regularizer: An optional kernel weight regularizer.
      **kwargs: keyword arguments to be passed.
    """
    self._self_setattr_tracking = False
    self._input_specs = input_specs
    self._params = params
    self._norm_activation = norm_activation
    self._act_fn = tf_utils.get_activation(self._norm_activation.activation)
    self._norm = functools.partial(
        layers.BatchNormalization,
        momentum=self._norm_activation.norm_momentum,
        epsilon=self._norm_activation.norm_epsilon,
        synchronized=self._norm_activation.use_sync_bn,
    )

    # [batch_size x num_frames x num_features]
    feature_size = input_specs.shape[-1]
    # shape 'excluding' batch_size
    model_input = tf.keras.Input(shape=self._input_specs.shape[1:])
    # normalize input features
    input_data = tf.nn.l2_normalize(model_input, -1)
    tf.summary.histogram("input_hist", input_data)

    # configure model
    if params.add_batch_norm:
      input_data = self._norm(name="input_bn")(input_data)

    # activation = reshaped input * cluster weights
    if params.cluster_size > 0:
      activation = layers.Dense(
          params.cluster_size,
          kernel_regularizer=l2_regularizer,
          kernel_initializer=tf.random_normal_initializer(
              stddev=1 / tf.sqrt(tf.cast(feature_size, tf.float32))
          ),
      )(input_data)
    else:
      activation = input_data

    if params.add_batch_norm:
      activation = self._norm(name="cluster_bn")(activation)
    else:
      cluster_biases = tf.Variable(
          tf.random_normal_initializer(stddev=1 / tf.math.sqrt(feature_size))(
              shape=[params.cluster_size]),
          name="cluster_biases")
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases

    activation = self._act_fn(activation)
    tf.summary.histogram("cluster_output", activation)

    if params.use_context_gate_cluster_layer:
      pooling_method = None
      norm_args = dict(name="context_gate_bn")
      activation = utils.context_gate(
          activation,
          normalizer_fn=self._norm,
          normalizer_params=norm_args,
          pooling_method=pooling_method,
          hidden_layer_size=params.context_gate_cluster_bottleneck_size,
          kernel_regularizer=l2_regularizer)

    activation = utils.frame_pooling(activation, params.pooling_method)

    # activation = activation * hidden1_weights
    activation = layers.Dense(
        params.hidden_size,
        kernel_regularizer=l2_regularizer,
        kernel_initializer=tf.random_normal_initializer(
            stddev=1 / tf.sqrt(tf.cast(params.cluster_size, tf.float32))))(
                activation)

    if params.add_batch_norm:
      activation = self._norm(name="hidden1_bn")(activation)

    else:
      hidden1_biases = tf.Variable(
          tf.random_normal_initializer(stddev=0.01)(shape=[params.hidden_size]),
          name="hidden1_biases")

      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases

    activation = self._act_fn(activation)
    tf.summary.histogram("hidden1_output", activation)

    super().__init__(inputs=model_input, outputs=activation, **kwargs)


@factory.register_backbone_builder("dbof")
def build_dbof(
    input_specs: tf.keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
    **kwargs,
) -> tf.keras.Model:
  """Builds a dbof backbone from a config."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == "dbof", f"Inconsistent backbone type {backbone_type}"

  return Dbof(
      input_specs=input_specs,
      params=backbone_cfg,
      norm_activation=norm_activation_config,
      l2_regularizer=l2_regularizer,
      **kwargs,
  )
