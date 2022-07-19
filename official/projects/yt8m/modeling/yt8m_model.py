# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""YT8M model definition."""
from typing import Optional

import tensorflow as tf

from official.modeling import tf_utils
from official.projects.yt8m.configs import yt8m as yt8m_cfg
from official.projects.yt8m.modeling import nn_layers
from official.projects.yt8m.modeling import yt8m_model_utils as utils

layers = tf.keras.layers


class DbofModel(tf.keras.Model):
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
      params: yt8m_cfg.DbofModel,
      num_frames: int = 30,
      num_classes: int = 3862,
      input_specs: layers.InputSpec = layers.InputSpec(
          shape=[None, None, 1152]),
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      activation: str = "relu",
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      **kwargs):
    """YT8M initialization function.

    Args:
      params: model configuration parameters
      num_frames: `int` number of frames in a single input.
      num_classes: `int` number of classes in dataset.
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
        [batch_size x num_frames x num_features]
      kernel_regularizer: tf.keras.regularizers.Regularizer object. Default to
        None.
      activation: A `str` of name of the activation function.
      use_sync_bn: If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      **kwargs: keyword arguments to be passed.
    """

    self._self_setattr_tracking = False
    self._config_dict = {
        "input_specs": input_specs,
        "num_classes": num_classes,
        "num_frames": num_frames,
        "params": params
    }
    self._num_classes = num_classes
    self._input_specs = input_specs
    self._act_fn = tf_utils.get_activation(activation)
    if use_sync_bn:
      self._norm = layers.experimental.SyncBatchNormalization
    else:
      self._norm = layers.BatchNormalization
    if tf.keras.backend.image_data_format() == "channels_last":
      bn_axis = -1
    else:
      bn_axis = 1

    # [batch_size x num_frames x num_features]
    feature_size = input_specs.shape[-1]
    # shape 'excluding' batch_size
    model_input = tf.keras.Input(shape=self._input_specs.shape[1:])
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", model_input)

    # configure model
    if params.add_batch_norm:
      reshaped_input = self._norm(
          axis=bn_axis,
          momentum=norm_momentum,
          epsilon=norm_epsilon,
          name="input_bn")(
              reshaped_input)

    # activation = reshaped input * cluster weights
    if params.cluster_size > 0:
      activation = layers.Dense(
          params.cluster_size,
          kernel_regularizer=kernel_regularizer,
          kernel_initializer=tf.random_normal_initializer(
              stddev=1 / tf.sqrt(tf.cast(feature_size, tf.float32))))(
                  reshaped_input)

    if params.add_batch_norm:
      activation = self._norm(
          axis=bn_axis,
          momentum=norm_momentum,
          epsilon=norm_epsilon,
          name="cluster_bn")(
              activation)
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
      norm_args = dict(
          axis=bn_axis,
          momentum=norm_momentum,
          epsilon=norm_epsilon,
          name="context_gate_bn")
      activation = utils.context_gate(
          activation,
          normalizer_fn=self._norm,
          normalizer_params=norm_args,
          pooling_method=pooling_method,
          hidden_layer_size=params.context_gate_cluster_bottleneck_size,
          kernel_regularizer=kernel_regularizer)
    activation = tf.reshape(activation, [-1, num_frames, params.cluster_size])
    activation = utils.frame_pooling(activation, params.pooling_method)

    # activation = activation * hidden1_weights
    activation = layers.Dense(
        params.hidden_size,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=tf.random_normal_initializer(
            stddev=1 / tf.sqrt(tf.cast(params.cluster_size, tf.float32))))(
                activation)

    if params.add_batch_norm:
      activation = self._norm(
          axis=bn_axis,
          momentum=norm_momentum,
          epsilon=norm_epsilon,
          name="hidden1_bn")(
              activation)

    else:
      hidden1_biases = tf.Variable(
          tf.random_normal_initializer(stddev=0.01)(shape=[params.hidden_size]),
          name="hidden1_biases")

      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases

    activation = self._act_fn(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(nn_layers,
                               params.yt8m_agg_classifier_model)
    norm_args = dict(axis=bn_axis, momentum=norm_momentum, epsilon=norm_epsilon)
    output = aggregated_model().create_model(
        model_input=activation,
        vocab_size=self._num_classes,
        num_mixtures=params.agg_model.num_mixtures,
        normalizer_fn=self._norm,
        normalizer_params=norm_args,
        l2_penalty=params.agg_model.l2_penalty)

    super().__init__(
        inputs=model_input, outputs=output.get("predictions"), **kwargs)

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    return dict()

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
