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

"""Build classification models."""

from typing import Any, Mapping, Optional
# Import libraries
import tensorflow as tf, tf_keras

layers = tf_keras.layers


@tf_keras.utils.register_keras_serializable(package='Vision')
class ClassificationModel(tf_keras.Model):
  """A classification class builder."""

  def __init__(
      self,
      backbone: tf_keras.Model,
      num_classes: int,
      input_specs: tf_keras.layers.InputSpec = layers.InputSpec(
          shape=[None, None, None, 3]),
      dropout_rate: float = 0.0,
      kernel_initializer: str = 'random_uniform',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      add_head_batch_norm: bool = False,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      skip_logits_layer: bool = False,
      **kwargs):
    """Classification initialization function.

    Args:
      backbone: a backbone network.
      num_classes: `int` number of classes in classification task.
      input_specs: `tf_keras.layers.InputSpec` specs of the input tensor.
      dropout_rate: `float` rate for dropout regularization.
      kernel_initializer: kernel initializer for the dense layer.
      kernel_regularizer: tf_keras.regularizers.Regularizer object. Default to
                          None.
      bias_regularizer: tf_keras.regularizers.Regularizer object. Default to
                          None.
      add_head_batch_norm: `bool` whether to add a batch normalization layer
        before pool.
      use_sync_bn: `bool` if True, use synchronized batch normalization.
      norm_momentum: `float` normalization momentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      skip_logits_layer: `bool`, whether to skip the prediction layer.
      **kwargs: keyword arguments to be passed.
    """
    norm = tf_keras.layers.BatchNormalization
    axis = -1 if tf_keras.backend.image_data_format() == 'channels_last' else 1

    inputs = tf_keras.Input(shape=input_specs.shape[1:], name=input_specs.name)
    endpoints = backbone(inputs)
    x = endpoints[max(endpoints.keys())]

    if add_head_batch_norm:
      x = norm(
          axis=axis,
          momentum=norm_momentum,
          epsilon=norm_epsilon,
          synchronized=use_sync_bn,
      )(x)

    # Depending on the backbone type, backbone's output can be
    # [batch_size, height, weight, channel_size] or
    # [batch_size, token_size, hidden_size].
    if len(x.shape) == 4:
      x = tf_keras.layers.GlobalAveragePooling2D()(x)
    elif len(x.shape) == 3:
      x = tf_keras.layers.GlobalAveragePooling1D()(x)

    if not skip_logits_layer:
      x = tf_keras.layers.Dropout(dropout_rate)(x)
      x = tf_keras.layers.Dense(
          num_classes,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)(
              x)

    super(ClassificationModel, self).__init__(
        inputs=inputs, outputs=x, **kwargs)
    self._config_dict = {
        'backbone': backbone,
        'num_classes': num_classes,
        'input_specs': input_specs,
        'dropout_rate': dropout_rate,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        'add_head_batch_norm': add_head_batch_norm,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
    }
    self._input_specs = input_specs
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._backbone = backbone
    self._norm = norm

  @property
  def checkpoint_items(self) -> Mapping[str, tf_keras.Model]:
    """Returns a dictionary of items to be additionally checkpointed."""
    return dict(backbone=self.backbone)

  @property
  def backbone(self) -> tf_keras.Model:
    return self._backbone

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
