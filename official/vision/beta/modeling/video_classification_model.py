# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Build video classification models."""
from typing import Any, Mapping, Optional, Union, List, Text

import tensorflow as tf

layers = tf.keras.layers


@tf.keras.utils.register_keras_serializable(package='Vision')
class VideoClassificationModel(tf.keras.Model):
  """A video classification class builder."""

  def __init__(
      self,
      backbone: tf.keras.Model,
      num_classes: int,
      input_specs: Optional[Mapping[str, tf.keras.layers.InputSpec]] = None,
      dropout_rate: float = 0.0,
      aggregate_endpoints: bool = False,
      kernel_initializer: str = 'random_uniform',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      require_endpoints: Optional[List[Text]] = None,
      **kwargs):
    """Video Classification initialization function.

    Args:
      backbone: a 3d backbone network.
      num_classes: `int` number of classes in classification task.
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
      dropout_rate: `float` rate for dropout regularization.
      aggregate_endpoints: `bool` aggregate all end ponits or only use the
        final end point.
      kernel_initializer: kernel initializer for the dense layer.
      kernel_regularizer: tf.keras.regularizers.Regularizer object. Default to
        None.
      bias_regularizer: tf.keras.regularizers.Regularizer object. Default to
        None.
      require_endpoints: the required endpoints for prediction. If None or
        empty, then only uses the final endpoint.
      **kwargs: keyword arguments to be passed.
    """
    if not input_specs:
      input_specs = {
          'image': layers.InputSpec(shape=[None, None, None, None, 3])
      }
    self._self_setattr_tracking = False
    self._config_dict = {
        'backbone': backbone,
        'num_classes': num_classes,
        'input_specs': input_specs,
        'dropout_rate': dropout_rate,
        'aggregate_endpoints': aggregate_endpoints,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        'require_endpoints': require_endpoints,
    }
    self._input_specs = input_specs
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._backbone = backbone

    inputs = {
        k: tf.keras.Input(shape=v.shape[1:]) for k, v in input_specs.items()
    }
    endpoints = backbone(inputs['image'])

    if aggregate_endpoints:
      pooled_feats = []
      for endpoint in endpoints.values():
        x_pool = tf.keras.layers.GlobalAveragePooling3D()(endpoint)
        pooled_feats.append(x_pool)
      x = tf.concat(pooled_feats, axis=1)
    else:
      if not require_endpoints:
        # Uses the last endpoint for prediction.
        x = endpoints[max(endpoints.keys())]
        x = tf.keras.layers.GlobalAveragePooling3D()(x)
      else:
        # Concats all the required endpoints for prediction.
        outputs = []
        for name in require_endpoints:
          x = endpoints[name]
          x = tf.keras.layers.GlobalAveragePooling3D()(x)
          outputs.append(x)
        x = tf.concat(outputs, axis=1)

    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(
        num_classes, kernel_initializer=kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(
            x)

    super(VideoClassificationModel, self).__init__(
        inputs=inputs, outputs=x, **kwargs)

  @property
  def checkpoint_items(
      self) -> Mapping[str, Union[tf.keras.Model, tf.keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    return dict(backbone=self.backbone)

  @property
  def backbone(self) -> tf.keras.Model:
    return self._backbone

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
