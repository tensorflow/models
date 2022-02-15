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

"""Build video classification models."""
from typing import Mapping, Optional

# Import libraries

import tensorflow as tf

from official.modeling import tf_utils
from official.vision.beta.modeling import backbones
from official.vision.beta.modeling import factory_3d as model_factory
from official.vision.projects.video_ssl.configs import video_ssl as video_ssl_cfg

layers = tf.keras.layers


class VideoSSLModel(tf.keras.Model):
  """A video ssl model class builder."""

  def __init__(self,
               backbone,
               normalize_feature,
               hidden_dim,
               hidden_layer_num,
               hidden_norm_args,
               projection_dim,
               input_specs: Optional[Mapping[str,
                                             tf.keras.layers.InputSpec]] = None,
               dropout_rate: float = 0.0,
               aggregate_endpoints: bool = False,
               kernel_initializer='random_uniform',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """Video Classification initialization function.

    Args:
      backbone: a 3d backbone network.
      normalize_feature: whether normalize backbone feature.
      hidden_dim: `int` number of hidden units in MLP.
      hidden_layer_num: `int` number of hidden layers in MLP.
      hidden_norm_args: `dict` for batchnorm arguments in MLP.
      projection_dim: `int` number of ouput dimension for MLP.
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
      dropout_rate: `float` rate for dropout regularization.
      aggregate_endpoints: `bool` aggregate all end ponits or only use the
        final end point.
      kernel_initializer: kernel initializer for the dense layer.
      kernel_regularizer: tf.keras.regularizers.Regularizer object. Default to
        None.
      bias_regularizer: tf.keras.regularizers.Regularizer object. Default to
        None.
      **kwargs: keyword arguments to be passed.
    """
    if not input_specs:
      input_specs = {
          'image': layers.InputSpec(shape=[None, None, None, None, 3])
      }
    self._self_setattr_tracking = False
    self._config_dict = {
        'backbone': backbone,
        'normalize_feature': normalize_feature,
        'hidden_dim': hidden_dim,
        'hidden_layer_num': hidden_layer_num,
        'use_sync_bn': hidden_norm_args.use_sync_bn,
        'norm_momentum': hidden_norm_args.norm_momentum,
        'norm_epsilon': hidden_norm_args.norm_epsilon,
        'activation': hidden_norm_args.activation,
        'projection_dim': projection_dim,
        'input_specs': input_specs,
        'dropout_rate': dropout_rate,
        'aggregate_endpoints': aggregate_endpoints,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
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
      x = endpoints[max(endpoints.keys())]
      x = tf.keras.layers.GlobalAveragePooling3D()(x)

    # L2 Normalize feature after backbone
    if normalize_feature:
      x = tf.nn.l2_normalize(x, axis=-1)

    # MLP hidden layers
    for _ in range(hidden_layer_num):
      x = tf.keras.layers.Dense(hidden_dim)(x)
      if self._config_dict['use_sync_bn']:
        x = tf.keras.layers.experimental.SyncBatchNormalization(
            momentum=self._config_dict['norm_momentum'],
            epsilon=self._config_dict['norm_epsilon'])(x)
      else:
        x = tf.keras.layers.BatchNormalization(
            momentum=self._config_dict['norm_momentum'],
            epsilon=self._config_dict['norm_epsilon'])(x)
      x = tf_utils.get_activation(self._config_dict['activation'])(x)

    # Projection head
    x = tf.keras.layers.Dense(projection_dim)(x)

    super(VideoSSLModel, self).__init__(
        inputs=inputs, outputs=x, **kwargs)

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    return dict(backbone=self.backbone)

  @property
  def backbone(self):
    return self._backbone

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


@model_factory.register_model_builder('video_ssl_model')
def build_video_ssl_pretrain_model(
    input_specs: tf.keras.layers.InputSpec,
    model_config: video_ssl_cfg.VideoSSLModel,
    num_classes: int,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None):
  """Builds the video classification model."""
  del num_classes
  input_specs_dict = {'image': input_specs}
  backbone = backbones.factory.build_backbone(
      input_specs=input_specs,
      backbone_config=model_config.backbone,
      norm_activation_config=model_config.norm_activation,
      l2_regularizer=l2_regularizer)

  # Norm layer type in the MLP head should same with backbone
  assert model_config.norm_activation.use_sync_bn == model_config.hidden_norm_activation.use_sync_bn

  model = VideoSSLModel(
      backbone=backbone,
      normalize_feature=model_config.normalize_feature,
      hidden_dim=model_config.hidden_dim,
      hidden_layer_num=model_config.hidden_layer_num,
      hidden_norm_args=model_config.hidden_norm_activation,
      projection_dim=model_config.projection_dim,
      input_specs=input_specs_dict,
      dropout_rate=model_config.dropout_rate,
      aggregate_endpoints=model_config.aggregate_endpoints,
      kernel_regularizer=l2_regularizer)
  return model
