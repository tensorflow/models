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

"""Contains the Tensorflow 2 version definition of S3D model.

S3D model is described in the following paper:
https://arxiv.org/abs/1712.04851.
"""
from typing import Any, Dict, Mapping, Optional, Sequence, Text, Tuple, Union

import tensorflow as tf, tf_keras

from official.modeling import hyperparams
from official.projects.s3d.configs import s3d as cfg
from official.projects.s3d.modeling import inception_utils
from official.projects.s3d.modeling import net_utils
from official.vision.modeling import factory_3d as model_factory
from official.vision.modeling.backbones import factory as backbone_factory

initializers = tf_keras.initializers
regularizers = tf_keras.regularizers


class S3D(tf_keras.Model):
  """Class to build S3D family model."""

  def __init__(self,
               input_specs: tf_keras.layers.InputSpec,
               final_endpoint: Text = 'Mixed_5c',
               first_temporal_kernel_size: int = 3,
               temporal_conv_start_at: Text = 'Conv2d_2c_3x3',
               gating_start_at: Text = 'Conv2d_2c_3x3',
               swap_pool_and_1x1x1: bool = True,
               gating_style: Text = 'CELL',
               use_sync_bn: bool = False,
               norm_momentum: float = 0.999,
               norm_epsilon: float = 0.001,
               temporal_conv_initializer: Union[
                   Text,
                   initializers.Initializer] = initializers.TruncatedNormal(
                       mean=0.0, stddev=0.01),
               temporal_conv_type: Text = '2+1d',
               kernel_initializer: Union[
                   Text,
                   initializers.Initializer] = initializers.TruncatedNormal(
                       mean=0.0, stddev=0.01),
               kernel_regularizer: Union[Text, regularizers.Regularizer] = 'l2',
               depth_multiplier: float = 1.0,
               **kwargs):
    """Constructor.

    Args:
      input_specs: `tf_keras.layers.InputSpec` specs of the input tensor.
      final_endpoint: Specifies the endpoint to construct the network up to.
      first_temporal_kernel_size: Temporal kernel size of the first convolution
        layer.
      temporal_conv_start_at: Specifies the endpoint where to start performimg
        temporal convolution from.
      gating_start_at: Specifies the endpoint where to start performimg self
        gating from.
      swap_pool_and_1x1x1: A boolean flag indicates that whether to swap the
        order of convolution and max pooling in Branch_3 of inception v1 cell.
      gating_style: A string that specifies self gating to be applied after each
        branch and/or after each cell. It can be one of ['BRANCH', 'CELL',
        'BRANCH_AND_CELL'].
      use_sync_bn: If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      temporal_conv_initializer: Weight initializer for temporal convolutional
        layers.
      temporal_conv_type: The type of parameterized convolution. Currently, we
        support '2d', '3d', '2+1d', '1+2d'.
      kernel_initializer: Weight initializer for convolutional layers other than
        temporal convolution.
      kernel_regularizer: Weight regularizer for all convolutional layers.
      depth_multiplier: A float to reduce/increase number of channels.
      **kwargs: keyword arguments to be passed.
    """

    self._input_specs = input_specs
    self._final_endpoint = final_endpoint
    self._first_temporal_kernel_size = first_temporal_kernel_size
    self._temporal_conv_start_at = temporal_conv_start_at
    self._gating_start_at = gating_start_at
    self._swap_pool_and_1x1x1 = swap_pool_and_1x1x1
    self._gating_style = gating_style
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._temporal_conv_initializer = temporal_conv_initializer
    self._temporal_conv_type = temporal_conv_type
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._depth_multiplier = depth_multiplier

    self._temporal_conv_endpoints = net_utils.make_set_from_start_endpoint(
        temporal_conv_start_at, inception_utils.INCEPTION_V1_CONV_ENDPOINTS)
    self._self_gating_endpoints = net_utils.make_set_from_start_endpoint(
        gating_start_at, inception_utils.INCEPTION_V1_CONV_ENDPOINTS)

    inputs = tf_keras.Input(shape=input_specs.shape[1:])
    net, end_points = inception_utils.inception_v1_stem_cells(
        inputs,
        depth_multiplier,
        final_endpoint,
        temporal_conv_endpoints=self._temporal_conv_endpoints,
        self_gating_endpoints=self._self_gating_endpoints,
        temporal_conv_type=self._temporal_conv_type,
        first_temporal_kernel_size=self._first_temporal_kernel_size,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon,
        temporal_conv_initializer=self._temporal_conv_initializer,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        parameterized_conv_layer=self._get_parameterized_conv_layer_impl(),
        layer_naming_fn=self._get_layer_naming_fn(),
    )

    for end_point, filters in inception_utils.INCEPTION_V1_ARCH_SKELETON:
      net, end_points = self._s3d_cell(net, end_point, end_points, filters)
      if end_point == final_endpoint:
        break

    if final_endpoint not in end_points:
      raise ValueError(
          'Unrecognized final endpoint %s (available endpoints: %s).' %
          (final_endpoint, end_points.keys()))

    super(S3D, self).__init__(inputs=inputs, outputs=end_points, **kwargs)

  def _s3d_cell(
      self,
      net: tf.Tensor,
      end_point: Text,
      end_points: Dict[Text, tf.Tensor],
      filters: Union[int, Sequence[Any]],
      non_local_block: Optional[tf_keras.layers.Layer] = None,
      attention_cell: Optional[tf_keras.layers.Layer] = None,
      attention_cell_super_graph: Optional[tf_keras.layers.Layer] = None
  ) -> Tuple[tf.Tensor, Dict[Text, tf.Tensor]]:
    if end_point.startswith('Mixed'):
      conv_type = (
          self._temporal_conv_type
          if end_point in self._temporal_conv_endpoints else '2d')
      use_self_gating_on_branch = (
          end_point in self._self_gating_endpoints and
          (self._gating_style == 'BRANCH' or
           self._gating_style == 'BRANCH_AND_CELL'))
      use_self_gating_on_cell = (
          end_point in self._self_gating_endpoints and
          (self._gating_style == 'CELL' or
           self._gating_style == 'BRANCH_AND_CELL'))
      net = self._get_inception_v1_cell_layer_impl()(
          branch_filters=net_utils.apply_depth_multiplier(
              filters, self._depth_multiplier),
          conv_type=conv_type,
          temporal_dilation_rate=1,
          swap_pool_and_1x1x1=self._swap_pool_and_1x1x1,
          use_self_gating_on_branch=use_self_gating_on_branch,
          use_self_gating_on_cell=use_self_gating_on_cell,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon,
          kernel_initializer=self._kernel_initializer,
          temporal_conv_initializer=self._temporal_conv_initializer,
          kernel_regularizer=self._kernel_regularizer,
          parameterized_conv_layer=self._get_parameterized_conv_layer_impl(),
          name=self._get_layer_naming_fn()(end_point))(
              net)
    else:
      net = tf_keras.layers.MaxPool3D(
          pool_size=filters[0],
          strides=filters[1],
          padding='same',
          name=self._get_layer_naming_fn()(end_point))(
              net)
    end_points[end_point] = net
    if non_local_block:
      # TODO(b/182299420): Implement non local block in TF2.
      raise NotImplementedError('Non local block is not implemented yet.')
    if attention_cell:
      # TODO(b/182299420): Implement attention cell in TF2.
      raise NotImplementedError('Attention cell is not implemented yet.')
    if attention_cell_super_graph:
      # TODO(b/182299420): Implement attention cell super graph in TF2.
      raise NotImplementedError('Attention cell super graph is not implemented'
                                ' yet.')
    return net, end_points

  def get_config(self):
    config_dict = {
        'input_specs': self._input_specs,
        'final_endpoint': self._final_endpoint,
        'first_temporal_kernel_size': self._first_temporal_kernel_size,
        'temporal_conv_start_at': self._temporal_conv_start_at,
        'gating_start_at': self._gating_start_at,
        'swap_pool_and_1x1x1': self._swap_pool_and_1x1x1,
        'gating_style': self._gating_style,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'temporal_conv_initializer': self._temporal_conv_initializer,
        'temporal_conv_type': self._temporal_conv_type,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'depth_multiplier': self._depth_multiplier
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs

  def _get_inception_v1_cell_layer_impl(self):
    return inception_utils.InceptionV1CellLayer

  def _get_parameterized_conv_layer_impl(self):
    return net_utils.ParameterizedConvLayer

  def _get_layer_naming_fn(self):
    return lambda end_point: None


class S3DModel(tf_keras.Model):
  """An S3D model builder."""

  def __init__(self,
               backbone: tf_keras.Model,
               num_classes: int,
               input_specs: Mapping[Text, tf_keras.layers.InputSpec],
               final_endpoint: Text = 'Mixed_5c',
               dropout_rate: float = 0.0,
               **kwargs):
    """Constructor.

    Args:
      backbone: S3D backbone Keras Model.
      num_classes: `int` number of possible classes for video classification.
      input_specs: input_specs: `tf_keras.layers.InputSpec` specs of the input
        tensor.
      final_endpoint: Specifies the endpoint to construct the network up to.
      dropout_rate: `float` between 0 and 1. Fraction of the input units to
        drop. Note that dropout_rate = 1.0 - dropout_keep_prob.
      **kwargs: keyword arguments to be passed.
    """
    self._self_setattr_tracking = False
    self._backbone = backbone
    self._num_classes = num_classes
    self._input_specs = input_specs
    self._final_endpoint = final_endpoint
    self._dropout_rate = dropout_rate
    self._config_dict = {
        'backbone': backbone,
        'num_classes': num_classes,
        'input_specs': input_specs,
        'final_endpoint': final_endpoint,
        'dropout_rate': dropout_rate,
    }

    inputs = {
        k: tf_keras.Input(shape=v.shape[1:]) for k, v in input_specs.items()
    }
    streams = self._backbone(inputs['image'])

    pool = tf.math.reduce_mean(streams[self._final_endpoint], axis=[1, 2, 3])
    fc = tf_keras.layers.Dropout(dropout_rate)(pool)
    logits = tf_keras.layers.Dense(**self._build_dense_layer_params())(fc)

    super(S3DModel, self).__init__(inputs=inputs, outputs=logits, **kwargs)

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

  def _build_dense_layer_params(self):
    return dict(units=self._num_classes, kernel_regularizer='l2')


@backbone_factory.register_backbone_builder('s3d')
def build_s3d(
    input_specs: tf_keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: tf_keras.regularizers.Regularizer = None
) -> tf_keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds S3D backbone."""

  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 's3d'
  del norm_activation_config

  backbone = S3D(
      input_specs=input_specs,
      final_endpoint=backbone_cfg.final_endpoint,
      first_temporal_kernel_size=backbone_cfg.first_temporal_kernel_size,
      temporal_conv_start_at=backbone_cfg.temporal_conv_start_at,
      gating_start_at=backbone_cfg.gating_start_at,
      swap_pool_and_1x1x1=backbone_cfg.swap_pool_and_1x1x1,
      gating_style=backbone_cfg.gating_style,
      use_sync_bn=backbone_cfg.use_sync_bn,
      norm_momentum=backbone_cfg.norm_momentum,
      norm_epsilon=backbone_cfg.norm_epsilon,
      temporal_conv_type=backbone_cfg.temporal_conv_type,
      kernel_regularizer=l2_regularizer,
      depth_multiplier=backbone_cfg.depth_multiplier)
  return backbone


@model_factory.register_model_builder('s3d')
def build_s3d_model(
    input_specs: tf_keras.layers.InputSpec,
    model_config: cfg.S3DModel,
    num_classes: int,
    l2_regularizer: tf_keras.regularizers.Regularizer = None
) -> tf_keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds S3D model with classification layer."""
  input_specs_dict = {'image': input_specs}
  backbone = build_s3d(input_specs, model_config.backbone,
                       model_config.norm_activation, l2_regularizer)

  model = S3DModel(
      backbone,
      num_classes=num_classes,
      input_specs=input_specs_dict,
      dropout_rate=model_config.dropout_rate)
  return model
