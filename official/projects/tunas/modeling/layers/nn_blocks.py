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

"""Compound layers, which are composition of common layers."""

import enum
from typing import Callable, Optional, Text, Tuple, Union

import pyglove as pg
from pyglove.tensorflow import keras
from pyglove.tensorflow import selections
from pyglove.tensorflow.keras import layers
from pyglove.tensorflow.keras.layers import modeling_utils
import tensorflow as tf


class OpOrder(enum.Enum):
  """Enum for operation order."""

  # Order in a sequence of operation, normalization and activation.
  OP_NORM_ACT = 0

  # Order in a sequence of operation, activation and normalization.
  OP_ACT_NORM = 1

  # Order in a sequence of activation, operation and normalization.
  ACT_OP_NORM = 2


def _op_sequence(op: tf.keras.layers.Layer,
                 norm: Optional[tf.keras.layers.Layer],
                 activation: Optional[tf.keras.layers.Layer],
                 op_order: OpOrder,
                 name: Optional[Text] = None):
  """Create a sequence of conv, norm, activation layers according the op_order.

  Args:
    op: A convolutional or linear layer.
    norm: An optional normalization layer.
    activation: An optional activation layer.
    op_order: A string of enum 'op-norm-activation', 'op-activation-norm' or
      'activation-op-norm'.
    name: Name of the graph block.

  Returns:
    `layer` if `norm` and `activation` are None, or a sequence of `layer`,
      `norm`, `activation` ordered according to the value of `op_order`.
  """
  if op_order == OpOrder.OP_NORM_ACT:
    net = [op, norm, activation]
  elif op_order == OpOrder.OP_ACT_NORM:
    net = [op, activation, norm]
  elif op_order == OpOrder.ACT_OP_NORM:
    net = [activation, op, norm]
  else:
    raise ValueError('Unsupported OpOrder %r' % op_order)

  net = [l for l in net if l is not None]
  if len(net) == 1:
    return net[0]
  return layers.Sequential(net, name=name)


Filters = Union[int, selections.IntSelection, Callable[[tf.Tensor], tf.Tensor]]
# The kernel selection should follow the following value specs
# - pg.typing.Int(min_value=1),
# - pg.typing.Tuple([pg.typing.Int(min_value=1),pg.typing.Int(min_value=1)])
KernelSize = Union[int, Tuple[int, int], selections.IntSelection,
                   selections.Selection]


@layers.compound
def conv2d(filters: Filters,
           kernel_size: KernelSize,
           strides: Union[int, Tuple[int, int]] = (1, 1),
           padding: Text = 'same',
           groups: Union[int, selections.IntSelection] = 1,
           kernel_initializer='glorot_uniform',
           kernel_regularizer=None,
           use_bias: bool = True,
           bias_initializer='zeros',
           bias_regularizer=None,
           normalization: Optional[tf.keras.layers.Layer] = None,
           activation: Optional[tf.keras.layers.Layer] = None,
           op_order: OpOrder = OpOrder.OP_NORM_ACT,
           data_format: Text = 'channels_last',
           name: Optional[Text] = None):
  """Create a Conv2D-Normalization-Activation layer."""
  if not selections.is_fixed(kernel_size):
    candidates = []
    for ks in selections.selection_candidates(kernel_size):
      candidates.append(
          conv2d(
              kernel_size=ks,
              filters=filters,
              strides=strides,
              groups=groups,
              kernel_initializer=kernel_initializer,
              kernel_regularizer=kernel_regularizer,
              use_bias=use_bias,
              bias_initializer=bias_initializer,
              bias_regularizer=bias_regularizer,
              normalization=normalization,
              activation=activation,
              data_format=data_format,
              op_order=op_order,
              name='branch_{:}'.format(len(candidates))))
    return layers.Switch(
        candidates=candidates,
        selected_index=selections.selection_index(kernel_size),
        name=name)
  if not selections.is_fixed(groups):
    candidates = []
    for gs in selections.selection_candidates(groups):
      candidates.append(
          conv2d(
              kernel_size=kernel_size,
              filters=filters,
              strides=strides,
              groups=gs,
              kernel_initializer=kernel_initializer,
              kernel_regularizer=kernel_regularizer,
              use_bias=use_bias,
              bias_initializer=bias_initializer,
              bias_regularizer=bias_regularizer,
              normalization=normalization,
              activation=activation,
              op_order=op_order,
              name='group_branch_{:}'.format(len(candidates))))
    return layers.Switch(
        candidates=candidates,
        selected_index=selections.selection_index(groups),
        name=name)
  conv = layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      groups=groups,
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer,
      use_bias=use_bias,
      bias_initializer=bias_initializer,
      bias_regularizer=bias_regularizer,
      data_format=data_format,
      name='conv2d')
  if normalization is not None:
    normalization = normalization.clone(override={'name': 'normalization'})
  return _op_sequence(conv, normalization, activation, op_order)


@layers.compound
def depthwise_conv2d(
    kernel_size: KernelSize,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: Text = 'same',
    depthwise_initializer='glorot_uniform',
    depthwise_regularizer=None,
    use_bias: bool = True,
    bias_initializer='zeros',
    bias_regularizer=None,
    normalization: Optional[tf.keras.layers.Layer] = None,
    activation: Optional[tf.keras.layers.Layer] = None,
    op_order: OpOrder = OpOrder.OP_NORM_ACT,
    data_format: Text = 'channels_last',
    name: Optional[Text] = None):
  """Creates a DepthwiseConv2D-Normalization-Activation layer."""
  if not selections.is_fixed(kernel_size):
    candidates = []
    for i, ks in enumerate(selections.selection_candidates(kernel_size)):
      candidates.append(
          depthwise_conv2d(
              kernel_size=ks,
              strides=strides,
              depthwise_initializer=depthwise_initializer,
              depthwise_regularizer=depthwise_regularizer,
              use_bias=use_bias,
              bias_initializer=bias_initializer,
              bias_regularizer=bias_regularizer,
              normalization=normalization,
              activation=activation,
              op_order=op_order,
              data_format=data_format,
              name='branch_%d' % i))
    return layers.Switch(
        candidates=candidates,
        selected_index=selections.selection_index(kernel_size),
        name=name)

  depthwise = layers.DepthwiseConv2D(
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      depthwise_initializer=depthwise_initializer,
      depthwise_regularizer=depthwise_regularizer,
      use_bias=use_bias,
      bias_initializer=bias_initializer,
      bias_regularizer=bias_regularizer,
      data_format=data_format,
      name='depthwise_conv2d')
  if normalization is not None:
    normalization = normalization.clone(override={'name': 'normalization'})
  return _op_sequence(depthwise, normalization, activation, op_order)


@pg.symbolize
def _expand_filters(
    input_filters_mask: tf.Tensor,
    is_input_filters_masked: bool,
    expansion_factor: float) -> Tuple[tf.Tensor, bool]:
  """Returns input filters mask multiplied by a factor."""
  assert input_filters_mask.shape.rank == 1, input_filters_mask
  output_filters_mask = tf.sequence_mask(
      tf.math.reduce_sum(
          tf.cast(input_filters_mask, tf.dtypes.int32)) * expansion_factor,
      input_filters_mask.shape[-1] * expansion_factor)
  return output_filters_mask, is_input_filters_masked


@layers.compound
def inverted_bottleneck(
    filters: Filters,
    kernel_size: KernelSize,
    expansion_factor: Union[int, selections.IntSelection] = 1,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    normalization: tf.keras.layers.Layer = layers.BatchNormalization(),
    activation: tf.keras.layers.Layer = layers.ReLU(),
    kernel_initializer='glorot_uniform',
    kernel_regularizer=None,
    depthwise_initializer='glorot_uniform',
    depthwise_regularizer=None,
    post_expansion: Optional[tf.keras.layers.Layer] = None,
    post_depthwise: Optional[tf.keras.layers.Layer] = None,
    post_projection: Optional[tf.keras.layers.Layer] = None,
    collapsed: bool = False,
    data_format: Text = 'channels_last',
    name: Optional[Text] = None):
  """Creates inverted bottleneck layer.

  Args:
    filters: output filters
    kernel_size: kernel size for the depthwise Conv2D.
    expansion_factor: The filters multiplier for the first Conv2D. If 1, the
      first Conv2D will be omitted.
    strides: Strides for the depthwise Conv2D.
    normalization: An optional normalization layer.
    activation: An optional activation layer.
    kernel_initializer: Kernel initializer used for Conv2D units.
    kernel_regularizer: Kernel regularizer for Conv2D units.
    depthwise_initializer: Kernel initializer used for depthwise Conv2D units.
    depthwise_regularizer: Kernel regularizer for depthwise Conv2D units.
    post_expansion: An optional layer that will be inserted after the first
      Conv2D.
    post_depthwise: An optional layer that will be inserted afther the depthwise
      Conv2D.
    post_projection: An optional layer that will be inserted after the last
      Conv2D.
    collapsed: If True, graph will collapse at convolutional units
      level on different kernel sizes.
    data_format: Data format used for Conv2D and depthwise Conv2D.
    name: Name of the layer.

  Returns:
    An inverted bottleneck layer as a compound layer.
  """
  if (not selections.is_fixed(expansion_factor)
      and 1 in selections.selection_candidates(expansion_factor)):
    raise ValueError(
        'Tunable `expansion_factor` with candidates 1 and values greater than 1'
        'is not supported: %r.' % expansion_factor)

  if not selections.is_fixed(kernel_size) and not collapsed:
    candidates = []
    for i, ks in enumerate(selections.selection_candidates(kernel_size)):
      candidates.append(inverted_bottleneck(
          kernel_size=ks,
          filters=filters,
          expansion_factor=expansion_factor,
          strides=strides,
          normalization=normalization,
          activation=activation,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          depthwise_initializer=depthwise_initializer,
          depthwise_regularizer=depthwise_regularizer,
          post_expansion=post_expansion,
          post_depthwise=post_depthwise,
          post_projection=post_projection,
          data_format=data_format,
          name='branch%d' % i))
    return layers.Switch(
        candidates=candidates,
        selected_index=selections.selection_index(kernel_size),
        name=name)

  if expansion_factor != 1:
    children = [
        conv2d(  # pylint: disable=unexpected-keyword-arg
            filters=_expand_filters(   # pylint: disable=no-value-for-parameter
                expansion_factor=expansion_factor),
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_bias=False,
            normalization=normalization,
            activation=activation,
            data_format=data_format,
            name='expansion')
    ]
  else:
    children = []
  if post_expansion:
    children.append(post_expansion)

  children.append(depthwise_conv2d(  # pylint: disable=unexpected-keyword-arg
      kernel_size=kernel_size,
      strides=strides,
      depthwise_initializer=depthwise_initializer,
      depthwise_regularizer=depthwise_regularizer,
      use_bias=False,
      normalization=normalization,
      activation=activation,
      data_format=data_format,
      name='depthwise'))
  if post_depthwise:
    children.append(post_depthwise)

  children.append(conv2d(   # pylint: disable=unexpected-keyword-arg
      filters=filters,
      kernel_size=(1, 1),
      strides=(1, 1),
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer,
      use_bias=False,
      normalization=normalization,
      activation=None,
      data_format=data_format,
      name='projection'))
  if post_projection:
    children.append(post_projection)
  return layers.Sequential(children)


def inverted_bottleneck_with_se(
    filters: Filters,
    kernel_size: KernelSize,
    expansion_factor: Union[int, selections.IntSelection] = 1,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    se_ratio: Optional[float] = None,
    filters_base: int = 8,
    normalization: tf.keras.layers.Layer = layers.BatchNormalization(),
    activation: tf.keras.layers.Layer = layers.ReLU(),
    kernel_initializer='glorot_uniform',
    kernel_regularizer=None,
    depthwise_initializer='glorot_uniform',
    depthwise_regularizer=None,
    name: Optional[Text] = None):
  """An inverted bottleneck layer with possibly squeeze excite."""
  post_depthwise = None
  if se_ratio:
    post_depthwise = SqueezeExcitation(
        ratio=se_ratio,
        filters_base=filters_base,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=name+'_se')
  return inverted_bottleneck(
      filters=filters,
      kernel_size=kernel_size,
      expansion_factor=expansion_factor,
      strides=strides,
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer,
      depthwise_initializer=depthwise_initializer,
      depthwise_regularizer=depthwise_regularizer,
      normalization=normalization,
      activation=activation,
      post_depthwise=post_depthwise,
      name=name)


@layers.compound
def fused_inverted_bottleneck(
    filters: Filters,
    kernel_size: KernelSize,
    expansion_factor: Union[int, selections.IntSelection] = 1,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    normalization: tf.keras.layers.Layer = layers.BatchNormalization(),
    activation: tf.keras.layers.Layer = layers.ReLU(),
    kernel_initializer='glorot_uniform',
    kernel_regularizer=None,
    post_fusion: Optional[tf.keras.layers.Layer] = None,
    post_projection: Optional[tf.keras.layers.Layer] = None,
    collapsed: bool = False,
    data_format: Text = 'channels_last',
    name: Optional[Text] = None):
  """Fused inverted bottleneck.

  Reference: https://arxiv.org/pdf/2003.02838.pdf

  Args:
    filters: output filters
    kernel_size: kernel size for the depthwise Conv2D.
    expansion_factor: The filters multiplier for the first Conv2D. If 1, the
      first Conv2D will be omitted.
    strides: Strides for the depthwise Conv2D.
    normalization: An optional normalization layer.
    activation: An optional activation layer.
    kernel_initializer: Kernel initializer used for Conv2D units.
    kernel_regularizer: Kernel regularizer for Conv2D units.
    post_fusion: An optional layer that will be inserted after the first
      Conv2D.
    post_projection: An optional layer that will be inserted after the last
      Conv2D.
    collapsed: If True, graph will collapse at convolutional units
      level on different kernel sizes.
    data_format: Data format used for Conv2D and depthwise Conv2D.
    name: Name of the layer.

  Returns:
    A fused inverted bottleneck layer as a compound layer.
  """
  if (not selections.is_fixed(expansion_factor)
      and 1 in selections.selection_candidates(expansion_factor)):
    raise ValueError(
        'Tunable `expansion_factor` with candidates 1 and values greater than 1'
        'is not supported: %r.' % expansion_factor)

  if not selections.is_fixed(kernel_size) and not collapsed:
    candidates = []
    for i, ks in enumerate(selections.selection_candidates(kernel_size)):
      candidates.append(fused_inverted_bottleneck(
          kernel_size=ks,
          filters=filters,
          expansion_factor=expansion_factor,
          strides=strides,
          normalization=normalization,
          activation=activation,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          post_fusion=post_fusion,
          post_projection=post_projection,
          data_format=data_format,
          name='branch%d' % i))
    return layers.Switch(
        candidates=candidates,
        selected_index=selections.selection_index(kernel_size),
        name=name)

  if expansion_factor != 1:
    children = [
        conv2d(  # pylint: disable=unexpected-keyword-arg
            filters=_expand_filters(   # pylint: disable=no-value-for-parameter
                expansion_factor=expansion_factor),
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_bias=False,
            normalization=normalization,
            activation=activation,
            data_format=data_format,
            name='expansion')
    ]
  else:
    children = []

  if post_fusion:
    children.append(post_fusion)

  children.append(conv2d(   # pylint: disable=unexpected-keyword-arg
      filters=filters,
      kernel_size=(1, 1),
      strides=(1, 1),
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer,
      use_bias=False,
      normalization=normalization,
      activation=None,
      data_format=data_format,
      name='fusion'))
  if post_projection:
    children.append(post_projection)
  return layers.Sequential(children)


@pg.symbolize
def _scale_filters(
    input_filters_mask: tf.Tensor,
    is_input_filters_masked: bool,
    ratio: Union[float, selections.FloatSelection],
    base: int) -> Tuple[tf.Tensor, bool]:
  """Returns input filters mask multiplied by a factor."""
  assert input_filters_mask.shape.rank == 1, input_filters_mask
  max_filters = modeling_utils.scale_filters(
      int(input_filters_mask.shape[-1]), ratio, base)
  effective_filters = modeling_utils.scale_filters(
      tf.math.reduce_sum(tf.cast(input_filters_mask, tf.dtypes.int32)),
      ratio, base)
  output_filters_mask = tf.sequence_mask(effective_filters, max_filters)
  return output_filters_mask, is_input_filters_masked


@layers.compound
def tucker_bottleneck(
    filters: Filters,
    kernel_size: Union[int, Tuple[int, int]],
    input_scale_ratio: Union[float, selections.FloatSelection],
    output_scale_ratio: Union[float, selections.FloatSelection],
    strides: Union[int, Tuple[int, int]] = (1, 1),
    normalization: tf.keras.layers.Layer = layers.BatchNormalization(),
    activation: tf.keras.layers.Layer = layers.ReLU(),
    kernel_initializer='glorot_uniform',
    kernel_regularizer=None,
    data_format: Text = 'channels_last',
    scale_filters_base: int = 8):
  """Fused inverted bottleneck.

  Reference: https://arxiv.org/pdf/2003.02838.pdf

  Args:
    filters: output filters
    kernel_size: kernel size for the depthwise Conv2D.
    input_scale_ratio: The filters scale ratio for the first Conv2D.
      If 0, the first Conv2D will be omitted.
    output_scale_ratio: The filters scale ratio for the last Conv2D.
    strides: Strides for the depthwise Conv2D.
    normalization: An optional normalization layer.
    activation: An optional activation layer.
    kernel_initializer: Kernel initializer used for Conv2D units.
    kernel_regularizer: Kernel regularizer for Conv2D units.
    data_format: Data format used for Conv2D and depthwise Conv2D.
    scale_filters_base: An integer as the base for scaling the filters. The
      scaled filters will always be multiple of the base.

  Returns:
    A fused inverted bottleneck layer as a compound layer.
  """
  if (not selections.is_fixed(input_scale_ratio)
      and 0 in selections.selection_candidates(input_scale_ratio)):
    raise ValueError(
        'Tunable `input_scale_ratio` with candidates 0 and values greater than '
        '0 is not supported: %r.' % input_scale_ratio)

  if (not selections.is_fixed(input_scale_ratio)
      or selections.selection_value(input_scale_ratio) > 0):
    children = [
        conv2d(   # pylint: disable=unexpected-keyword-arg
            filters=_scale_filters(   # pylint: disable=no-value-for-parameter
                ratio=input_scale_ratio, base=scale_filters_base),
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_bias=False,
            normalization=normalization,
            activation=activation,
            data_format=data_format,
            name='input_expansion')
    ]
  else:
    children = []

  children.append(conv2d(    # pylint: disable=unexpected-keyword-arg
      filters=modeling_utils.scale_filters(
          filters, output_scale_ratio, scale_filters_base),
      kernel_size=kernel_size,
      strides=strides,
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer,
      use_bias=False,
      normalization=normalization,
      activation=activation,
      data_format=data_format,
      name='output_expansion'))

  children.append(conv2d(   # pylint: disable=unexpected-keyword-arg
      filters=filters,
      kernel_size=(1, 1),
      strides=(1, 1),
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer,
      use_bias=False,
      normalization=normalization,
      activation=None,        # We do not have activation on the last Conv2D.
      data_format=data_format,
      name='projection'))
  return layers.Sequential(children)


class ScaleFiltersSaver(object):
  """Scale filters based on ratio and base, while save the input filters."""

  def __init__(self, ratio, base):
    self._input_filters_mask = None
    self._is_input_filters_masked = None
    self._call = _scale_filters(ratio=ratio, base=base)  # pylint: disable=no-value-for-parameter

  @property
  def value(self):
    if self._input_filters_mask is None:
      raise ValueError('self._input_filters is None.')
    return self._input_filters_mask, self._is_input_filters_masked

  @property
  def call(self):
    return self._call

  def __call__(self,
               input_filters_mask: tf.Tensor,
               is_input_filters_masked: bool):
    self._input_filters_mask = input_filters_mask
    self._is_input_filters_masked = is_input_filters_masked
    return self._call(input_filters_mask, is_input_filters_masked)  # pylint: disable=not-callable

  def __eq__(self, other: 'ScaleFiltersSaver'):
    if not isinstance(self, type(other)):
      return False
    else:
      return self.call == other.call


@pg.symbolize
def _return_saver_value(x, y, z):
  del x, y
  return z.value


@pg.symbolize
class SqueezeExcitation(keras.Model):
  """Mobile block."""

  def __init__(
      self,
      ratio: float,
      filters_base: int = 8,
      hidden_activation: tf.keras.layers.Layer = layers.ReLU(),
      output_activation: tf.keras.layers.Layer = layers.Activation('sigmoid'),
      kernel_initializer='glorot_uniform',
      kernel_regularizer=None,
      bias_initializer='zeros',
      bias_regularizer=None,
      name: Optional[Text] = None,
      **kwargs):
    """Mobile block.

    Args:
      ratio: Ratio to scale filters.
      filters_base: The number of filters will be rounded to a multiple of
        this value.
      hidden_activation: Activation for hidden convolutional layer.
      output_activation: Activation for output convolutional layer.
      kernel_initializer: Initializer for the kernel variables.
      kernel_regularizer: Regularizer function applied to the `kernel`
        weights matrix'.
      bias_initializer: Initializer for the `bias` weights matrix
      bias_regularizer: Regularizer function applied to the `bias`
        weights matrix.
      name: Name of the block.
      **kwargs: keyword arguments to be passed.

    Returns:
      A tuple of 2 tensors (block output, features-before-downsampling)
    """
    super().__init__(name=name, **kwargs)
    scale_filter_saver = ScaleFiltersSaver(ratio=ratio, base=filters_base)
    self._gap = layers.GlobalAveragePooling2D(keepdims=True)
    self._se_reduce = layers.Conv2D(
        kernel_size=(1, 1),
        filters=scale_filter_saver,
        strides=(1, 1),
        use_bias=True,
        activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        name=name + '-squeeze' if name is not None else None)
    self._se_expand = layers.Conv2D(
        kernel_size=(1, 1),
        filters=_return_saver_value(  # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
            z=scale_filter_saver, override_args=True),
        strides=(1, 1),
        use_bias=True,
        activation=output_activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        name=name + '-excite' if name is not None else None)
    self._multiply = layers.Multiply()

  def call(self, inputs, training=None):
    x = self._gap(inputs)
    x = self._se_reduce(x)
    x = self._se_expand(x)
    return self._multiply([x, inputs])
