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

# Lint as: python3
"""Contains definitions for the AssembleNet [1] models.

Requires the AssembleNet architecture to be specified in
FLAGS.model_structure (and optionally FLAGS.model_edge_weights).
This structure is a list corresponding to a graph representation of the
network, where a node is a convolutional block and an edge specifies a
connection from one block to another as described in [1].

Each node itself (in the structure list) is a list with the following format:
[block_level, [list_of_input_blocks], number_filter, temporal_dilation,
spatial_stride]. [list_of_input_blocks] should be the list of node indexes whose
values are less than the index of the node itself. The 'stems' of the network
directly taking raw inputs follow a different node format:
[stem_type, temporal_dilation]. The stem_type is -1 for RGB stem and is -2 for
optical flow stem.

Also note that the codes in this file could be used for one-shot differentiable
connection search by (1) giving an overly connected structure as
FLAGS.model_structure and by (2) setting FLAGS.model_edge_weights to be '[]'.
The 'agg_weights' variables will specify which connections are needed and which
are not, once trained.

[1] Michael S. Ryoo, AJ Piergiovanni, Mingxing Tan, Anelia Angelova,
    AssembleNet: Searching for Multi-Stream Neural Connectivity in Video
    Architectures. ICLR 2020
    https://arxiv.org/abs/1905.13209

It uses (2+1)D convolutions for video representations. The main AssembleNet
takes a 4-D (N*T)HWC tensor as an input (i.e., the batch dim and time dim are
mixed), and it reshapes a tensor to NT(H*W)C whenever a 1-D temporal conv. is
necessary. This is to run this on TPU efficiently.
"""

import functools
import math
from typing import Any, Mapping, List, Callable, Optional

from absl import logging
import numpy as np
import tensorflow as tf

from official.modeling import hyperparams
from official.vision.beta.modeling import factory_3d as model_factory
from official.vision.beta.modeling.backbones import factory as backbone_factory
from official.vision.beta.projects.assemblenet.configs import assemblenet as cfg
from official.vision.beta.projects.assemblenet.modeling import rep_flow_2d_layer as rf

layers = tf.keras.layers
intermediate_channel_size = [64, 128, 256, 512]


def fixed_padding(inputs, kernel_size):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or `[batch,
      height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
      operations. Should be a positive integer.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  data_format = tf.keras.backend.image_data_format()
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = tf.pad(
        inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def reshape_temporal_conv1d_bn(inputs: tf.Tensor,
                               filters: int,
                               kernel_size: int,
                               num_frames: int = 32,
                               temporal_dilation: int = 1,
                               bn_decay: float = rf.BATCH_NORM_DECAY,
                               bn_epsilon: float = rf.BATCH_NORM_EPSILON,
                               use_sync_bn: bool = False):
  """Performs 1D temporal conv.

  followed by batch normalization with reshaping.

  Args:
    inputs: `Tensor` of size `[batch*time, height, width, channels]`. Only
      supports 'channels_last' as the data format.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
      operations. Should be a positive integer.
    num_frames: `int` number of frames in the input tensor.
    temporal_dilation: `int` temporal dilatioin size for the 1D conv.
    bn_decay: `float` batch norm decay parameter to use.
    bn_epsilon: `float` batch norm epsilon parameter to use.
    use_sync_bn: use synchronized batch norm for TPU.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  data_format = tf.keras.backend.image_data_format()
  assert data_format == 'channels_last'

  feature_shape = inputs.shape

  inputs = tf.reshape(
      inputs,
      [-1, num_frames, feature_shape[1] * feature_shape[2], feature_shape[3]])

  if temporal_dilation == 1:
    inputs = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(kernel_size, 1),
        strides=1,
        padding='SAME',
        use_bias=False,
        kernel_initializer=tf.keras.initializers.VarianceScaling())(
            inputs=inputs)
  else:
    inputs = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(kernel_size, 1),
        strides=1,
        padding='SAME',
        dilation_rate=(temporal_dilation, 1),
        use_bias=False,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=math.sqrt(2.0 / (kernel_size * feature_shape[3]))))(
                inputs=inputs)

  num_channel = inputs.shape[3]
  inputs = tf.reshape(inputs,
                      [-1, feature_shape[1], feature_shape[2], num_channel])
  inputs = rf.build_batch_norm(
      bn_decay=bn_decay, bn_epsilon=bn_epsilon, use_sync_bn=use_sync_bn)(
          inputs)
  inputs = tf.nn.relu(inputs)

  return inputs


def conv2d_fixed_padding(inputs: tf.Tensor, filters: int, kernel_size: int,
                         strides: int):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.keras.layers.Conv2D` alone).

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size)

  return tf.keras.layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      kernel_initializer=tf.keras.initializers.VarianceScaling())(
          inputs=inputs)


def conv3d_same_padding(inputs: tf.Tensor,
                        filters: int,
                        kernel_size: int,
                        strides: int,
                        temporal_dilation: int = 1,
                        do_2d_conv: bool = False):
  """3D convolution layer wrapper.

  Uses conv3d function.

  Args:
    inputs: 5D `Tensor` following the data_format.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    temporal_dilation: `int` temporal dilatioin size for the 1D conv.
    do_2d_conv: `bool` indicating whether to do 2d conv. If false, do 3D conv.

  Returns:
    A `Tensor` of shape `[batch, time_in, height_in, width_in, channels]`.
  """
  if isinstance(kernel_size, int):
    if do_2d_conv:
      kernel_size = [1, kernel_size, kernel_size]
    else:
      kernel_size = [kernel_size, kernel_size, kernel_size]

  return tf.keras.layers.Conv3D(
      filters=filters,
      kernel_size=kernel_size,
      strides=[1, strides, strides],
      padding='SAME',
      dilation_rate=[temporal_dilation, 1, 1],
      use_bias=False,
      kernel_initializer=tf.keras.initializers.VarianceScaling())(
          inputs=inputs)


def bottleneck_block_interleave(inputs: tf.Tensor,
                                filters: int,
                                inter_filters: int,
                                strides: int,
                                use_projection: bool = False,
                                num_frames: int = 32,
                                temporal_dilation: int = 1,
                                bn_decay: float = rf.BATCH_NORM_DECAY,
                                bn_epsilon: float = rf.BATCH_NORM_EPSILON,
                                use_sync_bn: bool = False,
                                step=1):
  """Interleaves a standard 2D residual module and (2+1)D residual module.

  Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch*time, channels, height, width]`.
    filters: `int` number of filters for the first conv. layer. The last conv.
      layer will use 4 times as many filters.
    inter_filters: `int` number of filters for the second conv. layer.
    strides: `int` block stride. If greater than 1, this block will ultimately
      downsample the input spatially.
    use_projection: `bool` for whether this block should use a projection
      shortcut (versus the default identity shortcut). This is usually `True`
      for the first block of a block group, which may change the number of
      filters and the resolution.
    num_frames: `int` number of frames in the input tensor.
    temporal_dilation: `int` temporal dilatioin size for the 1D conv.
    bn_decay: `float` batch norm decay parameter to use.
    bn_epsilon: `float` batch norm epsilon parameter to use.
    use_sync_bn: use synchronized batch norm for TPU.
    step: `int` to decide whether to put 2D module or (2+1)D module.

  Returns:
    The output `Tensor` of the block.
  """
  if strides > 1 and not use_projection:
    raise ValueError('strides > 1 requires use_projections=True, otherwise the '
                     'inputs and shortcut will have shape mismatch')
  shortcut = inputs
  if use_projection:
    # Projection shortcut only in first block within a group. Bottleneck blocks
    # end with 4 times the number of filters.
    filters_out = 4 * filters
    shortcut = conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides)
    shortcut = rf.build_batch_norm(
        bn_decay=bn_decay, bn_epsilon=bn_epsilon, use_sync_bn=use_sync_bn)(
            shortcut)

  if step % 2 == 1:
    k = 3

    inputs = reshape_temporal_conv1d_bn(
        inputs=inputs,
        filters=filters,
        kernel_size=k,
        num_frames=num_frames,
        temporal_dilation=temporal_dilation,
        bn_decay=bn_decay,
        bn_epsilon=bn_epsilon,
        use_sync_bn=use_sync_bn)
  else:
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1)
    inputs = rf.build_batch_norm(
        bn_decay=bn_decay, bn_epsilon=bn_epsilon, use_sync_bn=use_sync_bn)(
            inputs)
    inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=inter_filters, kernel_size=3, strides=strides)
  inputs = rf.build_batch_norm(
      bn_decay=bn_decay, bn_epsilon=bn_epsilon, use_sync_bn=use_sync_bn)(
          inputs)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)
  inputs = rf.build_batch_norm(
      init_zero=True,
      bn_decay=bn_decay,
      bn_epsilon=bn_epsilon,
      use_sync_bn=use_sync_bn)(
          inputs)

  return tf.nn.relu(inputs + shortcut)


def block_group(inputs: tf.Tensor,
                filters: int,
                block_fn: Callable[..., tf.Tensor],
                blocks: int,
                strides: int,
                name,
                block_level,
                num_frames=32,
                temporal_dilation=1):
  """Creates one group of blocks for the AssembleNett model.

  Args:
    inputs: `Tensor` of size `[batch*time, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
      greater than 1, this layer will downsample the input.
    name: `str` name for the Tensor output of the block layer.
    block_level: `int` block level in AssembleNet.
    num_frames: `int` number of frames in the input tensor.
    temporal_dilation: `int` temporal dilatioin size for the 1D conv.

  Returns:
    The output `Tensor` of the block layer.
  """
  # Only the first block per block_group uses projection shortcut and strides.
  inputs = block_fn(
      inputs,
      filters,
      intermediate_channel_size[block_level],
      strides,
      use_projection=True,
      num_frames=num_frames,
      temporal_dilation=temporal_dilation,
      step=0)

  for i in range(1, blocks):
    inputs = block_fn(
        inputs,
        filters,
        intermediate_channel_size[block_level],
        1,
        num_frames=num_frames,
        temporal_dilation=temporal_dilation,
        step=i)

  return tf.identity(inputs, name)


def spatial_resize_and_concat(inputs):
  """Concatenates multiple different sized tensors channel-wise.

  Args:
    inputs: A list of `Tensors` of size `[batch*time, channels, height, width]`.

  Returns:
    The output `Tensor` after concatenation.
  """
  data_format = tf.keras.backend.image_data_format()
  assert data_format == 'channels_last'

  # Do nothing if only 1 input
  if len(inputs) == 1:
    return inputs[0]
  if data_format != 'channels_last':
    return inputs

  # get smallest spatial size and largest channels
  sm_size = [1000, 1000]
  for inp in inputs:
    # assume batch X height x width x channels
    sm_size[0] = min(sm_size[0], inp.shape[1])
    sm_size[1] = min(sm_size[1], inp.shape[2])

  for i in range(len(inputs)):
    if inputs[i].shape[1] != sm_size[0] or inputs[i].shape[2] != sm_size[1]:
      ratio = (inputs[i].shape[1] + 1) // sm_size[0]
      inputs[i] = tf.keras.layers.MaxPool2D([ratio, ratio],
                                            ratio,
                                            padding='same')(
                                                inputs[i])

  return tf.concat(inputs, 3)


class _ApplyEdgeWeight(layers.Layer):
  """Multiply weight on each input tensor.

  A weight is assigned for each connection (i.e., each input tensor). This layer
  is used by the multi_connection_fusion to compute the weighted inputs.
  """

  def __init__(self,
               weights_shape,
               index: Optional[int] = None,
               use_5d_mode: bool = False,
               model_edge_weights: Optional[List[Any]] = None,
               **kwargs):
    """Constructor.

    Args:
      weights_shape: shape of the weights. Should equals to [len(inputs)].
      index: `int` index of the block within the AssembleNet architecture. Used
        for summation weight initial loading.
      use_5d_mode: `bool` indicating whether the inputs are in 5D tensor or 4D.
      model_edge_weights: AssembleNet model structure connection weights in the
        string format.
      **kwargs: pass through arguments.
    """
    super(_ApplyEdgeWeight, self).__init__(**kwargs)

    self._weights_shape = weights_shape
    self._index = index
    self._use_5d_mode = use_5d_mode
    self._model_edge_weights = model_edge_weights
    data_format = tf.keras.backend.image_data_format()
    assert data_format == 'channels_last'

  def get_config(self):
    config = {
        'weights_shape': self._weights_shape,
        'index': self._index,
        'use_5d_mode': self._use_5d_mode,
        'model_edge_weights': self._model_edge_weights,
    }
    base_config = super(_ApplyEdgeWeight, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape: tf.TensorShape):
    if self._weights_shape[0] == 1:
      self._edge_weights = 1.0
      return

    if self._index is None or not self._model_edge_weights:
      self._edge_weights = self.add_weight(
          shape=self._weights_shape,
          initializer=tf.keras.initializers.TruncatedNormal(
              mean=0.0, stddev=0.01),
          trainable=True,
          name='agg_weights')
    else:
      initial_weights_after_sigmoid = np.asarray(
          self._model_edge_weights[self._index][0]).astype('float32')
      # Initial_weights_after_sigmoid is never 0, as the initial weights are
      # based the results of a successful connectivity search.
      initial_weights = -np.log(1. / initial_weights_after_sigmoid - 1.)
      self._edge_weights = self.add_weight(
          shape=self._weights_shape,
          initializer=tf.constant_initializer(initial_weights),
          trainable=False,
          name='agg_weights')

  def call(self,
           inputs: List[tf.Tensor],
           training: Optional[bool] = None) -> Mapping[Any, List[tf.Tensor]]:
    use_5d_mode = self._use_5d_mode
    dtype = inputs[0].dtype
    assert len(inputs) > 1

    if use_5d_mode:
      h_channel_loc = 2
    else:
      h_channel_loc = 1

    # get smallest spatial size and largest channels
    sm_size = [10000, 10000]
    lg_channel = 0
    for inp in inputs:
      # assume batch X height x width x channels
      sm_size[0] = min(sm_size[0], inp.shape[h_channel_loc])
      sm_size[1] = min(sm_size[1], inp.shape[h_channel_loc + 1])
      lg_channel = max(lg_channel, inp.shape[-1])

    # loads or creates weight variables to fuse multiple inputs
    weights = tf.math.sigmoid(tf.cast(self._edge_weights, dtype))

    # Compute weighted inputs. We group inputs with the same channels.
    per_channel_inps = dict({0: []})
    for i, inp in enumerate(inputs):
      if inp.shape[h_channel_loc] != sm_size[0] or inp.shape[h_channel_loc + 1] != sm_size[1]:  # pylint: disable=line-too-long
        assert sm_size[0] != 0
        ratio = (inp.shape[h_channel_loc] + 1) // sm_size[0]
        if use_5d_mode:
          inp = tf.keras.layers.MaxPool3D([1, ratio, ratio], [1, ratio, ratio],
                                          padding='same')(
                                              inp)
        else:
          inp = tf.keras.layers.MaxPool2D([ratio, ratio], ratio,
                                          padding='same')(
                                              inp)

      weights = tf.cast(weights, inp.dtype)
      if inp.shape[-1] in per_channel_inps:
        per_channel_inps[inp.shape[-1]].append(weights[i] * inp)
      else:
        per_channel_inps.update({inp.shape[-1]: [weights[i] * inp]})
    return per_channel_inps


def multi_connection_fusion(inputs: List[tf.Tensor],
                            index: Optional[int] = None,
                            use_5d_mode: bool = False,
                            model_edge_weights: Optional[List[Any]] = None):
  """Do weighted summation of multiple different sized tensors.

  A weight is assigned for each connection (i.e., each input tensor), and their
  summation weights are learned. Uses spatial max pooling and 1x1 conv.
  to match their sizes.

  Args:
    inputs: A `Tensor`. Either 4D or 5D, depending of use_5d_mode.
    index: `int` index of the block within the AssembleNet architecture. Used
      for summation weight initial loading.
    use_5d_mode: `bool` indicating whether the inputs are in 5D tensor or 4D.
    model_edge_weights: AssembleNet model structure connection weights in the
      string format.

  Returns:
    The output `Tensor` after concatenation.
  """

  if use_5d_mode:
    h_channel_loc = 2
    conv_function = conv3d_same_padding
  else:
    h_channel_loc = 1
    conv_function = conv2d_fixed_padding

  # If only 1 input.
  if len(inputs) == 1:
    return inputs[0]

  # get smallest spatial size and largest channels
  sm_size = [10000, 10000]
  lg_channel = 0
  for inp in inputs:
    # assume batch X height x width x channels
    sm_size[0] = min(sm_size[0], inp.shape[h_channel_loc])
    sm_size[1] = min(sm_size[1], inp.shape[h_channel_loc + 1])
    lg_channel = max(lg_channel, inp.shape[-1])

  per_channel_inps = _ApplyEdgeWeight(
      weights_shape=[len(inputs)],
      index=index,
      use_5d_mode=use_5d_mode,
      model_edge_weights=model_edge_weights)(
          inputs)

  # Adding 1x1 conv layers (to match channel size) and fusing all inputs.
  # We add inputs with the same channels first before applying 1x1 conv to save
  # memory.
  inps = []
  for key, channel_inps in per_channel_inps.items():
    if len(channel_inps) < 1:
      continue
    if len(channel_inps) == 1:
      if key == lg_channel:
        inp = channel_inps[0]
      else:
        inp = conv_function(
            channel_inps[0], lg_channel, kernel_size=1, strides=1)
      inps.append(inp)
    else:
      if key == lg_channel:
        inp = tf.add_n(channel_inps)
      else:
        inp = conv_function(
            tf.add_n(channel_inps), lg_channel, kernel_size=1, strides=1)
      inps.append(inp)

  return tf.add_n(inps)


def rgb_conv_stem(inputs,
                  num_frames,
                  filters,
                  temporal_dilation,
                  bn_decay: float = rf.BATCH_NORM_DECAY,
                  bn_epsilon: float = rf.BATCH_NORM_EPSILON,
                  use_sync_bn: bool = False):
  """Layers for a RGB stem.

  Args:
    inputs: A `Tensor` of size `[batch*time, height, width, channels]`.
    num_frames: `int` number of frames in the input tensor.
    filters: `int` number of filters in the convolution.
    temporal_dilation: `int` temporal dilatioin size for the 1D conv.
    bn_decay: `float` batch norm decay parameter to use.
    bn_epsilon: `float` batch norm epsilon parameter to use.
    use_sync_bn: use synchronized batch norm for TPU.

  Returns:
    The output `Tensor`.
  """
  data_format = tf.keras.backend.image_data_format()
  assert data_format == 'channels_last'

  if temporal_dilation < 1:
    temporal_dilation = 1

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=7, strides=2)
  inputs = tf.identity(inputs, 'initial_conv')
  inputs = rf.build_batch_norm(
      bn_decay=bn_decay, bn_epsilon=bn_epsilon, use_sync_bn=use_sync_bn)(
          inputs)
  inputs = tf.nn.relu(inputs)

  inputs = reshape_temporal_conv1d_bn(
      inputs=inputs,
      filters=filters,
      kernel_size=5,
      num_frames=num_frames,
      temporal_dilation=temporal_dilation,
      bn_decay=bn_decay,
      bn_epsilon=bn_epsilon,
      use_sync_bn=use_sync_bn)

  inputs = tf.keras.layers.MaxPool2D(
      pool_size=3, strides=2, padding='SAME')(
          inputs=inputs)
  inputs = tf.identity(inputs, 'initial_max_pool')

  return inputs


def flow_conv_stem(inputs,
                   filters,
                   temporal_dilation,
                   bn_decay: float = rf.BATCH_NORM_DECAY,
                   bn_epsilon: float = rf.BATCH_NORM_EPSILON,
                   use_sync_bn: bool = False):
  """Layers for an optical flow stem.

  Args:
    inputs: A `Tensor` of size `[batch*time, height, width, channels]`.
    filters: `int` number of filters in the convolution.
    temporal_dilation: `int` temporal dilatioin size for the 1D conv.
    bn_decay: `float` batch norm decay parameter to use.
    bn_epsilon: `float` batch norm epsilon parameter to use.
    use_sync_bn: use synchronized batch norm for TPU.

  Returns:
    The output `Tensor`.
  """

  if temporal_dilation < 1:
    temporal_dilation = 1

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=7, strides=2)
  inputs = tf.identity(inputs, 'initial_conv')
  inputs = rf.build_batch_norm(
      bn_decay=bn_decay, bn_epsilon=bn_epsilon, use_sync_bn=use_sync_bn)(
          inputs)
  inputs = tf.nn.relu(inputs)

  inputs = tf.keras.layers.MaxPool2D(
      pool_size=2, strides=2, padding='SAME')(
          inputs=inputs)
  inputs = tf.identity(inputs, 'initial_max_pool')

  return inputs


def multi_stream_heads(streams,
                       final_nodes,
                       num_frames,
                       num_classes,
                       max_pool_preditions: bool = False):
  """Layers for the classification heads.

  Args:
    streams: A list of 4D `Tensors` following the data_format.
    final_nodes: A list of `int` where classification heads will be added.
    num_frames: `int` number of frames in the input tensor.
    num_classes: `int` number of possible classes for video classification.
    max_pool_preditions: Use max-pooling on predictions instead of mean
      pooling on features. It helps if you have more than 32 frames.

  Returns:
    The output `Tensor`.
  """
  inputs = streams[final_nodes[0]]
  num_channels = inputs.shape[-1]

  def _pool_and_reshape(net):
    # The activation is 7x7 so this is a global average pool.
    net = tf.keras.layers.GlobalAveragePooling2D()(inputs=net)
    net = tf.identity(net, 'final_avg_pool0')

    net = tf.reshape(net, [-1, num_frames, num_channels])
    if not max_pool_preditions:
      net = tf.reduce_mean(net, 1)
    return net

  outputs = _pool_and_reshape(inputs)

  for i in range(1, len(final_nodes)):
    inputs = streams[final_nodes[i]]

    inputs = _pool_and_reshape(inputs)

    outputs = outputs + inputs

  if len(final_nodes) > 1:
    outputs = outputs / len(final_nodes)

  outputs = tf.keras.layers.Dense(
      units=num_classes,
      kernel_initializer=tf.random_normal_initializer(stddev=.01))(
          inputs=outputs)
  outputs = tf.identity(outputs, 'final_dense0')
  if max_pool_preditions:
    pre_logits = outputs / np.sqrt(num_frames)
    acts = tf.nn.softmax(pre_logits, axis=1)
    outputs = tf.math.multiply(outputs, acts)

    outputs = tf.reduce_sum(outputs, 1)

  return outputs


class AssembleNet(tf.keras.Model):
  """AssembleNet backbone."""

  def __init__(
      self,
      block_fn,
      num_blocks: List[int],
      num_frames: int,
      model_structure: List[Any],
      input_specs: layers.InputSpec = layers.InputSpec(
          shape=[None, None, None, None, 3]),
      model_edge_weights: Optional[List[Any]] = None,
      bn_decay: float = rf.BATCH_NORM_DECAY,
      bn_epsilon: float = rf.BATCH_NORM_EPSILON,
      use_sync_bn: bool = False,
      combine_method: str = 'sigmoid',
      **kwargs):
    """Generator for AssembleNet v1 models.

    Args:
      block_fn: `function` for the block to use within the model. Currently only
        has `bottleneck_block_interleave as its option`.
      num_blocks: list of 4 `int`s denoting the number of blocks to include in
        each of the 4 block groups. Each group consists of blocks that take
        inputs of the same resolution.
      num_frames: the number of frames in the input tensor.
      model_structure: AssembleNet model structure in the string format.
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
        Dimension should be `[batch*time, height, width, channels]`.
      model_edge_weights: AssembleNet model structure connection weights in the
        string format.
      bn_decay: `float` batch norm decay parameter to use.
      bn_epsilon: `float` batch norm epsilon parameter to use.
      use_sync_bn: use synchronized batch norm for TPU.
      combine_method: 'str' for the weighted summation to fuse different blocks.
      **kwargs: pass through arguments.
    """
    inputs = tf.keras.Input(shape=input_specs.shape[1:])
    data_format = tf.keras.backend.image_data_format()

    # Creation of the model graph.
    logging.info('model_structure=%r', model_structure)
    logging.info('model_structure=%r', model_structure)
    logging.info('model_edge_weights=%r', model_edge_weights)
    structure = model_structure

    original_num_frames = num_frames
    assert num_frames > 0, f'Invalid num_frames {num_frames}'

    grouping = {-3: [], -2: [], -1: [], 0: [], 1: [], 2: [], 3: []}
    for i in range(len(structure)):
      grouping[structure[i][0]].append(i)

    stem_count = len(grouping[-3]) + len(grouping[-2]) + len(grouping[-1])

    assert stem_count != 0
    stem_filters = 128 // stem_count

    original_inputs = inputs
    if len(input_specs.shape) == 5:
      first_dim = (
          input_specs.shape[0] * input_specs.shape[1]
          if input_specs.shape[0] and input_specs.shape[1] else -1)
      reshape_inputs = tf.reshape(inputs, (first_dim,) + input_specs.shape[2:])
    elif len(input_specs.shape) == 4:
      reshape_inputs = original_inputs
    else:
      raise ValueError(
          f'Expect input spec to be 4 or 5 dimensions {input_specs.shape}')
    if grouping[-2]:
      # Instead of loading optical flows as inputs from data pipeline, we are
      # applying the "Representation Flow" to RGB frames so that we can compute
      # the flow within TPU/GPU on fly. It's essentially optical flow since we
      # do it with RGBs.
      axis = 3 if data_format == 'channels_last' else 1
      flow_inputs = rf.RepresentationFlow(
          original_num_frames,
          depth=reshape_inputs.shape.as_list()[axis],
          num_iter=40,
          bottleneck=1)(
              reshape_inputs)
    streams = []

    for i in range(len(structure)):
      with tf.name_scope('Node_' + str(i)):
        if structure[i][0] == -1:
          inputs = rgb_conv_stem(
              reshape_inputs,
              original_num_frames,
              stem_filters,
              temporal_dilation=structure[i][1],
              bn_decay=bn_decay,
              bn_epsilon=bn_epsilon,
              use_sync_bn=use_sync_bn)
          streams.append(inputs)
        elif structure[i][0] == -2:
          inputs = flow_conv_stem(
              flow_inputs,
              stem_filters,
              temporal_dilation=structure[i][1],
              bn_decay=bn_decay,
              bn_epsilon=bn_epsilon,
              use_sync_bn=use_sync_bn)
          streams.append(inputs)

        else:
          num_frames = original_num_frames
          block_number = structure[i][0]

          combined_inputs = []
          if combine_method == 'concat':
            combined_inputs = [
                streams[structure[i][1][j]]
                for j in range(0, len(structure[i][1]))
            ]

            combined_inputs = spatial_resize_and_concat(combined_inputs)

          else:
            combined_inputs = [
                streams[structure[i][1][j]]
                for j in range(0, len(structure[i][1]))
            ]

            combined_inputs = multi_connection_fusion(
                combined_inputs, index=i, model_edge_weights=model_edge_weights)

          graph = block_group(
              inputs=combined_inputs,
              filters=structure[i][2],
              block_fn=block_fn,
              blocks=num_blocks[block_number],
              strides=structure[i][4],
              name='block_group' + str(i),
              block_level=structure[i][0],
              num_frames=num_frames,
              temporal_dilation=structure[i][3])

          streams.append(graph)

    super(AssembleNet, self).__init__(
        inputs=original_inputs, outputs=streams, **kwargs)


@tf.keras.utils.register_keras_serializable(package='Vision')
class AssembleNetModel(tf.keras.Model):
  """An AssembleNet model builder."""

  def __init__(self,
               backbone,
               num_classes,
               num_frames: int,
               model_structure: List[Any],
               input_specs: Optional[Mapping[str,
                                             tf.keras.layers.InputSpec]] = None,
               max_pool_preditions: bool = False,
               **kwargs):
    if not input_specs:
      input_specs = {
          'image': layers.InputSpec(shape=[None, None, None, None, 3])
      }
    self._self_setattr_tracking = False
    self._config_dict = {
        'backbone': backbone,
        'num_classes': num_classes,
        'num_frames': num_frames,
        'input_specs': input_specs,
        'model_structure': model_structure,
    }
    self._input_specs = input_specs
    self._backbone = backbone
    grouping = {-3: [], -2: [], -1: [], 0: [], 1: [], 2: [], 3: []}
    for i in range(len(model_structure)):
      grouping[model_structure[i][0]].append(i)

    inputs = {
        k: tf.keras.Input(shape=v.shape[1:]) for k, v in input_specs.items()
    }
    streams = self._backbone(inputs['image'])

    outputs = multi_stream_heads(
        streams,
        grouping[3],
        num_frames,
        num_classes,
        max_pool_preditions=max_pool_preditions)

    super(AssembleNetModel, self).__init__(
        inputs=inputs, outputs=outputs, **kwargs)

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


ASSEMBLENET_SPECS = {
    26: {
        'block': bottleneck_block_interleave,
        'num_blocks': [2, 2, 2, 2]
    },
    38: {
        'block': bottleneck_block_interleave,
        'num_blocks': [2, 4, 4, 2]
    },
    50: {
        'block': bottleneck_block_interleave,
        'num_blocks': [3, 4, 6, 3]
    },
    68: {
        'block': bottleneck_block_interleave,
        'num_blocks': [3, 4, 12, 3]
    },
    77: {
        'block': bottleneck_block_interleave,
        'num_blocks': [3, 4, 15, 3]
    },
    101: {
        'block': bottleneck_block_interleave,
        'num_blocks': [3, 4, 23, 3]
    },
}


def assemblenet_v1(assemblenet_depth: int,
                   num_classes: int,
                   num_frames: int,
                   model_structure: List[Any],
                   input_specs: layers.InputSpec = layers.InputSpec(
                       shape=[None, None, None, None, 3]),
                   model_edge_weights: Optional[List[Any]] = None,
                   max_pool_preditions: bool = False,
                   combine_method: str = 'sigmoid',
                   **kwargs):
  """Returns the AssembleNet model for a given size and number of output classes."""

  data_format = tf.keras.backend.image_data_format()
  assert data_format == 'channels_last'

  if assemblenet_depth not in ASSEMBLENET_SPECS:
    raise ValueError('Not a valid assemblenet_depth:', assemblenet_depth)

  input_specs_dict = {'image': input_specs}
  params = ASSEMBLENET_SPECS[assemblenet_depth]
  backbone = AssembleNet(
      block_fn=params['block'],
      num_blocks=params['num_blocks'],
      num_frames=num_frames,
      model_structure=model_structure,
      input_specs=input_specs,
      model_edge_weights=model_edge_weights,
      combine_method=combine_method,
      **kwargs)
  return AssembleNetModel(
      backbone,
      num_classes=num_classes,
      num_frames=num_frames,
      model_structure=model_structure,
      input_specs=input_specs_dict,
      max_pool_preditions=max_pool_preditions,
      **kwargs)


@backbone_factory.register_backbone_builder('assemblenet')
def build_assemblenet_v1(
    input_specs: tf.keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
) -> tf.keras.Model:
  """Builds assemblenet backbone."""
  del l2_regularizer

  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'assemblenet'

  assemblenet_depth = int(backbone_cfg.model_id)
  if assemblenet_depth not in ASSEMBLENET_SPECS:
    raise ValueError('Not a valid assemblenet_depth:', assemblenet_depth)
  model_structure, model_edge_weights = cfg.blocks_to_flat_lists(
      backbone_cfg.blocks)
  params = ASSEMBLENET_SPECS[assemblenet_depth]
  block_fn = functools.partial(
      params['block'],
      use_sync_bn=norm_activation_config.use_sync_bn,
      bn_decay=norm_activation_config.norm_momentum,
      bn_epsilon=norm_activation_config.norm_epsilon)
  backbone = AssembleNet(
      block_fn=block_fn,
      num_blocks=params['num_blocks'],
      num_frames=backbone_cfg.num_frames,
      model_structure=model_structure,
      input_specs=input_specs,
      model_edge_weights=model_edge_weights,
      combine_method=backbone_cfg.combine_method,
      use_sync_bn=norm_activation_config.use_sync_bn,
      bn_decay=norm_activation_config.norm_momentum,
      bn_epsilon=norm_activation_config.norm_epsilon)
  logging.info('Number of parameters in AssembleNet backbone: %f M.',
               backbone.count_params() / 10.**6)
  return backbone


@model_factory.register_model_builder('assemblenet')
def build_assemblenet_model(
    input_specs: tf.keras.layers.InputSpec,
    model_config: cfg.AssembleNetModel,
    num_classes: int,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None):
  """Builds assemblenet model."""
  input_specs_dict = {'image': input_specs}
  backbone = build_assemblenet_v1(input_specs, model_config.backbone,
                                  model_config.norm_activation, l2_regularizer)
  backbone_cfg = model_config.backbone.get()
  model_structure, _ = cfg.blocks_to_flat_lists(backbone_cfg.blocks)
  model = AssembleNetModel(
      backbone,
      num_classes=num_classes,
      num_frames=backbone_cfg.num_frames,
      model_structure=model_structure,
      input_specs=input_specs_dict,
      max_pool_preditions=model_config.max_pool_preditions)
  return model
