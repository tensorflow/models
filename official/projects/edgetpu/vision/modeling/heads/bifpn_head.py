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

"""Contains the definitions of Bi-Directional Feature Pyramid Networks (BiFPN)."""
import functools
import itertools

from typing import Text, Optional
# Import libraries

from absl import logging
import numpy as np
import tensorflow as tf, tf_keras

from official.projects.edgetpu.vision.modeling import common_modules


def activation_fn(features: tf.Tensor, act_type: Text):
  """Customized non-linear activation type."""
  if act_type in ('silu', 'swish'):
    return tf.nn.swish(features)
  elif act_type == 'swish_native':
    return features * tf.sigmoid(features)
  elif act_type == 'hswish':
    return features * tf.nn.relu6(features + 3) / 6
  elif act_type == 'relu':
    return tf.nn.relu(features)
  elif act_type == 'relu6':
    return tf.nn.relu6(features)
  else:
    raise ValueError('Unsupported act_type {}'.format(act_type))


def build_batch_norm(is_training_bn: bool,
                     beta_initializer: Text = 'zeros',
                     gamma_initializer: Text = 'ones',
                     data_format: Text = 'channels_last',
                     momentum: float = 0.99,
                     epsilon: float = 1e-3,
                     strategy: Optional[Text] = None,
                     name: Text = 'tpu_batch_normalization'):
  """Builds a batch normalization layer.

  Args:
    is_training_bn: `bool` for whether the model is training.
    beta_initializer: `str`, beta initializer.
    gamma_initializer: `str`, gamma initializer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    momentum: `float`, momentume of batch norm.
    epsilon: `float`, small value for numerical stability.
    strategy: `str`, whether to use tpu, gpus or other version of batch norm.
    name: the name of the batch normalization layer

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  axis = 1 if data_format == 'channels_first' else -1

  if is_training_bn:
    batch_norm_class = common_modules.get_batch_norm(strategy)
  else:
    batch_norm_class = tf_keras.layers.BatchNormalization

  bn_layer = batch_norm_class(
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=True,
      scale=True,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      name=name)

  return bn_layer


def bifpn_config(min_level, max_level):
  """A dynamic bifpn config that can adapt to different min/max levels."""
  p = {}

  # Node id starts from the input features and monotonically increase whenever
  # a new node is added. Here is an example for level P3 - P7:
  #     P7 (4)              P7" (12)
  #     P6 (3)    P6' (5)   P6" (11)
  #     P5 (2)    P5' (6)   P5" (10)
  #     P4 (1)    P4' (7)   P4" (9)
  #     P3 (0)              P3" (8)
  # So output would be like:
  # [
  #   {'feat_level': 6, 'inputs_offsets': [3, 4]},  # for P6'
  #   {'feat_level': 5, 'inputs_offsets': [2, 5]},  # for P5'
  #   {'feat_level': 4, 'inputs_offsets': [1, 6]},  # for P4'
  #   {'feat_level': 3, 'inputs_offsets': [0, 7]},  # for P3"
  #   {'feat_level': 4, 'inputs_offsets': [1, 7, 8]},  # for P4"
  #   {'feat_level': 5, 'inputs_offsets': [2, 6, 9]},  # for P5"
  #   {'feat_level': 6, 'inputs_offsets': [3, 5, 10]},  # for P6"
  #   {'feat_level': 7, 'inputs_offsets': [4, 11]},  # for P7"
  # ]
  num_levels = max_level - min_level + 1
  node_ids = {min_level + i: [i] for i in range(num_levels)}

  level_last_id = lambda level: node_ids[level][-1]
  level_all_ids = lambda level: node_ids[level]
  id_cnt = itertools.count(num_levels)

  p['nodes'] = []
  for i in range(max_level - 1, min_level - 1, -1):
    # top-down path.
    p['nodes'].append({
        'feat_level': i,
        'inputs_offsets': [level_last_id(i),
                           level_last_id(i + 1)]
    })
    node_ids[i].append(next(id_cnt))

  for i in range(min_level + 1, max_level + 1):
    # bottom-up path.
    p['nodes'].append({
        'feat_level': i,
        'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)]
    })
    node_ids[i].append(next(id_cnt))

  return p


def get_conv_op(conv_type):
  """Gets convlution op."""
  kernel_size = int(conv_type.split('_')[-1])
  if conv_type.startswith('sep'):
    conv_op = functools.partial(
        tf_keras.layers.SeparableConv2D,
        depth_multiplier=1,
        kernel_size=(kernel_size, kernel_size))
  elif conv_type.startswith('conv'):
    conv_op = functools.partial(
        tf_keras.layers.Conv2D, kernel_size=(kernel_size, kernel_size))
  else:
    raise ValueError('Unknown conv type: {}'.format(conv_type))
  return conv_op


def add_n(nodes):
  """A customized add_n to add up a list of tensors."""
  # tf.add_n is not supported by EdgeTPU, while tf.reduce_sum is not supported
  # by GPU and runs slow on EdgeTPU because of the 5-dimension op.
  with tf.name_scope('add_n'):
    new_node = nodes[0]
    for n in nodes[1:]:
      new_node = new_node + n
    return new_node


def resize_nearest_neighbor(data, height_scale, width_scale):
  """Nearest neighbor upsampling implementation."""
  with tf.name_scope('nearest_upsampling'):
    bs, h, w, c = data.get_shape().as_list()
    bs = -1 if bs is None else bs
    # Use reshape to quickly upsample the input.  The nearest pixel is selected
    # implicitly via broadcasting.
    data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones(
        [1, 1, height_scale, 1, width_scale, 1], dtype=data.dtype)
    return tf.reshape(data, [bs, h * height_scale, w * width_scale, c])


def resize(feat,
           target_height,
           target_width,
           strategy,
           training=False,
           method='bilinear'):
  """Resizes the spitial dimensions."""
  dtype = feat.dtype
  feat_shape = feat.get_shape()
  if method == 'bilinear':
    if strategy == 'tpu' and training:
      if dtype == tf.bfloat16:
        feat = tf.cast(feat, tf.float32)
        feat = tf.image.resize(feat, [target_height, target_width])
        feat = tf.cast(feat, dtype)
      elif feat_shape.is_fully_defined():
        # Batch dimension is known. Mimic resize[h,w] with
        # resize[h,1]+resize[1,w] to reduce HBM padding.
        b, h, w, c = feat_shape.as_list()
        feat = tf.reshape(feat, [b, h, 1, -1])
        feat = tf.image.resize(feat, [target_height, 1])
        feat = tf.reshape(feat, [-1, 1, w, c])
        feat = tf.image.resize(feat, [1, target_width])
        feat = tf.reshape(feat, [b, target_height, target_width, c])
      else:
        feat = tf.image.resize(feat, [target_height, target_width])
    else:
      feat = tf.image.resize(feat, [target_height, target_width])
  elif method == 'nearest':
    _, h, w, _ = feat_shape.as_list()
    if training and target_height % h == 0 and target_width % w == 0:

      feat = resize_nearest_neighbor(feat, target_height // h,
                                     target_width // w)
    else:
      feat = tf.cast(feat, tf.float32)
      feat = tf.image.resize(feat, [target_height, target_width],
                             tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  else:
    raise ValueError('Upsampling type {} is not supported.'.format(method))
  return tf.cast(feat, dtype)


class ResampleFeatureMap(tf_keras.layers.Layer):
  """Resamples feature map for downsampling or upsampling."""

  def __init__(self,
               feat_level,
               target_num_channels,
               apply_bn=False,
               is_training_bn=None,
               conv_after_downsample=False,
               strategy=None,
               data_format=None,
               pooling_type=None,
               upsampling_type=None,
               name='resample_p0'):
    super().__init__(name=name)
    self.apply_bn = apply_bn
    self.is_training_bn = is_training_bn
    self.data_format = data_format
    self.target_num_channels = target_num_channels
    self.feat_level = feat_level
    self.strategy = strategy
    self.conv_after_downsample = conv_after_downsample
    self.pooling_type = pooling_type or 'max'
    self.upsampling_type = upsampling_type or 'nearest'

  def _pool2d(self, inputs, height, width, target_height, target_width):
    """Pools the inputs to target height and width."""
    height_stride_size = int((height - 1) // target_height + 1)
    width_stride_size = int((width - 1) // target_width + 1)
    if self.pooling_type == 'max':
      return tf_keras.layers.MaxPooling2D(
          pool_size=[height_stride_size + 1, width_stride_size + 1],
          strides=[height_stride_size, width_stride_size],
          padding='SAME',
          data_format=self.data_format)(
              inputs)
    if self.pooling_type == 'avg':
      return tf_keras.layers.AveragePooling2D(
          pool_size=[height_stride_size + 1, width_stride_size + 1],
          strides=[height_stride_size, width_stride_size],
          padding='SAME',
          data_format=self.data_format)(
              inputs)
    raise ValueError('Unsupported pooling type {}.'.format(self.pooling_type))

  def _upsample2d(self, inputs, target_height, target_width, training):
    return resize(inputs, target_height, target_width, self.strategy, training,
                  self.upsampling_type)

  def _maybe_apply_1x1(self, feat, training, num_channels):
    """Applies 1x1 conv to change layer width if necessary."""
    target_num_channels = self.target_num_channels
    if target_num_channels is None or num_channels != target_num_channels:
      feat = self.conv2d(feat)
      if self.apply_bn:
        feat = self.bn(feat, training=training)
    return feat

  def build(self, feat_shape):
    num_channels = self.target_num_channels or feat_shape[-1]
    self.conv2d = tf_keras.layers.Conv2D(
        num_channels, (1, 1),
        padding='same',
        data_format=self.data_format,
        name='conv2d')
    self.bn = build_batch_norm(
        is_training_bn=self.is_training_bn,
        data_format=self.data_format,
        strategy=self.strategy,
        name='bn')
    self.built = True
    super().build(feat_shape)

  def call(self, feat, training, all_feats):
    hwc_idx = (2, 3, 1) if self.data_format == 'channels_first' else (1, 2, 3)
    height, width, num_channels = [feat.shape.as_list()[i] for i in hwc_idx]
    if all_feats:
      target_feat_shape = all_feats[self.feat_level].shape.as_list()
      target_height, target_width, _ = [target_feat_shape[i] for i in hwc_idx]
    else:
      # Default to downsampling if all_feats is empty.
      target_height, target_width = (height + 1) // 2, (width + 1) // 2

    # If conv_after_downsample is True, when downsampling, apply 1x1 after
    # downsampling for efficiency.
    if height > target_height and width > target_width:
      if not self.conv_after_downsample:
        feat = self._maybe_apply_1x1(feat, training, num_channels)
      feat = self._pool2d(feat, height, width, target_height, target_width)
      if self.conv_after_downsample:
        feat = self._maybe_apply_1x1(feat, training, num_channels)
    elif height <= target_height and width <= target_width:
      feat = self._maybe_apply_1x1(feat, training, num_channels)
      if height < target_height or width < target_width:
        feat = self._upsample2d(feat, target_height, target_width, training)
    else:
      raise ValueError(
          'Incompatible Resampling : feat shape {}x{} target_shape: {}x{}'
          .format(height, width, target_height, target_width))

    return feat


class FNode(tf_keras.layers.Layer):
  """A Keras Layer implementing BiFPN Node."""

  def __init__(self,
               feat_level,
               inputs_offsets,
               fpn_num_filters,
               apply_bn_for_resampling,
               is_training_bn,
               conv_after_downsample,
               conv_bn_act_pattern,
               conv_type,
               act_type,
               strategy,
               weight_method,
               data_format,
               pooling_type,
               upsampling_type,
               name='fnode'):
    super().__init__(name=name)
    self.feat_level = feat_level
    self.inputs_offsets = inputs_offsets
    self.fpn_num_filters = fpn_num_filters
    self.apply_bn_for_resampling = apply_bn_for_resampling
    self.conv_type = conv_type
    self.act_type = act_type
    self.is_training_bn = is_training_bn
    self.conv_after_downsample = conv_after_downsample
    self.strategy = strategy
    self.data_format = data_format
    self.weight_method = weight_method
    self.conv_bn_act_pattern = conv_bn_act_pattern
    self.pooling_type = pooling_type
    self.upsampling_type = upsampling_type
    self.resample_layers = []
    self.vars = []

  def fuse_features(self, nodes):
    """Fuses features from different resolutions and return a weighted sum.

    Args:
      nodes: a list of tensorflow features at different levels

    Returns:
      A tensor denoting the fused feature.
    """
    dtype = nodes[0].dtype

    if self.weight_method == 'attn':
      edge_weights = [tf.cast(var, dtype=dtype) for var in self.vars]
      normalized_weights = tf.nn.softmax(tf.stack(edge_weights))
      nodes = tf.stack(nodes, axis=-1)
      new_node = tf.reduce_sum(nodes * normalized_weights, -1)
    elif self.weight_method == 'fastattn':
      edge_weights = [
          tf.nn.relu(tf.cast(var, dtype=dtype)) for var in self.vars
      ]
      weights_sum = add_n(edge_weights)
      nodes = [
          nodes[i] * edge_weights[i] / (weights_sum + 0.0001)
          for i in range(len(nodes))
      ]
      new_node = add_n(nodes)
    elif self.weight_method == 'channel_attn':
      edge_weights = [tf.cast(var, dtype=dtype) for var in self.vars]
      normalized_weights = tf.nn.softmax(tf.stack(edge_weights, -1), axis=-1)
      nodes = tf.stack(nodes, axis=-1)
      new_node = tf.reduce_sum(nodes * normalized_weights, -1)
    elif self.weight_method == 'channel_fastattn':
      edge_weights = [
          tf.nn.relu(tf.cast(var, dtype=dtype)) for var in self.vars
      ]
      weights_sum = add_n(edge_weights)
      nodes = [
          nodes[i] * edge_weights[i] / (weights_sum + 0.0001)
          for i in range(len(nodes))
      ]
      new_node = add_n(nodes)
    elif self.weight_method == 'sum':
      new_node = add_n(nodes)
    else:
      raise ValueError('unknown weight_method %s' % self.weight_method)

    return new_node

  def _add_wsm(self, initializer, shape=None):
    for i, _ in enumerate(self.inputs_offsets):
      name = 'WSM' + ('' if i == 0 else '_' + str(i))
      self.vars.append(
          self.add_weight(initializer=initializer, name=name, shape=shape))

  def build(self, feats_shape):
    for i, input_offset in enumerate(self.inputs_offsets):
      name = 'resample_{}_{}_{}'.format(i, input_offset, len(feats_shape))
      self.resample_layers.append(
          ResampleFeatureMap(
              self.feat_level,
              self.fpn_num_filters,
              self.apply_bn_for_resampling,
              self.is_training_bn,
              self.conv_after_downsample,
              strategy=self.strategy,
              data_format=self.data_format,
              pooling_type=self.pooling_type,
              upsampling_type=self.upsampling_type,
              name=name))
    if self.weight_method == 'attn':
      self._add_wsm('ones')
    elif self.weight_method == 'fastattn':
      self._add_wsm('ones')
    elif self.weight_method == 'channel_attn':
      num_filters = int(self.fpn_num_filters)
      self._add_wsm(tf.ones, num_filters)
    elif self.weight_method == 'channel_fastattn':
      num_filters = int(self.fpn_num_filters)
      self._add_wsm(tf.ones, num_filters)
    self.op_after_combine = OpAfterCombine(
        self.is_training_bn,
        self.conv_bn_act_pattern,
        self.conv_type,
        self.fpn_num_filters,
        self.act_type,
        self.data_format,
        self.strategy,
        name='op_after_combine{}'.format(len(feats_shape)))
    self.built = True
    super().build(feats_shape)

  def call(self, feats, training):
    nodes = []
    for i, input_offset in enumerate(self.inputs_offsets):
      input_node = feats[input_offset]
      input_node = self.resample_layers[i](input_node, training, feats)
      nodes.append(input_node)
    new_node = self.fuse_features(nodes)
    new_node = self.op_after_combine(new_node)
    return feats + [new_node]


class OpAfterCombine(tf_keras.layers.Layer):
  """Operation after combining input features during feature fusiong."""

  def __init__(self,
               is_training_bn,
               conv_bn_act_pattern,
               conv_type,
               fpn_num_filters,
               act_type,
               data_format,
               strategy,
               name='op_after_combine'):
    super().__init__(name=name)
    self.conv_bn_act_pattern = conv_bn_act_pattern
    self.fpn_num_filters = fpn_num_filters
    self.act_type = act_type
    self.data_format = data_format
    self.strategy = strategy
    self.is_training_bn = is_training_bn
    self.conv_op = get_conv_op(conv_type)(
        filters=fpn_num_filters,
        padding='same',
        use_bias=not self.conv_bn_act_pattern,
        data_format=self.data_format,
        name='conv')
    self.bn = build_batch_norm(
        is_training_bn=self.is_training_bn,
        data_format=self.data_format,
        strategy=self.strategy,
        name='bn')

  def call(self, new_node, training):
    if not self.conv_bn_act_pattern:
      new_node = activation_fn(new_node, self.act_type)
    new_node = self.conv_op(new_node)
    new_node = self.bn(new_node, training=training)
    if self.conv_bn_act_pattern:
      new_node = activation_fn(new_node, self.act_type)
    return new_node


class FPNCells(tf_keras.layers.Layer):
  """FPN cells."""

  def __init__(self,
               min_level=3,
               max_level=8,
               fpn_num_filters=96,
               apply_bn_for_resampling=True,
               is_training_bn=True,
               conv_after_downsample=True,
               conv_bn_act_pattern=True,
               conv_type='sep_3',
               act_type='swish',
               strategy='tpu',
               fpn_weight_method='sum',
               data_format='channels_last',
               pooling_type='avg',
               upsampling_type='bilinear',
               fpn_name='bifpn',
               fpn_cell_repeats=4,
               **kwargs):
    super(FPNCells, self).__init__(**kwargs)
    self.min_level = min_level
    self.max_level = max_level
    if fpn_name != 'bifpn':
      raise ValueError('Only bifpn config is supported.')
    self.fpn_config = bifpn_config(min_level, max_level)
    self.cells = [
        FPNCell(  # pylint: disable=g-complex-comprehension
            min_level=min_level,
            max_level=max_level,
            fpn_num_filters=fpn_num_filters,
            apply_bn_for_resampling=apply_bn_for_resampling,
            is_training_bn=is_training_bn,
            conv_after_downsample=conv_after_downsample,
            conv_bn_act_pattern=conv_bn_act_pattern,
            conv_type=conv_type,
            act_type=act_type,
            strategy=strategy,
            fpn_weight_method=fpn_weight_method,
            data_format=data_format,
            pooling_type=pooling_type,
            upsampling_type=upsampling_type,
            fpn_name=fpn_name,
            name='cell_%d' % rep) for rep in range(fpn_cell_repeats)
    ]

  def call(self, feats, training):
    """Model call function."""
    for cell in self.cells:
      cell_feats = cell(feats, training)
      min_level = self.min_level
      max_level = self.max_level

      feats = []
      for level in range(min_level, max_level + 1):
        for i, fnode in enumerate(reversed(self.fpn_config['nodes'])):
          if fnode['feat_level'] == level:
            feats.append(cell_feats[-1 - i])
            break

    return feats


class FPNCell(tf_keras.layers.Layer):
  """A single FPN cell."""

  def __init__(self,
               min_level=3,
               max_level=7,
               fpn_num_filters=80,
               apply_bn_for_resampling=True,
               is_training_bn=True,
               conv_after_downsample=True,
               conv_bn_act_pattern=True,
               conv_type='sep_3',
               act_type='swish',
               strategy='tpu',
               fpn_weight_method='sum',
               data_format='channels_last',
               pooling_type='avg',
               upsampling_type='bilinear',
               fpn_name='bifpn',
               name='fpn_cell',
               **kwargs):
    super(FPNCell, self).__init__(**kwargs)
    if fpn_name != 'bifpn':
      raise ValueError('Only bifpn config is supported')
    self.fpn_config = bifpn_config(min_level, max_level)
    self.fnodes = []
    for i, fnode_cfg in enumerate(self.fpn_config['nodes']):
      logging.info('fnode %d : %s', i, fnode_cfg)
      fnode = FNode(
          fnode_cfg['feat_level'] - min_level,
          fnode_cfg['inputs_offsets'],
          fpn_num_filters=fpn_num_filters,
          apply_bn_for_resampling=apply_bn_for_resampling,
          is_training_bn=is_training_bn,
          conv_after_downsample=conv_after_downsample,
          conv_bn_act_pattern=conv_bn_act_pattern,
          conv_type=conv_type,
          act_type=act_type,
          strategy=strategy,
          weight_method=fpn_weight_method,
          data_format=data_format,
          pooling_type=pooling_type,
          upsampling_type=upsampling_type,
          name='fnode%d' % i)
      self.fnodes.append(fnode)

  def call(self, feats, training):
    def _call(feats):
      for fnode in self.fnodes:
        feats = fnode(feats, training)
      return feats
    return _call(feats)


class SegClassNet(tf_keras.layers.Layer):
  """Segmentation class prediction network."""

  def __init__(self,
               min_level=3,
               max_level=7,
               output_filters=256,
               apply_bn_for_resampling=True,
               is_training_bn=True,
               conv_after_downsample=True,
               conv_bn_act_pattern=True,
               head_conv_type='sep_3',
               act_type='swish',
               strategy='tpu',
               output_weight_method='attn',
               data_format='channels_last',
               pooling_type='avg',
               upsampling_type='bilinear',
               fullres_output=False,
               fullres_skip_connections=False,
               num_classes=32,
               name='seg_class_net'):
    """Initialize the SegClassNet.

    Args:
      min_level: minimum feature level to use in the head.
      max_level: maximum feature level to use in the head.
      output_filters: output filter size.
      apply_bn_for_resampling:
      whether to apply batch normalization for resampling.
      is_training_bn: is training mode.
      conv_after_downsample: whether to apply conv after downsample.
      conv_bn_act_pattern: conv batch norm activation pattern.
      head_conv_type: head convolution type.
      act_type: activation type.
      strategy: device strategy, eg. tpu.
      output_weight_method: output weight method.
      data_format: data format.
      pooling_type: pooling type.
      upsampling_type: upsamplihng type.
      fullres_output: full resolution output.
      fullres_skip_connections: full resolution skip connection.
      num_classes: number of classes.
      name: the name of this layer.
    """

    super().__init__(name=name)
    conv2d_layer = get_conv_op(head_conv_type)
    self.min_level = min_level
    self.max_level = max_level
    self.fullres_output = fullres_output
    self.fullres_skip_connections = fullres_skip_connections

    self.fnode = FNode(
        0,  # Always use the first level with highest resolution.
        list(range(max_level - min_level + 1)),
        output_filters,
        apply_bn_for_resampling,
        is_training_bn,
        conv_after_downsample,
        conv_bn_act_pattern,
        head_conv_type,
        act_type,
        strategy,
        output_weight_method,
        data_format,
        pooling_type,
        upsampling_type,
        name='seg_class_fusion')

    if fullres_output:
      self.fullres_conv_transpose = {}
      self.fullres_conv = {}
      for i in reversed(range(min_level)):
        num_filters = min(num_classes * 2**(i + 1),
                          output_filters)
        self.fullres_conv[str(i)] = conv2d_layer(
            filters=num_filters,
            data_format=data_format,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=act_type,
            name='fullres_conv_%d' % i)
        self.fullres_conv_transpose[str(i)] = tf_keras.layers.Conv2DTranspose(
            filters=num_filters,
            data_format=data_format,
            kernel_size=3,
            strides=2,
            padding='same',
            activation=act_type,
            name='fullres_conv_transpose_%d' % i)

    self.classes = conv2d_layer(
        num_classes,
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        padding='same',
        name='seg-class-predict')

  def call(self, inputs, backbone_feats, training):
    """Call SegClassNet."""

    seg_output = self.fnode(inputs, training)
    net = seg_output[-1]

    if self.fullres_output:
      for i in reversed(range(self.min_level)):
        if self.fullres_skip_connections:
          net = tf_keras.layers.Concatenate()([net, backbone_feats[i + 1]])
        net = self.fullres_conv[str(i)](net)
        net = self.fullres_conv_transpose[str(i)](net)

    class_outputs = self.classes(net)
    return class_outputs
