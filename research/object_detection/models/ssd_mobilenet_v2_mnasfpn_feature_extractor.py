# Lint as: python2, python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""SSD MobilenetV2 NAS-FPN Feature Extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
from six.moves import range
import tensorflow as tf

from tensorflow.contrib import slim as contrib_slim
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets.mobilenet import mobilenet
from nets.mobilenet import mobilenet_v2

slim = contrib_slim

Block = collections.namedtuple(
    'Block', ['inputs', 'output_level', 'kernel_size', 'expansion_size'])

_MNASFPN_CELL_CONFIG = [
    Block(inputs=(1, 2), output_level=4, kernel_size=3, expansion_size=256),
    Block(inputs=(0, 4), output_level=3, kernel_size=3, expansion_size=128),
    Block(inputs=(5, 4), output_level=4, kernel_size=3, expansion_size=128),
    Block(inputs=(4, 3), output_level=5, kernel_size=5, expansion_size=128),
    Block(inputs=(4, 3), output_level=6, kernel_size=3, expansion_size=96),
]

MNASFPN_DEF = dict(
    feature_levels=[3, 4, 5, 6],
    spec=[_MNASFPN_CELL_CONFIG] * 4,
)


def _maybe_pad(feature, use_explicit_padding, kernel_size=3):
  return ops.fixed_padding(feature,
                           kernel_size) if use_explicit_padding else feature


# Wrapper around mobilenet.depth_multiplier
def _apply_multiplier(d, multiplier, min_depth):
  p = {'num_outputs': d}
  mobilenet.depth_multiplier(
      p, multiplier=multiplier, divisible_by=8, min_depth=min_depth)
  return p['num_outputs']


def _apply_size_dependent_ordering(input_feature, feature_level, block_level,
                                   expansion_size, use_explicit_padding,
                                   use_native_resize_op):
  """Applies Size-Dependent-Ordering when resizing feature maps.

     See https://arxiv.org/abs/1912.01106

  Args:
    input_feature: input feature map to be resized.
    feature_level: the level of the input feature.
    block_level: the desired output level for the block.
    expansion_size: the expansion size for the block.
    use_explicit_padding: Whether to use explicit padding.
    use_native_resize_op: Whether to use native resize op.

  Returns:
    A transformed feature at the desired resolution and expansion size.
  """
  padding = 'VALID' if use_explicit_padding else 'SAME'
  if feature_level >= block_level:  # Perform 1x1 then upsampling.
    node = slim.conv2d(
        input_feature,
        expansion_size, [1, 1],
        activation_fn=None,
        normalizer_fn=slim.batch_norm,
        padding=padding,
        scope='Conv1x1')
    if feature_level == block_level:
      return node
    scale = 2**(feature_level - block_level)
    if use_native_resize_op:
      input_shape = shape_utils.combined_static_and_dynamic_shape(node)
      node = tf.image.resize_nearest_neighbor(
          node, [input_shape[1] * scale, input_shape[2] * scale])
    else:
      node = ops.nearest_neighbor_upsampling(node, scale=scale)
  else:  # Perform downsampling then 1x1.
    stride = 2**(block_level - feature_level)
    node = slim.max_pool2d(
        _maybe_pad(input_feature, use_explicit_padding), [3, 3],
        stride=[stride, stride],
        padding=padding,
        scope='Downsample')
    node = slim.conv2d(
        node,
        expansion_size, [1, 1],
        activation_fn=None,
        normalizer_fn=slim.batch_norm,
        padding=padding,
        scope='Conv1x1')
  return node


def _mnasfpn_cell(feature_maps,
                  feature_levels,
                  cell_spec,
                  output_channel=48,
                  use_explicit_padding=False,
                  use_native_resize_op=False,
                  multiplier_func=None):
  """Create a MnasFPN cell.

  Args:
    feature_maps: input feature maps.
    feature_levels: levels of the feature maps.
    cell_spec: A list of Block configs.
    output_channel: Number of features for the input, output and intermediate
      feature maps.
    use_explicit_padding: Whether to use explicit padding.
    use_native_resize_op: Whether to use native resize op.
    multiplier_func: Depth-multiplier function. If None, use identity function.

  Returns:
    A transformed list of feature maps at the same resolutions as the inputs.
  """
  # This is the level where multipliers are realized.
  if multiplier_func is None:
    multiplier_func = lambda x: x
  num_outputs = len(feature_maps)
  cell_features = list(feature_maps)
  cell_levels = list(feature_levels)
  padding = 'VALID' if use_explicit_padding else 'SAME'
  for bi, block in enumerate(cell_spec):
    with tf.variable_scope('block_{}'.format(bi)):
      block_level = block.output_level
      intermediate_feature = None
      for i, inp in enumerate(block.inputs):
        with tf.variable_scope('input_{}'.format(i)):
          input_level = cell_levels[inp]
          node = _apply_size_dependent_ordering(
              cell_features[inp], input_level, block_level,
              multiplier_func(block.expansion_size), use_explicit_padding,
              use_native_resize_op)
        # Add features incrementally to avoid producing AddN, which doesn't
        # play well with TfLite.
        if intermediate_feature is None:
          intermediate_feature = node
        else:
          intermediate_feature += node
      node = tf.nn.relu6(intermediate_feature)
      node = slim.separable_conv2d(
          _maybe_pad(node, use_explicit_padding, block.kernel_size),
          multiplier_func(output_channel),
          block.kernel_size,
          activation_fn=None,
          normalizer_fn=slim.batch_norm,
          padding=padding,
          scope='SepConv')
    cell_features.append(node)
    cell_levels.append(block_level)

  # Cell-wide residuals.
  out_idx = range(len(cell_features) - num_outputs, len(cell_features))
  for in_i, out_i in enumerate(out_idx):
    if cell_features[out_i].shape.as_list(
    ) == cell_features[in_i].shape.as_list():
      cell_features[out_i] += cell_features[in_i]

  return cell_features[-num_outputs:]


def mnasfpn(feature_maps,
            head_def,
            output_channel=48,
            use_explicit_padding=False,
            use_native_resize_op=False,
            multiplier_func=None):
  """Create the MnasFPN head given head_def."""
  features = feature_maps
  for ci, cell_spec in enumerate(head_def['spec']):
    with tf.variable_scope('cell_{}'.format(ci)):
      features = _mnasfpn_cell(features, head_def['feature_levels'], cell_spec,
                               output_channel, use_explicit_padding,
                               use_native_resize_op, multiplier_func)
  return features


def training_scope(l2_weight_decay=1e-4, is_training=None):
  """Arg scope for training MnasFPN."""
  with slim.arg_scope(
      [slim.conv2d],
      weights_initializer=tf.initializers.he_normal(),
      weights_regularizer=slim.l2_regularizer(l2_weight_decay)), \
      slim.arg_scope(
          [slim.separable_conv2d],
          weights_initializer=tf.initializers.truncated_normal(
              stddev=0.536),  # He_normal for 3x3 depthwise kernel.
          weights_regularizer=slim.l2_regularizer(l2_weight_decay)), \
      slim.arg_scope([slim.batch_norm],
                     is_training=is_training,
                     epsilon=0.01,
                     decay=0.99,
                     center=True,
                     scale=True) as s:
    return s


class SSDMobileNetV2MnasFPNFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD Feature Extractor using MobilenetV2 MnasFPN features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               fpn_min_level=3,
               fpn_max_level=6,
               additional_layer_depth=48,
               head_def=None,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               use_native_resize_op=False,
               override_base_feature_extractor_hyperparams=False,
               data_format='channels_last'):
    """SSD MnasFPN feature extractor based on Mobilenet v2 architecture.

    See https://arxiv.org/abs/1912.01106

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the base
        feature extractor.
      fpn_min_level: the highest resolution feature map to use in MnasFPN.
        Currently the only valid value is 3.
      fpn_max_level: the smallest resolution feature map to construct or use in
        MnasFPN. Currentl the only valid value is 6.
      additional_layer_depth: additional feature map layer channel depth for
        NAS-FPN.
      head_def: A dictionary specifying the MnasFPN head architecture. Default
        uses MNASFPN_DEF.
      reuse_weights: whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      use_native_resize_op: Whether to use native resize op. Default is False.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
      data_format: The ordering of the dimensions in the inputs, The valid
        values are {'channels_first', 'channels_last').
    """
    super(SSDMobileNetV2MnasFPNFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=(
            override_base_feature_extractor_hyperparams))
    if fpn_min_level != 3 or fpn_max_level != 6:
      raise ValueError('Min and max levels of MnasFPN must be 3 and 6 for now.')
    self._fpn_min_level = fpn_min_level
    self._fpn_max_level = fpn_max_level
    self._fpn_layer_depth = additional_layer_depth
    self._head_def = head_def if head_def else MNASFPN_DEF
    self._data_format = data_format
    self._use_native_resize_op = use_native_resize_op

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def _verify_config(self, inputs):
    """Verify that MnasFPN config and its inputs."""
    num_inputs = len(inputs)
    assert len(self._head_def['feature_levels']) == num_inputs

    base_width = inputs[0].shape.as_list(
    )[1] * 2**self._head_def['feature_levels'][0]
    for i in range(1, num_inputs):
      width = inputs[i].shape.as_list()[1]
      level = self._head_def['feature_levels'][i]
      expected_width = base_width // 2**level
      if width != expected_width:
        raise ValueError(
            'Resolution of input {} does not match its level {}.'.format(
                i, level))

    for cell_spec in self._head_def['spec']:
      # The last K nodes in a cell are the inputs to the next cell. Assert that
      # their feature maps are at the right level.
      for i in range(num_inputs):
        if cell_spec[-num_inputs +
                     i].output_level != self._head_def['feature_levels'][i]:
          raise ValueError(
              'Mismatch between node level {} and desired output level {}.'
              .format(cell_spec[-num_inputs + i].output_level,
                      self._head_def['feature_levels'][i]))
      # Assert that each block only uses precending blocks.
      for bi, block_spec in enumerate(cell_spec):
        for inp in block_spec.inputs:
          if inp >= bi + num_inputs:
            raise ValueError(
                'Block {} is trying to access uncreated block {}.'.format(
                    bi, inp))

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs = shape_utils.check_min_image_dim(
        33, preprocessed_inputs)
    with tf.variable_scope('MobilenetV2', reuse=self._reuse_weights) as scope:
      with slim.arg_scope(
          mobilenet_v2.training_scope(is_training=None, bn_decay=0.99)), \
          slim.arg_scope(
              [mobilenet.depth_multiplier], min_depth=self._min_depth):
        with slim.arg_scope(
            training_scope(l2_weight_decay=4e-5,
                           is_training=self._is_training)):

          _, image_features = mobilenet_v2.mobilenet_base(
              ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
              final_endpoint='layer_18',
              depth_multiplier=self._depth_multiplier,
              use_explicit_padding=self._use_explicit_padding,
              scope=scope)

    multiplier_func = functools.partial(
        _apply_multiplier,
        multiplier=self._depth_multiplier,
        min_depth=self._min_depth)
    with tf.variable_scope('MnasFPN', reuse=self._reuse_weights):
      with slim.arg_scope(
          training_scope(l2_weight_decay=1e-4, is_training=self._is_training)):
        # Create C6 by downsampling C5.
        c6 = slim.max_pool2d(
            _maybe_pad(image_features['layer_18'], self._use_explicit_padding),
            [3, 3],
            stride=[2, 2],
            padding='VALID' if self._use_explicit_padding else 'SAME',
            scope='C6_downsample')
        c6 = slim.conv2d(
            c6,
            multiplier_func(self._fpn_layer_depth),
            [1, 1],
            activation_fn=tf.identity,
            normalizer_fn=slim.batch_norm,
            weights_regularizer=None,  # this 1x1 has no kernel regularizer.
            padding='VALID',
            scope='C6_Conv1x1')
        image_features['C6'] = tf.identity(c6)  # Needed for quantization.
        for k in sorted(image_features.keys()):
          tf.logging.error('{}: {}'.format(k, image_features[k]))

        mnasfpn_inputs = [
            image_features['layer_7'],  # C3
            image_features['layer_14'],  # C4
            image_features['layer_18'],  # C5
            image_features['C6']  # C6
        ]
        self._verify_config(mnasfpn_inputs)
        feature_maps = mnasfpn(
            mnasfpn_inputs,
            head_def=self._head_def,
            output_channel=self._fpn_layer_depth,
            use_explicit_padding=self._use_explicit_padding,
            use_native_resize_op=self._use_native_resize_op,
            multiplier_func=multiplier_func)
    return feature_maps
