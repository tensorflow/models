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
"""SSDFeatureExtractor for MobileDet features."""

import functools
import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import ops
from object_detection.utils import shape_utils


BACKBONE_WEIGHT_DECAY = 4e-5


def _scale_filters(filters, multiplier, base=8):
  """Scale the filters accordingly to (multiplier, base)."""
  round_half_up = int(int(filters) * multiplier / base + 0.5)
  result = int(round_half_up * base)
  return max(result, base)


def _swish6(h):
  with tf.name_scope('swish6'):
    return h * tf.nn.relu6(h + np.float32(3)) * np.float32(1. / 6.)


def _conv(h, filters, kernel_size, strides=1,
          normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu6):
  if activation_fn is None:
    raise ValueError('Activation function cannot be None. Use tf.identity '
                     'instead to better support quantized training.')
  return slim.conv2d(
      h,
      filters,
      kernel_size,
      stride=strides,
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,
      weights_initializer=tf.initializers.he_normal(),
      weights_regularizer=slim.l2_regularizer(BACKBONE_WEIGHT_DECAY),
      padding='SAME')


def _separable_conv(
    h, filters, kernel_size, strides=1, activation_fn=tf.nn.relu6):
  """Separable convolution layer."""
  if activation_fn is None:
    raise ValueError('Activation function cannot be None. Use tf.identity '
                     'instead to better support quantized training.')
  # Depthwise variant of He initialization derived under the principle proposed
  # in the original paper. Note the original He normalization was designed for
  # full convolutions and calling tf.initializers.he_normal() can over-estimate
  # the fan-in of a depthwise kernel by orders of magnitude.
  stddev = (2.0 / kernel_size**2)**0.5 / .87962566103423978
  depthwise_initializer = tf.initializers.truncated_normal(stddev=stddev)
  return slim.separable_conv2d(
      h,
      filters,
      kernel_size,
      stride=strides,
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm,
      weights_initializer=depthwise_initializer,
      pointwise_initializer=tf.initializers.he_normal(),
      weights_regularizer=slim.l2_regularizer(BACKBONE_WEIGHT_DECAY),
      padding='SAME')


def _squeeze_and_excite(h, hidden_dim, activation_fn=tf.nn.relu6):
  with tf.variable_scope(None, default_name='SqueezeExcite'):
    height, width = h.shape[1], h.shape[2]
    u = slim.avg_pool2d(h, [height, width], stride=1, padding='VALID')
    u = _conv(u, hidden_dim, 1,
              normalizer_fn=None, activation_fn=activation_fn)
    u = _conv(u, h.shape[-1], 1,
              normalizer_fn=None, activation_fn=tf.nn.sigmoid)
    return u * h


def _inverted_bottleneck_no_expansion(
    h, filters, activation_fn=tf.nn.relu6,
    kernel_size=3, strides=1, use_se=False):
  """Inverted bottleneck layer without the first 1x1 expansion convolution."""
  with tf.variable_scope(None, default_name='IBNNoExpansion'):
    # Setting filters to None will make _separable_conv a depthwise conv.
    h = _separable_conv(
        h, None, kernel_size, strides=strides, activation_fn=activation_fn)
    if use_se:
      hidden_dim = _scale_filters(h.shape[-1], 0.25)
      h = _squeeze_and_excite(h, hidden_dim, activation_fn=activation_fn)
    h = _conv(h, filters, 1, activation_fn=tf.identity)
    return h


def _inverted_bottleneck(
    h, filters, activation_fn=tf.nn.relu6,
    kernel_size=3, expansion=8, strides=1, use_se=False, residual=True):
  """Inverted bottleneck layer."""
  with tf.variable_scope(None, default_name='IBN'):
    shortcut = h
    expanded_filters = int(h.shape[-1]) * expansion
    if expansion <= 1:
      raise ValueError('Expansion factor must be greater than 1.')
    h = _conv(h, expanded_filters, 1, activation_fn=activation_fn)
    # Setting filters to None will make _separable_conv a depthwise conv.
    h = _separable_conv(h, None, kernel_size, strides=strides,
                        activation_fn=activation_fn)
    if use_se:
      hidden_dim = _scale_filters(expanded_filters, 0.25)
      h = _squeeze_and_excite(h, hidden_dim, activation_fn=activation_fn)
    h = _conv(h, filters, 1, activation_fn=tf.identity)
    if residual:
      h = h + shortcut
    return h


def _fused_conv(
    h, filters, activation_fn=tf.nn.relu6,
    kernel_size=3, expansion=8, strides=1, use_se=False, residual=True):
  """Fused convolution layer."""
  with tf.variable_scope(None, default_name='FusedConv'):
    shortcut = h
    expanded_filters = int(h.shape[-1]) * expansion
    if expansion <= 1:
      raise ValueError('Expansion factor must be greater than 1.')
    h = _conv(h, expanded_filters, kernel_size, strides=strides,
              activation_fn=activation_fn)
    if use_se:
      hidden_dim = _scale_filters(expanded_filters, 0.25)
      h = _squeeze_and_excite(h, hidden_dim, activation_fn=activation_fn)
    h = _conv(h, filters, 1, activation_fn=tf.identity)
    if residual:
      h = h + shortcut
    return h


def _tucker_conv(
    h, filters, activation_fn=tf.nn.relu6,
    kernel_size=3, input_rank_ratio=0.25, output_rank_ratio=0.25,
    strides=1, residual=True):
  """Tucker convolution layer (generalized bottleneck)."""
  with tf.variable_scope(None, default_name='TuckerConv'):
    shortcut = h
    input_rank = _scale_filters(h.shape[-1], input_rank_ratio)
    h = _conv(h, input_rank, 1, activation_fn=activation_fn)
    output_rank = _scale_filters(filters, output_rank_ratio)
    h = _conv(h, output_rank, kernel_size, strides=strides,
              activation_fn=activation_fn)
    h = _conv(h, filters, 1, activation_fn=tf.identity)
    if residual:
      h = h + shortcut
    return h


def mobiledet_cpu_backbone(h, multiplier=1.0):
  """Build a MobileDet CPU backbone."""
  def _scale(filters):
    return _scale_filters(filters, multiplier)
  ibn = functools.partial(
      _inverted_bottleneck, use_se=True, activation_fn=_swish6)

  endpoints = {}
  h = _conv(h, _scale(16), 3, strides=2, activation_fn=_swish6)
  h = _inverted_bottleneck_no_expansion(
      h, _scale(8), use_se=True, activation_fn=_swish6)
  endpoints['C1'] = h
  h = ibn(h, _scale(16), expansion=4, strides=2, residual=False)
  endpoints['C2'] = h
  h = ibn(h, _scale(32), expansion=8, strides=2, residual=False)
  h = ibn(h, _scale(32), expansion=4)
  h = ibn(h, _scale(32), expansion=4)
  h = ibn(h, _scale(32), expansion=4)
  endpoints['C3'] = h
  h = ibn(h, _scale(72), kernel_size=5, expansion=8, strides=2, residual=False)
  h = ibn(h, _scale(72), expansion=8)
  h = ibn(h, _scale(72), kernel_size=5, expansion=4)
  h = ibn(h, _scale(72), expansion=4)
  h = ibn(h, _scale(72), expansion=8, residual=False)
  h = ibn(h, _scale(72), expansion=8)
  h = ibn(h, _scale(72), expansion=8)
  h = ibn(h, _scale(72), expansion=8)
  endpoints['C4'] = h
  h = ibn(h, _scale(104), kernel_size=5, expansion=8, strides=2, residual=False)
  h = ibn(h, _scale(104), kernel_size=5, expansion=4)
  h = ibn(h, _scale(104), kernel_size=5, expansion=4)
  h = ibn(h, _scale(104), expansion=4)
  h = ibn(h, _scale(144), expansion=8, residual=False)
  endpoints['C5'] = h
  return endpoints


def mobiledet_dsp_backbone(h, multiplier=1.0):
  """Build a MobileDet DSP backbone."""
  def _scale(filters):
    return _scale_filters(filters, multiplier)

  ibn = functools.partial(_inverted_bottleneck, activation_fn=tf.nn.relu6)
  fused = functools.partial(_fused_conv, activation_fn=tf.nn.relu6)
  tucker = functools.partial(_tucker_conv, activation_fn=tf.nn.relu6)

  endpoints = {}
  h = _conv(h, _scale(32), 3, strides=2, activation_fn=tf.nn.relu6)
  h = _inverted_bottleneck_no_expansion(
      h, _scale(24), activation_fn=tf.nn.relu6)
  endpoints['C1'] = h
  h = fused(h, _scale(32), expansion=4, strides=2, residual=False)
  h = fused(h, _scale(32), expansion=4)
  h = ibn(h, _scale(32), expansion=4)
  h = tucker(h, _scale(32), input_rank_ratio=0.25, output_rank_ratio=0.75)
  endpoints['C2'] = h
  h = fused(h, _scale(64), expansion=8, strides=2, residual=False)
  h = ibn(h, _scale(64), expansion=4)
  h = fused(h, _scale(64), expansion=4)
  h = fused(h, _scale(64), expansion=4)
  endpoints['C3'] = h
  h = fused(h, _scale(120), expansion=8, strides=2, residual=False)
  h = ibn(h, _scale(120), expansion=4)
  h = ibn(h, _scale(120), expansion=8)
  h = ibn(h, _scale(120), expansion=8)
  h = fused(h, _scale(144), expansion=8, residual=False)
  h = ibn(h, _scale(144), expansion=8)
  h = ibn(h, _scale(144), expansion=8)
  h = ibn(h, _scale(144), expansion=8)
  endpoints['C4'] = h
  h = ibn(h, _scale(160), expansion=4, strides=2, residual=False)
  h = ibn(h, _scale(160), expansion=4)
  h = fused(h, _scale(160), expansion=4)
  h = tucker(h, _scale(160), input_rank_ratio=0.75, output_rank_ratio=0.75)
  h = ibn(h, _scale(240), expansion=8, residual=False)
  endpoints['C5'] = h
  return endpoints


def mobiledet_edgetpu_backbone(h, multiplier=1.0):
  """Build a MobileDet EdgeTPU backbone."""
  def _scale(filters):
    return _scale_filters(filters, multiplier)

  ibn = functools.partial(_inverted_bottleneck, activation_fn=tf.nn.relu6)
  fused = functools.partial(_fused_conv, activation_fn=tf.nn.relu6)
  tucker = functools.partial(_tucker_conv, activation_fn=tf.nn.relu6)

  endpoints = {}
  h = _conv(h, _scale(32), 3, strides=2, activation_fn=tf.nn.relu6)
  h = tucker(h, _scale(16),
             input_rank_ratio=0.25, output_rank_ratio=0.75, residual=False)
  endpoints['C1'] = h
  h = fused(h, _scale(16), expansion=8, strides=2, residual=False)
  h = fused(h, _scale(16), expansion=4)
  h = fused(h, _scale(16), expansion=8)
  h = fused(h, _scale(16), expansion=4)
  endpoints['C2'] = h
  h = fused(h, _scale(40), expansion=8, kernel_size=5, strides=2,
            residual=False)
  h = fused(h, _scale(40), expansion=4)
  h = fused(h, _scale(40), expansion=4)
  h = fused(h, _scale(40), expansion=4)
  endpoints['C3'] = h
  h = ibn(h, _scale(72), expansion=8, strides=2, residual=False)
  h = ibn(h, _scale(72), expansion=8)
  h = fused(h, _scale(72), expansion=4)
  h = fused(h, _scale(72), expansion=4)
  h = ibn(h, _scale(96), expansion=8, kernel_size=5, residual=False)
  h = ibn(h, _scale(96), expansion=8, kernel_size=5)
  h = ibn(h, _scale(96), expansion=8)
  h = ibn(h, _scale(96), expansion=8)
  endpoints['C4'] = h
  h = ibn(h, _scale(120), expansion=8, kernel_size=5, strides=2, residual=False)
  h = ibn(h, _scale(120), expansion=8)
  h = ibn(h, _scale(120), expansion=4, kernel_size=5)
  h = ibn(h, _scale(120), expansion=8)
  h = ibn(h, _scale(384), expansion=8, kernel_size=5, residual=False)
  endpoints['C5'] = h
  return endpoints


def mobiledet_gpu_backbone(h, multiplier=1.0):
  """Build a MobileDet GPU backbone."""

  def _scale(filters):
    return _scale_filters(filters, multiplier)

  ibn = functools.partial(_inverted_bottleneck, activation_fn=tf.nn.relu6)
  fused = functools.partial(_fused_conv, activation_fn=tf.nn.relu6)
  tucker = functools.partial(_tucker_conv, activation_fn=tf.nn.relu6)

  endpoints = {}
  # block 0
  h = _conv(h, _scale(32), 3, strides=2, activation_fn=tf.nn.relu6)

  # block 1
  h = tucker(
      h,
      _scale(16),
      input_rank_ratio=0.25,
      output_rank_ratio=0.25,
      residual=False)
  endpoints['C1'] = h

  # block 2
  h = fused(h, _scale(32), expansion=8, strides=2, residual=False)
  h = tucker(h, _scale(32), input_rank_ratio=0.25, output_rank_ratio=0.25)
  h = tucker(h, _scale(32), input_rank_ratio=0.25, output_rank_ratio=0.25)
  h = tucker(h, _scale(32), input_rank_ratio=0.25, output_rank_ratio=0.25)
  endpoints['C2'] = h

  # block 3
  h = fused(
      h, _scale(64), expansion=8, kernel_size=3, strides=2, residual=False)
  h = fused(h, _scale(64), expansion=8)
  h = fused(h, _scale(64), expansion=8)
  h = fused(h, _scale(64), expansion=4)
  endpoints['C3'] = h

  # block 4
  h = fused(
      h, _scale(128), expansion=8, kernel_size=3, strides=2, residual=False)
  h = fused(h, _scale(128), expansion=4)
  h = fused(h, _scale(128), expansion=4)
  h = fused(h, _scale(128), expansion=4)

  # block 5
  h = fused(
      h, _scale(128), expansion=8, kernel_size=3, strides=1, residual=False)
  h = fused(h, _scale(128), expansion=8)
  h = fused(h, _scale(128), expansion=8)
  h = fused(h, _scale(128), expansion=8)
  endpoints['C4'] = h

  # block 6
  h = fused(
      h, _scale(128), expansion=4, kernel_size=3, strides=2, residual=False)
  h = fused(h, _scale(128), expansion=4)
  h = fused(h, _scale(128), expansion=4)
  h = fused(h, _scale(128), expansion=4)

  # block 7
  h = ibn(h, _scale(384), expansion=8, kernel_size=3, strides=1, residual=False)
  endpoints['C5'] = h
  return endpoints


class SSDMobileDetFeatureExtractorBase(ssd_meta_arch.SSDFeatureExtractor):
  """Base class of SSD feature extractor using MobileDet features."""

  def __init__(self,
               backbone_fn,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False,
               scope_name='MobileDet'):
    """MobileDet Feature Extractor for SSD Models.

    Reference:
      https://arxiv.org/abs/2004.14525

    Args:
      backbone_fn: function to construct the MobileDet backbone.
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: Integer, minimum feature extractor depth (number of filters).
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the base
        feature extractor.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features.
      use_depthwise: Whether to use depthwise convolutions in the SSD head.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
      scope_name: scope name (string) of network variables.
    """
    if use_explicit_padding:
      raise NotImplementedError(
          'Explicit padding is not yet supported in MobileDet backbones.')

    super(SSDMobileDetFeatureExtractorBase, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams
    )
    self._backbone_fn = backbone_fn
    self._scope_name = scope_name

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1]. The preprocessing assumes an input
    value range of [0, 255].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

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
    padded_inputs = ops.pad_to_multiple(
        preprocessed_inputs, self._pad_to_multiple)

    feature_map_layout = {
        'from_layer': ['C4', 'C5', '', '', '', ''],
        # Do not specify the layer depths (number of filters) for C4 and C5, as
        # their values are determined based on the backbone.
        'layer_depth': [-1, -1, 512, 256, 256, 128],
        'use_depthwise': self._use_depthwise,
        'use_explicit_padding': self._use_explicit_padding,
    }

    with tf.variable_scope(self._scope_name, reuse=self._reuse_weights):
      with slim.arg_scope([slim.batch_norm],
                          is_training=self._is_training,
                          epsilon=0.01, decay=0.99, center=True, scale=True):
        endpoints = self._backbone_fn(
            padded_inputs,
            multiplier=self._depth_multiplier)

      image_features = {'C4': endpoints['C4'], 'C5': endpoints['C5']}
      with slim.arg_scope(self._conv_hyperparams_fn()):
        feature_maps = feature_map_generators.multi_resolution_feature_maps(
            feature_map_layout=feature_map_layout,
            depth_multiplier=self._depth_multiplier,
            min_depth=self._min_depth,
            insert_1x1_conv=True,
            image_features=image_features)

    return list(feature_maps.values())


class SSDMobileDetCPUFeatureExtractor(SSDMobileDetFeatureExtractorBase):
  """MobileDet-CPU feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False,
               scope_name='MobileDetCPU'):
    super(SSDMobileDetCPUFeatureExtractor, self).__init__(
        backbone_fn=mobiledet_cpu_backbone,
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams,
        scope_name=scope_name)


class SSDMobileDetDSPFeatureExtractor(SSDMobileDetFeatureExtractorBase):
  """MobileDet-DSP feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False,
               scope_name='MobileDetDSP'):
    super(SSDMobileDetDSPFeatureExtractor, self).__init__(
        backbone_fn=mobiledet_dsp_backbone,
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams,
        scope_name=scope_name)


class SSDMobileDetEdgeTPUFeatureExtractor(SSDMobileDetFeatureExtractorBase):
  """MobileDet-EdgeTPU feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False,
               scope_name='MobileDetEdgeTPU'):
    super(SSDMobileDetEdgeTPUFeatureExtractor, self).__init__(
        backbone_fn=mobiledet_edgetpu_backbone,
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams,
        scope_name=scope_name)


class SSDMobileDetGPUFeatureExtractor(SSDMobileDetFeatureExtractorBase):
  """MobileDet-GPU feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False,
               scope_name='MobileDetGPU'):
    super(SSDMobileDetGPUFeatureExtractor, self).__init__(
        backbone_fn=mobiledet_gpu_backbone,
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams,
        scope_name=scope_name)
