# Lint as: python2, python3
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""SSDFeatureExtractor for PNASNet features.

Based on PNASNet ImageNet model: https://arxiv.org/abs/1712.00559
"""

import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import variables_helper
from nets.nasnet import pnasnet

slim = contrib_slim


def pnasnet_large_arg_scope_for_detection(is_batch_norm_training=False):
  """Defines the default arg scope for the PNASNet Large for object detection.

  This provides a small edit to switch batch norm training on and off.

  Args:
    is_batch_norm_training: Boolean indicating whether to train with batch norm.
    Default is False.

  Returns:
    An `arg_scope` to use for the PNASNet Large Model.
  """
  imagenet_scope = pnasnet.pnasnet_large_arg_scope()
  with slim.arg_scope(imagenet_scope):
    with slim.arg_scope([slim.batch_norm],
                        is_training=is_batch_norm_training) as sc:
      return sc


class SSDPNASNetFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD Feature Extractor using PNASNet features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               num_layers=6,
               override_base_feature_extractor_hyperparams=False):
    """PNASNet Feature Extractor for SSD Models.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      use_depthwise: Whether to use depthwise convolutions.
      num_layers: Number of SSD layers.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDPNASNetFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        num_layers=num_layers,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)

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

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """

    feature_map_layout = {
        'from_layer': ['Cell_7', 'Cell_11', '', '', '', ''][:self._num_layers],
        'layer_depth': [-1, -1, 512, 256, 256, 128][:self._num_layers],
        'use_explicit_padding': self._use_explicit_padding,
        'use_depthwise': self._use_depthwise,
    }

    with slim.arg_scope(
        pnasnet_large_arg_scope_for_detection(
            is_batch_norm_training=self._is_training)):
      with slim.arg_scope([slim.conv2d, slim.batch_norm, slim.separable_conv2d],
                          reuse=self._reuse_weights):
        with (slim.arg_scope(self._conv_hyperparams_fn())
              if self._override_base_feature_extractor_hyperparams else
              context_manager.IdentityContextManager()):
          _, image_features = pnasnet.build_pnasnet_large(
              ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
              num_classes=None,
              is_training=self._is_training,
              final_endpoint='Cell_11')
    with tf.variable_scope('SSD_feature_maps', reuse=self._reuse_weights):
      with slim.arg_scope(self._conv_hyperparams_fn()):
        feature_maps = feature_map_generators.multi_resolution_feature_maps(
            feature_map_layout=feature_map_layout,
            depth_multiplier=self._depth_multiplier,
            min_depth=self._min_depth,
            insert_1x1_conv=True,
            image_features=image_features)

    return list(feature_maps.values())

  def restore_from_classification_checkpoint_fn(self, feature_extractor_scope):
    """Returns a map of variables to load from a foreign checkpoint.

    Note that this overrides the default implementation in
    ssd_meta_arch.SSDFeatureExtractor which does not work for PNASNet
    checkpoints.

    Args:
      feature_extractor_scope: A scope name for the first stage feature
        extractor.

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    """
    variables_to_restore = {}
    for variable in variables_helper.get_global_variables_safely():
      if variable.op.name.startswith(feature_extractor_scope):
        var_name = variable.op.name.replace(feature_extractor_scope + '/', '')
        var_name += '/ExponentialMovingAverage'
        variables_to_restore[var_name] = variable
    return variables_to_restore
