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

"""SSDFeatureExtractor for MobilenetV1 features."""

import tensorflow.compat.v1 as tf
import tf_slim as slim

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets import mobilenet_v1


class SSDMobileNetV1FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD Feature Extractor using MobilenetV1 features."""

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
    """MobileNetV1 Feature Extractor for SSD Models.

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
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      num_layers: Number of SSD layers.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDMobileNetV1FeatureExtractor, self).__init__(
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
    preprocessed_inputs = shape_utils.check_min_image_dim(
        33, preprocessed_inputs)

    feature_map_layout = {
        'from_layer': ['Conv2d_11_pointwise', 'Conv2d_13_pointwise', '', '',
                       '', ''][:self._num_layers],
        'layer_depth': [-1, -1, 512, 256, 256, 128][:self._num_layers],
        'use_explicit_padding': self._use_explicit_padding,
        'use_depthwise': self._use_depthwise,
    }

    with tf.variable_scope('MobilenetV1',
                           reuse=self._reuse_weights) as scope:
      with slim.arg_scope(
          mobilenet_v1.mobilenet_v1_arg_scope(
              is_training=None, regularize_depthwise=True)):
        with (slim.arg_scope(self._conv_hyperparams_fn())
              if self._override_base_feature_extractor_hyperparams
              else context_manager.IdentityContextManager()):
          _, image_features = mobilenet_v1.mobilenet_v1_base(
              ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
              final_endpoint='Conv2d_13_pointwise',
              min_depth=self._min_depth,
              depth_multiplier=self._depth_multiplier,
              use_explicit_padding=self._use_explicit_padding,
              scope=scope)
      with slim.arg_scope(self._conv_hyperparams_fn()):
        feature_maps = feature_map_generators.multi_resolution_feature_maps(
            feature_map_layout=feature_map_layout,
            depth_multiplier=self._depth_multiplier,
            min_depth=self._min_depth,
            insert_1x1_conv=True,
            image_features=image_features)

    return list(feature_maps.values())
