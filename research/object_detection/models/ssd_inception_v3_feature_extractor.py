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

"""SSDFeatureExtractor for InceptionV3 features."""
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets import inception_v3

slim = contrib_slim


class SSDInceptionV3FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD Feature Extractor using InceptionV3 features."""

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
    """InceptionV3 Feature Extractor for SSD Models.

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
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      num_layers: Number of SSD layers.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.

    Raises:
      ValueError: If `override_base_feature_extractor_hyperparams` is False.
    """
    super(SSDInceptionV3FeatureExtractor, self).__init__(
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

    if not self._override_base_feature_extractor_hyperparams:
      raise ValueError('SSD Inception V3 feature extractor always uses'
                       'scope returned by `conv_hyperparams_fn` for both the '
                       'base feature extractor and the additional layers '
                       'added since there is no arg_scope defined for the base '
                       'feature extractor.')

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
        'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''
                      ][:self._num_layers],
        'layer_depth': [-1, -1, -1, 512, 256, 128][:self._num_layers],
        'use_explicit_padding': self._use_explicit_padding,
        'use_depthwise': self._use_depthwise,
    }

    with slim.arg_scope(self._conv_hyperparams_fn()):
      with tf.variable_scope('InceptionV3', reuse=self._reuse_weights) as scope:
        _, image_features = inception_v3.inception_v3_base(
            ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
            final_endpoint='Mixed_7c',
            min_depth=self._min_depth,
            depth_multiplier=self._depth_multiplier,
            scope=scope)
        feature_maps = feature_map_generators.multi_resolution_feature_maps(
            feature_map_layout=feature_map_layout,
            depth_multiplier=self._depth_multiplier,
            min_depth=self._min_depth,
            insert_1x1_conv=True,
            image_features=image_features)

    return list(feature_maps.values())
