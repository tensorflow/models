# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""SSDFeatureExtractor for MobileNetV3 features."""

import tensorflow.compat.v1 as tf
import tf_slim as slim

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets.mobilenet import mobilenet
from nets.mobilenet import mobilenet_v3


class SSDMobileNetV3FeatureExtractorBase(ssd_meta_arch.SSDFeatureExtractor):
  """Base class of SSD feature extractor using MobilenetV3 features."""

  def __init__(self,
               conv_defs,
               from_layer,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False,
               scope_name='MobilenetV3'):
    """MobileNetV3 Feature Extractor for SSD Models.

    MobileNet v3. Details found in:
    https://arxiv.org/abs/1905.02244

    Args:
      conv_defs: MobileNetV3 conv defs for backbone.
      from_layer: A cell of two layer names (string) to connect to the 1st and
        2nd inputs of the SSD head.
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the base
        feature extractor.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
      scope_name: scope name (string) of network variables.
    """
    super(SSDMobileNetV3FeatureExtractorBase, self).__init__(
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
    self._conv_defs = conv_defs
    self._from_layer = from_layer
    self._scope_name = scope_name

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
    Raises:
      ValueError if conv_defs is not provided or from_layer does not meet the
        size requirement.
    """

    if not self._conv_defs:
      raise ValueError('Must provide backbone conv defs.')

    if len(self._from_layer) != 2:
      raise ValueError('SSD input feature names are not provided.')

    preprocessed_inputs = shape_utils.check_min_image_dim(
        33, preprocessed_inputs)

    feature_map_layout = {
        'from_layer': [
            self._from_layer[0], self._from_layer[1], '', '', '', ''
        ],
        'layer_depth': [-1, -1, 512, 256, 256, 128],
        'use_depthwise': self._use_depthwise,
        'use_explicit_padding': self._use_explicit_padding,
    }

    with tf.variable_scope(
        self._scope_name, reuse=self._reuse_weights) as scope:
      with slim.arg_scope(
          mobilenet_v3.training_scope(is_training=None, bn_decay=0.9997)), \
          slim.arg_scope(
              [mobilenet.depth_multiplier], min_depth=self._min_depth):
        with (slim.arg_scope(self._conv_hyperparams_fn())
              if self._override_base_feature_extractor_hyperparams else
              context_manager.IdentityContextManager()):
          _, image_features = mobilenet_v3.mobilenet_base(
              ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
              conv_defs=self._conv_defs,
              final_endpoint=self._from_layer[1],
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


class SSDMobileNetV3LargeFeatureExtractor(SSDMobileNetV3FeatureExtractorBase):
  """Mobilenet V3-Large feature extractor."""

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
               scope_name='MobilenetV3'):
    super(SSDMobileNetV3LargeFeatureExtractor, self).__init__(
        conv_defs=mobilenet_v3.V3_LARGE_DETECTION,
        from_layer=['layer_14/expansion_output', 'layer_17'],
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams,
        scope_name=scope_name
    )


class SSDMobileNetV3SmallFeatureExtractor(SSDMobileNetV3FeatureExtractorBase):
  """Mobilenet V3-Small feature extractor."""

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
               scope_name='MobilenetV3'):
    super(SSDMobileNetV3SmallFeatureExtractor, self).__init__(
        conv_defs=mobilenet_v3.V3_SMALL_DETECTION,
        from_layer=['layer_10/expansion_output', 'layer_13'],
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams,
        scope_name=scope_name
        )


class SSDMobileNetV3SmallPrunedFeatureExtractor(
    SSDMobileNetV3FeatureExtractorBase):
  """Mobilenet V3-Small feature extractor."""

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
               scope_name='MobilenetV3'):
    super(SSDMobileNetV3SmallPrunedFeatureExtractor, self).__init__(
        conv_defs=mobilenet_v3.V3_SMALL_PRUNED_DETECTION,
        from_layer=['layer_9/expansion_output', 'layer_12'],
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
