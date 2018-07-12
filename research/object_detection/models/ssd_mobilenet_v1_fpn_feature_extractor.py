# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""SSD MobilenetV1 FPN Feature Extractor."""

import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets import mobilenet_v1

slim = tf.contrib.slim


class SSDMobileNetV1FpnFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD Feature Extractor using MobilenetV1 FPN features."""

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

      depth_fn = lambda d: max(int(d * self._depth_multiplier), self._min_depth)
      with slim.arg_scope(self._conv_hyperparams_fn()):
        with tf.variable_scope('fpn', reuse=self._reuse_weights):
          fpn_features = feature_map_generators.fpn_top_down_feature_maps(
              [(key, image_features[key])
               for key in ['Conv2d_5_pointwise', 'Conv2d_11_pointwise',
                           'Conv2d_13_pointwise']],
              depth=depth_fn(256))
          last_feature_map = fpn_features['top_down_Conv2d_13_pointwise']
          coarse_features = {}
          for i in range(14, 16):
            last_feature_map = slim.conv2d(
                last_feature_map,
                num_outputs=depth_fn(256),
                kernel_size=[3, 3],
                stride=2,
                padding='SAME',
                scope='bottom_up_Conv2d_{}'.format(i))
            coarse_features['bottom_up_Conv2d_{}'.format(i)] = last_feature_map
    return [fpn_features['top_down_Conv2d_5_pointwise'],
            fpn_features['top_down_Conv2d_11_pointwise'],
            fpn_features['top_down_Conv2d_13_pointwise'],
            coarse_features['bottom_up_Conv2d_14'],
            coarse_features['bottom_up_Conv2d_15']]
