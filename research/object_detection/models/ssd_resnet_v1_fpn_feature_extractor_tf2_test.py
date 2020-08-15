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
"""Tests for ssd resnet v1 FPN feature extractors."""
import unittest
import tensorflow.compat.v1 as tf

from object_detection.models import ssd_resnet_v1_fpn_feature_extractor_testbase
from object_detection.models import ssd_resnet_v1_fpn_keras_feature_extractor
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class SSDResnet50V1FeatureExtractorTest(
    ssd_resnet_v1_fpn_feature_extractor_testbase.
    SSDResnetFPNFeatureExtractorTestBase):
  """SSDResnet50v1Fpn feature extractor test."""

  def _create_feature_extractor(self, depth_multiplier, pad_to_multiple,
                                use_explicit_padding=False, min_depth=32,
                                use_keras=True):
    is_training = True
    return (ssd_resnet_v1_fpn_keras_feature_extractor.
            SSDResNet50V1FpnKerasFeatureExtractor(
                is_training=is_training,
                depth_multiplier=depth_multiplier,
                min_depth=min_depth,
                pad_to_multiple=pad_to_multiple,
                conv_hyperparams=self._build_conv_hyperparams(
                    add_batch_norm=False),
                freeze_batchnorm=False,
                inplace_batchnorm_update=False,
                name='ResNet50V1_FPN'))

  def _resnet_scope_name(self):
    return 'ResNet50V1_FPN'


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class SSDResnet101V1FeatureExtractorTest(
    ssd_resnet_v1_fpn_feature_extractor_testbase.
    SSDResnetFPNFeatureExtractorTestBase):
  """SSDResnet101v1Fpn feature extractor test."""

  def _create_feature_extractor(self, depth_multiplier, pad_to_multiple,
                                use_explicit_padding=False, min_depth=32,
                                use_keras=False):
    is_training = True
    return (ssd_resnet_v1_fpn_keras_feature_extractor.
            SSDResNet101V1FpnKerasFeatureExtractor(
                is_training=is_training,
                depth_multiplier=depth_multiplier,
                min_depth=min_depth,
                pad_to_multiple=pad_to_multiple,
                conv_hyperparams=self._build_conv_hyperparams(
                    add_batch_norm=False),
                freeze_batchnorm=False,
                inplace_batchnorm_update=False,
                name='ResNet101V1_FPN'))

  def _resnet_scope_name(self):
    return 'ResNet101V1_FPN'


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class SSDResnet152V1FeatureExtractorTest(
    ssd_resnet_v1_fpn_feature_extractor_testbase.
    SSDResnetFPNFeatureExtractorTestBase):
  """SSDResnet152v1Fpn feature extractor test."""

  def _create_feature_extractor(self, depth_multiplier, pad_to_multiple,
                                use_explicit_padding=False, min_depth=32,
                                use_keras=False):
    is_training = True
    return (ssd_resnet_v1_fpn_keras_feature_extractor.
            SSDResNet152V1FpnKerasFeatureExtractor(
                is_training=is_training,
                depth_multiplier=depth_multiplier,
                min_depth=min_depth,
                pad_to_multiple=pad_to_multiple,
                conv_hyperparams=self._build_conv_hyperparams(
                    add_batch_norm=False),
                freeze_batchnorm=False,
                inplace_batchnorm_update=False,
                name='ResNet152V1_FPN'))

  def _resnet_scope_name(self):
    return 'ResNet152V1_FPN'


if __name__ == '__main__':
  tf.test.main()
