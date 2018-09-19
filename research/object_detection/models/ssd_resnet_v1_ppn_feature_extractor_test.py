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
"""Tests for ssd resnet v1 feature extractors."""
import tensorflow as tf

from object_detection.models import ssd_resnet_v1_ppn_feature_extractor
from object_detection.models import ssd_resnet_v1_ppn_feature_extractor_testbase


class SSDResnet50V1PpnFeatureExtractorTest(
    ssd_resnet_v1_ppn_feature_extractor_testbase.
    SSDResnetPpnFeatureExtractorTestBase):
  """SSDResnet50v1 feature extractor test."""

  def _create_feature_extractor(self, depth_multiplier, pad_to_multiple,
                                use_explicit_padding=False):
    min_depth = 32
    is_training = True
    return ssd_resnet_v1_ppn_feature_extractor.SSDResnet50V1PpnFeatureExtractor(
        is_training,
        depth_multiplier,
        min_depth,
        pad_to_multiple,
        self.conv_hyperparams_fn,
        use_explicit_padding=use_explicit_padding)

  def _scope_name(self):
    return 'resnet_v1_50'


class SSDResnet101V1PpnFeatureExtractorTest(
    ssd_resnet_v1_ppn_feature_extractor_testbase.
    SSDResnetPpnFeatureExtractorTestBase):
  """SSDResnet101v1 feature extractor test."""

  def _create_feature_extractor(self, depth_multiplier, pad_to_multiple,
                                use_explicit_padding=False):
    min_depth = 32
    is_training = True
    return (
        ssd_resnet_v1_ppn_feature_extractor.SSDResnet101V1PpnFeatureExtractor(
            is_training,
            depth_multiplier,
            min_depth,
            pad_to_multiple,
            self.conv_hyperparams_fn,
            use_explicit_padding=use_explicit_padding))

  def _scope_name(self):
    return 'resnet_v1_101'


class SSDResnet152V1PpnFeatureExtractorTest(
    ssd_resnet_v1_ppn_feature_extractor_testbase.
    SSDResnetPpnFeatureExtractorTestBase):
  """SSDResnet152v1 feature extractor test."""

  def _create_feature_extractor(self, depth_multiplier, pad_to_multiple,
                                use_explicit_padding=False):
    min_depth = 32
    is_training = True
    return (
        ssd_resnet_v1_ppn_feature_extractor.SSDResnet152V1PpnFeatureExtractor(
            is_training,
            depth_multiplier,
            min_depth,
            pad_to_multiple,
            self.conv_hyperparams_fn,
            use_explicit_padding=use_explicit_padding))

  def _scope_name(self):
    return 'resnet_v1_152'


if __name__ == '__main__':
  tf.test.main()
