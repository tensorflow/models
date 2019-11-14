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
"""Tests for ssd_mobilenet_edgetpu_feature_extractor."""

import tensorflow as tf

from tensorflow.contrib import slim as contrib_slim
from object_detection.models import ssd_mobilenet_edgetpu_feature_extractor
from object_detection.models import ssd_mobilenet_edgetpu_feature_extractor_testbase

slim = contrib_slim


class SsdMobilenetEdgeTPUFeatureExtractorTest(
    ssd_mobilenet_edgetpu_feature_extractor_testbase
    ._SsdMobilenetEdgeTPUFeatureExtractorTestBase):

  def _get_input_sizes(self):
    """Return first two input feature map sizes."""
    return [384, 192]

  def _create_feature_extractor(self,
                                depth_multiplier,
                                pad_to_multiple,
                                use_explicit_padding=False,
                                use_keras=False):
    """Constructs a new MobileNetEdgeTPU feature extractor.

    Args:
      depth_multiplier: float depth multiplier for feature extractor
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      use_explicit_padding: use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      use_keras: if True builds a keras-based feature extractor, if False builds
        a slim-based one.

    Returns:
      an ssd_meta_arch.SSDFeatureExtractor object.
    """
    min_depth = 32
    return (ssd_mobilenet_edgetpu_feature_extractor
            .SSDMobileNetEdgeTPUFeatureExtractor(
                False,
                depth_multiplier,
                min_depth,
                pad_to_multiple,
                self.conv_hyperparams_fn,
                use_explicit_padding=use_explicit_padding))


if __name__ == '__main__':
  tf.test.main()
