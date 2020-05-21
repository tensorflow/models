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
"""Tests for model_builder under TensorFlow 1.X."""

from absl.testing import parameterized
import tensorflow as tf

from object_detection.builders import model_builder
from object_detection.builders import model_builder_test
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.protos import losses_pb2


class ModelBuilderTF1Test(model_builder_test.ModelBuilderTest):

  def default_ssd_feature_extractor(self):
    return 'ssd_resnet50_v1_fpn'

  def default_faster_rcnn_feature_extractor(self):
    return 'faster_rcnn_resnet101'

  def ssd_feature_extractors(self):
    return model_builder.SSD_FEATURE_EXTRACTOR_CLASS_MAP

  def faster_rcnn_feature_extractors(self):
    return model_builder.FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP



if __name__ == '__main__':
  tf.test.main()
