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
import unittest
from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from object_detection.builders import model_builder
from object_detection.builders import model_builder_test
from object_detection.meta_architectures import context_rcnn_meta_arch
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.protos import losses_pb2
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class ModelBuilderTF1Test(model_builder_test.ModelBuilderTest):

  def default_ssd_feature_extractor(self):
    return 'ssd_resnet50_v1_fpn'

  def default_faster_rcnn_feature_extractor(self):
    return 'faster_rcnn_resnet101'

  def ssd_feature_extractors(self):
    return model_builder.SSD_FEATURE_EXTRACTOR_CLASS_MAP

  def get_override_base_feature_extractor_hyperparams(self, extractor_type):
    return extractor_type in {'ssd_inception_v2', 'ssd_inception_v3'}

  def faster_rcnn_feature_extractors(self):
    return model_builder.FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP


  @parameterized.parameters(True, False)
  def test_create_context_rcnn_from_config_with_params(self, is_training):
    model_proto = self.create_default_faster_rcnn_model_proto()
    model_proto.faster_rcnn.context_config.attention_bottleneck_dimension = 10
    model_proto.faster_rcnn.context_config.attention_temperature = 0.5
    model = model_builder.build(model_proto, is_training=is_training)
    self.assertIsInstance(model, context_rcnn_meta_arch.ContextRCNNMetaArch)


if __name__ == '__main__':
  tf.test.main()
