# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for factory.py."""

from absl.testing import parameterized
import tensorflow as tf

# pylint: disable=unused-import
from official.projects.volumetric_models.configs import semantic_segmentation_3d as exp_cfg
from official.projects.volumetric_models.modeling import backbones
from official.projects.volumetric_models.modeling import decoders
from official.projects.volumetric_models.modeling import factory


class SegmentationModelBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(((128, 128, 128), 5e-5, True),
                            ((64, 64, 64), None, False))
  def test_unet3d_builder(self, input_size, weight_decay, use_bn):
    num_classes = 3
    input_specs = tf.keras.layers.InputSpec(
        shape=[None, input_size[0], input_size[1], input_size[2], 3])
    model_config = exp_cfg.SemanticSegmentationModel3D(num_classes=num_classes)
    model_config.head.use_batch_normalization = use_bn
    l2_regularizer = (
        tf.keras.regularizers.l2(weight_decay) if weight_decay else None)
    model = factory.build_segmentation_model_3d(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)
    self.assertIsInstance(
        model, tf.keras.Model,
        'Output should be a tf.keras.Model instance but got %s' % type(model))


if __name__ == '__main__':
  tf.test.main()
