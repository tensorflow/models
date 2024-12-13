# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for MobileNet."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.edgetpu.vision.modeling.backbones import mobilenet_edgetpu


class TestInputSpec:

  def __init__(self, shape):
    self.shape = shape


class TestBackboneConfig:

  def __init__(self, model_id):
    self.model_id = model_id
    self.freeze_large_filters = 99
    self.pretrained_checkpoint_path = None
    self.type = 'mobilenet_edgetpu'

  def get(self):
    return self


class MobileNetEdgeTPUTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('mobilenet_edgetpu_v2_s', (1, 512, 512, 3)),
      ('mobilenet_edgetpu_v2_l', (1, None, None, 3)),
      ('mobilenet_edgetpu', (1, 512, 512, 3)),
      ('mobilenet_edgetpu_dm1p25', (1, None, None, 3)),
  )
  def test_mobilenet_creation(self, model_id, input_shape):
    """Test creation of MobileNet family models."""
    tf_keras.backend.set_image_data_format('channels_last')

    test_model = mobilenet_edgetpu.build_mobilenet_edgetpu(
        input_specs=TestInputSpec(input_shape),
        backbone_config=TestBackboneConfig(model_id))
    self.assertGreater(len(test_model.outputs), 1)


if __name__ == '__main__':
  tf.test.main()
