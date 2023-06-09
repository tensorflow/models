# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for const_cl_model."""
import tensorflow as tf

from official.projects.const_cl.configs import const_cl as const_cl_cfg
from official.projects.const_cl.modeling import const_cl_model
# pylint: disable=unused-import
from official.projects.const_cl.modeling.backbones import resnet_3d
# pylint: enable=unused-import


class ConstClModelTest(tf.test.TestCase):

  def test_build_const_cl_pretrain_model(self):
    model_config = const_cl_cfg.ConstCLModel()
    images_input_specs = tf.keras.layers.InputSpec(
        shape=[None, 16, 224, 224, 4])
    boxes_input_specs = tf.keras.layers.InputSpec(shape=[None, 16, 8, 4])
    masks_input_specs = tf.keras.layers.InputSpec(shape=[None, 16, 8])

    input_specs_dict = {
        'image': images_input_specs,
        'instances_position': boxes_input_specs,
        'instances_mask': masks_input_specs,
    }
    model = const_cl_model.build_const_cl_pretrain_model(
        input_specs_dict=input_specs_dict,
        model_config=model_config,
        num_classes=500)
    self.assertIsInstance(model, const_cl_model.ConstCLModel)


if __name__ == '__main__':
  tf.test.main()
