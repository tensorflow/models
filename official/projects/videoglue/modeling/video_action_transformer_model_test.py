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

"""Tests for video_action_transformer_model."""

import tensorflow as tf

from official.projects.videoglue.configs import spatiotemporal_action_localization as cfg
from official.projects.videoglue.modeling import video_action_transformer_model


class VideoActionTransformerModelTest(tf.test.TestCase):

  def test_video_action_transformer_model_construction(self):
    model_config = cfg.VideoActionTransformerModel()
    input_specs = {
        'image': tf.keras.layers.InputSpec(shape=[None, 4, 20, 20, 3]),
        'instances_position': tf.keras.layers.InputSpec(shape=[None, 8, 4])
    }

    model = video_action_transformer_model.build_video_action_transformer_model(
        input_specs_dict=input_specs,
        model_config=model_config,
        num_classes=80)
    self.assertIsInstance(
        model, video_action_transformer_model.VideoActionTransformerModel)


if __name__ == '__main__':
  tf.test.main()
