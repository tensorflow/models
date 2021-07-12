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

# Lint as: python3
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

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from official.vision.beta.modeling import backbones
from official.vision.beta.projects.simclr.heads import simclr_head
from official.vision.beta.projects.simclr.modeling import simclr_model


class SimCLRModelTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (128, 3, 0),
      (128, 3, 1),
      (128, 1, 0),
      (128, 1, 1),
  )
  def test_model_creation(self, project_dim, num_proj_layers, ft_proj_idx):
    input_size = 224
    inputs = np.random.rand(2, input_size, input_size, 3)
    input_specs = tf.keras.layers.InputSpec(
        shape=[None, input_size, input_size, 3])

    tf.keras.backend.set_image_data_format('channels_last')

    backbone = backbones.ResNet(model_id=50, activation='relu',
                                input_specs=input_specs)
    projection_head = simclr_head.ProjectionHead(
        proj_output_dim=project_dim,
        num_proj_layers=num_proj_layers,
        ft_proj_idx=ft_proj_idx
    )
    num_classes = 10
    supervised_head = simclr_head.ClassificationHead(
        num_classes=10
    )

    model = simclr_model.SimCLRModel(
        input_specs=input_specs,
        backbone=backbone,
        projection_head=projection_head,
        supervised_head=supervised_head,
        mode=simclr_model.PRETRAIN
    )
    outputs = model(inputs)
    projection_outputs = outputs[simclr_model.PROJECTION_OUTPUT_KEY]
    supervised_outputs = outputs[simclr_model.SUPERVISED_OUTPUT_KEY]

    self.assertAllEqual(projection_outputs.shape.as_list(),
                        [2, project_dim])
    self.assertAllEqual([2, num_classes],
                        supervised_outputs.numpy().shape)


if __name__ == '__main__':
  tf.test.main()
