# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from official.vision.beta.projects.simclr.heads import simclr_head


class ProjectionHeadTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (0, None),
      (1, 128),
      (2, 128),
  )
  def test_head_creation(self, num_proj_layers, proj_output_dim):
    test_layer = simclr_head.ProjectionHead(
        num_proj_layers=num_proj_layers,
        proj_output_dim=proj_output_dim)

    input_dim = 64
    x = tf.keras.Input(shape=(input_dim,))
    proj_head_output, proj_finetune_output = test_layer(x)

    proj_head_output_dim = input_dim
    if num_proj_layers > 0:
      proj_head_output_dim = proj_output_dim
    self.assertAllEqual(proj_head_output.shape.as_list(),
                        [None, proj_head_output_dim])

    if num_proj_layers > 0:
      proj_finetune_output_dim = input_dim
      self.assertAllEqual(proj_finetune_output.shape.as_list(),
                          [None, proj_finetune_output_dim])

  @parameterized.parameters(
      (0, None, 0),
      (1, 128, 0),
      (2, 128, 1),
      (2, 128, 2),
  )
  def test_outputs(self, num_proj_layers, proj_output_dim, ft_proj_idx):
    test_layer = simclr_head.ProjectionHead(
        num_proj_layers=num_proj_layers,
        proj_output_dim=proj_output_dim,
        ft_proj_idx=ft_proj_idx
    )

    input_dim = 64
    batch_size = 2
    inputs = np.random.rand(batch_size, input_dim)
    proj_head_output, proj_finetune_output = test_layer(inputs)

    if num_proj_layers == 0:
      self.assertAllClose(inputs, proj_head_output)
      self.assertAllClose(inputs, proj_finetune_output)
    else:
      self.assertAllEqual(proj_head_output.shape.as_list(),
                          [batch_size, proj_output_dim])
      if ft_proj_idx == 0:
        self.assertAllClose(inputs, proj_finetune_output)
      elif ft_proj_idx < num_proj_layers:
        self.assertAllEqual(proj_finetune_output.shape.as_list(),
                            [batch_size, input_dim])
      else:
        self.assertAllEqual(proj_finetune_output.shape.as_list(),
                            [batch_size, proj_output_dim])


class ClassificationHeadTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      10, 20
  )
  def test_head_creation(self, num_classes):
    test_layer = simclr_head.ClassificationHead(num_classes=num_classes)

    input_dim = 64
    x = tf.keras.Input(shape=(input_dim,))
    out_x = test_layer(x)

    self.assertAllEqual(out_x.shape.as_list(),
                        [None, num_classes])


if __name__ == '__main__':
  tf.test.main()
