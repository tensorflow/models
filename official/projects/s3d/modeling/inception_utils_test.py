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
import tensorflow as tf

from official.projects.s3d.modeling import inception_utils


class InceptionUtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((1.0, 3, {'Conv2d_1a_7x7', 'Conv2d_2c_3x3'}),
                            (0.5, 5, {'Conv2d_1a_7x7', 'Conv2d_2c_3x3'}),
                            (0.25, 7, {'Conv2d_1a_7x7', 'Conv2d_2c_3x3'}))
  def test_s3d_stem_cells(self, depth_multiplier, first_temporal_kernel_size,
                          temporal_conv_endpoints):
    batch_size = 1
    num_frames = 64
    height, width = 224, 224

    inputs = tf.keras.layers.Input(
        shape=(num_frames, height, width, 3), batch_size=batch_size)

    outputs, output_endpoints = inception_utils.inception_v1_stem_cells(
        inputs,
        depth_multiplier,
        'Mixed_5c',
        temporal_conv_endpoints=temporal_conv_endpoints,
        self_gating_endpoints={'Conv2d_2c_3x3'},
        first_temporal_kernel_size=first_temporal_kernel_size)
    self.assertListEqual(outputs.shape.as_list(),
                         [batch_size, 32, 28, 28, int(192 * depth_multiplier)])

    expected_endpoints = {
        'Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1', 'Conv2d_2c_3x3',
        'MaxPool_3a_3x3'
    }
    self.assertSetEqual(expected_endpoints, set(output_endpoints.keys()))

  @parameterized.parameters(
      ('3d', True, True, True),
      ('2d', False, False, True),
      ('1+2d', True, False, False),
      ('2+1d', False, True, False),
  )
  def test_inception_v1_cell_endpoint_match(self, conv_type,
                                            swap_pool_and_1x1x1,
                                            use_self_gating_on_branch,
                                            use_self_gating_on_cell):
    batch_size = 5
    num_frames = 32
    channels = 128
    height, width = 28, 28

    inputs = tf.keras.layers.Input(
        shape=(num_frames, height, width, channels), batch_size=batch_size)

    inception_v1_cell_layer = inception_utils.InceptionV1CellLayer(
        [[64], [96, 128], [16, 32], [32]],
        conv_type=conv_type,
        swap_pool_and_1x1x1=swap_pool_and_1x1x1,
        use_self_gating_on_branch=use_self_gating_on_branch,
        use_self_gating_on_cell=use_self_gating_on_cell,
        name='test')
    outputs = inception_v1_cell_layer(inputs)

    # self.assertTrue(net.op.name.startswith('test'))
    self.assertListEqual(outputs.shape.as_list(),
                         [batch_size, 32, 28, 28, 256])

if __name__ == '__main__':
  tf.test.main()
