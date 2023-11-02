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


from absl import logging
from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.s3d.modeling import net_utils


class Tf2NetUtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('3d', [2, 1, 1], [5, 16, 28, 28, 256]),
      ('3d', [2, 2, 2], [5, 16, 14, 14, 256]),
      ('3d', [1, 2, 1], [5, 32, 14, 28, 256]),
      ('2d', [2, 2, 2], [5, 32, 14, 14, 256]),
      ('2d', [1, 1, 2], [5, 32, 28, 14, 256]),
      ('1+2d', [2, 2, 2], [5, 16, 14, 14, 256]),
      ('1+2d', [2, 1, 1], [5, 16, 28, 28, 256]),
      ('1+2d', [1, 1, 1], [5, 32, 28, 28, 256]),
      ('1+2d', [1, 1, 2], [5, 32, 28, 14, 256]),
      ('2+1d', [2, 2, 2], [5, 16, 14, 14, 256]),
      ('2+1d', [1, 1, 1], [5, 32, 28, 28, 256]),
      ('2+1d', [2, 1, 2], [5, 16, 28, 14, 256]),
      ('1+1+1d', [2, 2, 2], [5, 16, 14, 14, 256]),
      ('1+1+1d', [1, 1, 1], [5, 32, 28, 28, 256]),
      ('1+1+1d', [2, 1, 2], [5, 16, 28, 14, 256]),
  )
  def test_parameterized_conv_layer_creation(self, conv_type, strides,
                                             expected_shape):
    batch_size = 5
    temporal_size = 32
    spatial_size = 28
    channels = 128

    kernel_size = 3
    filters = 256
    rates = [1, 1, 1]

    name = 'ParameterizedConv'

    inputs = tf_keras.Input(
        shape=(temporal_size, spatial_size, spatial_size, channels),
        batch_size=batch_size)
    parameterized_conv_layer = net_utils.ParameterizedConvLayer(
        conv_type, kernel_size, filters, strides, rates, name=name)

    features = parameterized_conv_layer(inputs)
    logging.info(features.shape.as_list())
    logging.info([w.name for w in parameterized_conv_layer.weights])

    self.assertAllEqual(features.shape.as_list(), expected_shape)

if __name__ == '__main__':
  tf.test.main()
