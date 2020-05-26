# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.contrib.slim.nets.cyclegan."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from nets import cyclegan


# TODO(joelshor): Add a test to check generator endpoints.
class CycleganTest(tf.test.TestCase):

  def test_generator_inference(self):
    """Check one inference step."""
    img_batch = tf.zeros([2, 32, 32, 3])
    model_output, _ = cyclegan.cyclegan_generator_resnet(img_batch)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(model_output)

  def _test_generator_graph_helper(self, shape):
    """Check that generator can take small and non-square inputs."""
    output_imgs, _ = cyclegan.cyclegan_generator_resnet(tf.ones(shape))
    self.assertAllEqual(shape, output_imgs.shape.as_list())

  def test_generator_graph_small(self):
    self._test_generator_graph_helper([4, 32, 32, 3])

  def test_generator_graph_medium(self):
    self._test_generator_graph_helper([3, 128, 128, 3])

  def test_generator_graph_nonsquare(self):
    self._test_generator_graph_helper([2, 80, 400, 3])

  def test_generator_unknown_batch_dim(self):
    """Check that generator can take unknown batch dimension inputs."""
    img = tf.placeholder(tf.float32, shape=[None, 32, None, 3])
    output_imgs, _ = cyclegan.cyclegan_generator_resnet(img)

    self.assertAllEqual([None, 32, None, 3], output_imgs.shape.as_list())

  def _input_and_output_same_shape_helper(self, kernel_size):
    img_batch = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    output_img_batch, _ = cyclegan.cyclegan_generator_resnet(
        img_batch, kernel_size=kernel_size)

    self.assertAllEqual(img_batch.shape.as_list(),
                        output_img_batch.shape.as_list())

  def input_and_output_same_shape_kernel3(self):
    self._input_and_output_same_shape_helper(3)

  def input_and_output_same_shape_kernel4(self):
    self._input_and_output_same_shape_helper(4)

  def input_and_output_same_shape_kernel5(self):
    self._input_and_output_same_shape_helper(5)

  def input_and_output_same_shape_kernel6(self):
    self._input_and_output_same_shape_helper(6)

  def _error_if_height_not_multiple_of_four_helper(self, height):
    self.assertRaisesRegexp(
        ValueError, 'The input height must be a multiple of 4.',
        cyclegan.cyclegan_generator_resnet,
        tf.placeholder(tf.float32, shape=[None, height, 32, 3]))

  def test_error_if_height_not_multiple_of_four_height29(self):
    self._error_if_height_not_multiple_of_four_helper(29)

  def test_error_if_height_not_multiple_of_four_height30(self):
    self._error_if_height_not_multiple_of_four_helper(30)

  def test_error_if_height_not_multiple_of_four_height31(self):
    self._error_if_height_not_multiple_of_four_helper(31)

  def _error_if_width_not_multiple_of_four_helper(self, width):
    self.assertRaisesRegexp(
        ValueError, 'The input width must be a multiple of 4.',
        cyclegan.cyclegan_generator_resnet,
        tf.placeholder(tf.float32, shape=[None, 32, width, 3]))

  def test_error_if_width_not_multiple_of_four_width29(self):
    self._error_if_width_not_multiple_of_four_helper(29)

  def test_error_if_width_not_multiple_of_four_width30(self):
    self._error_if_width_not_multiple_of_four_helper(30)

  def test_error_if_width_not_multiple_of_four_width31(self):
    self._error_if_width_not_multiple_of_four_helper(31)


if __name__ == '__main__':
  tf.test.main()
