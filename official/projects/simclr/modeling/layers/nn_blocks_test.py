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

from absl.testing import parameterized

import tensorflow as tf, tf_keras

from official.projects.simclr.modeling.layers import nn_blocks


class DenseBNTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (64, True, True),
      (64, True, False),
      (64, False, True),
  )
  def test_pass_through(self, output_dim, use_bias, use_normalization):
    test_layer = nn_blocks.DenseBN(
        output_dim=output_dim,
        use_bias=use_bias,
        use_normalization=use_normalization
    )

    x = tf_keras.Input(shape=(64,))
    out_x = test_layer(x)

    self.assertAllEqual(out_x.shape.as_list(), [None, output_dim])

    # kernel of the dense layer
    train_var_len = 1
    if use_normalization:
      if use_bias:
        # batch norm introduce two trainable variables
        train_var_len += 2
      else:
        # center is set to False if not use bias
        train_var_len += 1
    else:
      if use_bias:
        # bias of dense layer
        train_var_len += 1
    self.assertLen(test_layer.trainable_variables, train_var_len)


if __name__ == '__main__':
  tf.test.main()
