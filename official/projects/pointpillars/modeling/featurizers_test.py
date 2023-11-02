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

"""Tests for backbones."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.pointpillars.modeling import featurizers


class FeaturizerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ([32, 32], [16, 4, 2], 4, 2, 1),
      ([32, 16], [1, 3, 1], 2, 2, 3),
  )
  def test_network_creation(self, image_size, pillars_size, train_batch_size,
                            eval_batch_size, num_blocks):
    num_channels = 3
    h, w = image_size
    n, _, _ = pillars_size
    featurizer = featurizers.Featurizer(image_size, pillars_size,
                                        train_batch_size, eval_batch_size,
                                        num_blocks, num_channels)

    # Train mode.
    pillars = tf_keras.Input(shape=pillars_size, batch_size=train_batch_size)
    indices = tf_keras.Input(
        shape=[n, 2], batch_size=train_batch_size, dtype=tf.int32)
    image = featurizer(pillars, indices, training=True)
    self.assertAllEqual([train_batch_size, h, w, num_channels],
                        image.shape.as_list())

    # Evaluation mode.
    pillars = tf_keras.Input(shape=pillars_size, batch_size=eval_batch_size)
    indices = tf_keras.Input(
        shape=[n, 2], batch_size=eval_batch_size, dtype=tf.int32)
    image = featurizer(pillars, indices, training=False)
    self.assertAllEqual([eval_batch_size, h, w, num_channels],
                        image.shape.as_list())

    # Test mode, batch size must be 1.
    pillars = tf_keras.Input(shape=pillars_size, batch_size=1)
    indices = tf_keras.Input(
        shape=[n, 2], batch_size=1, dtype=tf.int32)
    image = featurizer(pillars, indices, training=None)
    self.assertAllEqual([1, h, w, num_channels],
                        image.shape.as_list())

  def test_serialization(self):
    kwargs = dict(
        image_size=[4, 4],
        pillars_size=[4, 5, 6],
        train_batch_size=4,
        eval_batch_size=2,
        num_blocks=3,
        num_channels=4,
        kernel_regularizer=None,
    )
    net = featurizers.Featurizer(**kwargs)
    expected_config = kwargs
    self.assertEqual(net.get_config(), expected_config)

    new_net = featurizers.Featurizer.from_config(net.get_config())
    self.assertAllEqual(net.get_config(), new_net.get_config())


if __name__ == '__main__':
  tf.test.main()
