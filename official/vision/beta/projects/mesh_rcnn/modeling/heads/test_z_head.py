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
"""Tests for Mesh R-CNN Heads."""

from absl.testing import parameterized
import tensorflow as tf

# from official.vision.beta.projects.mesh_rcnn.modeling.heads import z_head
import z_head

class ZHeadTest(parameterized.TestCase, tf.test.TestCase):
    '''Test for Mesh R-CNN Z head'''

    def test_output_shape(self,
        num_fc: int,
        fc_dim: int,
        cls_agnostic: bool,
        num_classes: int):
        '''Check that Z head output is of correct shape'''

        (batch_size, height, width, channels) = (64, 14, 14, 256)
        head = z_head.ZHead(num_fc, fc_dim, cls_agnostic, num_classes)

        test_input = tf.zeros((batch_size, height, width, channels))
        output = head(test_input)
        expected_output = tf.zeros((batch_size, num_classes))
        self.assertAllEqual(output,expected_output)

    def test_serialize_deserialize(self,
        num_fc: int,
        fc_dim: int,
        cls_agnostic: bool,
        num_classes: int):
        """Create a network object that sets all of its config options."""

        (batch_size, height, width, channels) = (64, 14, 14, 256)
        head = z_head.ZHead(num_fc, fc_dim, cls_agnostic, num_classes)
        test_input = tf.zeros((batch_size, height, width, channels))
        _ = head(test_input)

        serialized = head.get_config()
        deserialized = z_head.ZHead.from_config(serialized)

        self.assertAllEqual(head.get_config(), deserialized.get_config())

    def test_gradient_pass_through(self,
        num_fc: int,
        fc_dim: int,
        cls_agnostic: bool,
        num_classes: int):
        '''Check that gradients are not None'''

        (batch_size, height, width, channels) = (64, 14, 14, 256)
        head = z_head.ZHead(num_fc, fc_dim, cls_agnostic, num_classes)

        loss = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.SGD()

        init = tf.random_normal_initializer()
        input_shape = (batch_size, height, width, channels)
        test_input = tf.Variable(initial_value = init(shape=input_shape, dtype=tf.float32))

        output_shape = head(test_input).shape
        test_output = tf.Variable(initial_value = init(shape=output_shape, dtype=tf.float32))

        with tf.GradientTape() as tape:
            y_hat = head(test_input)
            grad_loss = loss(y_hat, test_output)
        grad = tape.gradient(grad_loss, head.trainable_variables)
        optimizer.apply_gradients(zip(grad, head.trainable_variables))

        self.assertNotIn(None, grad)

if __name__ == '__main__':
    print("Unit Testing Z Head")
    zht = ZHeadTest()
    zht.test_output_shape(2, 1024, False, 100)
    zht.test_serialize_deserialize(2, 1024, False, 100)
    zht.test_gradient_pass_through(2, 1024, False, 100)
    print("Tests Successful")
