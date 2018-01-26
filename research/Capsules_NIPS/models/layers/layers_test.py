# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Tests for layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

from models.layers import layers


class LayersTest(tf.test.TestCase):

  def testSquashRankSix(self):
    """Checks the value and shape of the squash output given a rank 6 input."""
    input_tensor = tf.ones((1, 1, 1, 1, 1, 1))
    squashed = layers._squash(input_tensor)
    self.assertEqual(len(squashed.get_shape()), 6)
    with self.test_session() as sess:
      r_squashed = sess.run(squashed)
    scale = 0.5
    self.assertEqual(np.array(r_squashed).shape, input_tensor.get_shape())
    self.assertAllClose(np.linalg.norm(r_squashed, axis=2), [[[[[scale]]]]])

  def testLeakyRoutingRankThree(self):
    """Checks the shape of the leaky routing output given a rank 3 input.

    When using leaky routing the some of routing logits should be always less
    than 1 because some portion of the probability is leaked.
    """
    logits = tf.ones((2, 3, 4))
    leaky = layers._leaky_routing(logits, 4)
    self.assertEqual(len(leaky.get_shape()), 3)
    with self.test_session() as sess:
      r_leaky = sess.run(leaky)
    self.assertEqual(np.array(r_leaky).shape, logits.get_shape())
    self.assertTrue(np.less(np.sum(r_leaky, axis=2), 1.0).all(keepdims=False))

  def testUpdateRouting(self):
    """Tests the correct shape of the output of update_routing function.

    Checks that routing iterations change the activation value.
    """
    votes = np.reshape(np.arange(8, dtype=np.float32), (1, 2, 2, 2))
    biases = tf.zeros((2, 2))
    logit_shape = (1, 2, 2)
    activations_1 = layers._update_routing(
        votes,
        biases,
        logit_shape,
        num_dims=4,
        input_dim=2,
        num_routing=1,
        output_dim=2,
        leaky=False)
    activations_2 = layers._update_routing(
        votes,
        biases,
        logit_shape,
        num_dims=4,
        input_dim=2,
        num_routing=1,
        output_dim=2,
        leaky=False)
    activations_3 = layers._update_routing(
        votes,
        biases,
        logit_shape,
        num_dims=4,
        input_dim=2,
        num_routing=30,
        output_dim=2,
        leaky=False)
    self.assertEqual(len(activations_3.get_shape()), 3)
    self.assertEqual(len(activations_1.get_shape()), 3)
    with self.test_session() as sess:
      act_1, act_2, act_3 = sess.run(
          [activations_1, activations_2, activations_3])
    self.assertNotAlmostEquals(np.sum(act_1), np.sum(act_3))
    self.assertAlmostEquals(np.sum(act_1), np.sum(act_2))

  def testCapsule(self):
    """Tests the correct output and variable declaration of layers.capsule."""
    input_tensor = tf.random_uniform((4, 3, 2))
    output = layers.capsule(
        input_tensor=input_tensor,
        input_dim=3,
        output_dim=2,
        layer_name='capsule',
        input_atoms=2,
        output_atoms=5,
        num_routing=3,
        leaky=False)
    self.assertListEqual(output.get_shape().as_list(), [4, 2, 5])
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    self.assertEqual(len(trainable_vars), 2)
    self.assertStartsWith(trainable_vars[0].name, 'capsule')

  def testConvSlimCapsule(self):
    """Tests the creation of layers.conv_slim_capsule.

    Tests the correct number of variables are declared and shape of the output
    is as a 5D list with the correct numbers.
    """
    input_tensor = tf.random_uniform((6, 4, 2, 3, 3))
    output = layers.conv_slim_capsule(
        input_tensor=input_tensor,
        input_dim=4,
        output_dim=2,
        layer_name='conv_capsule',
        input_atoms=2,
        output_atoms=5,
        stride=1,
        kernel_size=2,
        padding='SAME',
        num_routing=3,
        leaky=False)
    self.assertListEqual(output.get_shape().as_list(), [None, 2, 5, 3, 3])
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    self.assertEqual(len(trainable_vars), 2)
    self.assertStartsWith(trainable_vars[0].name, 'conv_capsule')

  def testMarginLoss(self):
    """Checks the correct margin loss output for a simple scenario.

    In the first example it should only penalize the second logit.
    In the second example it should penalize second and third logit.
    """
    labels = [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    logits = [[-0.3, 0.3, 0.9], [1.2, 0.5, -0.5]]
    costs = [[0, 0.5 * 0.2 * 0.2, 0], [0, 0.4 * 0.4, 1.4 * 1.4]]
    sum_costs = np.sum(costs)
    margin_output = layers._margin_loss(
        labels=tf.constant(labels), raw_logits=tf.constant(logits))
    with self.test_session() as sess:
      output = sess.run(margin_output)
    self.assertAlmostEqual(0.5 * sum_costs, np.sum(output), places=6)

  def testEvaluateSingle(self):
    """Tests layers.evaluate for loss collection and evaluation metrics.

    Checks whether the size of loss collection is increased correctly and
    the accuracy is correct in single digit scenario.

    We have a correct prediction for first example and a wrong one for the
    second example. In case of third example since all three logits are the same
    it outputs the first one as the prediction which is a correct one.
    """
    labels = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    logits = [[-0.3, 0.3, 0.9], [1.2, 0.5, -0.5], [0.0, 0.0, 0.0]]
    _, correct_sum, almost_correct_sum = layers.evaluate(
        logits=tf.constant(logits),
        labels=tf.constant(labels),
        num_targets=1,
        scope='',
        loss_type='softmax')
    losses = tf.get_collection('losses')
    self.assertEqual(len(losses), 1)
    with self.test_session() as sess:
      r_correct, r_almost = sess.run([correct_sum, almost_correct_sum])
    self.assertEqual(2, r_correct)
    self.assertEqual(2, r_almost)

  def testEvaluateMulti(self):
    """Tests layers.evaluate for loss collection and evaluation metrics.

    Checks whether the size of loss collection is increased correctly and
    the accuracy is correct in multi digit scenario.

    We have an almost correct prediction for first example and a wrong one for
    the
    second example. In case of third example since all three logits are the same
    it outputs the first two as the prediction which is a correct one.
    """
    labels = [[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]
    logits = [[-0.3, 0.3, 0.9, 0.0], [1.2, 0.5, -0.5, 0.0],
              [0.0, 0.0, 0.0, 0.0]]
    _, correct_sum, almost_correct_sum = layers.evaluate(
        logits=tf.constant(logits),
        labels=tf.constant(labels),
        num_targets=2,
        scope='',
        loss_type='softmax')
    losses = tf.get_collection('losses')
    self.assertEqual(len(losses), 1)
    with self.test_session() as sess:
      r_correct, r_almost = sess.run([correct_sum, almost_correct_sum])
    self.assertEqual(1, r_correct)
    self.assertEqual(2, r_almost)

  def testReconstruction(self):
    """Tests layers.reconstruction output and variable declaration.

    Checks the correct number of variables are added to the trainable collection
    and the output size is the same as image.

    Reconstruction layer addes 3 fully connected layers therefore it should add
    6 (3 weights, 3 biases) to the trainable variable collection.
    """
    image = tf.random_uniform([2, 9])
    target = [2, 7]
    embedding = tf.random_uniform([2, 10, 4])
    reconstruction = layers.reconstruction(
        capsule_mask=tf.one_hot(target, 10),
        num_atoms=4,
        capsule_embedding=embedding,
        layer_sizes=(5, 10),
        num_pixels=9,
        reuse=False,
        image=image,
        balance_factor=0.1)
    self.assertListEqual([2, 9], reconstruction.get_shape().as_list())
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    self.assertEqual(len(train_vars), 6)

  def testReconstructionMasking(self):
    """Tests layers.reconstruction masking mechanism.

    Masking enforces that only logit values at the reconstruction target affects
    the reconstruction. Therefore, since capsule_output1 and capsule_output3
    both has values at other digits they result in the same
    reconstruction. While capsule_output2 results in a different reconstruction.
    capsule_output2 is the only one with different logit values at the
    reconstruction target.
    """
    image = tf.zeros([1, 9], dtype=tf.float32)
    capsule_mask = tf.one_hot([2], 10)
    embedding1 = np.zeros((1, 10, 4), dtype=np.float32)
    embedding1[:, 1, :] = 1
    embedding2 = np.zeros((1, 10, 4), dtype=np.float32)
    embedding2[:, 2, :] = 1
    embedding3 = np.zeros((1, 10, 4), dtype=np.float32)
    embedding3[:, 3, :] = 10
    reconstruction1 = layers.reconstruction(
        capsule_mask=capsule_mask,
        num_atoms=4,
        capsule_embedding=embedding1,
        layer_sizes=(5, 10),
        num_pixels=9,
        reuse=False,
        image=image,
        balance_factor=0.1)
    reconstruction2 = layers.reconstruction(
        capsule_mask=capsule_mask,
        num_atoms=4,
        capsule_embedding=embedding2,
        layer_sizes=(5, 10),
        num_pixels=9,
        reuse=True,
        image=image,
        balance_factor=0.1)
    reconstruction3 = layers.reconstruction(
        capsule_mask=capsule_mask,
        num_atoms=4,
        capsule_embedding=embedding3,
        layer_sizes=(5, 10),
        num_pixels=9,
        reuse=True,
        image=image,
        balance_factor=0.1)
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    self.assertEqual(len(train_vars), 6)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      r_1, r_2, r_3 = sess.run(
          [reconstruction1, reconstruction2, reconstruction3])
    self.assertAlmostEqual(np.sum(r_1), np.sum(r_3))
    self.assertNotAlmostEqual(np.sum(r_2), np.sum(r_1))


if __name__ == '__main__':
  tf.test.main()
