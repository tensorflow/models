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

"""Tests for losses."""

import tensorflow as tf
from official.projects.const_cl.losses import losses


class LossesTest(tf.test.TestCase):

  def test_constrative_loss(self):
    contrastive_loss = losses.ContrastiveLoss(normalize_inputs=True,
                                              temperature=0.1)
    inputs1 = tf.constant(
        [[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1], [8, 7, 6, 5]],
        dtype=tf.float32)
    inputs2 = tf.constant(
        [[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8], [8, 7, 6, 5]],
        dtype=tf.float32)
    inputs = tf.concat([inputs1, inputs2], axis=0)
    contrastive_loss_dict = contrastive_loss(inputs)

    self.assertAlmostEqual(contrastive_loss_dict['contrastive_accuracy'], 0.5)
    self.assertAlmostEqual(contrastive_loss_dict['loss'], 4.136947, places=4)

  def test_instance_constrative_loss(self):
    instance_contrastive_loss = losses.InstanceContrastiveLoss(
        normalize_inputs=True, temperature=0.1)
    inst_a = tf.constant(
        [[[1, 2, 3, 4], [5, 6, 7, 8], [-1, -1, -1, -1], [-1, -1, -1, -1]],
         [[-1, -1, -1, -1], [-1, -1, -1, -1], [4, 3, 2, 1], [8, 7, 6, 5]],
         [[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1], [8, 7, 6, 5]]],
        dtype=tf.float32)
    inst_b = tf.constant([[[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5],
                           [-1, -1, -1, -1], [-1, -1, -1, -1]],
                          [[-1, -1, -1, -1], [-1, -1, -1, -1],
                           [5.5, 6.5, 7.5, 8.5], [8.5, 7.5, 6.5, 5.5]],
                          [[1.5, 2.5, 3.5, 4.5], [4.5, 3.5, 2.5, 1.5],
                           [5.5, 6.5, 7.5, 8.5], [8.5, 7.5, 6.5, 5.5]]],
                         dtype=tf.float32)

    inst_a2b = inst_b
    inst_b2a = inst_a

    masks_a = tf.constant(
        [[True, True, False, False],
         [False, False, True, True],
         [True, True, True, True]], dtype=tf.bool)
    masks_b = tf.constant(
        [[True, True, False, False],
         [False, False, True, True],
         [True, True, True, True]], dtype=tf.bool)

    predictions = {
        'instances_a': inst_a,
        'instances_b': inst_b,
        'instances_a2b': inst_a2b,
        'instances_b2a': inst_b2a,
        'masks_a': masks_a,
        'masks_b': masks_b}
    contrastive_loss_dict = instance_contrastive_loss(
        predictions=predictions)

    self.assertContainsSubset(
        list(contrastive_loss_dict.keys()), [
            'loss', 'positive_similarity_mean', 'positive_similarity_min',
            'positive_similarity_max', 'negative_similarity_mean',
            'negative_similarity_min', 'negative_similarity_max'
        ])

if __name__ == '__main__':
  tf.test.main()
