# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
"""Tests for utils.py."""

import numpy as np
import tensorflow as tf

from deeplab.core import utils


class UtilsTest(tf.test.TestCase):

  def testScaleDimensionOutput(self):
    self.assertEqual(161, utils.scale_dimension(321, 0.5))
    self.assertEqual(193, utils.scale_dimension(321, 0.6))
    self.assertEqual(241, utils.scale_dimension(321, 0.75))

  def testGetLabelWeightMask_withFloatLabelWeights(self):
    labels = tf.constant([0, 4, 1, 3, 2])
    ignore_label = 4
    num_classes = 5
    label_weights = 0.5
    expected_label_weight_mask = np.array([0.5, 0.0, 0.5, 0.5, 0.5],
                                          dtype=np.float32)

    with self.test_session() as sess:
      label_weight_mask = utils.get_label_weight_mask(
          labels, ignore_label, num_classes, label_weights=label_weights)
      label_weight_mask = sess.run(label_weight_mask)
      self.assertAllEqual(label_weight_mask, expected_label_weight_mask)

  def testGetLabelWeightMask_withListLabelWeights(self):
    labels = tf.constant([0, 4, 1, 3, 2])
    ignore_label = 4
    num_classes = 5
    label_weights = [0.0, 0.1, 0.2, 0.3, 0.4]
    expected_label_weight_mask = np.array([0.0, 0.0, 0.1, 0.3, 0.2],
                                          dtype=np.float32)

    with self.test_session() as sess:
      label_weight_mask = utils.get_label_weight_mask(
          labels, ignore_label, num_classes, label_weights=label_weights)
      label_weight_mask = sess.run(label_weight_mask)
      self.assertAllEqual(label_weight_mask, expected_label_weight_mask)

  def testGetLabelWeightMask_withInvalidLabelWeightsType(self):
    labels = tf.constant([0, 4, 1, 3, 2])
    ignore_label = 4
    num_classes = 5

    self.assertRaisesWithRegexpMatch(
        ValueError,
        '^The type of label_weights is invalid, it must be a float or a list',
        utils.get_label_weight_mask,
        labels=labels,
        ignore_label=ignore_label,
        num_classes=num_classes,
        label_weights=None)

  def testGetLabelWeightMask_withInvalidLabelWeightsLength(self):
    labels = tf.constant([0, 4, 1, 3, 2])
    ignore_label = 4
    num_classes = 5
    label_weights = [0.0, 0.1, 0.2]

    self.assertRaisesWithRegexpMatch(
        ValueError,
        '^Length of label_weights must be equal to num_classes if it is a list',
        utils.get_label_weight_mask,
        labels=labels,
        ignore_label=ignore_label,
        num_classes=num_classes,
        label_weights=label_weights)


if __name__ == '__main__':
  tf.test.main()
