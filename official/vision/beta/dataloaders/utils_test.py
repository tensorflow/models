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

"""Tests for dataloader utils functions."""

# Import libraries

from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.dataloaders import utils


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_process_empty_source_id(self):
    source_id = tf.constant([], dtype=tf.int64)
    source_id = tf.strings.as_string(source_id)
    self.assertEqual(-1, utils.process_source_id(source_id=source_id))

  @parameterized.parameters(
      ([128, 256], [128, 256]),
      ([128, 32, 16], [128, 32, 16]),
  )
  def test_process_source_id(self, source_id, expected_result):
    source_id = tf.constant(source_id, dtype=tf.int64)
    source_id = tf.strings.as_string(source_id)
    self.assertSequenceAlmostEqual(expected_result,
                                   utils.process_source_id(source_id=source_id))

  @parameterized.parameters(
      ([[10, 20, 30, 40]], [[100]], [[0]], 10),
      ([[0.1, 0.2, 0.5, 0.6]], [[0.5]], [[1]], 2),
  )
  def test_pad_groundtruths_to_fixed_size(self, boxes, area, classes, size):
    groundtruths = {}
    groundtruths['boxes'] = tf.constant(boxes)
    groundtruths['is_crowds'] = tf.constant([[0]])
    groundtruths['areas'] = tf.constant(area)
    groundtruths['classes'] = tf.constant(classes)

    actual_result = utils.pad_groundtruths_to_fixed_size(
        groundtruths=groundtruths, size=size)

    # Check that the first dimension is padded to the expected size.
    for key in actual_result:
      pad_shape = actual_result[key].shape[0]
      self.assertEqual(size, pad_shape)


if __name__ == '__main__':
  tf.test.main()
