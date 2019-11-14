# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
"""Tests for test_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from absl.testing import absltest
import numpy as np

from deeplab.evaluation import test_utils


class TestUtilsTest(absltest.TestCase):

  def test_read_test_image(self):
    image_array = test_utils.read_test_image('team_pred_class.png')
    self.assertSequenceEqual(image_array.shape, (231, 345, 4))

  def test_reads_segmentation_with_color_map(self):
    rgb_to_semantic_label = {(0, 0, 0): 0, (0, 0, 255): 1, (255, 0, 0): 23}
    labels = test_utils.read_segmentation_with_rgb_color_map(
        'team_pred_class.png', rgb_to_semantic_label)

    input_image = test_utils.read_test_image('team_pred_class.png')
    np.testing.assert_array_equal(
        labels == 0,
        np.logical_and(input_image[:, :, 0] == 0, input_image[:, :, 2] == 0))
    np.testing.assert_array_equal(labels == 1, input_image[:, :, 2] == 255)
    np.testing.assert_array_equal(labels == 23, input_image[:, :, 0] == 255)

  def test_reads_gt_segmentation(self):
    instance_label_to_semantic_label = {
        0: 0,
        47: 1,
        97: 1,
        133: 1,
        150: 1,
        174: 1,
        198: 23,
        215: 1,
        244: 1,
        255: 1,
    }
    instances, classes = test_utils.panoptic_segmentation_with_class_map(
        'team_gt_instance.png', instance_label_to_semantic_label)

    expected_label_shape = (231, 345)
    self.assertSequenceEqual(instances.shape, expected_label_shape)
    self.assertSequenceEqual(classes.shape, expected_label_shape)
    np.testing.assert_array_equal(instances == 0, classes == 0)
    np.testing.assert_array_equal(instances == 198, classes == 23)
    np.testing.assert_array_equal(
        np.logical_and(instances != 0, instances != 198), classes == 1)


if __name__ == '__main__':
  absltest.main()
