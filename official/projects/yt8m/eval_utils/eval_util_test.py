# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
import numpy as np
import tensorflow as tf

from official.projects.yt8m.eval_utils.average_precision_calculator import AveragePrecisionCalculator


class YT8MAveragePrecisionCalculatorTest(parameterized.TestCase,
                                         tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.prediction = np.array([
        [0.98, 0.88, 0.77, 0.65, 0.64, 0.59, 0.45, 0.43, 0.20, 0.05],
        [0.878, 0.832, 0.759, 0.621, 0.458, 0.285, 0.134],
        [0.98],
        [0.56],
    ])
    self.raw_prediction = np.random.rand(5, 10) + np.random.randint(
        low=0, high=10, size=(5, 10))
    self.ground_truth = np.array([[1, 1, 0, 0, 0, 1, 1, 0, 0, 1],
                                  [1, 0, 1, 0, 0, 1, 0], [1], [0]])

    self.expected_ap = np.array([
        0.714,
        0.722,
        1.000,
        0.000,
    ])

  def test_ap_calculator_ap(self):

    # Compare Expected Average Precision with function expected
    for i, _ in enumerate(self.ground_truth):
      calculator = AveragePrecisionCalculator()
      ap = calculator.ap(self.prediction[i], self.ground_truth[i])
      logging.info('DEBUG %dth AP: %r', i + 1, ap)

  def test_ap_calculator_zero_one_normalize(self):
    for i, _ in enumerate(self.raw_prediction):
      calculator = AveragePrecisionCalculator()
      logging.error('%r', self.raw_prediction[i])
      normalized_score = calculator._zero_one_normalize(self.raw_prediction[i])
      self.assertAllInRange(normalized_score, lower_bound=0.0, upper_bound=1.0)

  @parameterized.parameters((None,), (3,), (5,), (10,), (20,))
  def test_ap_calculator_ap_at_n(self, n):
    for i, _ in enumerate(self.ground_truth):
      calculator = AveragePrecisionCalculator(n)
      ap = calculator.ap_at_n(self.prediction[i], self.ground_truth[i], n)
      logging.info('DEBUG %dth AP: %r', i + 1, ap)


if __name__ == '__main__':
  tf.test.main()
