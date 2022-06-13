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

# Lint as: python3
from absl.testing import parameterized
from absl import logging
import tensorflow as tf
from official.projects.yt8m.eval_utils.average_precision_calculator import AveragePrecisionCalculator
import numpy as np

from official.projects.yt8m.eval_utils.eval_util import calculate_precision_at_equal_recall_rate



'''
p = np.array([random.random() for _ in xrange(10)])
a = np.array([random.choice([0, 1]) for _ in xrange(10)])

ap = average_precision_calculator.AveragePrecisionCalculator.ap(p, a)

p1 = np.array([random.random() for _ in xrange(5)])
a1 = np.array([random.choice([0, 1]) for _ in xrange(5)])
p2 = np.array([random.random() for _ in xrange(5)])
a2 = np.array([random.choice([0, 1]) for _ in xrange(5)])

# interpolated average precision at 10 using 1000 break points
calculator = average_precision_calculator.AveragePrecisionCalculator(10)
calculator.accumulate(p1, a1)
calculator.accumulate(p2, a2)
ap3 = calculator.peek_ap_at_n()
'''

class YT8MAveragePrecisionCalculatorTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.prediction = np.array([
      [0.98, 0.88, 0.77, 0.65, 0.64, 0.59, 0.45, 0.43, 0.20, 0.05],
      [0.878, 0.832, 0.759, 0.621, 0.458, 0.285, 0.134],
      [0.98],
      [0.56],
    ])
    self.raw_prediction = np.random.rand(5,10) + \
    np.random.randint(low=0, high =10, size=(5,10))
    self.ground_truth = np.array([
      [1,1,0,0,0,1,1,0,0,1],
      [1,0,1,0,0,1,0],
      [1],
      [0]
    ])

    self.expected_ap = np.array([
      0.714,
      0.722,
      1.000,
      0.000,
    ])
    # self.prediction_expand = np.random.rand(10)
    # self.ground_truth_expand = np.random.choice(2,10,True)

  def test_ap_calculator_ap(self):
    
    # Compare Expected Average Precision with function expected
    for i,_ in enumerate(self.ground_truth):
      calculator = AveragePrecisionCalculator()
      ap = calculator.ap(self.prediction[i], self.ground_truth[i])
      self.assertNear(ap,self.expected_ap[i], err=1e-3)
      logging.info(f'DEBUG {i+1}th AP: {ap}')

  def test_ap_calculator_zero_one_normalize(self):
    for i,_ in enumerate(self.raw_prediction):
      calculator = AveragePrecisionCalculator()
      logging.error(f'{self.raw_prediction[i]}')
      normalized_score = calculator._zero_one_normalize(self.raw_prediction[i])
      self.assertAllInRange(normalized_score, lower_bound=0.0, upper_bound=1.0)
  
  @parameterized.parameters(
    (None, None), (3, None), (5,None), (10,None), (20,None)
  )
  def test_ap_calculator_ap_at_n(self, n, total_num_positives ):
    for i,_ in enumerate(self.ground_truth):
      calculator = AveragePrecisionCalculator(n)
      ap = calculator.ap_at_n(self.prediction[i], self.ground_truth[i],n)
      logging.info(f'DEBUG {i+1}th AP: {ap}')


if __name__ == '__main__':
  tf.test.main()