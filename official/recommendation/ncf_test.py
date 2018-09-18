# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests NCF."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf

from official.recommendation import ncf_main


class NcfTest(tf.test.TestCase):
  def test_hit_rate_and_ndcg(self):
    # Test with no duplicate items
    predictions = np.array([
        [1., 2., 0.],  # In top 2
        [2., 1., 0.],  # In top 1
        [0., 2., 1.],  # In top 3
        [2., 3., 4.]   # In top 3
    ])
    items = np.array([
        [1, 2, 3],
        [2, 3, 1],
        [3, 2, 1],
        [2, 1, 3],
    ])
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 1)
    self.assertAlmostEqual(hr, 1 / 4)
    self.assertAlmostEqual(ndcg, 1 / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 2)
    self.assertAlmostEqual(hr, 2 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3)) / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 3)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3) +
                                  2 * math.log(2) / math.log(4)) / 4)

    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 1,
                                              match_mlperf=True)
    self.assertAlmostEqual(hr, 1 / 4)
    self.assertAlmostEqual(ndcg, 1 / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 2,
                                              match_mlperf=True)
    self.assertAlmostEqual(hr, 2 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3)) / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 3,
                                              match_mlperf=True)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3) +
                                  2 * math.log(2) / math.log(4)) / 4)

    # Test with duplicate items. In the MLPerf case, we treat the duplicates as
    # a single item. Otherwise, we treat the duplicates as separate items.
    predictions = np.array([
        [1., 2., 2., 3.],  # In top 4. MLPerf: In top 3
        [3., 1., 0., 2.],  # In top 1. MLPerf: In top 1
        [0., 2., 3., 2.],  # In top 4. MLPerf: In top 3
        [3., 2., 4., 2.]   # In top 2. MLPerf: In top 2
    ])
    items = np.array([
        [1, 2, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 2],
        [4, 3, 2, 1],
    ])
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 1)
    self.assertAlmostEqual(hr, 1 / 4)
    self.assertAlmostEqual(ndcg, 1 / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 2)
    self.assertAlmostEqual(hr, 2 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3)) / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 3)
    self.assertAlmostEqual(hr, 2 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3)) / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 4)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3) +
                                  2 * math.log(2) / math.log(5)) / 4)

    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 1,
                                              match_mlperf=True)
    self.assertAlmostEqual(hr, 1 / 4)
    self.assertAlmostEqual(ndcg, 1 / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 2,
                                              match_mlperf=True)
    self.assertAlmostEqual(hr, 2 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3)) / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 3,
                                              match_mlperf=True)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3) +
                                  2 * math.log(2) / math.log(4)) / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 4,
                                              match_mlperf=True)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (1 + math.log(2) / math.log(3) +
                                  2 * math.log(2) / math.log(4)) / 4)

    # Test with duplicate items, where the predictions for the same item can
    # differ. In the MLPerf case, we should take the first prediction.
    predictions = np.array([
        [3., 2., 4., 4.],  # In top 3. MLPerf: In top 2
        [3., 4., 2., 4.],  # In top 3. MLPerf: In top 3
        [2., 3., 4., 1.],  # In top 3. MLPerf: In top 2
        [4., 3., 5., 2.]   # In top 2. MLPerf: In top 1
    ])
    items = np.array([
        [1, 2, 2, 3],
        [4, 3, 3, 2],
        [2, 1, 1, 1],
        [4, 2, 2, 1],
    ])
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 1)
    self.assertAlmostEqual(hr, 0 / 4)
    self.assertAlmostEqual(ndcg, 0 / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 2)
    self.assertAlmostEqual(hr, 1 / 4)
    self.assertAlmostEqual(ndcg, (math.log(2) / math.log(3)) / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 3)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (math.log(2) / math.log(3) +
                                  3 * math.log(2) / math.log(4)) / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 4)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (math.log(2) / math.log(3) +
                                  3 * math.log(2) / math.log(4)) / 4)

    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 1,
                                              match_mlperf=True)
    self.assertAlmostEqual(hr, 1 / 4)
    self.assertAlmostEqual(ndcg, 1 / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 2,
                                              match_mlperf=True)
    self.assertAlmostEqual(hr, 3 / 4)
    self.assertAlmostEqual(ndcg, (1 + 2 * math.log(2) / math.log(3)) / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 3,
                                              match_mlperf=True)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (1 + 2 * math.log(2) / math.log(3) +
                                  math.log(2) / math.log(4)) / 4)
    hr, ndcg = ncf_main.get_hit_rate_and_ndcg(predictions, items, 4,
                                              match_mlperf=True)
    self.assertAlmostEqual(hr, 4 / 4)
    self.assertAlmostEqual(ndcg, (1 + 2 * math.log(2) / math.log(3) +
                                  math.log(2) / math.log(4)) / 4)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
