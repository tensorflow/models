# Copyright 2018 The TensorFlow Authors.
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

"""Tests the Python wrapping of the median_filter library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from light_curve_util.cc.python import median_filter


class MedianFilterTest(absltest.TestCase):

  def testError(self):
    x = [2, 0, 1]
    y = [1, 2, 3]
    with self.assertRaises(ValueError):
      median_filter.median_filter(
          x, y, num_bins=2, bin_width=1, x_min=0, x_max=2)

  def testMedianFilter(self):
    x = np.arange(-6, 7)
    y = np.arange(1, 14)

    result = median_filter.median_filter(
        x, y, num_bins=5, bin_width=2, x_min=-5, x_max=5)

    expected = [2.5, 4.5, 6.5, 8.5, 10.5]
    np.testing.assert_almost_equal(result, expected)


if __name__ == "__main__":
  absltest.main()
