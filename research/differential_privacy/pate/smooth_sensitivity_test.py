# Copyright 2017 The 'Scalable Private Learning with PATE' Authors All Rights Reserved.
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

"""Tests for pate.smooth_sensitivity."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

import smooth_sensitivity as pate_ss


class PateSmoothSensitivityTest(unittest.TestCase):

  def test_check_conditions(self):
    self.assertEqual(pate_ss.check_conditions(20, 10, 25.), (True, False))
    self.assertEqual(pate_ss.check_conditions(30, 10, 25.), (True, True))

  def _assert_all_close(self, x, y):
    """Asserts that two numpy arrays are close."""
    self.assertEqual(len(x), len(y))
    self.assertTrue(np.allclose(x, y, rtol=1e-8, atol=0))

  def test_compute_local_sensitivity_bounds_gnmax(self):
    counts1 = np.array([10, 0, 0])
    sigma1 = .5
    order1 = 1.5

    answer1 = np.array(
        [3.13503646e-17, 1.60178280e-08, 5.90681786e-03] + [5.99981308e+00] * 7)

    # Test for "going right" in the smooth sensitivity computation.
    out1 = pate_ss.compute_local_sensitivity_bounds_gnmax(
        counts1, 10, sigma1, order1)

    self._assert_all_close(out1, answer1)

    counts2 = np.array([1000, 500, 300, 200, 0])
    sigma2 = 250.
    order2 = 10.

    # Test for "going left" in the smooth sensitivity computation.
    out2 = pate_ss.compute_local_sensitivity_bounds_gnmax(
        counts2, 2000, sigma2, order2)

    answer2 = np.array([0.] * 298 + [2.77693450548e-7, 2.10853979548e-6] +
                       [2.73113623988e-6] * 1700)
    self._assert_all_close(out2, answer2)

  def test_compute_local_sensitivity_bounds_threshold(self):
    counts1_3 = np.array([20, 10, 0])
    num_teachers = sum(counts1_3)
    t1 = 16  # high threshold
    sigma = 2
    order = 10

    out1 = pate_ss.compute_local_sensitivity_bounds_threshold(
        counts1_3, num_teachers, t1, sigma, order)
    answer1 = np.array([0] * 3 + [
        1.48454129e-04, 1.47826870e-02, 3.94153241e-02, 6.45775697e-02,
        9.01543247e-02, 1.16054002e-01, 1.42180452e-01, 1.42180452e-01,
        1.48454129e-04, 1.47826870e-02, 3.94153241e-02, 6.45775697e-02,
        9.01543266e-02, 1.16054000e-01, 1.42180452e-01, 1.68302106e-01,
        1.93127860e-01
    ] + [0] * 10)
    self._assert_all_close(out1, answer1)

    t2 = 2  # low threshold

    out2 = pate_ss.compute_local_sensitivity_bounds_threshold(
        counts1_3, num_teachers, t2, sigma, order)
    answer2 = np.array([
        1.60212079e-01, 2.07021132e-01, 2.07021132e-01, 1.93127860e-01,
        1.68302106e-01, 1.42180452e-01, 1.16054002e-01, 9.01543247e-02,
        6.45775697e-02, 3.94153241e-02, 1.47826870e-02, 1.48454129e-04
    ] + [0] * 18)
    self._assert_all_close(out2, answer2)

    t3 = 50  # very high threshold (larger than the number of teachers).

    out3 = pate_ss.compute_local_sensitivity_bounds_threshold(
        counts1_3, num_teachers, t3, sigma, order)

    answer3 = np.array([
        1.35750725752e-19, 1.88990500499e-17, 2.05403154065e-15,
        1.74298153642e-13, 1.15489723995e-11, 5.97584949325e-10,
        2.41486826748e-08, 7.62150641922e-07, 1.87846248741e-05,
        0.000360973025976, 0.000360973025976, 2.76377015215e-50,
        1.00904975276e-53, 2.87254164748e-57, 6.37583360761e-61,
        1.10331620211e-64, 1.48844393335e-68, 1.56535552444e-72,
        1.28328011060e-76, 8.20047697109e-81
    ] + [0] * 10)

    self._assert_all_close(out3, answer3)

    # Fractional values.
    counts4 = np.array([19.5, -5.1, 0])
    t4 = 10.1
    out4 = pate_ss.compute_local_sensitivity_bounds_threshold(
        counts4, num_teachers, t4, sigma, order)

    answer4 = np.array([
        0.0620410301, 0.0875807131, 0.113451958, 0.139561671, 0.1657074530,
        0.1908244840, 0.2070270720, 0.207027072, 0.169718100, 0.0575152142,
        0.00678695871
    ] + [0] * 6 + [0.000536304908, 0.0172181073, 0.041909870] + [0] * 10)
    self._assert_all_close(out4, answer4)


if __name__ == "__main__":
  unittest.main()
