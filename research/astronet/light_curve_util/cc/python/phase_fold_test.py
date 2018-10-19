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

"""Tests the Python wrapping of the phase_fold library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from light_curve_util.cc.python import phase_fold


class PhaseFoldTimeTest(absltest.TestCase):

  def testEmpty(self):
    result = phase_fold.phase_fold_time(time=[], period=1, t0=0.45)
    self.assertEmpty(result)

  def testSimple(self):
    time = np.arange(0, 2, 0.1)
    result = phase_fold.phase_fold_time(time, period=1, t0=0.45)
    expected = [
        -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, -0.45,
        -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45
    ]
    np.testing.assert_almost_equal(result, expected)


class PhaseFoldAndSortLightCurveTest(absltest.TestCase):

  def testError(self):
    with self.assertRaises(ValueError):
      phase_fold.phase_fold_and_sort_light_curve(
          time=[1, 2, 3], flux=[7.5, 8.6], period=1, t0=0.5)

  def testFoldAndSort(self):
    time = np.arange(0, 2, 0.1)
    flux = np.arange(0, 20, 1)

    folded_time, folded_flux = phase_fold.phase_fold_and_sort_light_curve(
        time, flux, period=2, t0=0.15)

    expected_time = [
        -0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05,
        0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95
    ]
    np.testing.assert_almost_equal(folded_time, expected_time)

    expected_flux = [
        12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    ]
    np.testing.assert_almost_equal(folded_flux, expected_flux)


if __name__ == "__main__":
  absltest.main()
