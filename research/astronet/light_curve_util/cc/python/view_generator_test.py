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

"""Tests the Python wrapping of the view_generator library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from light_curve_util.cc.python import view_generator


class ViewGeneratorTest(absltest.TestCase):

  def testPrivateConstructorNotVisible(self):
    time = [1, 2, 3]
    flux = [2, 3]
    with self.assertRaises(ValueError):
      view_generator.ViewGenerator(time, flux)

  def testCreationError(self):
    time = [1, 2, 3]
    flux = [2, 3]
    with self.assertRaises(ValueError):
      view_generator.create_view_generator(time, flux, period=1, t0=0.5)

  def testGenerateViews(self):
    time = np.arange(0, 2, 0.1)
    flux = np.arange(0, 20, 1)

    vg = view_generator.create_view_generator(time, flux, period=2, t0=0.15)

    with self.assertRaises(ValueError):
      vg.generate_view(
          num_bins=10, bin_width=0.2, t_min=-1, t_max=-1, normalize=False)

    # Global view, unnormalized.
    result = vg.generate_view(
        num_bins=10, bin_width=0.2, t_min=-1, t_max=1, normalize=False)
    expected = [12.5, 14.5, 16.5, 18.5, 0.5, 2.5, 4.5, 6.5, 8.5, 10.5]
    np.testing.assert_almost_equal(result, expected)

    # Global view, normalized.
    result = vg.generate_view(
        num_bins=10, bin_width=0.2, t_min=-1, t_max=1, normalize=True)
    expected = [
        3.0 / 9, 5.0 / 9, 7.0 / 9, 9.0 / 9, -9.0 / 9, -7.0 / 9, -5.0 / 9,
        -3.0 / 9, -1.0 / 9, 1.0 / 9
    ]
    np.testing.assert_almost_equal(result, expected)

    # Local view, unnormalized.
    result = vg.generate_view(
        num_bins=5, bin_width=0.2, t_min=-0.5, t_max=0.5, normalize=False)
    expected = [17.5, 9.5, 1.5, 3.5, 5.5]
    np.testing.assert_almost_equal(result, expected)

    # Local view, normalized.
    result = vg.generate_view(
        num_bins=5, bin_width=0.2, t_min=-0.5, t_max=0.5, normalize=True)
    expected = [3, 1, -1, -0.5, 0]
    np.testing.assert_almost_equal(result, expected)


if __name__ == '__main__':
  absltest.main()
