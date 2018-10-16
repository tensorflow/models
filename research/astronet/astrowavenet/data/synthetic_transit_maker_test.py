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

"""Tests for synthetic_transit_maker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from astrowavenet.data import synthetic_transit_maker


class SyntheticTransitMakerTest(absltest.TestCase):

  def testBadRangesRaiseExceptions(self):

    # Period range cannot contain negative values.
    with self.assertRaisesRegexp(ValueError, "Period"):
      synthetic_transit_maker.SyntheticTransitMaker(period_range=(-1, 10))

    # Amplitude range cannot contain negative values.
    with self.assertRaisesRegexp(ValueError, "Amplitude"):
      synthetic_transit_maker.SyntheticTransitMaker(amplitude_range=(-10, -1))

    # Threshold ratio range must be contained in the half-open interval [0, 1).
    with self.assertRaisesRegexp(ValueError, "Threshold ratio"):
      synthetic_transit_maker.SyntheticTransitMaker(
          threshold_ratio_range=(0, 1))

    # Noise standard deviation range must only contain nonnegative values.
    with self.assertRaisesRegexp(ValueError, "Noise standard deviation"):
      synthetic_transit_maker.SyntheticTransitMaker(noise_sd_range=(-1, 1))

    # End of range may not be less than start.
    invalid_range = (0.2, 0.1)
    range_args = [
        "period_range", "threshold_ratio_range", "amplitude_range",
        "noise_sd_range", "phase_range"
    ]
    for range_arg in range_args:
      with self.assertRaisesRegexp(ValueError, "may not be less"):
        synthetic_transit_maker.SyntheticTransitMaker(
            **{range_arg: invalid_range})

  def testStochasticLightCurveGeneration(self):
    transit_maker = synthetic_transit_maker.SyntheticTransitMaker()

    time = np.arange(100)
    flux, mask = transit_maker.random_light_curve(time, mask_prob=0.4)
    self.assertEqual(len(flux), 100)
    self.assertEqual(len(mask), 100)

  def testDeterministicLightCurveGeneration(self):
    gold_flux = np.array([
        0., 0., 0., 0., 0., 0., 0., -0.85099258, -2.04776251, -2.65829632,
        -2.53014378, -1.69530454, -0.36223792, 0., 0., 0., 0., 0., 0.,
        -0.2110405, -1.57757635, -2.47528153, -2.67999913, -2.14061117,
        -0.9918028, 0., 0., 0., 0., 0., 0., 0., -1.01475559, -2.15534176,
        -2.68282928, -2.46550457, -1.55763357, -0.18591162, 0., 0., 0., 0., 0.,
        0., -0.3870683, -1.71426199, -2.53849461, -2.65395535, -2.03181367,
        -0.82741829, 0., 0., 0., 0., 0., 0., 0., -1.17380391, -2.2541162,
        -2.69666588, -2.39094831, -1.41330116, -0.00784284, 0., 0., 0., 0., 0.,
        0., -0.56063229, -1.84372452, -2.59152891, -2.61731875, -1.91465433,
        -0.65899089, 0., 0., 0., 0., 0., 0., 0., -1.3275672, -2.34373163,
        -2.69975648, -2.30674237, -1.26282489, 0., 0., 0., 0., 0., 0., 0.,
        -0.73111006, -1.9654997, -2.63419424, -2.5702207, -1.78955328,
        -0.48712456
    ])

    # Use ranges containing one value for determinism.
    transit_maker = synthetic_transit_maker.SyntheticTransitMaker(
        period_range=(2, 2),
        amplitude_range=(3, 3),
        threshold_ratio_range=(.1, .1),
        phase_range=(0, 0),
        noise_sd_range=(0, 0))

    time = np.linspace(0, 100, 100)

    flux, mask = transit_maker.random_light_curve(time)
    np.testing.assert_array_almost_equal(flux, gold_flux)
    np.testing.assert_array_almost_equal(mask, np.ones(100))

  def testRandomLightCurveGenerator(self):
    transit_maker = synthetic_transit_maker.SyntheticTransitMaker()
    time = np.linspace(0, 100, 100)
    generator = transit_maker.random_light_curve_generator(
        time, mask_prob=0.3)()
    for _ in range(5):
      flux, mask = next(generator)
      self.assertEqual(len(flux), 100)
      self.assertEqual(len(mask), 100)


if __name__ == "__main__":
  absltest.main()
