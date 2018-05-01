"""Tests for kepler_spline.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from third_party.kepler_spline import kepler_spline


class KeplerSplineTest(absltest.TestCase):

  def testKeplerSplineSine(self):
    # Fit a sine wave.
    time = np.arange(0, 10, 0.1)
    flux = np.sin(time)

    # Expect very close fit with no outliers removed.
    spline, mask = kepler_spline.kepler_spline(time, flux, bkspace=0.5)
    rmse = np.sqrt(np.mean((flux[mask] - spline[mask])**2))
    self.assertLess(rmse, 1e-4)
    self.assertTrue(np.all(mask))

    # Add some outliers.
    flux[35] = 10
    flux[77] = -3
    flux[95] = 2.9

    # Expect a close fit with outliers removed.
    spline, mask = kepler_spline.kepler_spline(time, flux, bkspace=0.5)
    rmse = np.sqrt(np.mean((flux[mask] - spline[mask])**2))
    self.assertLess(rmse, 1e-4)
    self.assertEqual(np.sum(mask), 97)
    self.assertFalse(mask[35])
    self.assertFalse(mask[77])
    self.assertFalse(mask[95])

    # Increase breakpoint spacing. Fit is not quite as close.
    spline, mask = kepler_spline.kepler_spline(time, flux, bkspace=1)
    rmse = np.sqrt(np.mean((flux[mask] - spline[mask])**2))
    self.assertLess(rmse, 2e-3)
    self.assertEqual(np.sum(mask), 97)
    self.assertFalse(mask[35])
    self.assertFalse(mask[77])
    self.assertFalse(mask[95])

  def testKeplerSplineCubic(self):
    # Fit a cubic polynomial.
    time = np.arange(0, 10, 0.1)
    flux = (time - 5)**3 + 2 * (time - 5)**2 + 10

    # Expect very close fit with no outliers removed. We choose maxiter=1,
    # because a cubic spline will fit a cubic polynomial ~exactly, so the
    # standard deviation of residuals will be ~0, which will cause some closely
    # fit points to be rejected.
    spline, mask = kepler_spline.kepler_spline(
        time, flux, bkspace=0.5, maxiter=1)
    rmse = np.sqrt(np.mean((flux[mask] - spline[mask])**2))
    self.assertLess(rmse, 1e-12)
    self.assertTrue(np.all(mask))

  def testKeplerSplineError(self):
    # Big gap.
    time = np.concatenate([np.arange(0, 1, 0.1), [2]])
    flux = np.sin(time)

    with self.assertRaises(kepler_spline.SplineError):
      kepler_spline.kepler_spline(time, flux, bkspace=0.5)

  def testChooseKeplerSpline(self):
    # High frequency sine wave.
    time = [np.arange(0, 100, 0.1), np.arange(100, 200, 0.1)]
    flux = [np.sin(t) for t in time]

    # Logarithmically sample candidate break point spacings.
    bkspaces = np.logspace(np.log10(0.5), np.log10(5), num=20)

    def _rmse(all_flux, all_spline):
      f = np.concatenate(all_flux)
      s = np.concatenate(all_spline)
      return np.sqrt(np.mean((f - s)**2))

    # Penalty coefficient 1.0.
    spline, mask, bkspace, bad_bkspaces = kepler_spline.choose_kepler_spline(
        time, flux, bkspaces, penalty_coeff=1.0)
    self.assertAlmostEqual(_rmse(flux, spline), 0.013013)
    self.assertTrue(np.all(mask))
    self.assertAlmostEqual(bkspace, 1.67990914314)
    self.assertEmpty(bad_bkspaces)

    # Decrease penalty coefficient; allow smaller spacing for closer fit.
    spline, mask, bkspace, bad_bkspaces = kepler_spline.choose_kepler_spline(
        time, flux, bkspaces, penalty_coeff=0.1)
    self.assertAlmostEqual(_rmse(flux, spline), 0.0066376)
    self.assertTrue(np.all(mask))
    self.assertAlmostEqual(bkspace, 1.48817572082)
    self.assertEmpty(bad_bkspaces)

    # Increase penalty coefficient; require larger spacing at the cost of worse
    # fit.
    spline, mask, bkspace, bad_bkspaces = kepler_spline.choose_kepler_spline(
        time, flux, bkspaces, penalty_coeff=2)
    self.assertAlmostEqual(_rmse(flux, spline), 0.026215449)
    self.assertTrue(np.all(mask))
    self.assertAlmostEqual(bkspace, 1.89634509537)
    self.assertEmpty(bad_bkspaces)


if __name__ == "__main__":
  absltest.main()
