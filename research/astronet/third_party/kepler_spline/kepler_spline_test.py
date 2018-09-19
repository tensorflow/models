"""Tests for kepler_spline.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from third_party.kepler_spline import kepler_spline


class KeplerSplineTest(absltest.TestCase):

  def testFitSine(self):
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

  def testFitCubic(self):
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

  def testInsufficientPointsError(self):
    # Empty light curve.
    time = np.array([])
    flux = np.array([])

    with self.assertRaises(kepler_spline.InsufficientPointsError):
      kepler_spline.kepler_spline(time, flux, bkspace=0.5)

    # Only 3 points.
    time = np.array([0.1, 0.2, 0.3])
    flux = np.sin(time)

    with self.assertRaises(kepler_spline.InsufficientPointsError):
      kepler_spline.kepler_spline(time, flux, bkspace=0.5)


class ChooseKeplerSplineTest(absltest.TestCase):

  def testEmptyInput(self):
    # Logarithmically sample candidate break point spacings.
    bkspaces = np.logspace(np.log10(0.5), np.log10(5), num=20)

    spline, metadata = kepler_spline.choose_kepler_spline(
        all_time=[],
        all_flux=[],
        bkspaces=bkspaces,
        penalty_coeff=1.0,
        verbose=False)
    np.testing.assert_array_equal(spline, [])
    np.testing.assert_array_equal(metadata.light_curve_mask, [])

  def testNoPoints(self):
    all_time = [np.array([])]
    all_flux = [np.array([])]

    # Logarithmically sample candidate break point spacings.
    bkspaces = np.logspace(np.log10(0.5), np.log10(5), num=20)

    spline, metadata = kepler_spline.choose_kepler_spline(
        all_time, all_flux, bkspaces, penalty_coeff=1.0, verbose=False)
    np.testing.assert_array_equal(spline, [[]])
    np.testing.assert_array_equal(metadata.light_curve_mask, [[]])

  def testTooFewPoints(self):
    # Sine wave with segments of 1, 2, 3 points.
    all_time = [
        np.array([0.1]),
        np.array([0.2, 0.3]),
        np.array([0.4, 0.5, 0.6])
    ]
    all_flux = [np.sin(t) for t in all_time]

    # Logarithmically sample candidate break point spacings.
    bkspaces = np.logspace(np.log10(0.5), np.log10(5), num=20)

    spline, metadata = kepler_spline.choose_kepler_spline(
        all_time, all_flux, bkspaces, penalty_coeff=1.0, verbose=False)

    # All segments are NaN.
    self.assertTrue(np.all(np.isnan(np.concatenate(spline))))
    self.assertFalse(np.any(np.concatenate(metadata.light_curve_mask)))
    self.assertIsNone(metadata.bkspace)
    self.assertEmpty(metadata.bad_bkspaces)
    self.assertIsNone(metadata.likelihood_term)
    self.assertIsNone(metadata.penalty_term)
    self.assertIsNone(metadata.bic)

    # Add a longer segment.
    all_time.append(np.arange(0.7, 2.0, 0.1))
    all_flux.append(np.sin(all_time[-1]))

    spline, metadata = kepler_spline.choose_kepler_spline(
        all_time, all_flux, bkspaces, penalty_coeff=1.0, verbose=False)

    # First 3 segments are NaN.
    for i in range(3):
      self.assertTrue(np.all(np.isnan(spline[i])))
      self.assertFalse(np.any(metadata.light_curve_mask[i]))

    # Final segment is a good fit.
    self.assertTrue(np.all(np.isfinite(spline[3])))
    self.assertTrue(np.all(metadata.light_curve_mask[3]))
    self.assertEmpty(metadata.bad_bkspaces)
    self.assertAlmostEqual(metadata.likelihood_term, -58.0794069927957)
    self.assertAlmostEqual(metadata.penalty_term, 7.69484807238461)
    self.assertAlmostEqual(metadata.bic, -50.3845589204111)

  def testFitSine(self):
    # High frequency sine wave.
    all_time = [np.arange(0, 100, 0.1), np.arange(100, 200, 0.1)]
    all_flux = [np.sin(t) for t in all_time]

    # Logarithmically sample candidate break point spacings.
    bkspaces = np.logspace(np.log10(0.5), np.log10(5), num=20)

    def _rmse(all_flux, all_spline):
      f = np.concatenate(all_flux)
      s = np.concatenate(all_spline)
      return np.sqrt(np.mean((f - s)**2))

    # Penalty coefficient 1.0.
    spline, metadata = kepler_spline.choose_kepler_spline(
        all_time, all_flux, bkspaces, penalty_coeff=1.0)
    self.assertAlmostEqual(_rmse(all_flux, spline), 0.013013)
    self.assertTrue(np.all(metadata.light_curve_mask))
    self.assertAlmostEqual(metadata.bkspace, 1.67990914314)
    self.assertEmpty(metadata.bad_bkspaces)
    self.assertAlmostEqual(metadata.likelihood_term, -6685.64217856480)
    self.assertAlmostEqual(metadata.penalty_term, 942.51190498322)
    self.assertAlmostEqual(metadata.bic, -5743.13027358158)

    # Decrease penalty coefficient; allow smaller spacing for closer fit.
    spline, metadata = kepler_spline.choose_kepler_spline(
        all_time, all_flux, bkspaces, penalty_coeff=0.1)
    self.assertAlmostEqual(_rmse(all_flux, spline), 0.0066376)
    self.assertTrue(np.all(metadata.light_curve_mask))
    self.assertAlmostEqual(metadata.bkspace, 1.48817572082)
    self.assertEmpty(metadata.bad_bkspaces)
    self.assertAlmostEqual(metadata.likelihood_term, -6731.59913975551)
    self.assertAlmostEqual(metadata.penalty_term, 1064.12634433589)
    self.assertAlmostEqual(metadata.bic, -6625.18650532192)

    # Increase penalty coefficient; require larger spacing at the cost of worse
    # fit.
    spline, metadata = kepler_spline.choose_kepler_spline(
        all_time, all_flux, bkspaces, penalty_coeff=2)
    self.assertAlmostEqual(_rmse(all_flux, spline), 0.026215449)
    self.assertTrue(np.all(metadata.light_curve_mask))
    self.assertAlmostEqual(metadata.bkspace, 1.89634509537)
    self.assertEmpty(metadata.bad_bkspaces)
    self.assertAlmostEqual(metadata.likelihood_term, -6495.65564287904)
    self.assertAlmostEqual(metadata.penalty_term, 836.099270549629)
    self.assertAlmostEqual(metadata.bic, -4823.45710177978)


if __name__ == "__main__":
  absltest.main()
