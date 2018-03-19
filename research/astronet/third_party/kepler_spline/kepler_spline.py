"""Functions for computing normalization splines for Kepler light curves."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np
from pydl.pydlutils import bspline

from third_party.robust_mean import robust_mean


class SplineError(Exception):
  """Error when fitting a Kepler spline."""
  pass


def kepler_spline(time, flux, bkspace=1.5, maxiter=5, outlier_cut=3):
  """Computes a best-fit spline curve for a light curve segment.

  The spline is fit using an iterative process to remove outliers that may cause
  the spline to be "pulled" by discrepent points. In each iteration the spline
  is fit, and if there are any points where the absolute deviation from the
  median residual is at least 3*sigma (where sigma is a robust estimate of the
  standard deviation of the residuals), those points are removed and the spline
  is re-fit.

  Args:
    time: Numpy array; the time values of the light curve.
    flux: Numpy array; the flux (brightness) values of the light curve.
    bkspace: Spline break point spacing in time units.
    maxiter: Maximum number of attempts to fit the spline after removing badly
        fit points.
    outlier_cut: The maximum number of standard deviations from the median
        spline residual before a point is considered an outlier.

  Returns:
    spline: The values of the fitted spline corresponding to the input time
        values.
    mask: Boolean mask indicating the points used to fit the final spline.
  """
  # Rescale time into [0, 1].
  t_min = np.min(time)
  t_max = np.max(time)
  time = (time - t_min) / (t_max - t_min)
  bkspace /= (t_max - t_min)  # Rescale bucket spacing.

  # Values of the best fitting spline evaluated at the time points.
  spline = None

  # Mask indicating the points used to fit the spline.
  mask = None

  for _ in range(maxiter):
    if spline is None:
      mask = np.ones_like(time, dtype=np.bool)  # Try to fit all points.
    else:
      # Choose points where the absolute deviation from the median residual is
      # less than 3*sigma, where sigma is a robust estimate of the standard
      # deviation of the residuals from the previous spline.
      residuals = flux - spline
      _, _, new_mask = robust_mean.robust_mean(residuals, cut=outlier_cut)

      if np.all(new_mask == mask):
        break  # Spline converged.

      mask = new_mask

    try:
      with warnings.catch_warnings():
        # Suppress warning messages printed by pydlutils.bspline. Instead we
        # catch any exception and raise a more informative error.
        warnings.simplefilter("ignore")

        # Fit the spline on non-outlier points.
        curve = bspline.iterfit(time[mask], flux[mask], bkspace=bkspace)[0]

      # Evaluate spline at the time points.
      spline = curve.value(time)[0]
    except (IndexError, TypeError) as e:
      raise SplineError(
          "Fitting spline failed with error: '%s'. This might be caused by the "
          "breakpoint spacing being too small, and/or there being insufficient "
          "points to fit the spline in one of the intervals." % e)

  return spline, mask


def choose_kepler_spline(all_time,
                         all_flux,
                         bkspaces,
                         maxiter=5,
                         penalty_coeff=1.0,
                         verbose=True):
  """Computes the best-fit Kepler spline across a break-point spacings.

  Some Kepler light curves have low-frequency variability, while others have
  very high-frequency variability (e.g. due to rapid rotation). Therefore, it is
  suboptimal to use the same break-point spacing for every star. This function
  computes the best-fit spline by fitting splines with different break-point
  spacings, calculating the Bayesian Information Criterion (BIC) for each
  spline, and choosing the break-point spacing that minimizes the BIC.

  This function assumes a piecewise light curve, that is, a light curve that is
  divided into different segments (e.g. split by quarter breaks or gaps in the
  in the data). A separate spline is fit for each segment.

  Args:
    all_time: List of 1D numpy arrays; the time values of the light curve.
    all_flux: List of 1D numpy arrays; the flux (brightness) values of the light
        curve.
    bkspaces: List of break-point spacings to try.
    maxiter: Maximum number of attempts to fit each spline after removing badly
        fit points.
    penalty_coeff: Coefficient of the penalty term for using more parameters in
        the Bayesian Information Criterion. Decreasing this value will allow
        more parameters to be used (i.e. smaller break-point spacing), and
        vice-versa.
    verbose: Whether to log individual spline errors. Note that if bkspaces
        contains many values (particularly small ones) then this may cause
        logging pollution if calling this function for many light curves.

  Returns:
    spline: List of numpy arrays; values of the best-fit spline corresponding to
        to the input flux arrays.
    spline_mask: List of boolean numpy arrays indicating which points in the
        flux arrays were used to fit the best-fit spline.
    bkspace: The break-point spacing used for the best-fit spline.
    bad_bkspaces: List of break-point spacing values that failed.
  """
  # Compute the assumed standard deviation of Gaussian white noise about the
  # spline model.
  abs_deviations = np.concatenate([np.abs(f[1:] - f[:-1]) for f in all_flux])
  sigma = np.median(abs_deviations) * 1.48 / np.sqrt(2)

  best_bic = None
  best_spline = None
  best_spline_mask = None
  best_bkspace = None
  bad_bkspaces = []
  for bkspace in bkspaces:
    nparams = 0  # Total number of free parameters in the piecewise spline.
    npoints = 0  # Total number of data points used to fit the piecewise spline.
    ssr = 0  # Sum of squared residuals between the model and the spline.

    spline = []
    spline_mask = []
    bad_bkspace = False  # Indicates that the current bkspace should be skipped.
    for time, flux in zip(all_time, all_flux):
      # Don't fit a spline on less than 4 points.
      if len(time) < 4:
        spline.append(flux)
        spline_mask.append(np.ones_like(flux), dtype=np.bool)
        continue

      # Fit B-spline to this light-curve segment.
      try:
        spline_piece, mask = kepler_spline(
            time, flux, bkspace=bkspace, maxiter=maxiter)

      # It's expected to get a SplineError occasionally for small values of
      # bkspace.
      except SplineError as e:
        if verbose:
          warnings.warn("Bad bkspace %.4f: %s" % (bkspace, e))
        bad_bkspaces.append(bkspace)
        bad_bkspace = True
        break

      spline.append(spline_piece)
      spline_mask.append(mask)

      # Accumulate the number of free parameters.
      total_time = np.max(time) - np.min(time)
      nknots = int(total_time / bkspace) + 1  # From the bspline implementation.
      nparams += nknots + 3 - 1  # number of knots + degree of spline - 1

      # Accumulate the number of points and the squared residuals.
      npoints += np.sum(mask)
      ssr += np.sum((flux[mask] - spline_piece[mask])**2)

    if bad_bkspace:
      continue

    # The following term is -2*ln(L), where L is the likelihood of the data
    # given the model, under the assumption that the model errors are iid
    # Gaussian with mean 0 and standard deviation sigma.
    likelihood_term = npoints * np.log(2 * np.pi * sigma**2) + ssr / sigma**2

    # Bayesian information criterion.
    bic = likelihood_term + penalty_coeff * nparams * np.log(npoints)

    if best_bic is None or bic < best_bic:
      best_bic = bic
      best_spline = spline
      best_spline_mask = spline_mask
      best_bkspace = bkspace

  return best_spline, best_spline_mask, best_bkspace, bad_bkspaces
