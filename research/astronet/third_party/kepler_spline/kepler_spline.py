"""Functions for computing normalization splines for Kepler light curves."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np
from pydl.pydlutils import bspline

from third_party.robust_mean import robust_mean


class InsufficientPointsError(Exception):
  """Indicates that insufficient points were available for spline fitting."""
  pass


class SplineError(Exception):
  """Indicates an error in the underlying spline-fitting implementation."""
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

  Raises:
    InsufficientPointsError: If there were insufficient points (after removing
        outliers) for spline fitting.
    SplineError: If the spline could not be fit, for example if the breakpoint
        spacing is too small.
  """
  if len(time) < 4:
    raise InsufficientPointsError(
        "Cannot fit a spline on less than 4 points. Got {} points.".format(
            len(time)))

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
      # less than outlier_cut*sigma, where sigma is a robust estimate of the
      # standard deviation of the residuals from the previous spline.
      residuals = flux - spline
      new_mask = robust_mean.robust_mean(residuals, cut=outlier_cut)[2]

      if np.all(new_mask == mask):
        break  # Spline converged.

      mask = new_mask

    if np.sum(mask) < 4:
      # Fewer than 4 points after removing outliers. We could plausibly return
      # the spline from the previous iteration because it was fit with at least
      # 4 points. However, since the outliers were such a significant fraction
      # of the curve, the spline from the previous iteration is probably junk,
      # and we consider this a fatal error.
      raise InsufficientPointsError(
          "Cannot fit a spline on less than 4 points. After removing "
          "outliers, got {} points.".format(np.sum(mask)))

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
          "Fitting spline failed with error: '{}'. This might be caused by the "
          "breakpoint spacing being too small, and/or there being insufficient "
          "points to fit the spline in one of the intervals.".format(e))

  return spline, mask


class SplineMetadata(object):
  """Metadata about a spline fit.

  Attributes:
    light_curve_mask: List of boolean numpy arrays indicating which points in
      the light curve were used to fit the best-fit spline.
    bkspace: The break-point spacing used for the best-fit spline.
    bad_bkspaces: List of break-point spacing values that failed.
    likelihood_term: The likelihood term of the Bayesian Information Criterion;
      -2*ln(L), where L is the likelihood of the data given the model.
    penalty_term: The penalty term for the number of parameters in the Bayesian
      Information Criterion.
    bic: The value of the Bayesian Information Criterion; equal to
      likelihood_term + penalty_coeff * penalty_term.
  """

  def __init__(self):
    self.light_curve_mask = None
    self.bkspace = None
    self.bad_bkspaces = []
    self.likelihood_term = None
    self.penalty_term = None
    self.bic = None


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
    all_flux: List of 1D numpy arrays; the flux values of the light curve.
    bkspaces: List of break-point spacings to try.
    maxiter: Maximum number of attempts to fit each spline after removing badly
      fit points.
    penalty_coeff: Coefficient of the penalty term for using more parameters in
      the Bayesian Information Criterion. Decreasing this value will allow more
      parameters to be used (i.e. smaller break-point spacing), and vice-versa.
    verbose: Whether to log individual spline errors. Note that if bkspaces
      contains many values (particularly small ones) then this may cause logging
      pollution if calling this function for many light curves.

  Returns:
    spline: List of numpy arrays; values of the best-fit spline corresponding to
        to the input flux arrays.
    metadata: Object containing metadata about the spline fit.
  """
  # Initialize outputs.
  best_spline = None
  metadata = SplineMetadata()

  # Compute the assumed standard deviation of Gaussian white noise about the
  # spline model. We assume that each flux value f[i] is a Gaussian random
  # variable f[i] ~ N(s[i], sigma^2), where s is the value of the true spline
  # model and sigma is the constant standard deviation for all flux values.
  # Moreover, we assume that s[i] ~= s[i+1]. Therefore,
  # (f[i+1] - f[i]) / sqrt(2) ~ N(0, sigma^2).
  scaled_diffs = [np.diff(f) / np.sqrt(2) for f in all_flux]
  scaled_diffs = np.concatenate(scaled_diffs) if scaled_diffs else np.array([])
  if not scaled_diffs.size:
    best_spline = [np.array([np.nan] * len(f)) for f in all_flux]
    metadata.light_curve_mask = [
        np.zeros_like(f, dtype=np.bool) for f in all_flux
    ]
    return best_spline, metadata

  # Compute the median absoute deviation as a robust estimate of sigma. The
  # conversion factor of 1.48 takes the median absolute deviation to the
  # standard deviation of a normal distribution. See, e.g.
  # https://www.mathworks.com/help/stats/mad.html.
  sigma = np.median(np.abs(scaled_diffs)) * 1.48

  for bkspace in bkspaces:
    nparams = 0  # Total number of free parameters in the piecewise spline.
    npoints = 0  # Total number of data points used to fit the piecewise spline.
    ssr = 0  # Sum of squared residuals between the model and the spline.

    spline = []
    light_curve_mask = []
    bad_bkspace = False  # Indicates that the current bkspace should be skipped.
    for time, flux in zip(all_time, all_flux):
      # Fit B-spline to this light-curve segment.
      try:
        spline_piece, mask = kepler_spline(
            time, flux, bkspace=bkspace, maxiter=maxiter)
      except InsufficientPointsError as e:
        # It's expected to occasionally see intervals with insufficient points,
        # especially if periodic signals have been removed from the light curve.
        # Skip this interval, but continue fitting the spline.
        if verbose:
          warnings.warn(str(e))
        spline.append(np.array([np.nan] * len(flux)))
        light_curve_mask.append(np.zeros_like(flux, dtype=np.bool))
        continue
      except SplineError as e:
        # It's expected to get a SplineError occasionally for small values of
        # bkspace. Skip this bkspace.
        if verbose:
          warnings.warn("Bad bkspace {}: {}".format(bkspace, e))
        metadata.bad_bkspaces.append(bkspace)
        bad_bkspace = True
        break

      spline.append(spline_piece)
      light_curve_mask.append(mask)

      # Accumulate the number of free parameters.
      total_time = np.max(time) - np.min(time)
      nknots = int(total_time / bkspace) + 1  # From the bspline implementation.
      nparams += nknots + 3 - 1  # number of knots + degree of spline - 1

      # Accumulate the number of points and the squared residuals.
      npoints += np.sum(mask)
      ssr += np.sum((flux[mask] - spline_piece[mask])**2)

    if bad_bkspace or not npoints:
      continue

    # The following term is -2*ln(L), where L is the likelihood of the data
    # given the model, under the assumption that the model errors are iid
    # Gaussian with mean 0 and standard deviation sigma.
    likelihood_term = npoints * np.log(2 * np.pi * sigma**2) + ssr / sigma**2

    # Penalty term for the number of parameters used to fit the model.
    penalty_term = nparams * np.log(npoints)

    # Bayesian information criterion.
    bic = likelihood_term + penalty_coeff * penalty_term

    if best_spline is None or bic < metadata.bic:
      best_spline = spline
      metadata.light_curve_mask = light_curve_mask
      metadata.bkspace = bkspace
      metadata.likelihood_term = likelihood_term
      metadata.penalty_term = penalty_term
      metadata.bic = bic

  if best_spline is None:
    # All bkspaces resulted in a SplineError, or all light curve intervals had
    # insufficient points.
    best_spline = [np.array([np.nan] * len(f)) for f in all_flux]
    metadata.light_curve_mask = [
        np.zeros_like(f, dtype=np.bool) for f in all_flux
    ]

  return best_spline, metadata


def fit_kepler_spline(all_time,
                      all_flux,
                      bkspace_min=0.5,
                      bkspace_max=20,
                      bkspace_num=20,
                      maxiter=5,
                      penalty_coeff=1.0,
                      verbose=True):
  """Fits a Kepler spline with logarithmically-sampled breakpoint spacings.

  Args:
    all_time: List of 1D numpy arrays; the time values of the light curve.
    all_flux: List of 1D numpy arrays; the flux values of the light curve.
    bkspace_min: Minimum breakpoint spacing to try.
    bkspace_max: Maximum breakpoint spacing to try.
    bkspace_num: Number of breakpoint spacings to try.
    maxiter: Maximum number of attempts to fit each spline after removing badly
      fit points.
    penalty_coeff: Coefficient of the penalty term for using more parameters in
      the Bayesian Information Criterion. Decreasing this value will allow more
      parameters to be used (i.e. smaller break-point spacing), and vice-versa.
    verbose: Whether to log individual spline errors. Note that if bkspaces
      contains many values (particularly small ones) then this may cause logging
      pollution if calling this function for many light curves.

  Returns:
    spline: List of numpy arrays; values of the best-fit spline corresponding to
        to the input flux arrays.
    metadata: Object containing metadata about the spline fit.
  """
  # Logarithmically sample bkspace_num candidate break point spacings between
  # bkspace_min and bkspace_max.
  bkspaces = np.logspace(
      np.log10(bkspace_min), np.log10(bkspace_max), num=bkspace_num)

  return choose_kepler_spline(
      all_time,
      all_flux,
      bkspaces,
      maxiter=maxiter,
      penalty_coeff=penalty_coeff,
      verbose=verbose)
