# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A standalone utility for tracking the RDP accountant.

The utility for computing Renyi differential privacy. Its public interface
consists of two methods:
    compute_rdp(q, sigma, T, order) computes RDP with sampling probability q,
                                    noise sigma, T steps at order.
    get_privacy_spent computes delta (or eps) given RDP and eps (or delta).

Example use:

Suppose that we have run an algorithm with parameters, an array of
(q1, sigma1, T1) ... (qk, sigmak, Tk), and we wish to compute eps for a given
delta. The example code would be:

  max_order = 32
  orders = range(1, max_order + 1)
  rdp_list = []
  for order in orders:
    rdp = 0
    for q, sigma, T in parameters:
      rdp += compute_rdp(q, sigma, T, order)
    rdp_list.append((order, rdp))
  eps, delta, opt_order = get_privacy_spent(rdp_list, target_delta=delta)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import numpy as np
from scipy import special

######################
# FLOAT64 ARITHMETIC #
######################


def _log_add(logx, logy):
  """Add two numbers in the log space."""
  a, b = min(logx, logy), max(logx, logy)
  if a == -np.inf:  # adding 0
    return b
  # Apply exp(a) + exp(b) = (exp(a - b) + 1.) * exp(b)
  return np.log(np.exp(a - b) + 1.) + b


def _log_sub(logx, logy):
  """Subtract two numbers in the log space. Answer must be positive."""
  if logy == -np.inf:  # subtracting 0
    return logx
  assert logx > logy
  with np.errstate(over="raise"):
    try:
      return np.log(np.exp(logx - logy) - 1.) + logy
    except FloatingPointError:
      return logx


def _log_print(logx):
  """Pretty print."""
  if logx < np.log(sys.float_info.max):
    return "{}".format(np.exp(logx))
  else:
    return "exp({})".format(logx)


def _compute_log_a_int(q, sigma, alpha, verbose=False):
  """Compute log(A_alpha) for integer alpha."""
  assert isinstance(alpha, (int, long))
  log_a1, log_a2 = -np.inf, -np.inf  # log of the first and second terms of A_alpha

  for i in range(alpha + 1):
    # Do computation in the log space. Extra care needed for q = 0 or 1.
    log_coef_i = np.log(special.binom(alpha, i))
    if q > 0:
      log_coef_i += i * np.log(q)
    elif i > 0:
      continue  # the term is 0, skip the rest

    if q < 1.0:
      log_coef_i += (alpha - i) * np.log(1 - q)
    elif i < alpha:
      continue  # the term is 0, skip the rest

    s1 = log_coef_i + (i * i - i) / (2.0 * (sigma**2))
    s2 = log_coef_i + (i * i + i) / (2.0 * (sigma**2))
    log_a1 = _log_add(log_a1, s1)
    log_a2 = _log_add(log_a2, s2)

  log_a = _log_add(np.log(1.0 - q) + log_a1, np.log(q) + log_a2)
  if verbose:
    print("A: by binomial expansion    {} = {} + {}".format(
        _log_print(log_a),
        _log_print(np.log(1.0 - q) + log_a1), _log_print(np.log(q) + log_a2)))
  return np.float64(log_a)


def _compute_log_a_frac(q, sigma, alpha, verbose=False):
  """Compute log(A_alpha) for fractional alpha."""
  # The four parts of A_alpha:
  log_a11, log_a12 = -np.inf, -np.inf
  log_a21, log_a22 = -np.inf, -np.inf
  i = 0

  z0, _ = _compute_zs(sigma, q)

  while i == 0 or max(log_s11, log_s21, log_s21, log_s22) > -30:
    coef = special.binom(alpha, i)
    log_coef = np.log(abs(coef))
    j = alpha - i

    log_t1 = log_coef + i * np.log(q) + j * np.log(1 - q)
    log_t2 = log_coef + j * np.log(q) + i * np.log(1 - q)

    log_e11 = np.log(.5) + _log_erfc((i - z0) / (2**.5 * sigma))
    log_e12 = np.log(.5) + _log_erfc((z0 - j) / (2**.5 * sigma))
    log_e21 = np.log(.5) + _log_erfc((i - (z0 - 1)) / (2**.5 * sigma))
    log_e22 = np.log(.5) + _log_erfc((z0 - 1 - j) / (2**.5 * sigma))

    log_s11 = log_t1 + (i * i - i) / (2.0 * (sigma**2)) + log_e11
    log_s12 = log_t2 + (j * j - j) / (2.0 * (sigma**2)) + log_e12
    log_s21 = log_t1 + (i * i + i) / (2.0 * (sigma**2)) + log_e21
    log_s22 = log_t2 + (j * j + j) / (2.0 * (sigma**2)) + log_e22

    if coef > 0:
      log_a11 = _log_add(log_a11, log_s11)
      log_a12 = _log_add(log_a12, log_s12)
      log_a21 = _log_add(log_a21, log_s21)
      log_a22 = _log_add(log_a22, log_s22)
    else:
      log_a11 = _log_sub(log_a11, log_s11)
      log_a12 = _log_sub(log_a12, log_s12)
      log_a21 = _log_sub(log_a21, log_s21)
      log_a22 = _log_sub(log_a22, log_s22)

    i += 1

  log_a = _log_add(
      np.log(1. - q) + _log_add(log_a11, log_a12),
      np.log(q) + _log_add(log_a21, log_a22))
  return np.float64(log_a)


def compute_log_a(q, sigma, alpha, verbose=False):
  if float(alpha).is_integer():
    return _compute_log_a_int(q, sigma, int(alpha), verbose)
  else:
    return _compute_log_a_frac(q, sigma, alpha, verbose)


def _log_erfc(x):
  # Can be replaced with a single call to log_ntdr if available:
  # return np.log(2.) + special.log_ntdr(-x * 2**.5)
  r = special.erfc(x)
  if r == 0.0:
    # Using the Laurent series at infinity for the tail of the erfc function:
    #     erfc(x) ~ exp(-x^2-.5/x^2+.625/x^4)/(x*pi^.5)
    # To verify in Mathematica:
    #     Series[Log[Erfc[x]] + Log[x] + Log[Pi]/2 + x^2, {x, Infinity, 6}]
    return (-np.log(np.pi) / 2 - np.log(x) - x**2 - .5 * x**-2 + .625 * x**-4 -
            37. / 24. * x**-6 + 353. / 64. * x**-8)
  else:
    return np.log(r)


def _compute_zs(sigma, q):
  z0 = sigma**2 * np.log(1. / q - 1) + .5
  z1 = min(z0 - 2, z0 / 2)
  return z0, z1


def _compute_log_b0(sigma, q, alpha, z1):
  """Return an approximation to B0 or None if failed to converge."""
  z0, _ = _compute_zs(sigma, q)
  s, log_term, log_b0, k, sign, max_log_term = 0, 1., 0, 0, 1, -np.inf
  # Keep adding new terms until precision is no longer preserved.
  # Don't stop on the negative.
  while k < alpha or (log_term > max_log_term - 36 and log_term > -30
  ) or sign < 0.:
    log_b1 = k * (k - 2 * z0) / (2 * sigma**2)
    log_b2 = _log_erfc((k - z1) / (np.sqrt(2) * sigma))
    log_term = log_b0 + log_b1 + log_b2
    max_log_term = max(max_log_term, log_term)
    s += sign * np.exp(log_term)
    k += 1
    # Maintain invariant: sign * exp(log_b0) = {-alpha choose k}
    log_b0 += np.log(np.abs(-alpha - k + 1)) - np.log(k)
    sign *= -1

  if s == 0:  # May happen if all terms are < 1e-324.
    return -np.inf
  if s < 0 or np.log(s) < max_log_term - 25:  # the series failed to converge
    return None
  c = np.log(.5) - np.log(1 - q) * alpha
  return c + np.log(s)


def _bound_log_b1(sigma, q, alpha, z1):
  log_c = _log_add(np.log(1 - q), np.log(q) + (2 * z1 - 1.) / (2 * sigma**2))
  return np.log(.5) - log_c * alpha + _log_erfc(z1 / (2**.5 * sigma))


def bound_log_b(q, sigma, alpha, verbose=False):
  """Compute a numerically stable bound on log(B_alpha)."""
  if q == 1.:  # If the sampling rate is 100%, A and B are symmetric.
    return compute_log_a(q, sigma, alpha, verbose)

  z0, z1 = _compute_zs(sigma, q)
  log_b_bound = np.inf

  # log_b1 cannot be less than its value at z0
  log_lb_b1 = _bound_log_b1(sigma, q, alpha, z0)

  while z0 - z1 > 1e-3:
    m = (z0 + z1) / 2
    log_b0 = _compute_log_b0(sigma, q, alpha, m)
    if log_b0 is None:
      z0 = m
      continue
    log_b1 = _bound_log_b1(sigma, q, alpha, m)
    log_b_bound = min(log_b_bound, _log_add(log_b0, log_b1))
    log_b_min_bound = _log_add(log_b0, log_lb_b1)
    if log_b_bound < 0 or log_b_min_bound < 0 or log_b_bound > log_b_min_bound + .01:
      # If the bound is likely to be too loose, move z1 closer to z0 and repeat.
      z1 = m
    else:
      break

  return np.float64(log_b_bound)


def _log_bound_b_elementary(q, alpha):
  return -np.log(1 - q) * alpha


def _compute_delta(log_moments, eps):
  """Compute delta for given log_moments and eps.

  Args:
    log_moments: the log moments of privacy loss, in the form of pairs
      of (moment_order, log_moment)
    eps: the target epsilon.
  Returns:
    delta, order
  """
  min_delta, opt_order = 1.0, float("NaN")
  for moment_order, log_moment in log_moments:
    if moment_order == 0:
      continue
    if math.isinf(log_moment) or math.isnan(log_moment):
      sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
      continue
    if log_moment < moment_order * eps:
      delta = math.exp(log_moment - moment_order * eps)
      if delta < min_delta:
        min_delta, opt_order = delta, moment_order
  return min_delta, opt_order


def _compute_eps(log_moments, delta):
  """Compute epsilon for given log_moments and delta.

  Args:
    log_moments: the log moments of privacy loss, in the form of pairs
      of (moment_order, log_moment)
    delta: the target delta.
  Returns:
    epsilon, order
  """
  min_eps, opt_order = float("inf"), float("NaN")
  for moment_order, log_moment in log_moments:
    if moment_order == 0:
      continue
    if math.isinf(log_moment) or math.isnan(log_moment):
      sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
      continue
    eps = (log_moment - math.log(delta)) / moment_order
    if eps < min_eps:
      min_eps, opt_order = eps, moment_order
  return min_eps, opt_order


def compute_log_moment(q, sigma, steps, alpha, verbose=False):
  """Compute the log moment of Gaussian mechanism for given parameters.

  Args:
    q: the sampling ratio.
    sigma: the noise sigma.
    steps: the number of steps.
    alpha: the moment order.
    verbose: if True, print out debug information.
  Returns:
    the log moment with type np.float64, could be np.inf.
  """
  log_moment_a = compute_log_a(q, sigma, alpha, verbose=verbose)

  log_bound_b = _log_bound_b_elementary(q, alpha)  # does not require sigma

  if log_bound_b < log_moment_a:
    if verbose:
      print("Elementary bound suffices   : {} < {}".format(
          _log_print(log_bound_b), _log_print(log_moment_a)))
  else:
    log_bound_b2 = bound_log_b(q, sigma, alpha, verbose=verbose)
    if np.isnan(log_bound_b2):
      if verbose:
        print("B bound failed to converge")
    else:
      if verbose and (log_bound_b2 < log_bound_b):
        print("Elementary bound is stronger: {} < {}".format(
            _log_print(log_bound_b2), _log_print(log_bound_b)))
      log_bound_b = min(log_bound_b, log_bound_b2)

  return max(log_moment_a, log_bound_b) * steps


def get_privacy_spent(log_moments, target_eps=None, target_delta=None):
  """Compute delta (or eps) for given eps (or delta) from log moments.

  Args:
    log_moments: array of (moment_order, log_moment) pairs.
    target_eps: if not None, the epsilon for which we would like to compute
      corresponding delta value.
    target_delta: if not None, the delta for which we would like to compute
      corresponding epsilon value. Exactly one of target_eps and target_delta
      is None.
  Returns:
    eps, delta, opt_order
  """
  assert bool(target_eps is None) ^ bool(target_delta is None)
  if target_eps is not None:
    delta, opt_order = _compute_delta(log_moments, target_eps)
    return target_eps, delta, opt_order
  else:
    eps, opt_order = _compute_eps(log_moments, target_delta)
    return eps, target_delta, opt_order
