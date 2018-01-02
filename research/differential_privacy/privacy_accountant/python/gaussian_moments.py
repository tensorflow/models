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

"""A standalone utility for computing the log moments.

The utility for computing the log moments. It consists of two methods.
compute_log_moment(q, sigma, T, lmbd) computes the log moment with sampling
probability q, noise sigma, order lmbd, and T steps. get_privacy_spent computes
delta (or eps) given log moments and eps (or delta).

Example use:

Suppose that we have run an algorithm with parameters, an array of
(q1, sigma1, T1) ... (qk, sigmak, Tk), and we wish to compute eps for a given
delta. The example code would be:

  max_lmbd = 32
  lmbds = xrange(1, max_lmbd + 1)
  log_moments = []
  for lmbd in lmbds:
    log_moment = 0
    for q, sigma, T in parameters:
      log_moment += compute_log_moment(q, sigma, T, lmbd)
    log_moments.append((lmbd, log_moment))
  eps, delta = get_privacy_spent(log_moments, target_delta=delta)

To verify that the I1 >= I2 (see comments in GaussianMomentsAccountant in
accountant.py for the context), run the same loop above with verify=True
passed to compute_log_moment.
"""
import math
import sys

import numpy as np
import scipy.integrate as integrate
import scipy.stats
from sympy.mpmath import mp


def _to_np_float64(v):
  if math.isnan(v) or math.isinf(v):
    return np.inf
  return np.float64(v)


######################
# FLOAT64 ARITHMETIC #
######################


def pdf_gauss(x, sigma, mean=0):
  return scipy.stats.norm.pdf(x, loc=mean, scale=sigma)


def cropped_ratio(a, b):
  if a < 1E-50 and b < 1E-50:
    return 1.
  else:
    return a / b


def integral_inf(fn):
  integral, _ = integrate.quad(fn, -np.inf, np.inf)
  return integral


def integral_bounded(fn, lb, ub):
  integral, _ = integrate.quad(fn, lb, ub)
  return integral


def distributions(sigma, q):
  mu0 = lambda y: pdf_gauss(y, sigma=sigma, mean=0.0)
  mu1 = lambda y: pdf_gauss(y, sigma=sigma, mean=1.0)
  mu = lambda y: (1 - q) * mu0(y) + q * mu1(y)
  return mu0, mu1, mu


def compute_a(sigma, q, lmbd, verbose=False):
  lmbd_int = int(math.ceil(lmbd))
  if lmbd_int == 0:
    return 1.0

  a_lambda_first_term_exact = 0
  a_lambda_second_term_exact = 0
  for i in xrange(lmbd_int + 1):
    coef_i = scipy.special.binom(lmbd_int, i) * (q ** i)
    s1, s2 = 0, 0
    for j in xrange(i + 1):
      coef_j = scipy.special.binom(i, j) * (-1) ** (i - j)
      s1 += coef_j * np.exp((j * j - j) / (2.0 * (sigma ** 2)))
      s2 += coef_j * np.exp((j * j + j) / (2.0 * (sigma ** 2)))
    a_lambda_first_term_exact += coef_i * s1
    a_lambda_second_term_exact += coef_i * s2

  a_lambda_exact = ((1.0 - q) * a_lambda_first_term_exact +
                    q * a_lambda_second_term_exact)
  if verbose:
    print "A: by binomial expansion    {} = {} + {}".format(
        a_lambda_exact,
        (1.0 - q) * a_lambda_first_term_exact,
        q * a_lambda_second_term_exact)
  return _to_np_float64(a_lambda_exact)


def compute_b(sigma, q, lmbd, verbose=False):
  mu0, _, mu = distributions(sigma, q)

  b_lambda_fn = lambda z: mu0(z) * np.power(cropped_ratio(mu0(z), mu(z)), lmbd)
  b_lambda = integral_inf(b_lambda_fn)
  m = sigma ** 2 * (np.log((2. - q) / (1. - q)) + 1. / (2 * sigma ** 2))

  b_fn = lambda z: (np.power(mu0(z) / mu(z), lmbd) -
                    np.power(mu(-z) / mu0(z), lmbd))
  if verbose:
    print "M =", m
    print "f(-M) = {} f(M) = {}".format(b_fn(-m), b_fn(m))
    assert b_fn(-m) < 0 and b_fn(m) < 0

  b_lambda_int1_fn = lambda z: (mu0(z) *
                                np.power(cropped_ratio(mu0(z), mu(z)), lmbd))
  b_lambda_int2_fn = lambda z: (mu0(z) *
                                np.power(cropped_ratio(mu(z), mu0(z)), lmbd))
  b_int1 = integral_bounded(b_lambda_int1_fn, -m, m)
  b_int2 = integral_bounded(b_lambda_int2_fn, -m, m)

  a_lambda_m1 = compute_a(sigma, q, lmbd - 1)
  b_bound = a_lambda_m1 + b_int1 - b_int2

  if verbose:
    print "B: by numerical integration", b_lambda
    print "B must be no more than     ", b_bound
  print b_lambda, b_bound
  return _to_np_float64(b_lambda)


###########################
# MULTIPRECISION ROUTINES #
###########################


def pdf_gauss_mp(x, sigma, mean):
  return mp.mpf(1.) / mp.sqrt(mp.mpf("2.") * sigma ** 2 * mp.pi) * mp.exp(
      - (x - mean) ** 2 / (mp.mpf("2.") * sigma ** 2))


def integral_inf_mp(fn):
  integral, _ = mp.quad(fn, [-mp.inf, mp.inf], error=True)
  return integral


def integral_bounded_mp(fn, lb, ub):
  integral, _ = mp.quad(fn, [lb, ub], error=True)
  return integral


def distributions_mp(sigma, q):
  mu0 = lambda y: pdf_gauss_mp(y, sigma=sigma, mean=mp.mpf(0))
  mu1 = lambda y: pdf_gauss_mp(y, sigma=sigma, mean=mp.mpf(1))
  mu = lambda y: (1 - q) * mu0(y) + q * mu1(y)
  return mu0, mu1, mu


def compute_a_mp(sigma, q, lmbd, verbose=False):
  lmbd_int = int(math.ceil(lmbd))
  if lmbd_int == 0:
    return 1.0

  mu0, mu1, mu = distributions_mp(sigma, q)
  a_lambda_fn = lambda z: mu(z) * (mu(z) / mu0(z)) ** lmbd_int
  a_lambda_first_term_fn = lambda z: mu0(z) * (mu(z) / mu0(z)) ** lmbd_int
  a_lambda_second_term_fn = lambda z: mu1(z) * (mu(z) / mu0(z)) ** lmbd_int

  a_lambda = integral_inf_mp(a_lambda_fn)
  a_lambda_first_term = integral_inf_mp(a_lambda_first_term_fn)
  a_lambda_second_term = integral_inf_mp(a_lambda_second_term_fn)

  if verbose:
    print "A: by numerical integration {} = {} + {}".format(
        a_lambda,
        (1 - q) * a_lambda_first_term,
        q * a_lambda_second_term)

  return _to_np_float64(a_lambda)


def compute_b_mp(sigma, q, lmbd, verbose=False):
  lmbd_int = int(math.ceil(lmbd))
  if lmbd_int == 0:
    return 1.0

  mu0, _, mu = distributions_mp(sigma, q)

  b_lambda_fn = lambda z: mu0(z) * (mu0(z) / mu(z)) ** lmbd_int
  b_lambda = integral_inf_mp(b_lambda_fn)

  m = sigma ** 2 * (mp.log((2 - q) / (1 - q)) + 1 / (2 * (sigma ** 2)))
  b_fn = lambda z: ((mu0(z) / mu(z)) ** lmbd_int -
                    (mu(-z) / mu0(z)) ** lmbd_int)
  if verbose:
    print "M =", m
    print "f(-M) = {} f(M) = {}".format(b_fn(-m), b_fn(m))
    assert b_fn(-m) < 0 and b_fn(m) < 0

  b_lambda_int1_fn = lambda z: mu0(z) * (mu0(z) / mu(z)) ** lmbd_int
  b_lambda_int2_fn = lambda z: mu0(z) * (mu(z) / mu0(z)) ** lmbd_int
  b_int1 = integral_bounded_mp(b_lambda_int1_fn, -m, m)
  b_int2 = integral_bounded_mp(b_lambda_int2_fn, -m, m)

  a_lambda_m1 = compute_a_mp(sigma, q, lmbd - 1)
  b_bound = a_lambda_m1 + b_int1 - b_int2

  if verbose:
    print "B by numerical integration", b_lambda
    print "B must be no more than    ", b_bound
  assert b_lambda < b_bound + 1e-5
  return _to_np_float64(b_lambda)


def _compute_delta(log_moments, eps):
  """Compute delta for given log_moments and eps.

  Args:
    log_moments: the log moments of privacy loss, in the form of pairs
      of (moment_order, log_moment)
    eps: the target epsilon.
  Returns:
    delta
  """
  min_delta = 1.0
  for moment_order, log_moment in log_moments:
    if moment_order == 0:
      continue
    if math.isinf(log_moment) or math.isnan(log_moment):
      sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
      continue
    if log_moment < moment_order * eps:
      min_delta = min(min_delta,
                      math.exp(log_moment - moment_order * eps))
  return min_delta


def _compute_eps(log_moments, delta):
  """Compute epsilon for given log_moments and delta.

  Args:
    log_moments: the log moments of privacy loss, in the form of pairs
      of (moment_order, log_moment)
    delta: the target delta.
  Returns:
    epsilon
  """
  min_eps = float("inf")
  for moment_order, log_moment in log_moments:
    if moment_order == 0:
      continue
    if math.isinf(log_moment) or math.isnan(log_moment):
      sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
      continue
    min_eps = min(min_eps, (log_moment - math.log(delta)) / moment_order)
  return min_eps


def compute_log_moment(q, sigma, steps, lmbd, verify=False, verbose=False):
  """Compute the log moment of Gaussian mechanism for given parameters.

  Args:
    q: the sampling ratio.
    sigma: the noise sigma.
    steps: the number of steps.
    lmbd: the moment order.
    verify: if False, only compute the symbolic version. If True, computes
      both symbolic and numerical solutions and verifies the results match.
    verbose: if True, print out debug information.
  Returns:
    the log moment with type np.float64, could be np.inf.
  """
  moment = compute_a(sigma, q, lmbd, verbose=verbose)
  if verify:
    mp.dps = 50
    moment_a_mp = compute_a_mp(sigma, q, lmbd, verbose=verbose)
    moment_b_mp = compute_b_mp(sigma, q, lmbd, verbose=verbose)
    np.testing.assert_allclose(moment, moment_a_mp, rtol=1e-10)
    if not np.isinf(moment_a_mp):
      # The following test fails for (1, np.inf)!
      np.testing.assert_array_less(moment_b_mp, moment_a_mp)
  if np.isinf(moment):
    return np.inf
  else:
    return np.log(moment) * steps


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
    eps, delta pair
  """
  assert (target_eps is None) ^ (target_delta is None)
  assert not ((target_eps is None) and (target_delta is None))
  if target_eps is not None:
    return (target_eps, _compute_delta(log_moments, target_eps))
  else:
    return (_compute_eps(log_moments, target_delta), target_delta)
