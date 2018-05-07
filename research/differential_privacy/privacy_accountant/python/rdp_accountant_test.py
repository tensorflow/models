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
"""Tests for rdp_accountant.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import mpmath as mp
from absl.testing import absltest

import rdp_accountant


class TestGaussianMoments(absltest.TestCase):
  # MULTI-PRECISION ROUTINES

  def _pdf_gauss_mp(self, x, sigma, mean):
    return 1. / mp.sqrt(2. * sigma ** 2 * mp.pi) * mp.exp(-(x - mean)
                                                           ** 2 / (
                                                                2. * sigma ** 2))

  def _integral_inf_mp(self, fn):
    integral, _ = mp.quad(
        fn, [-mp.inf, mp.inf], error=True,
        maxdegree=7)  # maxdegree = 6 is not enough
    return integral

  def _integral_bounded_mp(self, fn, lb, ub):
    integral, _ = mp.quad(fn, [lb, ub], error=True)
    return integral

  def _distributions_mp(self, sigma, q):
    mu0 = lambda y: self._pdf_gauss_mp(y, sigma=sigma, mean=0)
    mu1 = lambda y: self._pdf_gauss_mp(y, sigma=sigma, mean=1.)
    mu = lambda y: (1 - q) * mu0(y) + q * mu1(y)
    return mu0, mu1, mu

  def compute_a_mp(self, sigma, q, order, verbose=False):
    """Compute A_lambda for arbitrary lambda by numerical integration."""
    mp.dps = 100
    mu0, mu1, mu = self._distributions_mp(sigma, q)
    a_lambda_fn = lambda z: mu(z) * (mu(z) / mu0(z)) ** order
    a_lambda = self._integral_inf_mp(a_lambda_fn)

    if verbose:
      a_lambda_first_term_fn = lambda z: mu0(z) * (mu(z) / mu0(z)) ** order
      a_lambda_second_term_fn = lambda z: mu1(z) * (mu(z) / mu0(z)) ** order

      a_lambda_first_term = self._integral_inf_mp(a_lambda_first_term_fn)
      a_lambda_second_term = self._integral_inf_mp(a_lambda_second_term_fn)

      print("A: by numerical integration {} = {} + {}".format(
          a_lambda, (1 - q) * a_lambda_first_term, q * a_lambda_second_term))

    return a_lambda

  def compute_b_mp(self, sigma, q, order, verbose=False):
    """Compute B_lambda for arbitrary lambda by numerical integration."""
    mu0, _, mu = self._distributions_mp(sigma, q)
    b_lambda_fn = lambda z: mu0(z) * (mu0(z) / mu(z)) ** order
    b_numeric = self._integral_inf_mp(b_lambda_fn)

    if verbose:
      z0, z1 = rdp_accountant._compute_zs(sigma, q)
      print("z1 = ", z1)
      print("x in the Taylor series = ", q / (1 - q) * np.exp(
          (2 * z1 - 1) / (2 * sigma ** 2)))

      b0_numeric = self._integral_bounded_mp(b_lambda_fn, -np.inf, z1)
      b1_numeric = self._integral_bounded_mp(b_lambda_fn, z1, +np.inf)

      print("B: numerically {} = {} + {}".format(b_numeric, b0_numeric,
                                                 b1_numeric))
    return np.float64(b_numeric)

  def _compute_log_moment_mp(self, q, sigma, order):
    log_a_mp = np.float64(mp.log(self.compute_a_mp(sigma, q, order)))
    log_b_mp = np.float64(mp.log(self.compute_b_mp(sigma, q, order)))
    return log_a_mp, log_b_mp

  # TEST ROUTINES
  def _almost_equal(self, a, b, rtol):
    # Analogue of np.testing.assert_allclose(a, b, rtol).
    self.assertBetween(a, b * (1 - rtol), b * (1 + rtol))

  def _compare_bounds(self, q, sigma, order):
    log_a_mp, log_b_mp = self._compute_log_moment_mp(q, sigma, order)
    log_a = rdp_accountant.compute_log_a(q, sigma, order)
    log_bound_b = rdp_accountant.bound_log_b(q, sigma, order)

    if log_a_mp < 1000 and log_a_mp > 1e-6:
      self._almost_equal(log_a, log_a_mp, rtol=1e-6)

    else:  # be more tolerant for _very_ large or small logarithms
      if log_a_mp > 1e-12:
        self._almost_equal(log_a, log_a_mp, rtol=1e-2)
      else:
        print("Bounds on A are too small to compare: {}, {}".format(
            log_a, log_a_mp))
    if np.isfinite(log_bound_b) and log_bound_b > 1e-12:
      # Ignore divergence between the bound and exact value of B if
      # they don't matter anyway (bound on A is larger) or q > .5
      if log_bound_b > log_a and q <= .5:
        self._almost_equal(log_b_mp, log_bound_b, rtol=1e-2)

    if np.isfinite(log_a_mp) and np.isfinite(log_b_mp):
      # We hypothesize that this assertion is always true; no proof yet.
      self.assertLessEqual(log_b_mp, log_a_mp + 1e-6)

  def test_compute_log_moments(self):
    log_moment = rdp_accountant.compute_log_moment(0.1, 2, 10, 4)
    self.assertAlmostEqual(log_moment, 0.30948, places=5)

    self._compare_bounds(q=.01, sigma=.1, order=.5)
    self._compare_bounds(q=.1, sigma=1., order=5)
    self._compare_bounds(q=.5, sigma=2., order=32.5)

    # Compare the cheap computation with expensive, multi-precision
    # computation for a few parameters. Takes about a minute.
    # for q in (1e-6, .1, .999):
    #   for sigma in (.1, 10.):
    #     for order in (.5, 1., 1.5, 256.):
    #       self._compare_bounds(q, sigma, order)

  def test_get_privacy_spent(self):
    orders = range(1, 33)
    log_moments = []
    for order in orders:
      log_moment = rdp_accountant.compute_log_moment(0.01, 4, 10000, order)
      log_moments.append((order, log_moment))
    eps, delta, opt_order = rdp_accountant.get_privacy_spent(log_moments,
                                                             target_delta=1e-5)
    self.assertAlmostEqual(eps, 1.258575, places=5)
    self.assertEqual(opt_order, 19)

    eps, delta, _ = rdp_accountant.get_privacy_spent(log_moments,
                                                     target_eps=1.258575)
    self.assertAlmostEqual(delta, 1e-5)

  def test_compute_privacy_loss(self):
    parameters = [(0.01, 4, 10000), (0.1, 2, 100)]
    delta = 1e-5
    orders = (
    1, 1.25, 1.5, 1.75, 2., 2.5, 3., 4., 5., 6., 7., 8., 10., 12., 14.,
    16., 20., 24., 28., 32., 64., 256.)
    log_moments = []
    for order in orders:
      log_moment = 0
      for q, sigma, steps in parameters:
        log_moment += rdp_accountant.compute_log_moment(q, sigma, steps, order)
      log_moments.append((order, log_moment))
    eps, delta, opt_order = rdp_accountant.get_privacy_spent(
        log_moments, target_delta=delta)
    self.assertAlmostEqual(eps, 3.276237, places=5)
    self.assertEqual(opt_order, 7)


if __name__ == "__main__":
  absltest.main()
