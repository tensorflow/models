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

from absl.testing import absltest
from absl.testing import parameterized

import mpmath as mp
import numpy as np
import sys

import rdp_accountant


class TestGaussianMoments(parameterized.TestCase):
  #################################
  # HELPER FUNCTIONS:             #
  # Exact computations using      #
  # multi-precision arithmetic.   #
  #################################

  def _log_float_mp(self, x):
    # Convert multi-precision input to float log space.
    if x >= sys.float_info.min:
      return float(mp.log(x))
    else:
      return -np.inf

  def _integral_mp(self, fn, bounds=(-mp.inf, mp.inf)):
    integral, _ = mp.quad(fn, bounds, error=True, maxdegree=8)
    return integral

  def _distributions_mp(self, sigma, q):

    def _mu0(x):
      return mp.npdf(x, mu=0, sigma=sigma)

    def _mu1(x):
      return mp.npdf(x, mu=1, sigma=sigma)

    def _mu(x):
      return (1 - q) * _mu0(x) + q * _mu1(x)

    return _mu0, _mu  # Closure!

  def _mu1_over_mu0(self, x, sigma):
    # Closed-form expression for N(1, sigma^2) / N(0, sigma^2) at x.
    return mp.exp((2 * x - 1) / (2 * sigma**2))

  def _mu_over_mu0(self, x, q, sigma):
    return (1 - q) + q * self._mu1_over_mu0(x, sigma)

  def _compute_a_mp(self, sigma, q, alpha):
    """Compute A_alpha for arbitrary alpha by numerical integration."""
    mu0, _ = self._distributions_mp(sigma, q)
    a_alpha_fn = lambda z: mu0(z) * self._mu_over_mu0(z, q, sigma)**alpha
    a_alpha = self._integral_mp(a_alpha_fn)
    return a_alpha

  # TEST ROUTINES
  def test_compute_rdp_no_data(self):
    # q = 0
    self.assertEqual(rdp_accountant.compute_rdp(0, 10, 1, 20), 0)

  def test_compute_rdp_no_sampling(self):
    # q = 1, RDP = alpha/2 * sigma^2
    self.assertEqual(rdp_accountant.compute_rdp(1, 10, 1, 20), 0.1)

  def test_compute_rdp_scalar(self):
    rdp_scalar = rdp_accountant.compute_rdp(0.1, 2, 10, 5)
    self.assertAlmostEqual(rdp_scalar, 0.07737, places=5)

  def test_compute_rdp_sequence(self):
    rdp_vec = rdp_accountant.compute_rdp(0.01, 2.5, 50,
                                         [1.5, 2.5, 5, 50, 100, np.inf])
    self.assertSequenceAlmostEqual(
        rdp_vec, [0.00065, 0.001085, 0.00218075, 0.023846, 167.416307, np.inf],
        delta=1e-5)

  params = ({'q': 1e-7, 'sigma': .1, 'order': 1.01},
            {'q': 1e-6, 'sigma': .1, 'order': 256},
            {'q': 1e-5, 'sigma': .1, 'order': 256.1},
            {'q': 1e-6, 'sigma': 1, 'order': 27},
            {'q': 1e-4, 'sigma': 1., 'order': 1.5},
            {'q': 1e-3, 'sigma': 1., 'order': 2},
            {'q': .01, 'sigma': 10, 'order': 20},
            {'q': .1, 'sigma': 100, 'order': 20.5},
            {'q': .99, 'sigma': .1, 'order': 256},
            {'q': .999, 'sigma': 100, 'order': 256.1})

  # pylint:disable=undefined-variable
  @parameterized.parameters(p for p in params)
  def test_compute_log_a_equals_mp(self, q, sigma, order):
    # Compare the cheap computation of log(A) with an expensive, multi-precision
    # computation.
    log_a = rdp_accountant._compute_log_a(q, sigma, order)
    log_a_mp = self._log_float_mp(self._compute_a_mp(sigma, q, order))
    np.testing.assert_allclose(log_a, log_a_mp, rtol=1e-4)

  def test_get_privacy_spent_check_target_delta(self):
    orders = range(2, 33)
    rdp = rdp_accountant.compute_rdp(0.01, 4, 10000, orders)
    eps, _, opt_order = rdp_accountant.get_privacy_spent(
        orders, rdp, target_delta=1e-5)
    self.assertAlmostEqual(eps, 1.258575, places=5)
    self.assertEqual(opt_order, 20)

  def test_get_privacy_spent_check_target_eps(self):
    orders = range(2, 33)
    rdp = rdp_accountant.compute_rdp(0.01, 4, 10000, orders)
    _, delta, opt_order = rdp_accountant.get_privacy_spent(
        orders, rdp, target_eps=1.258575)
    self.assertAlmostEqual(delta, 1e-5)
    self.assertEqual(opt_order, 20)

  def test_check_composition(self):
    orders = (1.25, 1.5, 1.75, 2., 2.5, 3., 4., 5., 6., 7., 8., 10., 12., 14.,
              16., 20., 24., 28., 32., 64., 256.)

    rdp = rdp_accountant.compute_rdp(q=1e-4,
                                     stddev_to_sensitivity_ratio=.4,
                                     steps=40000,
                                     orders=orders)

    eps, _, opt_order = rdp_accountant.get_privacy_spent(orders, rdp,
                                                         target_delta=1e-6)

    rdp += rdp_accountant.compute_rdp(q=0.1,
                                      stddev_to_sensitivity_ratio=2,
                                      steps=100,
                                      orders=orders)
    eps, _, opt_order = rdp_accountant.get_privacy_spent(orders, rdp,
                                                         target_delta=1e-5)
    self.assertAlmostEqual(eps, 8.509656, places=5)
    self.assertEqual(opt_order, 2.5)


if __name__ == '__main__':
  absltest.main()