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
"""RDP analysis of the Sampled Gaussian mechanism.

Functionality for computing Renyi differential privacy (RDP) of an additive
Sampled Gaussian mechanism (SGM). Its public interface consists of two methods:
  compute_rdp(q, stddev_to_sensitivity_ratio, T, orders) computes RDP with for
                                   SGM iterated T times.
  get_privacy_spent(orders, rdp, target_eps, target_delta) computes delta
                                   (or eps) given RDP at multiple orders and
                                   a target value for eps (or delta).

Example use:

Suppose that we have run an SGM applied to a function with l2-sensitivity 1.
Its parameters are given as a list of tuples (q1, sigma1, T1), ...,
(qk, sigma_k, Tk), and we wish to compute eps for a given delta.
The example code would be:

  max_order = 32
  orders = range(2, max_order + 1)
  rdp = np.zeros_like(orders, dtype=float)
  for q, sigma, T in parameters:
   rdp += rdp_accountant.compute_rdp(q, sigma, T, orders)
  eps, _, opt_order = rdp_accountant.get_privacy_spent(rdp, target_delta=delta)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import app
from absl import flags

import math
import sys

import numpy as np
from scipy import special

########################
# LOG-SPACE ARITHMETIC #
########################


def _log_add(logx, logy):
  """Add two numbers in the log space."""
  a, b = min(logx, logy), max(logx, logy)
  if a == -np.inf:  # adding 0
    return b
  # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
  return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx, logy):
  """Subtract two numbers in the log space. Answer must be non-negative."""
  if logx < logy:
    raise ValueError("The result of subtraction must be non-negative .")
  if logy == -np.inf:  # subtracting 0
    return logx
  if logx == logy:
    return -np.inf  # 0 is represented as -np.inf in the log space.

  try:
    # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
    return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
  except OverflowError:
    return logx


def _log_print(logx):
  """Pretty print."""
  if logx < math.log(sys.float_info.max):
    return "{}".format(math.exp(logx))
  else:
    return "exp({})".format(logx)


def _compute_log_a_int(q, sigma, alpha):
  """Compute log(A_alpha) for integer alpha. 0 < q < 1."""
  assert isinstance(alpha, (int, long))

  # Initialize with 0 in the log space.
  log_a = -np.inf

  for i in range(alpha + 1):
    log_coef_i = (
        math.log(special.binom(alpha, i)) + i * math.log(q) +
        (alpha - i) * math.log(1 - q))

    s = log_coef_i + (i * i - i) / (2 * (sigma**2))
    log_a = _log_add(log_a, s)

  return float(log_a)


def _compute_log_a_frac(q, sigma, alpha):
  """Compute log(A_alpha) for fractional alpha. 0 < q < 1."""
  # The two parts of A_alpha, integrals over (-inf,z0] and (z0, +inf), are
  # initialized to 0 in the log space:
  log_a0, log_a1 = -np.inf, -np.inf
  i = 0

  z0 = sigma**2 * math.log(1 / q - 1) + .5

  while True:  # do ... until loop
    coef = special.binom(alpha, i)
    log_coef = math.log(abs(coef))
    j = alpha - i

    log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
    log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

    log_e0 = math.log(.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
    log_e1 = math.log(.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

    log_s0 = log_t0 + (i * i - i) / (2 * (sigma**2)) + log_e0
    log_s1 = log_t1 + (j * j - j) / (2 * (sigma**2)) + log_e1

    if coef > 0:
      log_a0 = _log_add(log_a0, log_s0)
      log_a1 = _log_add(log_a1, log_s1)
    else:
      log_a0 = _log_sub(log_a0, log_s0)
      log_a1 = _log_sub(log_a1, log_s1)

    i += 1
    if max(log_s0, log_s1) < -30:
      break

  return _log_add(log_a0, log_a1)


def _compute_log_a(q, sigma, alpha):
  """Compute log(A_alpha) for any positive finite alpha."""
  if float(alpha).is_integer():
    return _compute_log_a_int(q, sigma, int(alpha))
  else:
    return _compute_log_a_frac(q, sigma, alpha)


def _log_erfc(x):
  try:
    return math.log(2) + special.log_ndtr(-x * 2**.5)
  except NameError:
    # If log_ndtr is not available, approximate as follows:
    r = special.erfc(x)
    if r == 0.0:
      # Using the Laurent series at infinity for the tail of the erfc function:
      #     erfc(x) ~ exp(-x^2-.5/x^2+.625/x^4)/(x*pi^.5)
      # To verify in Mathematica:
      #     Series[Log[Erfc[x]] + Log[x] + Log[Pi]/2 + x^2, {x, Infinity, 6}]
      return (-math.log(math.pi) / 2 - math.log(x) - x**2 - .5 * x**-2 +
              .625 * x**-4 - 37. / 24. * x**-6 + 353. / 64. * x**-8)
    else:
      return math.log(r)


def _compute_delta(orders, rdp, eps):
  """Compute delta given an RDP curve and target epsilon.

  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    eps: The target epsilon.

  Returns:
    Pair of (delta, optimal_order).

  Raises:
    ValueError: If input is malformed.

  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  deltas = np.exp((rdp_vec - eps) * (orders_vec - 1))
  idx_opt = np.argmin(deltas)
  return min(deltas[idx_opt], 1.), orders_vec[idx_opt]


def _compute_eps(orders, rdp, delta):
  """Compute epsilon given an RDP curve and target delta.

  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    delta: The target delta.

  Returns:
    Pair of (eps, optimal_order).

  Raises:
    ValueError: If input is malformed.

  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  eps = rdp_vec - math.log(delta) / (orders_vec - 1)

  idx_opt = np.nanargmin(eps)  # Ignore NaNs
  return eps[idx_opt], orders_vec[idx_opt]


def _compute_rdp(q, sigma, alpha):
  """Compute RDP of the Sampled Gaussian mechanism at order alpha.

  Args:
    q: The sampling rate.
    sigma: The std of the additive Gaussian noise.
    alpha: The order at which RDP is computed.

  Returns:
    RDP at alpha, can be np.inf.
  """
  if q == 0:
    return 0

  if q == 1.:
    return alpha / (2 * sigma**2)

  if np.isinf(alpha):
    return np.inf

  return _compute_log_a(q, sigma, alpha) / (alpha - 1)


def compute_rdp(q, stddev_to_sensitivity_ratio, steps, orders):
  """Compute RDP of the Sampled Gaussian Mechanism for given parameters.

  Args:
    q: The sampling rate.
    stddev_to_sensitivity_ratio: The ratio of std of the Gaussian noise to the
      l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.

  Returns:
    The RDPs at all orders, can be np.inf.
  """

  if np.isscalar(orders):
    rdp = _compute_rdp(q, stddev_to_sensitivity_ratio, orders)
  else:
    rdp = np.array([_compute_rdp(q, stddev_to_sensitivity_ratio, order)
                    for order in orders])

  return rdp * steps


def get_privacy_spent(orders, rdp, target_eps=None, target_delta=None):
  """Compute delta (or eps) for given eps (or delta) from the RDP curve.

  Args:
    orders: An array (or a scalar) of RDP orders.
    rdp: An array of RDP values. Must be of the same length as the orders list.
    target_eps: If not None, the epsilon for which we compute the corresponding
              delta.
    target_delta: If not None, the delta for which we compute the corresponding
              epsilon. Exactly one of target_eps and target_delta must be None.
  Returns:
    eps, delta, opt_order.

  Raises:
    ValueError: If target_eps and target_delta are messed up.
  """
  if target_eps is None and target_delta is None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (Both are).")

  if target_eps is not None and target_delta is not None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (None is).")

  if target_eps is not None:
    delta, opt_order = _compute_delta(orders, rdp, target_eps)
    return target_eps, delta, opt_order
  else:
    eps, opt_order = _compute_eps(orders, rdp, target_delta)
    return eps, target_delta, opt_order
