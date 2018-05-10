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
"""RDP analysis of the Gaussian-with-sampling mechanism.

Functionality for computing Renyi differential privacy of an additive Gaussian
mechanism with sampling. Its public interface consists of two methods:
  compute_rdp(q, sigma, T, orders) computes RDP with the sampling rate q,
                                   noise sigma, T steps at the list of orders.
  get_privacy_spent(orders, rdp, target_eps, target_delta) computes delta
                                   (or eps) given RDP at multiple orders and
                                   a target value for eps (or delta).

Example use:

Suppose that we have run an algorithm with parameters, an array of
(q1, sigma1, T1) ... (qk, sigma_k, Tk), and we wish to compute eps for a given
delta. The example code would be:

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
import numpy as np
from scipy import special

FLAGS = flags.FLAGS
flags.DEFINE_boolean("rdp_verbose", False,
                     "Output intermediate results for RDP computation.")
FLAGS(sys.argv)  # Load the flags (including on import)


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
  """Subtract two numbers in the log space. Answer must be positive."""
  if logy == -np.inf:  # subtracting 0
    return logx
  assert logx > logy

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
  """Compute log(A_alpha) for integer alpha."""
  assert isinstance(alpha, (int, long))

  # The first and second terms of A_alpha in the log space:
  log_a1, log_a2 = -np.inf, -np.inf

  for i in range(alpha + 1):
    # Compute in the log space. Extra care needed for q = 0 or 1.
    log_coef_i = math.log(special.binom(alpha, i))
    if q > 0:
      log_coef_i += i * math.log(q)
    elif i > 0:
      continue  # The term is 0, skip the rest.

    if q < 1.0:
      log_coef_i += (alpha - i) * math.log(1 - q)
    elif i < alpha:
      continue  # The term is 0, skip the rest.

    s1 = log_coef_i + (i * i - i) / (2.0 * (sigma ** 2))
    s2 = log_coef_i + (i * i + i) / (2.0 * (sigma ** 2))
    log_a1 = _log_add(log_a1, s1)
    log_a2 = _log_add(log_a2, s2)

  log_a = _log_add(math.log(1 - q) + log_a1, math.log(q) + log_a2)
  if FLAGS.rdp_verbose:
    print("A: by binomial expansion    {} = {} + {}".format(
        _log_print(log_a),
        _log_print(math.log(1 - q) + log_a1), _log_print(math.log(q) + log_a2)))
  return float(log_a)


def _compute_log_a_frac(q, sigma, alpha):
  """Compute log(A_alpha) for fractional alpha."""
  # The four parts of A_alpha in the log space:
  log_a11, log_a12 = -np.inf, -np.inf
  log_a21, log_a22 = -np.inf, -np.inf
  i = 0

  z0, _ = _compute_zs(sigma, q)

  while True:  # do ... until loop
    coef = special.binom(alpha, i)
    log_coef = math.log(abs(coef))
    j = alpha - i

    log_t1 = log_coef + i * math.log(q) + j * math.log(1 - q)
    log_t2 = log_coef + j * math.log(q) + i * math.log(1 - q)

    log_e11 = math.log(.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
    log_e12 = math.log(.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))
    log_e21 = math.log(.5) + _log_erfc((i - (z0 - 1)) / (math.sqrt(2) * sigma))
    log_e22 = math.log(.5) + _log_erfc((z0 - 1 - j) / (math.sqrt(2) * sigma))

    log_s11 = log_t1 + (i * i - i) / (2 * (sigma ** 2)) + log_e11
    log_s12 = log_t2 + (j * j - j) / (2 * (sigma ** 2)) + log_e12
    log_s21 = log_t1 + (i * i + i) / (2 * (sigma ** 2)) + log_e21
    log_s22 = log_t2 + (j * j + j) / (2 * (sigma ** 2)) + log_e22

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
    if max(log_s11, log_s21, log_s21, log_s22) < -30:
      break

  log_a = _log_add(
      math.log(1. - q) + _log_add(log_a11, log_a12),
      math.log(q) + _log_add(log_a21, log_a22))
  return log_a


def _compute_log_a(q, sigma, alpha):
  """Compute log(A_alpha) for any positive finite alpha."""
  if float(alpha).is_integer():
    return _compute_log_a_int(q, sigma, int(alpha))
  else:
    return _compute_log_a_frac(q, sigma, alpha)


def _log_erfc(x):
  # Can be replaced with a single call to log_ntdr if available:
  # return np.log(2.) + special.log_ntdr(-x * 2**.5)
  r = special.erfc(x)
  if r == 0.0:
    # Using the Laurent series at infinity for the tail of the erfc function:
    #     erfc(x) ~ exp(-x^2-.5/x^2+.625/x^4)/(x*pi^.5)
    # To verify in Mathematica:
    #     Series[Log[Erfc[x]] + Log[x] + Log[Pi]/2 + x^2, {x, Infinity, 6}]
    return (-math.log(math.pi) / 2 - math.log(x) - x ** 2 - .5 * x ** -2 +
            .625 * x ** -4 - 37. / 24. * x ** -6 + 353. / 64. * x ** -8)
  else:
    return math.log(r)


def _compute_zs(sigma, q):
  z0 = sigma ** 2 * math.log(1 / q - 1) + .5
  z1 = min(z0 - 2, z0 / 2)
  return z0, z1


def _compute_log_b0(sigma, q, alpha, z1):
  """Return an approximation to log(B0) or None if failed to converge."""
  z0, _ = _compute_zs(sigma, q)
  s, log_term, log_b0, k, sign, max_log_term = 0, 1., 0, 0, 1, -np.inf
  # Keep adding new terms until precision is no longer preserved.
  # Don't stop on the negative.
  while (k < alpha or (log_term > max_log_term - 36 and log_term > -30) or
         sign < 0.):
    log_b1 = k * (k - 2 * z0) / (2 * sigma ** 2)
    log_b2 = _log_erfc((k - z1) / (math.sqrt(2) * sigma))
    log_term = log_b0 + log_b1 + log_b2
    max_log_term = max(max_log_term, log_term)
    s += sign * math.exp(log_term)
    k += 1
    # Maintain invariant: sign * exp(log_b0) = {-alpha choose k}
    log_b0 += math.log(abs(-alpha - k + 1)) - math.log(k)
    sign *= -1

  if s == 0:  # May happen if all terms are < 1e-324.
    return -np.inf
  if s < 0 or math.log(s) < max_log_term - 25:  # The series failed to converge.
    return None
  c = math.log(.5) - math.log(1 - q) * alpha
  return c + math.log(s)


def _bound_log_b1(sigma, q, alpha, z1):
  log_c = _log_add(math.log(1 - q),
                   math.log(q) + (2 * z1 - 1.) / (2 * sigma ** 2))
  return math.log(.5) - log_c * alpha + _log_erfc(z1 / (math.sqrt(2) * sigma))


def _bound_log_b(q, sigma, alpha):
  """Compute a numerically stable bound on log(B_alpha)."""
  if q == 1.:  # If the sampling rate is 100%, A and B are symmetric.
    return _compute_log_a(q, sigma, alpha)

  z0, z1 = _compute_zs(sigma, q)
  log_b_bound = np.inf

  # Puts a lower bound on B1: it cannot be less than its value at z0.
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
    if (log_b_bound < 0 or
        log_b_min_bound < 0 or
        log_b_bound > log_b_min_bound + .01):
      # If the bound is likely to be too loose, move z1 closer to z0 and repeat.
      z1 = m
    else:
      break

  return log_b_bound


def _log_bound_b_elementary(q, alpha):
  return -math.log(1 - q) * alpha


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
  """Compute RDP of the Gaussian mechanism with sampling at order alpha.

  Args:
    q: The sampling rate.
    sigma: The std of the additive Gaussian noise.
    alpha: The order at which RDP is computed.

  Returns:
    RDP at alpha, can be np.inf.
  """
  if np.isinf(alpha):
    return np.inf

  log_moment_a = _compute_log_a(q, sigma, alpha - 1)

  log_bound_b = _log_bound_b_elementary(q, alpha - 1)  # does not require sigma

  if log_bound_b < log_moment_a:
    if FLAGS.rdp_verbose:
      print("Elementary bound suffices   : {} < {}".format(
          _log_print(log_bound_b), _log_print(log_moment_a)))
  else:
    log_bound_b2 = _bound_log_b(q, sigma, alpha - 1)
    if math.isnan(log_bound_b2):
      if FLAGS.rdp_verbose:
        print("B bound failed to converge")
    else:
      if FLAGS.rdp_verbose and (log_bound_b2 < log_bound_b):
        print("Elementary bound is stronger: {} < {}".format(
            _log_print(log_bound_b2), _log_print(log_bound_b)))
      log_bound_b = min(log_bound_b, log_bound_b2)

  return max(log_moment_a, log_bound_b) / (alpha - 1)


def compute_rdp(q, sigma, steps, orders):
  """Compute RDP of Gaussian mechanism with sampling for given parameters.

  Args:
    q: The sampling rate.
    sigma: The std of the additive Gaussian noise.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.

  Returns:
    The RDPs at all orders, can be np.inf.
  """

  if np.isscalar(orders):
    rdp = _compute_rdp(q, sigma, orders)
  else:
    rdp = np.array([_compute_rdp(q, sigma, order) for order in orders])

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


def main(_):
  pass


if __name__ == "__main__":
  app.run(main)
