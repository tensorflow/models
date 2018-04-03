# Copyright 2017 The 'Scalable Private Learning with PATE' Authors All Rights Reserved.
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

"""Core functions for RDP analysis in PATE framework.

This library comprises the core functions for doing differentially private
analysis of the PATE architecture and its various Noisy Max and other
mechanisms.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
import numpy as np
import scipy.stats


def _logaddexp(x):
  """Addition in the log space. Analogue of numpy.logaddexp for a list."""
  m = max(x)
  return m + math.log(sum(np.exp(x - m)))


def _log1mexp(x):
  """Numerically stable computation of log(1-exp(x))."""
  if x < -1:
    return math.log1p(-math.exp(x))
  elif x < 0:
    return math.log(-math.expm1(x))
  elif x == 0:
    return -np.inf
  else:
    raise ValueError("Argument must be non-positive.")


def compute_eps_from_delta(orders, rdp, delta):
  """Translates between RDP and (eps, delta)-DP.

  Args:
    orders: A list (or a scalar) of orders.
    rdp: A list of RDP guarantees (of the same length as orders).
    delta: Target delta.

  Returns:
    Pair of (eps, optimal_order).

  Raises:
    ValueError: If input is malformed.
  """
  if len(orders) != len(rdp):
    raise ValueError("Input lists must have the same length.")
  eps = np.array(rdp) - math.log(delta) / (np.array(orders) - 1)
  idx_opt = np.argmin(eps)
  return eps[idx_opt], orders[idx_opt]


#####################
# RDP FOR THE GNMAX #
#####################


def compute_logq_gaussian(counts, sigma):
  """Returns an upper bound on ln Pr[outcome != argmax] for GNMax.

  Implementation of Proposition 7.

  Args:
    counts: A numpy array of scores.
    sigma: The standard deviation of the Gaussian noise in the GNMax mechanism.

  Returns:
    logq: Natural log of the probability that outcome is different from argmax.
  """
  n = len(counts)
  variance = sigma**2
  idx_max = np.argmax(counts)
  counts_normalized = counts[idx_max] - counts
  counts_rest = counts_normalized[np.arange(n) != idx_max]  # exclude one index
  # Upper bound q via a union bound rather than a more precise calculation.
  logq = _logaddexp(
      scipy.stats.norm.logsf(counts_rest, scale=math.sqrt(2 * variance)))

  # A sketch of a more accurate estimate, which is currently disabled for two
  # reasons:
  # 1. Numerical instability;
  # 2. Not covered by smooth sensitivity analysis.
  # covariance = variance * (np.ones((n - 1, n - 1)) + np.identity(n - 1))
  # logq = np.log1p(-statsmodels.sandbox.distributions.extras.mvnormcdf(
  #     counts_rest, np.zeros(n - 1), covariance, maxpts=1e4))

  return min(logq, math.log(1 - (1 / n)))


def rdp_data_independent_gaussian(sigma, orders):
  """Computes a data-independent RDP curve for GNMax.

  Implementation of Proposition 8.

  Args:
    sigma: Standard deviation of Gaussian noise.
    orders: An array_like list of Renyi orders.

  Returns:
    Upper bound on RPD for all orders. A scalar if orders is a scalar.

  Raises:
    ValueError: If the input is malformed.
  """
  if sigma < 0 or np.any(orders <= 1):  # not defined for alpha=1
    raise ValueError("Inputs are malformed.")

  variance = sigma**2
  if np.isscalar(orders):
    return orders / variance
  else:
    return np.atleast_1d(orders) / variance


def rdp_gaussian(logq, sigma, orders):
  """Bounds RDP from above of GNMax given an upper bound on q (Theorem 6).

  Args:
    logq: Natural logarithm of the probability of a non-argmax outcome.
    sigma: Standard deviation of Gaussian noise.
    orders: An array_like list of Renyi orders.

  Returns:
    Upper bound on RPD for all orders. A scalar if orders is a scalar.

  Raises:
    ValueError: If the input is malformed.
  """
  if logq > 0 or sigma < 0 or np.any(orders <= 1):  # not defined for alpha=1
    raise ValueError("Inputs are malformed.")

  if np.isneginf(logq):  # If the mechanism's output is fixed, it has 0-DP.
    if np.isscalar(orders):
      return 0.
    else:
      return np.full_like(orders, 0., dtype=np.float)

  variance = sigma**2

  # Use two different higher orders: mu_hi1 and mu_hi2 computed according to
  # Proposition 10.
  mu_hi2 = math.sqrt(variance * -logq)
  mu_hi1 = mu_hi2 + 1

  orders_vec = np.atleast_1d(orders)

  ret = orders_vec / variance  # baseline: data-independent bound

  # Filter out entries where data-dependent bound does not apply.
  mask = np.logical_and(mu_hi1 > orders_vec, mu_hi2 > 1)

  rdp_hi1 = mu_hi1 / variance
  rdp_hi2 = mu_hi2 / variance

  log_a2 = (mu_hi2 - 1) * rdp_hi2

  # Make sure q is in the increasing wrt q range and A is positive.
  if (np.any(mask) and logq <= log_a2 - mu_hi2 *
      (math.log(1 + 1 / (mu_hi1 - 1)) + math.log(1 + 1 / (mu_hi2 - 1))) and
      -logq > rdp_hi2):
    # Use log1p(x) = log(1 + x) to avoid catastrophic cancellations when x ~ 0.
    log1q = _log1mexp(logq)  # log1q = log(1-q)
    log_a = (orders - 1) * (
        log1q - _log1mexp((logq + rdp_hi2) * (1 - 1 / mu_hi2)))
    log_b = (orders - 1) * (rdp_hi1 - logq / (mu_hi1 - 1))

    # Use logaddexp(x, y) = log(e^x + e^y) to avoid overflow for large x, y.
    log_s = np.logaddexp(log1q + log_a, logq + log_b)
    ret[mask] = np.minimum(ret, log_s / (orders - 1))[mask]

  assert np.all(ret >= 0)

  if np.isscalar(orders):
    return np.asscalar(ret)
  else:
    return ret


def is_data_independent_always_opt_gaussian(num_teachers, num_classes, sigma,
                                            orders):
  """Tests whether data-ind bound is always optimal for GNMax.

  Args:
    num_teachers: Number of teachers.
    num_classes: Number of classes.
    sigma: Standard deviation of the Gaussian noise.
    orders: An array_like list of Renyi orders.

  Returns:
    Boolean array of length |orders| (a scalar if orders is a scalar). True if
    the data-independent bound is always the same as the data-dependent bound.

  """
  unanimous = np.array([num_teachers] + [0] * (num_classes - 1))
  logq = compute_logq_gaussian(unanimous, sigma)

  rdp_dep = rdp_gaussian(logq, sigma, orders)
  rdp_ind = rdp_data_independent_gaussian(sigma, orders)
  return np.isclose(rdp_dep, rdp_ind)


###################################
# RDP FOR THE THRESHOLD MECHANISM #
###################################


def compute_logpr_answered(t, sigma, counts):
  """Computes log of the probability that a noisy threshold is crossed.

  Args:
    t: The threshold.
    sigma: The stdev of the Gaussian noise added to the threshold.
    counts: An array of votes.

  Returns:
    Natural log of the probability that max is larger than a noisy threshold.
  """
  # Compared to the paper, max(counts) is rounded to the nearest integer. This
  # is done to facilitate computation of smooth sensitivity for the case of
  # the interactive mechanism, where votes are not necessarily integer.
  return scipy.stats.norm.logsf(t - round(max(counts)), scale=sigma)


def compute_rdp_data_independent_threshold(sigma, orders):
  # The input to the threshold mechanism has stability 1, compared to
  # GNMax, which has stability = 2. Hence the sqrt(2) factor below.
  return rdp_data_independent_gaussian(2**.5 * sigma, orders)


def compute_rdp_threshold(log_pr_answered, sigma, orders):
  logq = min(log_pr_answered, _log1mexp(log_pr_answered))
  # The input to the threshold mechanism has stability 1, compared to
  # GNMax, which has stability = 2. Hence the sqrt(2) factor below.
  return rdp_gaussian(logq, 2**.5 * sigma, orders)


def is_data_independent_always_opt_threshold(num_teachers, threshold, sigma,
                                             orders):
  """Tests whether data-ind bound is always optimal for the threshold mechanism.

  Args:
    num_teachers: Number of teachers.
    threshold: The cut-off threshold.
    sigma: Standard deviation of the Gaussian noise.
    orders: An array_like list of Renyi orders.

  Returns:
    Boolean array of length |orders| (a scalar if orders is a scalar). True if
    the data-independent bound is always the same as the data-dependent bound.
  """

  # Since the data-dependent bound depends only on max(votes), it suffices to
  # check whether the data-dependent bounds are better than data-independent
  # bounds in the extreme cases when max(votes) is minimal or maximal.
  # For both Confident GNMax and Interactive GNMax it holds that
  #   0 <= max(votes) <= num_teachers.
  # The upper bound is trivial in both cases.
  # The lower bound is trivial for Confident GNMax (and a stronger one, based on
  # the pigeonhole principle, is possible).
  # For Interactive GNMax (Algorithm 2), the lower bound follows from the
  # following argument. Since the votes vector is the difference between the
  # actual teachers' votes and the student's baseline, we need to argue that
  # max(n_j - M * p_j) >= 0.
  # The bound holds because sum_j n_j = sum M * p_j = M. Thus,
  # sum_j (n_j - M * p_j) = 0, and max_j (n_j - M * p_j) >= 0 as needed.
  logq1 = compute_logpr_answered(threshold, sigma, [0])
  logq2 = compute_logpr_answered(threshold, sigma, [num_teachers])

  rdp_dep1 = compute_rdp_threshold(logq1, sigma, orders)
  rdp_dep2 = compute_rdp_threshold(logq2, sigma, orders)

  rdp_ind = compute_rdp_data_independent_threshold(sigma, orders)
  return np.isclose(rdp_dep1, rdp_ind) and np.isclose(rdp_dep2, rdp_ind)


#############################
# RDP FOR THE LAPLACE NOISE #
#############################


def compute_logq_laplace(counts, lmbd):
  """Computes an upper bound on log Pr[outcome != argmax] for LNMax.

  Args:
    counts: A list of scores.
    lmbd: The lambda parameter of the Laplace distribution ~exp(-|x| / lambda).

  Returns:
    logq: Natural log of the probability that outcome is different from argmax.
  """
  # For noisy max, we only get an upper bound via the union bound. See Lemma 4
  # in https://arxiv.org/abs/1610.05755.
  #
  # Pr[ j beats i*] = (2+gap(j,i*))/ 4 exp(gap(j,i*)
  # proof at http://mathoverflow.net/questions/66763/

  idx_max = np.argmax(counts)
  counts_normalized = (counts - counts[idx_max]) / lmbd
  counts_rest = np.array(
      [counts_normalized[i] for i in range(len(counts)) if i != idx_max])

  logq = _logaddexp(np.log(2 - counts_rest) + math.log(.25) + counts_rest)

  return min(logq, math.log(1 - (1 / len(counts))))


def rdp_pure_eps(logq, pure_eps, orders):
  """Computes the RDP value given logq and pure privacy eps.

  Implementation of https://arxiv.org/abs/1610.05755, Theorem 3.

  The bound used is the min of three terms. The first term is from
  https://arxiv.org/pdf/1605.02065.pdf.
  The second term is based on the fact that when event has probability (1-q) for
  q close to zero, q can only change by exp(eps), which corresponds to a
  much smaller multiplicative change in (1-q)
  The third term comes directly from the privacy guarantee.

  Args:
    logq: Natural logarithm of the probability of a non-optimal outcome.
    pure_eps: eps parameter for DP
    orders: array_like list of moments to compute.

  Returns:
    Array of upper bounds on rdp (a scalar if orders is a scalar).
  """
  orders_vec = np.atleast_1d(orders)
  q = math.exp(logq)
  log_t = np.full_like(orders_vec, np.inf)
  if q <= 1 / (math.exp(pure_eps) + 1):
    logt_one = math.log1p(-q) + (
        math.log1p(-q) - _log1mexp(pure_eps + logq)) * (
            orders_vec - 1)
    logt_two = logq + pure_eps * (orders_vec - 1)
    log_t = np.logaddexp(logt_one, logt_two)

  ret = np.minimum(
      np.minimum(0.5 * pure_eps * pure_eps * orders_vec,
                 log_t / (orders_vec - 1)), pure_eps)
  if np.isscalar(orders):
    return np.asscalar(ret)
  else:
    return ret


def main(argv):
  del argv  # Unused.


if __name__ == "__main__":
  app.run(main)
