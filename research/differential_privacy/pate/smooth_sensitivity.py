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

"""Functions for smooth sensitivity analysis for PATE mechanisms.

This library implements functionality for doing smooth sensitivity analysis
for Gaussian Noise Max (GNMax), Threshold with Gaussian noise, and Gaussian
Noise with Smooth Sensitivity (GNSS) mechanisms.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from absl import app
import numpy as np
import scipy
import sympy as sp

import core as pate

################################
# SMOOTH SENSITIVITY FOR GNMAX #
################################

# Global dictionary for storing cached q0 values keyed by (sigma, order).
_logq0_cache = {}


def _compute_logq0(sigma, order):
  key = (sigma, order)
  if key in _logq0_cache:
    return _logq0_cache[key]

  logq0 = compute_logq0_gnmax(sigma, order)

  _logq0_cache[key] = logq0  # Update the global variable.
  return logq0


def _compute_logq1(sigma, order, num_classes):
  logq0 = _compute_logq0(sigma, order)  # Most likely already cached.
  logq1 = math.log(_compute_bl_gnmax(math.exp(logq0), sigma, num_classes))
  assert logq1 <= logq0
  return logq1


def _compute_mu1_mu2_gnmax(sigma, logq):
  # Computes mu1, mu2 according to Proposition 10.
  mu2 = sigma * math.sqrt(-logq)
  mu1 = mu2 + 1
  return mu1, mu2


def _compute_data_dep_bound_gnmax(sigma, logq, order):
  # Applies Theorem 6 in Appendix without checking that logq satisfies necessary
  # constraints. The pre-conditions must be assured by comparing logq against
  # logq0 by the caller.
  variance = sigma**2
  mu1, mu2 = _compute_mu1_mu2_gnmax(sigma, logq)
  eps1 = mu1 / variance
  eps2 = mu2 / variance

  log1q = np.log1p(-math.exp(logq))  # log1q = log(1-q)
  log_a = (order - 1) * (
      log1q - (np.log1p(-math.exp((logq + eps2) * (1 - 1 / mu2)))))
  log_b = (order - 1) * (eps1 - logq / (mu1 - 1))

  return np.logaddexp(log1q + log_a, logq + log_b) / (order - 1)


def _compute_rdp_gnmax(sigma, logq, order):
  logq0 = _compute_logq0(sigma, order)
  if logq >= logq0:
    return pate.rdp_data_independent_gaussian(sigma, order)
  else:
    return _compute_data_dep_bound_gnmax(sigma, logq, order)


def compute_logq0_gnmax(sigma, order):
  """Computes the point where we start using data-independent bounds.

  Args:
    sigma: std of the Gaussian noise
    order: Renyi order lambda

  Returns:
    logq0: the point above which the data-ind bound overtakes data-dependent
    bound.
  """

  def _check_validity_conditions(logq):
    # Function returns true iff logq is in the range where data-dependent bound
    # is valid. (Theorem 6 in Appendix.)
    mu1, mu2 = _compute_mu1_mu2_gnmax(sigma, logq)
    if mu1 < order:
      return False
    eps2 = mu2 / sigma**2
    # Do computation in the log space. The condition below comes from Lemma 9
    # from Appendix.
    return (logq <= (mu2 - 1) * eps2 - mu2 * math.log(mu1 / (mu1 - 1) * mu2 /
                                                      (mu2 - 1)))

  def _compare_dep_vs_ind(logq):
    return (_compute_data_dep_bound_gnmax(sigma, logq, order) -
            pate.rdp_data_independent_gaussian(sigma, order))

  # Natural upper bounds on q0.
  logub = min(-(1 + 1. / sigma)**2, -((order - .99) / sigma)**2, -1 / sigma**2)
  assert _check_validity_conditions(logub)

  # If data-dependent bound is already better, we are done already.
  if _compare_dep_vs_ind(logub) < 0:
    return logub

  # Identifying a reasonable lower bound to bracket logq0.
  loglb = 2 * logub  # logub is negative, and thus loglb < logub.
  while _compare_dep_vs_ind(loglb) > 0:
    assert loglb > -10000, "The lower bound on q0 is way too low."
    loglb *= 1.5

  logq0, r = scipy.optimize.brentq(
      _compare_dep_vs_ind, loglb, logub, full_output=True)
  assert r.converged, "The root finding procedure failed to converge."
  assert _check_validity_conditions(logq0)  # just in case.

  return logq0


def _compute_bl_gnmax(q, sigma, num_classes):
  return ((num_classes - 1) / 2 * scipy.special.erfc(
      1 / sigma + scipy.special.erfcinv(2 * q / (num_classes - 1))))


def _compute_bu_gnmax(q, sigma, num_classes):
  return min(1, (num_classes - 1) / 2 * scipy.special.erfc(
      -1 / sigma + scipy.special.erfcinv(2 * q / (num_classes - 1))))


def _compute_local_sens_gnmax(logq, sigma, num_classes, order):
  """Implements Algorithm 3 (computes an upper bound on local sensitivity).

  (See Proposition 13 for proof of correctness.)
  """
  logq0 = _compute_logq0(sigma, order)
  logq1 = _compute_logq1(sigma, order, num_classes)
  if logq1 <= logq <= logq0:
    logq = logq1

  beta = _compute_rdp_gnmax(sigma, logq, order)
  beta_bu_q = _compute_rdp_gnmax(
      sigma, math.log(_compute_bu_gnmax(math.exp(logq), sigma, num_classes)),
      order)
  beta_bl_q = _compute_rdp_gnmax(
      sigma, math.log(_compute_bl_gnmax(math.exp(logq), sigma, num_classes)),
      order)
  return max(beta_bu_q - beta, beta - beta_bl_q)


def compute_local_sensitivity_bounds_gnmax(votes, num_teachers, sigma, order):
  """Computes a list of max-LS-at-distance-d for the GNMax mechanism.

  A more efficient implementation of Algorithms 4 and 5 working in time
  O(teachers*classes). A naive implementation is O(teachers^2*classes) or worse.

  Args:
    votes: A numpy array of votes.
    num_teachers: Total number of voting teachers.
    sigma: Standard deviation of the Guassian noise.
    order: The Renyi order.

  Returns:
    A numpy array of local sensitivities at distances d, 0 <= d <= num_teachers.
  """

  num_classes = len(votes)  # Called m in the paper.

  logq0 = _compute_logq0(sigma, order)
  logq1 = _compute_logq1(sigma, order, num_classes)
  logq = pate.compute_logq_gaussian(votes, sigma)
  plateau = _compute_local_sens_gnmax(logq1, sigma, num_classes, order)

  res = np.full(num_teachers, plateau)

  if logq1 <= logq <= logq0:
    return res

  # Invariant: votes is sorted in the non-increasing order.
  votes = sorted(votes, reverse=True)

  res[0] = _compute_local_sens_gnmax(logq, sigma, num_classes, order)
  curr_d = 0

  go_left = logq > logq0  # Otherwise logq < logq1 and we go right.

  # Iterate while the following is true:
  #    1. If we are going left, logq is still larger than logq0 and we may still
  #       increase the gap between votes[0] and votes[1].
  #    2. If we are going right, logq is still smaller than logq1.
  while ((go_left and logq > logq0 and votes[1] > 0) or
         (not go_left and logq < logq1)):
    curr_d += 1
    if go_left:  # Try decreasing logq.
      votes[0] += 1
      votes[1] -= 1
      idx = 1
      # Restore the invariant. (Can be implemented more efficiently by keeping
      # track of the range of indices equal to votes[1]. Does not seem to matter
      # for the overall running time.)
      while idx < len(votes) - 1 and votes[idx] < votes[idx + 1]:
        votes[idx], votes[idx + 1] = votes[idx + 1], votes[idx]
        idx += 1
    else:  # Go right, i.e., try increasing logq.
      votes[0] -= 1
      votes[1] += 1  # The invariant holds since otherwise logq >= logq1.

    logq = pate.compute_logq_gaussian(votes, sigma)
    res[curr_d] = _compute_local_sens_gnmax(logq, sigma, num_classes, order)

  return res


##################################################
# SMOOTH SENSITIVITY FOR THE THRESHOLD MECHANISM #
##################################################

# A global dictionary of RDPs for various threshold values. Indexed by a 4-tuple
# (num_teachers, threshold, sigma, order).
_rdp_thresholds = {}


def _compute_rdp_list_threshold(num_teachers, threshold, sigma, order):
  key = (num_teachers, threshold, sigma, order)
  if key in _rdp_thresholds:
    return _rdp_thresholds[key]

  res = np.zeros(num_teachers + 1)
  for v in range(0, num_teachers + 1):
    logp = scipy.stats.norm.logsf(threshold - v, scale=sigma)
    res[v] = pate.compute_rdp_threshold(logp, sigma, order)

  _rdp_thresholds[key] = res
  return res


def compute_local_sensitivity_bounds_threshold(counts, num_teachers, threshold,
                                               sigma, order):
  """Computes a list of max-LS-at-distance-d for the threshold mechanism."""

  def _compute_ls(v):
    ls_step_up, ls_step_down = None, None
    if v > 0:
      ls_step_down = abs(rdp_list[v - 1] - rdp_list[v])
    if v < num_teachers:
      ls_step_up = abs(rdp_list[v + 1] - rdp_list[v])
    return max(ls_step_down, ls_step_up)  # Rely on max(x, None) = x.

  cur_max = int(round(max(counts)))
  rdp_list = _compute_rdp_list_threshold(num_teachers, threshold, sigma, order)

  ls = np.zeros(num_teachers)
  for d in range(max(cur_max, num_teachers - cur_max)):
    ls_up, ls_down = None, None
    if cur_max + d <= num_teachers:
      ls_up = _compute_ls(cur_max + d)
    if cur_max - d >= 0:
      ls_down = _compute_ls(cur_max - d)
    ls[d] = max(ls_up, ls_down)
  return ls


#############################################
# PROCEDURES FOR SMOOTH SENSITIVITY RELEASE #
#############################################

# A global dictionary of exponentially decaying arrays. Indexed by beta.
dict_beta_discount = {}


def compute_discounted_max(beta, a):
  n = len(a)

  if beta not in dict_beta_discount or (len(dict_beta_discount[beta]) < n):
    dict_beta_discount[beta] = np.exp(-beta * np.arange(n))

  return max(a * dict_beta_discount[beta][:n])


def compute_smooth_sensitivity_gnmax(beta, counts, num_teachers, sigma, order):
  """Computes smooth sensitivity of a single application of GNMax."""

  ls = compute_local_sensitivity_bounds_gnmax(counts, sigma, order,
                                              num_teachers)
  return compute_discounted_max(beta, ls)


def compute_rdp_of_smooth_sensitivity_gaussian(beta, sigma, order):
  """Computes the RDP curve for the GNSS mechanism.

  Implements Theorem 23 (https://arxiv.org/pdf/1802.08908.pdf).
  """
  if beta > 0 and not 1 < order < 1 / (2 * beta):
    raise ValueError("Order outside the (1, 1/(2*beta)) range.")

  return order * math.exp(2 * beta) / sigma**2 + (
      -.5 * math.log(1 - 2 * order * beta) + beta * order) / (
          order - 1)


def compute_params_for_ss_release(eps, delta):
  """Computes sigma for additive Gaussian noise scaled by smooth sensitivity.

  Presently not used. (We proceed via RDP analysis.)

  Compute beta, sigma for applying Lemma 2.6 (full version of Nissim et al.) via
  Lemma 2.10.
  """
  # Rather than applying Lemma 2.10 directly, which would give suboptimal alpha,
  # (see http://www.cse.psu.edu/~ads22/pubs/NRS07/NRS07-full-draft-v1.pdf),
  # we extract a sufficient condition on alpha from its proof.
  #
  # Let a = rho_(delta/2)(Z_1). Then solve for alpha such that
  # 2 alpha a + alpha^2 = eps/2.
  a = scipy.special.ndtri(1 - delta / 2)
  alpha = math.sqrt(a**2 + eps / 2) - a

  beta = eps / (2 * scipy.special.chdtri(1, delta / 2))

  return alpha, beta


#######################################################
# SYMBOLIC-NUMERIC VERIFICATION OF CONDITIONS C5--C6. #
#######################################################


def _construct_symbolic_beta(q, sigma, order):
  mu2 = sigma * sp.sqrt(sp.log(1 / q))
  mu1 = mu2 + 1
  eps1 = mu1 / sigma**2
  eps2 = mu2 / sigma**2
  a = (1 - q) / (1 - (q * sp.exp(eps2))**(1 - 1 / mu2))
  b = sp.exp(eps1) / q**(1 / (mu1 - 1))
  s = (1 - q) * a**(order - 1) + q * b**(order - 1)
  return (1 / (order - 1)) * sp.log(s)


def _construct_symbolic_bu(q, sigma, m):
  return (m - 1) / 2 * sp.erfc(sp.erfcinv(2 * q / (m - 1)) - 1 / sigma)


def _is_non_decreasing(fn, q, bounds):
  """Verifies whether the function is non-decreasing within a range.

  Args:
    fn: Symbolic function of a single variable.
    q: The name of f's variable.
    bounds: Pair of (lower_bound, upper_bound) reals.

  Returns:
    True iff the function is non-decreasing in the range.
  """
  diff_fn = sp.diff(fn, q)  # Symbolically compute the derivative.
  diff_fn_lambdified = sp.lambdify(
      q,
      diff_fn,
      modules=[
          "numpy", {
              "erfc": scipy.special.erfc,
              "erfcinv": scipy.special.erfcinv
          }
      ])
  r = scipy.optimize.minimize_scalar(
      diff_fn_lambdified, bounds=bounds, method="bounded")
  assert r.success, "Minimizer failed to converge."
  return r.fun >= 0  # Check whether the derivative is non-negative.


def check_conditions(sigma, m, order):
  """Checks conditions C5 and C6 (Section B.4.2 in Appendix)."""
  q = sp.symbols("q", positive=True, real=True)

  beta = _construct_symbolic_beta(q, sigma, order)
  q0 = math.exp(compute_logq0_gnmax(sigma, order))

  cond5 = _is_non_decreasing(beta, q, (0, q0))

  if cond5:
    bl_q0 = _compute_bl_gnmax(q0, sigma, m)

    bu = _construct_symbolic_bu(q, sigma, m)
    delta_beta = beta.subs(q, bu) - beta

    cond6 = _is_non_decreasing(delta_beta, q, (0, bl_q0))
  else:
    cond6 = False  # Skip the check, since Condition 5 is false already.

  return (cond5, cond6)


def main(argv):
  del argv  # Unused.


if __name__ == "__main__":
  app.run(main)
