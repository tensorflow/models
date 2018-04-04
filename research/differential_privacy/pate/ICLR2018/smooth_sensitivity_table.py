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

"""Performs privacy analysis of GNMax with smooth sensitivity.

A script in support of the paper "Scalable Private Learning with PATE" by
Nicolas Papernot, Shuang Song, Ilya Mironov, Ananth Raghunathan, Kunal Talwar,
Ulfar Erlingsson (https://arxiv.org/abs/1802.08908).

Several flavors of the GNMax algorithm can be analyzed.
  - Plain GNMax (argmax w/ Gaussian noise) is assumed when arguments threshold
    and sigma2 are missing.
  - Confident GNMax (thresholding + argmax w/ Gaussian noise) is used when
    threshold, sigma1, and sigma2 are given.
  - Interactive GNMax (two- or multi-round) is triggered by specifying
    baseline_file, which provides baseline values for votes selection in Step 1.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys

sys.path.append('..')  # Main modules reside in the parent directory.

from absl import app
from absl import flags
import numpy as np
import core as pate
import smooth_sensitivity as pate_ss

FLAGS = flags.FLAGS

flags.DEFINE_string('counts_file', None, 'Counts file.')
flags.DEFINE_string('baseline_file', None, 'File with baseline scores.')
flags.DEFINE_boolean('data_independent', False,
                     'Force data-independent bounds.')
flags.DEFINE_float('threshold', None, 'Threshold for step 1 (selection).')
flags.DEFINE_float('sigma1', None, 'Sigma for step 1 (selection).')
flags.DEFINE_float('sigma2', None, 'Sigma for step 2 (argmax).')
flags.DEFINE_integer('queries', None, 'Number of queries made by the student.')
flags.DEFINE_float('delta', 1e-8, 'Target delta.')
flags.DEFINE_float(
    'order', None,
    'Fixes a Renyi DP order (if unspecified, finds an optimal order from a '
    'hardcoded list).')
flags.DEFINE_integer(
    'teachers', None,
    'Number of teachers (if unspecified, derived from the counts file).')

flags.mark_flag_as_required('counts_file')
flags.mark_flag_as_required('sigma2')


def _check_conditions(sigma, num_classes, orders):
  """Symbolic-numeric verification of conditions C5 and C6.

  The conditions on the beta function are verified by constructing the beta
  function symbolically, and then checking that its derivative (computed
  symbolically) is non-negative within the interval of conjectured monotonicity.
  The last check is performed numerically.
  """

  print('Checking conditions C5 and C6 for all orders.')
  sys.stdout.flush()
  conditions_hold = True

  for order in orders:
    cond5, cond6 = pate_ss.check_conditions(sigma, num_classes, order)
    conditions_hold &= cond5 and cond6
    if not cond5:
      print('Condition C5 does not hold for order =', order)
    elif not cond6:
      print('Condition C6 does not hold for order =', order)

  if conditions_hold:
    print('Conditions C5-C6 hold for all orders.')
  sys.stdout.flush()
  return conditions_hold


def _compute_rdp(votes, baseline, threshold, sigma1, sigma2, delta, orders,
                 data_ind):
  """Computes the (data-dependent) RDP curve for Confident GNMax."""
  rdp_cum = np.zeros(len(orders))
  rdp_sqrd_cum = np.zeros(len(orders))
  answered = 0

  for i, v in enumerate(votes):
    if threshold is None:
      logq_step1 = 0  # No thresholding, always proceed to step 2.
      rdp_step1 = np.zeros(len(orders))
    else:
      logq_step1 = pate.compute_logpr_answered(threshold, sigma1,
                                               v - baseline[i,])
      if data_ind:
        rdp_step1 = pate.compute_rdp_data_independent_threshold(sigma1, orders)
      else:
        rdp_step1 = pate.compute_rdp_threshold(logq_step1, sigma1, orders)

    if data_ind:
      rdp_step2 = pate.rdp_data_independent_gaussian(sigma2, orders)
    else:
      logq_step2 = pate.compute_logq_gaussian(v, sigma2)
      rdp_step2 = pate.rdp_gaussian(logq_step2, sigma2, orders)

    q_step1 = np.exp(logq_step1)
    rdp = rdp_step1 + rdp_step2 * q_step1
    # The expression below evaluates
    #     E[(cost_of_step_1 + Bernoulli(pr_of_step_2) * cost_of_step_2)^2]
    rdp_sqrd = (
        rdp_step1**2 + 2 * rdp_step1 * q_step1 * rdp_step2 +
        q_step1 * rdp_step2**2)
    rdp_sqrd_cum += rdp_sqrd

    rdp_cum += rdp
    answered += q_step1
    if ((i + 1) % 1000 == 0) or (i == votes.shape[0] - 1):
      rdp_var = rdp_sqrd_cum / i - (
          rdp_cum / i)**2  # Ignore Bessel's correction.
      eps_total, order_opt = pate.compute_eps_from_delta(orders, rdp_cum, delta)
      order_opt_idx = np.searchsorted(orders, order_opt)
      eps_std = ((i + 1) * rdp_var[order_opt_idx])**.5  # Std of the sum.
      print(
          'queries = {}, E[answered] = {:.2f}, E[eps] = {:.3f} (std = {:.5f}) '
          'at order = {:.2f} (contribution from delta = {:.3f})'.format(
              i + 1, answered, eps_total, eps_std, order_opt,
              -math.log(delta) / (order_opt - 1)))
      sys.stdout.flush()

    _, order_opt = pate.compute_eps_from_delta(orders, rdp_cum, delta)

  return order_opt


def _find_optimal_smooth_sensitivity_parameters(
    votes, baseline, num_teachers, threshold, sigma1, sigma2, delta, ind_step1,
    ind_step2, order):
  """Optimizes smooth sensitivity parameters by minimizing a cost function.

  The cost function is
        exact_eps + cost of GNSS + two stds of noise,
  which captures that upper bound of the confidence interval of the sanitized
  privacy budget.

  Since optimization is done with full view of sensitive data, the results
  cannot be released.
  """
  rdp_cum = 0
  answered_cum = 0
  ls_cum = 0

  # Define a plausible range for the beta values.
  betas = np.arange(.3 / order, .495 / order, .01 / order)
  cost_delta = math.log(1 / delta) / (order - 1)

  for i, v in enumerate(votes):
    if threshold is None:
      log_pr_answered = 0
      rdp1 = 0
      ls_step1 = np.zeros(num_teachers)
    else:
      log_pr_answered = pate.compute_logpr_answered(threshold, sigma1,
                                                    v - baseline[i,])
      if ind_step1:  # apply data-independent bound for step 1 (thresholding).
        rdp1 = pate.compute_rdp_data_independent_threshold(sigma1, order)
        ls_step1 = np.zeros(num_teachers)
      else:
        rdp1 = pate.compute_rdp_threshold(log_pr_answered, sigma1, order)
        ls_step1 = pate_ss.compute_local_sensitivity_bounds_threshold(
            v - baseline[i,], num_teachers, threshold, sigma1, order)

    pr_answered = math.exp(log_pr_answered)
    answered_cum += pr_answered

    if ind_step2:  # apply data-independent bound for step 2 (GNMax).
      rdp2 = pate.rdp_data_independent_gaussian(sigma2, order)
      ls_step2 = np.zeros(num_teachers)
    else:
      logq_step2 = pate.compute_logq_gaussian(v, sigma2)
      rdp2 = pate.rdp_gaussian(logq_step2, sigma2, order)
      # Compute smooth sensitivity.
      ls_step2 = pate_ss.compute_local_sensitivity_bounds_gnmax(
          v, num_teachers, sigma2, order)

    rdp_cum += rdp1 + pr_answered * rdp2
    ls_cum += ls_step1 + pr_answered * ls_step2  # Expected local sensitivity.

    if ind_step1 and ind_step2:
      # Data-independent bounds.
      cost_opt, beta_opt, ss_opt, sigma_ss_opt = None, 0., 0., np.inf
    else:
      # Data-dependent bounds.
      cost_opt, beta_opt, ss_opt, sigma_ss_opt = np.inf, None, None, None

      for beta in betas:
        ss = pate_ss.compute_discounted_max(beta, ls_cum)

        # Solution to the minimization problem:
        #   min_sigma {order * exp(2 * beta)/ sigma^2 + 2 * ss * sigma}
        sigma_ss = ((order * math.exp(2 * beta)) / ss)**(1 / 3)
        cost_ss = pate_ss.compute_rdp_of_smooth_sensitivity_gaussian(
            beta, sigma_ss, order)

        # Cost captures exact_eps + cost of releasing SS + two stds of noise.
        cost = rdp_cum + cost_ss + 2 * ss * sigma_ss
        if cost < cost_opt:
          cost_opt, beta_opt, ss_opt, sigma_ss_opt = cost, beta, ss, sigma_ss

    if ((i + 1) % 100 == 0) or (i == votes.shape[0] - 1):
      eps_before_ss = rdp_cum + cost_delta
      eps_with_ss = (
          eps_before_ss + pate_ss.compute_rdp_of_smooth_sensitivity_gaussian(
              beta_opt, sigma_ss_opt, order))
      print('{}: E[answered queries] = {:.1f}, RDP at {} goes from {:.3f} to '
            '{:.3f} +/- {:.3f} (ss = {:.4}, beta = {:.4f}, sigma_ss = {:.3f})'.
            format(i + 1, answered_cum, order, eps_before_ss, eps_with_ss,
                   ss_opt * sigma_ss_opt, ss_opt, beta_opt, sigma_ss_opt))
      sys.stdout.flush()

  # Return optimal parameters for the last iteration.
  return beta_opt, ss_opt, sigma_ss_opt


####################
# HELPER FUNCTIONS #
####################


def _load_votes(counts_file, baseline_file, queries):
  counts_file_expanded = os.path.expanduser(counts_file)
  print('Reading raw votes from ' + counts_file_expanded)
  sys.stdout.flush()

  votes = np.load(counts_file_expanded)
  print('Shape of the votes matrix = {}'.format(votes.shape))

  if baseline_file is not None:
    baseline_file_expanded = os.path.expanduser(baseline_file)
    print('Reading baseline values from ' + baseline_file_expanded)
    sys.stdout.flush()
    baseline = np.load(baseline_file_expanded)
    if votes.shape != baseline.shape:
      raise ValueError(
          'Counts file and baseline file must have the same shape. Got {} and '
          '{} instead.'.format(votes.shape, baseline.shape))
  else:
    baseline = np.zeros_like(votes)

  if queries is not None:
    if votes.shape[0] < queries:
      raise ValueError('Expect {} rows, got {} in {}'.format(
          queries, votes.shape[0], counts_file))
    # Truncate the votes matrix to the number of queries made.
    votes = votes[:queries,]
    baseline = baseline[:queries,]
  else:
    print('Process all {} input rows. (Use --queries flag to truncate.)'.format(
        votes.shape[0]))

  return votes, baseline


def _count_teachers(votes):
  s = np.sum(votes, axis=1)
  num_teachers = int(max(s))
  if min(s) != num_teachers:
    raise ValueError(
        'Matrix of votes is malformed: the number of votes is not the same '
        'across rows.')
  return num_teachers


def _is_data_ind_step1(num_teachers, threshold, sigma1, orders):
  if threshold is None:
    return True
  return np.all(
      pate.is_data_independent_always_opt_threshold(num_teachers, threshold,
                                                    sigma1, orders))


def _is_data_ind_step2(num_teachers, num_classes, sigma, orders):
  return np.all(
      pate.is_data_independent_always_opt_gaussian(num_teachers, num_classes,
                                                   sigma, orders))


def main(argv):
  del argv  # Unused.

  if (FLAGS.threshold is None) != (FLAGS.sigma1 is None):
    raise ValueError(
        '--threshold flag and --sigma1 flag must be present or absent '
        'simultaneously.')

  if FLAGS.order is None:
    # Long list of orders.
    orders = np.concatenate((np.arange(2, 100 + 1, .5),
                             np.logspace(np.log10(100), np.log10(500),
                                         num=100)))
    # Short list of orders.
    # orders = np.round(
    #     np.concatenate((np.arange(2, 50 + 1, 1),
    #                     np.logspace(np.log10(50), np.log10(1000), num=20))))
  else:
    orders = np.array([FLAGS.order])

  votes, baseline = _load_votes(FLAGS.counts_file, FLAGS.baseline_file,
                                FLAGS.queries)

  if FLAGS.teachers is None:
    num_teachers = _count_teachers(votes)
  else:
    num_teachers = FLAGS.teachers

  num_classes = votes.shape[1]

  order = _compute_rdp(votes, baseline, FLAGS.threshold, FLAGS.sigma1,
                       FLAGS.sigma2, FLAGS.delta, orders,
                       FLAGS.data_independent)

  ind_step1 = _is_data_ind_step1(num_teachers, FLAGS.threshold, FLAGS.sigma1,
                                 order)

  ind_step2 = _is_data_ind_step2(num_teachers, num_classes, FLAGS.sigma2, order)

  if FLAGS.data_independent or (ind_step1 and ind_step2):
    print('Nothing to do here, all analyses are data-independent.')
    return

  if not _check_conditions(FLAGS.sigma2, num_classes, [order]):
    return  # Quit early: sufficient conditions for correctness fail to hold.

  beta_opt, ss_opt, sigma_ss_opt = _find_optimal_smooth_sensitivity_parameters(
      votes, baseline, num_teachers, FLAGS.threshold, FLAGS.sigma1,
      FLAGS.sigma2, FLAGS.delta, ind_step1, ind_step2, order)

  print('Optimal beta = {:.4f}, E[SS_beta] = {:.4}, sigma_ss = {:.2f}'.format(
      beta_opt, ss_opt, sigma_ss_opt))


if __name__ == '__main__':
  app.run(main)
