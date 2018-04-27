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

"""Plots three graphs illustrating cost of privacy per answered query.

A script in support of the paper "Scalable Private Learning with PATE" by
Nicolas Papernot, Shuang Song, Ilya Mironov, Ananth Raghunathan, Kunal Talwar,
Ulfar Erlingsson (https://arxiv.org/abs/1802.08908).

The input is a file containing a numpy array of votes, one query per row, one
class per column. Ex:
  43, 1821, ..., 3
  31, 16, ..., 0
  ...
  0, 86, ..., 438
The output is written to a specified directory and consists of three pdf files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import pickle
import sys

sys.path.append('..')  # Main modules reside in the parent directory.

from absl import app
from absl import flags
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import core as pate

plt.style.use('ggplot')

FLAGS = flags.FLAGS
flags.DEFINE_boolean('cache', False,
                     'Read results of privacy analysis from cache.')
flags.DEFINE_string('counts_file', None, 'Counts file.')
flags.DEFINE_string('figures_dir', '', 'Path where figures are written to.')

flags.mark_flag_as_required('counts_file')

def run_analysis(votes, mechanism, noise_scale, params):
  """Computes data-dependent privacy.

  Args:
    votes: A matrix of votes, where each row contains votes in one instance.
    mechanism: A name of the mechanism ('lnmax', 'gnmax', or 'gnmax_conf')
    noise_scale: A mechanism privacy parameter.
    params: Other privacy parameters.

  Returns:
    Four lists: cumulative privacy cost epsilon, how privacy budget is split,
    how many queries were answered, optimal order.
  """

  def compute_partition(order_opt, eps):
    order_opt_idx = np.searchsorted(orders, order_opt)
    if mechanism == 'gnmax_conf':
      p = (rdp_select_cum[order_opt_idx],
           rdp_cum[order_opt_idx] - rdp_select_cum[order_opt_idx],
           -math.log(delta) / (order_opt - 1))
    else:
      p = (rdp_cum[order_opt_idx], -math.log(delta) / (order_opt - 1))
    return [x / eps for x in p]  # Ensures that sum(x) == 1

  # Short list of orders.
  # orders = np.round(np.concatenate((np.arange(2, 50 + 1, 1),
  #                   np.logspace(np.log10(50), np.log10(1000), num=20))))

  # Long list of orders.
  orders = np.concatenate((np.arange(2, 100 + 1, .5),
                           np.logspace(np.log10(100), np.log10(500), num=100)))
  delta = 1e-8

  n = votes.shape[0]
  eps_total = np.zeros(n)
  partition = [None] * n
  order_opt = np.full(n, np.nan, dtype=float)
  answered = np.zeros(n, dtype=float)

  rdp_cum = np.zeros(len(orders))
  rdp_sqrd_cum = np.zeros(len(orders))
  rdp_select_cum = np.zeros(len(orders))
  answered_sum = 0

  for i in range(n):
    v = votes[i,]
    if mechanism == 'lnmax':
      logq_lnmax = pate.compute_logq_laplace(v, noise_scale)
      rdp_query = pate.rdp_pure_eps(logq_lnmax, 2. / noise_scale, orders)
      rdp_sqrd = rdp_query ** 2
      pr_answered = 1
    elif mechanism == 'gnmax':
      logq_gmax = pate.compute_logq_gaussian(v, noise_scale)
      rdp_query = pate.rdp_gaussian(logq_gmax, noise_scale, orders)
      rdp_sqrd = rdp_query ** 2
      pr_answered = 1
    elif mechanism == 'gnmax_conf':
      logq_step1 = pate.compute_logpr_answered(params['t'], params['sigma1'], v)
      logq_step2 = pate.compute_logq_gaussian(v, noise_scale)
      q_step1 = np.exp(logq_step1)
      logq_step1_min = min(logq_step1, math.log1p(-q_step1))
      rdp_gnmax_step1 = pate.rdp_gaussian(logq_step1_min,
                                          2 ** .5 * params['sigma1'], orders)
      rdp_gnmax_step2 = pate.rdp_gaussian(logq_step2, noise_scale, orders)
      rdp_query = rdp_gnmax_step1 + q_step1 * rdp_gnmax_step2
      # The expression below evaluates
      #     E[(cost_of_step_1 + Bernoulli(pr_of_step_2) * cost_of_step_2)^2]
      rdp_sqrd = (
          rdp_gnmax_step1 ** 2 + 2 * rdp_gnmax_step1 * q_step1 * rdp_gnmax_step2
          + q_step1 * rdp_gnmax_step2 ** 2)
      rdp_select_cum += rdp_gnmax_step1
      pr_answered = q_step1
    else:
      raise ValueError(
          'Mechanism must be one of ["lnmax", "gnmax", "gnmax_conf"]')

    rdp_cum += rdp_query
    rdp_sqrd_cum += rdp_sqrd
    answered_sum += pr_answered

    answered[i] = answered_sum
    eps_total[i], order_opt[i] = pate.compute_eps_from_delta(
        orders, rdp_cum, delta)
    partition[i] = compute_partition(order_opt[i], eps_total[i])

    if i > 0 and (i + 1) % 1000 == 0:
      rdp_var = rdp_sqrd_cum / i - (
          rdp_cum / i) ** 2  # Ignore Bessel's correction.
      order_opt_idx = np.searchsorted(orders, order_opt[i])
      eps_std = ((i + 1) * rdp_var[order_opt_idx]) ** .5  # Std of the sum.
      print(
          'queries = {}, E[answered] = {:.2f}, E[eps] = {:.3f} (std = {:.5f}) '
          'at order = {:.2f} (contribution from delta = {:.3f})'.format(
              i + 1, answered_sum, eps_total[i], eps_std, order_opt[i],
              -math.log(delta) / (order_opt[i] - 1)))
      sys.stdout.flush()

  return eps_total, partition, answered, order_opt


def print_plot_small(figures_dir, eps_lap, eps_gnmax, answered_gnmax):
  """Plots a graph of LNMax vs GNMax.

  Args:
    figures_dir: A name of the directory where to save the plot.
    eps_lap: The cumulative privacy costs of the Laplace mechanism.
    eps_gnmax: The cumulative privacy costs of the Gaussian mechanism
    answered_gnmax: The cumulative count of queries answered.
  """
  xlim = 6000
  x_axis = range(0, int(xlim), 10)
  y_lap = np.zeros(len(x_axis), dtype=float)
  y_gnmax = np.full(len(x_axis), np.nan, dtype=float)

  for i in range(len(x_axis)):
    x = x_axis[i]
    y_lap[i] = eps_lap[x]
    idx = np.searchsorted(answered_gnmax, x)
    if idx < len(eps_gnmax):
      y_gnmax[i] = eps_gnmax[idx]

  fig, ax = plt.subplots()
  fig.set_figheight(4.5)
  fig.set_figwidth(4.7)
  ax.plot(
      x_axis, y_lap, color='r', ls='--', label='LNMax', alpha=.5, linewidth=5)
  ax.plot(
      x_axis,
      y_gnmax,
      color='g',
      ls='-',
      label='Confident-GNMax',
      alpha=.5,
      linewidth=5)
  plt.xticks(np.arange(0, 7000, 1000))
  plt.xlim([0, 6000])
  plt.ylim([0, 6.])
  plt.xlabel('Number of queries answered', fontsize=16)
  plt.ylabel(r'Privacy cost $\varepsilon$ at $\delta=10^{-8}$', fontsize=16)
  plt.legend(loc=2, fontsize=13)  # loc=2 -- upper left
  ax.tick_params(labelsize=14)
  fout_name = os.path.join(figures_dir, 'lnmax_vs_gnmax.pdf')
  print('Saving the graph to ' + fout_name)
  fig.savefig(fout_name, bbox_inches='tight')
  plt.show()


def print_plot_large(figures_dir, eps_lap, eps_gnmax1, answered_gnmax1,
    eps_gnmax2, partition_gnmax2, answered_gnmax2):
  """Plots a graph of LNMax vs GNMax with two parameters.

  Args:
    figures_dir: A name of the  directory where to save the plot.
    eps_lap: The cumulative privacy costs of the Laplace mechanism.
    eps_gnmax1: The cumulative privacy costs of the Gaussian mechanism (set 1).
    answered_gnmax1: The cumulative count of queries answered (set 1).
    eps_gnmax2: The cumulative privacy costs of the Gaussian mechanism (set 2).
    partition_gnmax2: Allocation of eps for set 2.
    answered_gnmax2: The cumulative count of queries answered (set 2).
  """
  xlim = 6000
  x_axis = range(0, int(xlim), 10)
  lenx = len(x_axis)
  y_lap = np.zeros(lenx)
  y_gnmax1 = np.full(lenx, np.nan, dtype=float)
  y_gnmax2 = np.full(lenx, np.nan, dtype=float)
  y1_gnmax2 = np.full(lenx, np.nan, dtype=float)

  for i in range(lenx):
    x = x_axis[i]
    y_lap[i] = eps_lap[x]
    idx1 = np.searchsorted(answered_gnmax1, x)
    if idx1 < len(eps_gnmax1):
      y_gnmax1[i] = eps_gnmax1[idx1]
    idx2 = np.searchsorted(answered_gnmax2, x)
    if idx2 < len(eps_gnmax2):
      y_gnmax2[i] = eps_gnmax2[idx2]
      fraction_step1, fraction_step2, _ = partition_gnmax2[idx2]
      y1_gnmax2[i] = eps_gnmax2[idx2] * fraction_step1 / (
          fraction_step1 + fraction_step2)

  fig, ax = plt.subplots()
  fig.set_figheight(4.5)
  fig.set_figwidth(4.7)
  ax.plot(
      x_axis,
      y_lap,
      color='r',
      ls='dashed',
      label='LNMax',
      alpha=.5,
      linewidth=5)
  ax.plot(
      x_axis,
      y_gnmax1,
      color='g',
      ls='-',
      label='Confident-GNMax (moderate)',
      alpha=.5,
      linewidth=5)
  ax.plot(
      x_axis,
      y_gnmax2,
      color='b',
      ls='-',
      label='Confident-GNMax (aggressive)',
      alpha=.5,
      linewidth=5)
  ax.fill_between(
      x_axis, [0] * lenx,
      y1_gnmax2.tolist(),
      facecolor='b',
      alpha=.3,
      hatch='\\')
  ax.plot(
      x_axis,
      y1_gnmax2,
      color='b',
      ls='-',
      label='_nolegend_',
      alpha=.5,
      linewidth=1)
  ax.fill_between(
      x_axis, y1_gnmax2.tolist(), y_gnmax2.tolist(), facecolor='b', alpha=.3)
  plt.xticks(np.arange(0, 7000, 1000))
  plt.xlim([0, xlim])
  plt.ylim([0, 1.])
  plt.xlabel('Number of queries answered', fontsize=16)
  plt.ylabel(r'Privacy cost $\varepsilon$ at $\delta=10^{-8}$', fontsize=16)
  plt.legend(loc=2, fontsize=13)  # loc=2 -- upper left
  ax.tick_params(labelsize=14)
  fout_name = os.path.join(figures_dir, 'lnmax_vs_2xgnmax_large.pdf')
  print('Saving the graph to ' + fout_name)
  fig.savefig(fout_name, bbox_inches='tight')
  plt.show()


def run_all_analyses(votes, lambda_laplace, gnmax_parameters, sigma2):
  """Sequentially runs all analyses.

  Args:
    votes: A matrix of votes, where each row contains votes in one instance.
    lambda_laplace: The scale of the Laplace noise (lambda).
    gnmax_parameters: A list of parameters for GNMax.
    sigma2: Shared parameter for the GNMax mechanisms.

  Returns:
    Five lists whose length is the number of queries.
  """
  print('=== Laplace Mechanism ===')
  eps_lap, _, _, _ = run_analysis(votes, 'lnmax', lambda_laplace, None)
  print()

  # Does not go anywhere, for now
  # print('=== Gaussian Mechanism (simple) ===')
  # eps, _, _, _ = run_analysis(votes[:n,], 'gnmax', sigma1, None)

  eps_gnmax = [[] for p in gnmax_parameters]
  partition_gmax = [[] for p in gnmax_parameters]
  answered = [[] for p in gnmax_parameters]
  order_opt = [[] for p in gnmax_parameters]
  for i, p in enumerate(gnmax_parameters):
    print('=== Gaussian Mechanism (confident) {}: ==='.format(p))
    eps_gnmax[i], partition_gmax[i], answered[i], order_opt[i] = run_analysis(
        votes, 'gnmax_conf', sigma2, p)
    print()

  return eps_lap, eps_gnmax, partition_gmax, answered, order_opt


def main(argv):
  del argv  # Unused.
  lambda_laplace = 50.  # corresponds to eps = 1. / lambda_laplace

  # Paramaters of the GNMax
  gnmax_parameters = ({
                        't': 1000,
                        'sigma1': 500
                      }, {
                        't': 3500,
                        'sigma1': 1500
                      }, {
                        't': 5000,
                        'sigma1': 1500
                      })
  sigma2 = 100  # GNMax parameters differ only in Step 1 (selection).
  ftemp_name = '/tmp/precomputed.pkl'

  figures_dir = os.path.expanduser(FLAGS.figures_dir)

  if FLAGS.cache and os.path.isfile(ftemp_name):
    print('Reading from cache ' + ftemp_name)
    with open(ftemp_name, 'rb') as f:
      (eps_lap, eps_gnmax, partition_gmax, answered_gnmax,
       orders_opt_gnmax) = pickle.load(f)
  else:
    fin_name = os.path.expanduser(FLAGS.counts_file)
    print('Reading raw votes from ' + fin_name)
    sys.stdout.flush()

    votes = np.load(fin_name)

    (eps_lap, eps_gnmax, partition_gmax,
     answered_gnmax, orders_opt_gnmax) = run_all_analyses(
        votes, lambda_laplace, gnmax_parameters, sigma2)

    print('Writing to cache ' + ftemp_name)
    with open(ftemp_name, 'wb') as f:
      pickle.dump((eps_lap, eps_gnmax, partition_gmax, answered_gnmax,
                   orders_opt_gnmax), f)

  print_plot_small(figures_dir, eps_lap, eps_gnmax[0], answered_gnmax[0])
  print_plot_large(figures_dir, eps_lap, eps_gnmax[1], answered_gnmax[1],
                   eps_gnmax[2], partition_gmax[2], answered_gnmax[2])
  plt.close('all')


if __name__ == '__main__':
  app.run(main)
