"""Produces two plots. One compares aggregators and their analyses. The other
illustrates sources of privacy loss for Confident-GNMax.

A script in support of the paper "Scalable Private Learning with PATE" by
Nicolas Papernot, Shuang Song, Ilya Mironov, Ananth Raghunathan, Kunal Talwar,
Ulfar Erlingsson (https://arxiv.org/abs/1802.08908).

The input is a file containing a numpy array of votes, one query per row, one
class per column. Ex:
  43, 1821, ..., 3
  31, 16, ..., 0
  ...
  0, 86, ..., 438
The output is written to a specified directory and consists of two files.
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
from collections import namedtuple
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import core as pate
import smooth_sensitivity as pate_ss

plt.style.use('ggplot')

FLAGS = flags.FLAGS
flags.DEFINE_boolean('cache', False,
                     'Read results of privacy analysis from cache.')
flags.DEFINE_string('counts_file', None, 'Counts file.')
flags.DEFINE_string('figures_dir', '', 'Path where figures are written to.')
flags.DEFINE_float('threshold', None, 'Threshold for step 1 (selection).')
flags.DEFINE_float('sigma1', None, 'Sigma for step 1 (selection).')
flags.DEFINE_float('sigma2', None, 'Sigma for step 2 (argmax).')
flags.DEFINE_integer('queries', None, 'Number of queries made by the student.')
flags.DEFINE_float('delta', 1e-8, 'Target delta.')

flags.mark_flag_as_required('counts_file')
flags.mark_flag_as_required('threshold')
flags.mark_flag_as_required('sigma1')
flags.mark_flag_as_required('sigma2')

Partition = namedtuple('Partition', ['step1', 'step2', 'ss', 'delta'])


def analyze_gnmax_conf_data_ind(votes, threshold, sigma1, sigma2, delta):
  orders = np.logspace(np.log10(1.5), np.log10(500), num=100)
  n = votes.shape[0]

  rdp_total = np.zeros(len(orders))
  answered_total = 0
  answered = np.zeros(n)
  eps_cum = np.full(n, None, dtype=float)

  for i in range(n):
    v = votes[i,]
    if threshold is not None and sigma1 is not None:
      q_step1 = np.exp(pate.compute_logpr_answered(threshold, sigma1, v))
      rdp_total += pate.rdp_data_independent_gaussian(sigma1, orders)
    else:
      q_step1 = 1.  # always answer

    answered_total += q_step1
    answered[i] = answered_total

    rdp_total += q_step1 * pate.rdp_data_independent_gaussian(sigma2, orders)

    eps_cum[i], order_opt = pate.compute_eps_from_delta(orders, rdp_total,
                                                        delta)

    if i > 0 and (i + 1) % 1000 == 0:
      print('queries = {}, E[answered] = {:.2f}, E[eps] = {:.3f} '
            'at order = {:.2f}.'.format(
          i + 1,
          answered[i],
          eps_cum[i],
          order_opt))
      sys.stdout.flush()

  return eps_cum, answered


def analyze_gnmax_conf_data_dep(votes, threshold, sigma1, sigma2, delta):
  # Short list of orders.
  # orders = np.round(np.logspace(np.log10(20), np.log10(200), num=20))

  # Long list of orders.
  orders = np.concatenate((np.arange(20, 40, .2),
                           np.arange(40, 75, .5),
                            np.logspace(np.log10(75), np.log10(200), num=20)))

  n = votes.shape[0]
  num_classes = votes.shape[1]
  num_teachers = int(sum(votes[0,]))

  if threshold is not None and sigma1 is not None:
    is_data_ind_step1 = pate.is_data_independent_always_opt_gaussian(
        num_teachers, num_classes, sigma1, orders)
  else:
    is_data_ind_step1 = [True] * len(orders)

  is_data_ind_step2 = pate.is_data_independent_always_opt_gaussian(
      num_teachers, num_classes, sigma2, orders)

  eps_partitioned = np.full(n, None, dtype=Partition)
  order_opt = np.full(n, None, dtype=float)
  ss_std_opt = np.full(n, None, dtype=float)
  answered = np.zeros(n)

  rdp_step1_total = np.zeros(len(orders))
  rdp_step2_total = np.zeros(len(orders))

  ls_total = np.zeros((len(orders), num_teachers))
  answered_total = 0

  for i in range(n):
    v = votes[i,]

    if threshold is not None and sigma1 is not None:
      logq_step1 = pate.compute_logpr_answered(threshold, sigma1, v)
      rdp_step1_total += pate.compute_rdp_threshold(logq_step1, sigma1, orders)
    else:
      logq_step1 = 0.  # always answer

    pr_answered = np.exp(logq_step1)
    logq_step2 = pate.compute_logq_gaussian(v, sigma2)
    rdp_step2_total += pr_answered * pate.rdp_gaussian(logq_step2, sigma2,
                                                       orders)

    answered_total += pr_answered

    rdp_ss = np.zeros(len(orders))
    ss_std = np.zeros(len(orders))

    for j, order in enumerate(orders):
      if not is_data_ind_step1[j]:
        ls_step1 = pate_ss.compute_local_sensitivity_bounds_threshold(v,
            num_teachers, threshold, sigma1, order)
      else:
        ls_step1 = np.full(num_teachers, 0, dtype=float)

      if not is_data_ind_step2[j]:
        ls_step2 = pate_ss.compute_local_sensitivity_bounds_gnmax(
            v, num_teachers, sigma2, order)
      else:
        ls_step2 = np.full(num_teachers, 0, dtype=float)

      ls_total[j,] += ls_step1 + pr_answered * ls_step2

      beta_ss = .49 / order

      ss = pate_ss.compute_discounted_max(beta_ss, ls_total[j,])
      sigma_ss = ((order * math.exp(2 * beta_ss)) / ss) ** (1 / 3)
      rdp_ss[j] = pate_ss.compute_rdp_of_smooth_sensitivity_gaussian(
          beta_ss, sigma_ss, order)
      ss_std[j] = ss * sigma_ss

    rdp_total = rdp_step1_total + rdp_step2_total + rdp_ss

    answered[i] = answered_total
    _, order_opt[i] = pate.compute_eps_from_delta(orders, rdp_total, delta)
    order_idx = np.searchsorted(orders, order_opt[i])

    # Since optimal orders are always non-increasing, shrink orders array
    # and all cumulative arrays to speed up computation.
    if order_idx < len(orders):
      orders = orders[:order_idx + 1]
      rdp_step1_total = rdp_step1_total[:order_idx + 1]
      rdp_step2_total = rdp_step2_total[:order_idx + 1]

    eps_partitioned[i] = Partition(step1=rdp_step1_total[order_idx],
                                   step2=rdp_step2_total[order_idx],
                                   ss=rdp_ss[order_idx],
                                   delta=-math.log(delta) / (order_opt[i] - 1))
    ss_std_opt[i] = ss_std[order_idx]
    if i > 0 and (i + 1) % 1 == 0:
      print('queries = {}, E[answered] = {:.2f}, E[eps] = {:.3f} +/- {:.3f} '
            'at order = {:.2f}. Contributions: delta = {:.3f}, step1 = {:.3f}, '
            'step2 = {:.3f}, ss = {:.3f}'.format(
          i + 1,
          answered[i],
          sum(eps_partitioned[i]),
          ss_std_opt[i],
          order_opt[i],
          eps_partitioned[i].delta,
          eps_partitioned[i].step1,
          eps_partitioned[i].step2,
          eps_partitioned[i].ss))
      sys.stdout.flush()

  return eps_partitioned, answered, ss_std_opt, order_opt


def plot_comparison(figures_dir, simple_ind, conf_ind, simple_dep, conf_dep):
  """Plots variants of GNMax algorithm and their analyses.
  """

  def pivot(x_axis, eps, answered):
    y = np.full(len(x_axis), None, dtype=float)  # delta
    for i, x in enumerate(x_axis):
      idx = np.searchsorted(answered, x)
      if idx < len(eps):
        y[i] = eps[idx]
    return y

  def pivot_dep(x_axis, data_dep):
    eps_partitioned, answered, _, _ = data_dep
    eps = [sum(p) for p in eps_partitioned]  # Flatten eps
    return pivot(x_axis, eps, answered)

  xlim = 10000
  x_axis = range(0, xlim, 10)

  y_simple_ind = pivot(x_axis, *simple_ind)
  y_conf_ind = pivot(x_axis, *conf_ind)

  y_simple_dep = pivot_dep(x_axis, simple_dep)
  y_conf_dep = pivot_dep(x_axis, conf_dep)

  # plt.close('all')
  fig, ax = plt.subplots()
  fig.set_figheight(4.5)
  fig.set_figwidth(4.7)

  ax.plot(x_axis, y_simple_ind, ls='--', color='r', lw=3, label=r'Simple GNMax, data-ind analysis')
  ax.plot(x_axis, y_conf_ind, ls='--', color='b', lw=3, label=r'Confident GNMax, data-ind analysis')
  ax.plot(x_axis, y_simple_dep, ls='-', color='r', lw=3, label=r'Simple GNMax, data-dep analysis')
  ax.plot(x_axis, y_conf_dep, ls='-', color='b', lw=3, label=r'Confident GNMax, data-dep analysis')

  plt.xticks(np.arange(0, xlim + 1000, 2000))
  plt.xlim([0, xlim])
  plt.ylim(bottom=0)
  plt.legend(fontsize=16)
  ax.set_xlabel('Number of queries answered', fontsize=16)
  ax.set_ylabel(r'Privacy cost $\varepsilon$ at $\delta=10^{-8}$', fontsize=16)

  ax.tick_params(labelsize=14)
  plot_filename = os.path.join(figures_dir, 'comparison.pdf')
  print('Saving the graph to ' + plot_filename)
  fig.savefig(plot_filename, bbox_inches='tight')
  plt.show()


def plot_partition(figures_dir, gnmax_conf, print_order):
  """Plots an expert version of the privacy-per-answered-query graph.

  Args:
    figures_dir: A name of the directory where to save the plot.
    eps: The cumulative privacy cost.
    partition: Allocation of the privacy cost.
    answered: Cumulative number of queries answered.
    order_opt: The list of optimal orders.
  """
  eps_partitioned, answered, ss_std_opt, order_opt = gnmax_conf

  xlim = 10000
  x = range(0, int(xlim), 10)
  lenx = len(x)
  y0 = np.full(lenx, np.nan, dtype=float)  # delta
  y1 = np.full(lenx, np.nan, dtype=float)  # delta + step1
  y2 = np.full(lenx, np.nan, dtype=float)  # delta + step1 + step2
  y3 = np.full(lenx, np.nan, dtype=float)  # delta + step1 + step2 + ss
  noise_std = np.full(lenx, np.nan, dtype=float)

  y_right = np.full(lenx, np.nan, dtype=float)

  for i in range(lenx):
    idx = np.searchsorted(answered, x[i])
    if idx < len(eps_partitioned):
      y0[i] = eps_partitioned[idx].delta
      y1[i] = y0[i] + eps_partitioned[idx].step1
      y2[i] = y1[i] + eps_partitioned[idx].step2
      y3[i] = y2[i] + eps_partitioned[idx].ss

      noise_std[i] = ss_std_opt[idx]
      y_right[i] = order_opt[idx]

  # plt.close('all')
  fig, ax = plt.subplots()
  fig.set_figheight(4.5)
  fig.set_figwidth(4.7)
  fig.patch.set_alpha(0)

  l1 = ax.plot(
      x, y3, color='b', ls='-', label=r'Total privacy cost', linewidth=1).pop()

  for y in (y0, y1, y2):
    ax.plot(x, y, color='b', ls='-', label=r'_nolegend_', alpha=.5, linewidth=1)

  ax.fill_between(x, [0] * lenx, y0.tolist(), facecolor='b', alpha=.5)
  ax.fill_between(x, y0.tolist(), y1.tolist(), facecolor='b', alpha=.4)
  ax.fill_between(x, y1.tolist(), y2.tolist(), facecolor='b', alpha=.3)
  ax.fill_between(x, y2.tolist(), y3.tolist(), facecolor='b', alpha=.2)

  ax.fill_between(x, (y3 - noise_std).tolist(), (y3 + noise_std).tolist(),
                  facecolor='r', alpha=.5)


  plt.xticks(np.arange(0, xlim + 1000, 2000))
  plt.xlim([0, xlim])
  ax.set_ylim([0, 3.])

  ax.set_xlabel('Number of queries answered', fontsize=16)
  ax.set_ylabel(r'Privacy cost $\varepsilon$ at $\delta=10^{-8}$', fontsize=16)

  # Merging legends.
  if print_order:
    ax2 = ax.twinx()
    l2 = ax2.plot(
        x, y_right, 'r', ls='-', label=r'Optimal order', linewidth=5,
        alpha=.5).pop()
    ax2.grid(False)
    # ax2.set_ylabel(r'Optimal Renyi order', fontsize=16)
    ax2.set_ylim([0, 200.])
    # ax.legend((l1, l2), (l1.get_label(), l2.get_label()), loc=0, fontsize=13)

  ax.tick_params(labelsize=14)
  plot_filename = os.path.join(figures_dir, 'partition.pdf')
  print('Saving the graph to ' + plot_filename)
  fig.savefig(plot_filename, bbox_inches='tight', dpi=800)
  plt.show()


def run_all_analyses(votes, threshold, sigma1, sigma2, delta):
  simple_ind = analyze_gnmax_conf_data_ind(votes, None, None, sigma2,
                                           delta)

  conf_ind = analyze_gnmax_conf_data_ind(votes, threshold, sigma1, sigma2,
                                         delta)

  simple_dep = analyze_gnmax_conf_data_dep(votes, None, None, sigma2,
                                           delta)

  conf_dep = analyze_gnmax_conf_data_dep(votes, threshold, sigma1, sigma2,
                                         delta)

  return (simple_ind, conf_ind, simple_dep, conf_dep)


def run_or_load_all_analyses():
  temp_filename = os.path.expanduser('~/tmp/partition_cached.pkl')

  if FLAGS.cache and os.path.isfile(temp_filename):
    print('Reading from cache ' + temp_filename)
    with open(temp_filename, 'rb') as f:
      all_analyses = pickle.load(f)
  else:
    fin_name = os.path.expanduser(FLAGS.counts_file)
    print('Reading raw votes from ' + fin_name)
    sys.stdout.flush()

    votes = np.load(fin_name)

    if FLAGS.queries is not None:
      if votes.shape[0] < FLAGS.queries:
        raise ValueError('Expect {} rows, got {} in {}'.format(
            FLAGS.queries, votes.shape[0], fin_name))
      # Truncate the votes matrix to the number of queries made.
      votes = votes[:FLAGS.queries, ]

    all_analyses = run_all_analyses(votes, FLAGS.threshold, FLAGS.sigma1,
                                    FLAGS.sigma2, FLAGS.delta)

    print('Writing to cache ' + temp_filename)
    with open(temp_filename, 'wb') as f:
      pickle.dump(all_analyses, f)

  return all_analyses


def main(argv):
  del argv  # Unused.

  simple_ind, conf_ind, simple_dep, conf_dep = run_or_load_all_analyses()

  figures_dir = os.path.expanduser(FLAGS.figures_dir)

  plot_comparison(figures_dir, simple_ind, conf_ind, simple_dep, conf_dep)
  plot_partition(figures_dir, conf_dep, True)
  plt.close('all')


if __name__ == '__main__':
  app.run(main)
