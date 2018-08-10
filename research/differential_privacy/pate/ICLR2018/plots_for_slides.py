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

"""Plots graphs for the slide deck.

A script in support of the PATE2 paper. The input is a file containing a numpy
array of votes, one query per row, one class per column. Ex:
  43, 1821, ..., 3
  31, 16, ..., 0
  ...
  0, 86, ..., 438
The output graphs are visualized using the TkAgg backend.
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
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import core as pate
import random

plt.style.use('ggplot')

FLAGS = flags.FLAGS
flags.DEFINE_string('counts_file', None, 'Counts file.')
flags.DEFINE_string('figures_dir', '', 'Path where figures are written to.')
flags.DEFINE_boolean('transparent', False, 'Set background to transparent.')

flags.mark_flag_as_required('counts_file')


def setup_plot():
  fig, ax = plt.subplots()
  fig.set_figheight(4.5)
  fig.set_figwidth(4.7)

  if FLAGS.transparent:
    fig.patch.set_alpha(0)

  return fig, ax


def plot_rdp_curve_per_example(votes, sigmas):
  orders = np.linspace(1., 100., endpoint=True, num=1000)
  orders[0] = 1.001
  fig, ax = setup_plot()

  for i in range(votes.shape[0]):
    for sigma in sigmas:
      logq = pate.compute_logq_gaussian(votes[i,], sigma)
      rdp = pate.rdp_gaussian(logq, sigma, orders)
      ax.plot(
          orders,
          rdp,
          alpha=1.,
          label=r'Data-dependent bound, $\sigma$={}'.format(int(sigma)),
          linewidth=5)

  for sigma in sigmas:
    ax.plot(
        orders,
        pate.rdp_data_independent_gaussian(sigma, orders),
        alpha=.3,
        label=r'Data-independent bound, $\sigma$={}'.format(int(sigma)),
        linewidth=10)

  plt.xlim(xmin=1, xmax=100)
  plt.ylim(ymin=0)
  plt.xticks([1, 20, 40, 60, 80, 100])
  plt.yticks([0, .0025, .005, .0075, .01])
  plt.xlabel(r'Order $\alpha$', fontsize=16)
  plt.ylabel(r'RDP value $\varepsilon$ at $\alpha$', fontsize=16)
  ax.tick_params(labelsize=14)

  plt.legend(loc=0, fontsize=13)
  plt.show()


def plot_rdp_of_sigma(v, order):
  sigmas = np.linspace(1., 1000., endpoint=True, num=1000)
  fig, ax = setup_plot()

  y = np.zeros(len(sigmas))

  for i, sigma in enumerate(sigmas):
    logq = pate.compute_logq_gaussian(v, sigma)
    y[i] = pate.rdp_gaussian(logq, sigma, order)

  ax.plot(sigmas, y, alpha=.8, linewidth=5)

  plt.xlim(xmin=1, xmax=1000)
  plt.ylim(ymin=0)
  # plt.yticks([0, .0004, .0008, .0012])
  ax.tick_params(labelleft='off')
  plt.xlabel(r'Noise $\sigma$', fontsize=16)
  plt.ylabel(r'RDP at order $\alpha={}$'.format(order), fontsize=16)
  ax.tick_params(labelsize=14)

  # plt.legend(loc=0, fontsize=13)
  plt.show()


def compute_rdp_curve(votes, threshold, sigma1, sigma2, orders,
    target_answered):
  rdp_cum = np.zeros(len(orders))
  answered = 0
  for i, v in enumerate(votes):
    v = sorted(v, reverse=True)
    q_step1 = math.exp(pate.compute_logpr_answered(threshold, sigma1, v))
    logq_step2 = pate.compute_logq_gaussian(v, sigma2)
    rdp = pate.rdp_gaussian(logq_step2, sigma2, orders)
    rdp_cum += q_step1 * rdp

    answered += q_step1
    if answered >= target_answered:
      print('Processed {} queries to answer {}.'.format(i, target_answered))
      return rdp_cum

  assert False, 'Never reached {} answered queries.'.format(target_answered)


def plot_rdp_total(votes, sigmas):
  orders = np.linspace(1., 100., endpoint=True, num=100)
  orders[0] = 1.1

  fig, ax = setup_plot()

  target_answered = 2000

  for sigma in sigmas:
    rdp = compute_rdp_curve(votes, 5000, 1000, sigma, orders, target_answered)
    ax.plot(
        orders,
        rdp,
        alpha=.8,
        label=r'Data-dependent bound, $\sigma$={}'.format(int(sigma)),
        linewidth=5)

  # for sigma in sigmas:
  #   ax.plot(
  #       orders,
  #       target_answered * pate.rdp_data_independent_gaussian(sigma, orders),
  #       alpha=.3,
  #       label=r'Data-independent bound, $\sigma$={}'.format(int(sigma)),
  #       linewidth=10)

  plt.xlim(xmin=1, xmax=100)
  plt.ylim(ymin=0)
  plt.xticks([1, 20, 40, 60, 80, 100])
  plt.yticks([0, .0005, .001, .0015, .002])

  plt.xlabel(r'Order $\alpha$', fontsize=16)
  plt.ylabel(r'RDP value $\varepsilon$ at $\alpha$', fontsize=16)
  ax.tick_params(labelsize=14)

  plt.legend(loc=0, fontsize=13)
  plt.show()


def plot_data_ind_curve():
  fig, ax = setup_plot()

  orders = np.linspace(1., 10., endpoint=True, num=1000)
  orders[0] = 1.01

  ax.plot(
      orders,
      pate.rdp_data_independent_gaussian(1., orders),
      alpha=.5,
      color='gray',
      linewidth=10)

  # plt.yticks([])
  plt.xlim(xmin=1, xmax=10)
  plt.ylim(ymin=0)
  plt.xticks([1, 3, 5, 7, 9])
  ax.tick_params(labelsize=14)
  plt.show()


def plot_two_data_ind_curves():
  orders = np.linspace(1., 100., endpoint=True, num=1000)
  orders[0] = 1.001

  fig, ax = setup_plot()

  for sigma in [100, 150]:
    ax.plot(
        orders,
        pate.rdp_data_independent_gaussian(sigma, orders),
        alpha=.3,
        label=r'Data-independent bound, $\sigma$={}'.format(int(sigma)),
        linewidth=10)

  plt.xlim(xmin=1, xmax=100)
  plt.ylim(ymin=0)
  plt.xticks([1, 20, 40, 60, 80, 100])
  plt.yticks([0, .0025, .005, .0075, .01])
  plt.xlabel(r'Order $\alpha$', fontsize=16)
  plt.ylabel(r'RDP value $\varepsilon$ at $\alpha$', fontsize=16)
  ax.tick_params(labelsize=14)

  plt.legend(loc=0, fontsize=13)
  plt.show()


def scatter_plot(votes, threshold, sigma1, sigma2, order):
  fig, ax = setup_plot()
  x = []
  y = []
  for i, v in enumerate(votes):
    if threshold is not None and sigma1 is not None:
      q_step1 = math.exp(pate.compute_logpr_answered(threshold, sigma1, v))
    else:
      q_step1 = 1.
    if random.random() < q_step1:
      logq_step2 = pate.compute_logq_gaussian(v, sigma2)
      x.append(max(v))
      y.append(pate.rdp_gaussian(logq_step2, sigma2, order))

  print('Selected {} queries.'.format(len(x)))
  # Plot the data-independent curve:
  # data_ind = pate.rdp_data_independent_gaussian(sigma, order)
  # plt.plot([0, 5000], [data_ind, data_ind], color='tab:blue', linestyle='-', linewidth=2)
  ax.set_yscale('log')
  plt.xlim(xmin=0, xmax=5000)
  plt.ylim(ymin=1e-300, ymax=1)
  plt.yticks([1, 1e-100, 1e-200, 1e-300])
  plt.scatter(x, y, s=1, alpha=0.5)
  plt.ylabel(r'RDP at $\alpha={}$'.format(order), fontsize=16)
  plt.xlabel(r'max count', fontsize=16)
  ax.tick_params(labelsize=14)
  plt.show()


def main(argv):
  del argv  # Unused.
  fin_name = os.path.expanduser(FLAGS.counts_file)
  print('Reading raw votes from ' + fin_name)
  sys.stdout.flush()

  plot_data_ind_curve()
  plot_two_data_ind_curves()

  v1 = [2550, 2200, 250]  # based on votes[2,]
  # v2 = [2600, 2200, 200]  # based on votes[381,]
  plot_rdp_curve_per_example(np.array([v1]), (100., 150.))

  plot_rdp_of_sigma(np.array(v1), 20.)

  votes = np.load(fin_name)

  plot_rdp_total(votes[:12000, ], (100., 150.))
  scatter_plot(votes[:6000, ], None, None, 100, 20)  # w/o thresholding
  scatter_plot(votes[:6000, ], 3500, 1500, 100, 20)  # with thresholding


if __name__ == '__main__':
  app.run(main)
