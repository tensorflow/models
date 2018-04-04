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

"""Plots two graphs illustrating cost of privacy per answered query.

PRESENTLY NOT USED.

A script in support of the PATE2 paper. The input is a file containing a numpy
array of votes, one query per row, one class per column. Ex:
  43, 1821, ..., 3
  31, 16, ..., 0
  ...
  0, 86, ..., 438
The output is written to a specified directory and consists of two pdf files.
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

plt.style.use('ggplot')

FLAGS = flags.FLAGS
flags.DEFINE_string('counts_file', '', 'Counts file.')
flags.DEFINE_string('figures_dir', '', 'Path where figures are written to.')


def plot_rdp_curve_per_example(votes, sigmas):
  orders = np.linspace(1., 100., endpoint=True, num=1000)
  orders[0] = 1.5

  fig, ax = plt.subplots()
  fig.set_figheight(4.5)
  fig.set_figwidth(4.7)

  styles = [':', '-']
  labels = ['ex1', 'ex2']

  for i in xrange(votes.shape[0]):
    print(sorted(votes[i,], reverse=True)[:10])
    for sigma in sigmas:
      logq = pate.compute_logq_gaussian(votes[i,], sigma)
      rdp = pate.rdp_gaussian(logq, sigma, orders)
      ax.plot(
          orders,
          rdp,
          label=r'{} $\sigma$={}'.format(labels[i], int(sigma)),
          linestyle=styles[i],
          linewidth=5)

  for sigma in sigmas:
    ax.plot(
        orders,
        pate.rdp_data_independent_gaussian(sigma, orders),
        alpha=.3,
        label=r'Data-ind bound $\sigma$={}'.format(int(sigma)),
        linewidth=10)

  plt.yticks([0, .01])
  plt.xlabel(r'Order $\lambda$', fontsize=16)
  plt.ylabel(r'RDP value $\varepsilon$ at $\lambda$', fontsize=16)
  ax.tick_params(labelsize=14)

  fout_name = os.path.join(FLAGS.figures_dir, 'rdp_flow1.pdf')
  print('Saving the graph to ' + fout_name)
  fig.savefig(fout_name, bbox_inches='tight')
  plt.legend(loc=0, fontsize=13)
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
  orders[0] = 1.5

  fig, ax = plt.subplots()
  fig.set_figheight(4.5)
  fig.set_figwidth(4.7)

  for sigma in sigmas:
    rdp = compute_rdp_curve(votes, 5000, 1000, sigma, orders, 2000)
    ax.plot(
        orders,
        rdp,
        alpha=.8,
        label=r'$\sigma$={}'.format(int(sigma)),
        linewidth=5)

  plt.xlabel(r'Order $\lambda$', fontsize=16)
  plt.ylabel(r'RDP value $\varepsilon$ at $\lambda$', fontsize=16)
  ax.tick_params(labelsize=14)

  fout_name = os.path.join(FLAGS.figures_dir, 'rdp_flow2.pdf')
  print('Saving the graph to ' + fout_name)
  fig.savefig(fout_name, bbox_inches='tight')
  plt.legend(loc=0, fontsize=13)
  plt.show()


def main(argv):
  del argv  # Unused.
  fin_name = os.path.expanduser(FLAGS.counts_file)
  print('Reading raw votes from ' + fin_name)
  sys.stdout.flush()

  votes = np.load(fin_name)
  votes = votes[:12000,]  # truncate to 4000 samples

  v1 = [2550, 2200, 250]  # based on votes[2,]
  v2 = [2600, 2200, 200]  # based on votes[381,]
  plot_rdp_curve_per_example(np.array([v1, v2]), (100., 150.))

  plot_rdp_total(votes, (100., 150.))


if __name__ == '__main__':
  app.run(main)
