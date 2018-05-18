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

"""Illustrates how noisy thresholding check changes distribution of queries.

A script in support of the paper "Scalable Private Learning with PATE" by
Nicolas Papernot, Shuang Song, Ilya Mironov, Ananth Raghunathan, Kunal Talwar,
Ulfar Erlingsson (https://arxiv.org/abs/1802.08908).

The input is a file containing a numpy array of votes, one query per row, one
class per column. Ex:
  43, 1821, ..., 3
  31, 16, ..., 0
  ...
  0, 86, ..., 438
The output is one of two graphs depending on the setting of the plot variable.
The output is written to a pdf file.
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
flags.DEFINE_enum('plot', 'small', ['small', 'large'], 'Selects which of'
                  'the two plots is produced.')
flags.DEFINE_string('counts_file', None, 'Counts file.')
flags.DEFINE_string('plot_file', '', 'Plot file to write.')

flags.mark_flag_as_required('counts_file')


def compute_count_per_bin(bin_num, votes):
  """Tabulates number of examples in each bin.

  Args:
    bin_num: Number of bins.
    votes: A matrix of votes, where each row contains votes in one instance.

  Returns:
    Array of counts of length bin_num.
  """
  sums = np.sum(votes, axis=1)
  # Check that all rows contain the same number of votes.
  assert max(sums) == min(sums)

  s = max(sums)

  counts = np.zeros(bin_num)
  n = votes.shape[0]

  for i in xrange(n):
    v = votes[i,]
    bin_idx = int(math.floor(max(v) * bin_num / s))
    assert 0 <= bin_idx < bin_num
    counts[bin_idx] += 1

  return counts


def compute_privacy_cost_per_bins(bin_num, votes, sigma2, order):
  """Outputs average privacy cost per bin.

  Args:
    bin_num: Number of bins.
    votes: A matrix of votes, where each row contains votes in one instance.
    sigma2: The scale (std) of the Gaussian noise. (Same as sigma_2 in
            Algorithms 1 and 2.)
    order: The Renyi order for which privacy cost is computed.

  Returns:
    Expected eps of RDP (ignoring delta) per example in each bin.
  """
  n = votes.shape[0]

  bin_counts = np.zeros(bin_num)
  bin_rdp = np.zeros(bin_num)  # RDP at order=order

  for i in xrange(n):
    v = votes[i,]
    logq = pate.compute_logq_gaussian(v, sigma2)
    rdp_at_order = pate.rdp_gaussian(logq, sigma2, order)

    bin_idx = int(math.floor(max(v) * bin_num / sum(v)))
    assert 0 <= bin_idx < bin_num
    bin_counts[bin_idx] += 1
    bin_rdp[bin_idx] += rdp_at_order
    if (i + 1) % 1000 == 0:
      print('example {}'.format(i + 1))
      sys.stdout.flush()

  return bin_rdp / bin_counts


def compute_expected_answered_per_bin(bin_num, votes, threshold, sigma1):
  """Computes expected number of answers per bin.

  Args:
    bin_num: Number of bins.
    votes: A matrix of votes, where each row contains votes in one instance.
    threshold: The threshold against which check is performed.
    sigma1: The std of the Gaussian noise with which check is performed. (Same
      as sigma_1 in Algorithms 1 and 2.)

  Returns:
    Expected number of queries answered per bin.
  """
  n = votes.shape[0]

  bin_answered = np.zeros(bin_num)

  for i in xrange(n):
    v = votes[i,]
    p = math.exp(pate.compute_logpr_answered(threshold, sigma1, v))
    bin_idx = int(math.floor(max(v) * bin_num / sum(v)))
    assert 0 <= bin_idx < bin_num
    bin_answered[bin_idx] += p
    if (i + 1) % 1000 == 0:
      print('example {}'.format(i + 1))
      sys.stdout.flush()

  return bin_answered


def main(argv):
  del argv  # Unused.
  fin_name = os.path.expanduser(FLAGS.counts_file)
  print('Reading raw votes from ' + fin_name)
  sys.stdout.flush()

  votes = np.load(fin_name)
  votes = votes[:4000,]  # truncate to 4000 samples

  if FLAGS.plot == 'small':
    bin_num = 5
    m_check = compute_expected_answered_per_bin(bin_num, votes, 3500, 1500)
  elif FLAGS.plot == 'large':
    bin_num = 10
    m_check = compute_expected_answered_per_bin(bin_num, votes, 3500, 1500)
    a_check = compute_expected_answered_per_bin(bin_num, votes, 5000, 1500)
    eps = compute_privacy_cost_per_bins(bin_num, votes, 100, 50)
  else:
    raise ValueError('--plot flag must be one of ["small", "large"]')

  counts = compute_count_per_bin(bin_num, votes)
  bins = np.linspace(0, 100, num=bin_num, endpoint=False)

  plt.close('all')
  fig, ax = plt.subplots()
  if FLAGS.plot == 'small':
    fig.set_figheight(5)
    fig.set_figwidth(5)
    ax.bar(
        bins,
        counts,
        20,
        color='orangered',
        linestyle='dotted',
        linewidth=5,
        edgecolor='red',
        fill=False,
        alpha=.5,
        align='edge',
        label='LNMax answers')
    ax.bar(
        bins,
        m_check,
        20,
        color='g',
        alpha=.5,
        linewidth=0,
        edgecolor='g',
        align='edge',
        label='Confident-GNMax\nanswers')
  elif FLAGS.plot == 'large':
    fig.set_figheight(4.7)
    fig.set_figwidth(7)
    ax.bar(
        bins,
        counts,
        10,
        linestyle='dashed',
        linewidth=5,
        edgecolor='red',
        fill=False,
        alpha=.5,
        align='edge',
        label='LNMax answers')
    ax.bar(
        bins,
        m_check,
        10,
        color='g',
        alpha=.5,
        linewidth=0,
        edgecolor='g',
        align='edge',
        label='Confident-GNMax\nanswers (moderate)')
    ax.bar(
        bins,
        a_check,
        10,
        color='b',
        alpha=.5,
        align='edge',
        label='Confident-GNMax\nanswers (aggressive)')
    ax2 = ax.twinx()
    bin_centers = [x + 5 for x in bins]
    ax2.plot(bin_centers, eps, 'ko', alpha=.8)
    ax2.set_ylim([1e-200, 1.])
    ax2.set_yscale('log')
    ax2.grid(False)
    ax2.set_yticks([1e-3, 1e-50, 1e-100, 1e-150, 1e-200])
    plt.tick_params(which='minor', right='off')
    ax2.set_ylabel(r'Per query privacy cost $\varepsilon$', fontsize=16)

  plt.xlim([0, 100])
  ax.set_ylim([0, 2500])
  # ax.set_yscale('log')
  ax.set_xlabel('Percentage of teachers that agree', fontsize=16)
  ax.set_ylabel('Number of queries answered', fontsize=16)
  vals = ax.get_xticks()
  ax.set_xticklabels([str(int(x)) + '%' for x in vals])
  ax.tick_params(labelsize=14, bottom=True, top=True, left=True, right=True)
  ax.legend(loc=2, prop={'size': 16})

  # simple: 'figures/noisy_thresholding_check_perf.pdf')
  # detailed: 'figures/noisy_thresholding_check_perf_details.pdf'

  print('Saving the graph to ' + FLAGS.plot_file)
  plt.savefig(os.path.expanduser(FLAGS.plot_file), bbox_inches='tight')
  plt.show()


if __name__ == '__main__':
  app.run(main)
