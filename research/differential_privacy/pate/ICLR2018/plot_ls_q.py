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

"""Plots LS(q).

A script in support of the PATE2 paper. NOT PRESENTLY USED.

The output is written to a specified directory as a pdf file.
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
import smooth_sensitivity as pate_ss

plt.style.use('ggplot')

FLAGS = flags.FLAGS

flags.DEFINE_string('figures_dir', '', 'Path where the output is written to.')


def compute_ls_q(sigma, order, num_classes):

  def beta(q):
    return pate_ss._compute_rdp_gnmax(sigma, math.log(q), order)

  def bu(q):
    return pate_ss._compute_bu_gnmax(q, sigma, order)

  def bl(q):
    return pate_ss._compute_bl_gnmax(q, sigma, order)

  def delta_beta(q):
    if q == 0 or q > .8:
      return 0
    beta_q = beta(q)
    beta_bu_q = beta(bu(q))
    beta_bl_q = beta(bl(q))
    assert beta_bl_q <= beta_q <= beta_bu_q
    return beta_bu_q - beta_q  # max(beta_bu_q - beta_q, beta_q - beta_bl_q)

  logq0 = pate_ss.compute_logq0_gnmax(sigma, order)
  logq1 = pate_ss._compute_logq1(sigma, order, num_classes)
  print(math.exp(logq1), math.exp(logq0))
  xs = np.linspace(0, .1, num=1000, endpoint=True)
  ys = [delta_beta(x) for x in xs]
  return xs, ys


def main(argv):
  del argv  # Unused.

  sigma = 20
  order = 20.
  num_classes = 10

  # sigma = 20
  # order = 25.
  # num_classes = 10

  x_axis, ys = compute_ls_q(sigma, order, num_classes)

  fig, ax = plt.subplots()
  fig.set_figheight(4.5)
  fig.set_figwidth(4.7)

  ax.plot(x_axis, ys, alpha=.8, linewidth=5)
  plt.xlabel('Number of queries answered', fontsize=16)
  plt.ylabel(r'Privacy cost $\varepsilon$ at $\delta=10^{-8}$', fontsize=16)
  ax.tick_params(labelsize=14)
  fout_name = os.path.join(FLAGS.figures_dir, 'ls_of_q.pdf')
  print('Saving the graph to ' + fout_name)
  plt.show()

  plt.close('all')


if __name__ == '__main__':
  app.run(main)
