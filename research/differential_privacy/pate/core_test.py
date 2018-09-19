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

"""Tests for pate.core."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest
import numpy as np

import core as pate


class PateTest(unittest.TestCase):

  def _test_rdp_gaussian_value_errors(self):
    # Test for ValueErrors.
    with self.assertRaises(ValueError):
      pate.rdp_gaussian(1.0, 1.0, np.array([2, 3, 4]))
    with self.assertRaises(ValueError):
      pate.rdp_gaussian(np.log(0.5), -1.0, np.array([2, 3, 4]))
    with self.assertRaises(ValueError):
      pate.rdp_gaussian(np.log(0.5), 1.0, np.array([1, 3, 4]))

  def _test_rdp_gaussian_as_function_of_q(self):
    # Test for data-independent and data-dependent ranges over q.
    # The following corresponds to orders 1.1, 2.5, 32, 250
    # sigmas 1.5, 15, 1500, 15000.
    # Hand calculated -log(q0)s arranged in a 'sigma major' ordering.
    neglogq0s = [
        2.8, 2.6, 427, None, 4.8, 4.0, 4.7, 275, 9.6, 8.8, 6.0, 4, 12, 11.2,
        8.6, 6.4
    ]
    idx_neglogq0s = 0  # To iterate through neglogq0s.
    orders = [1.1, 2.5, 32, 250]
    sigmas = [1.5, 15, 1500, 15000]
    for sigma in sigmas:
      for order in orders:
        curr_neglogq0 = neglogq0s[idx_neglogq0s]
        idx_neglogq0s += 1
        if curr_neglogq0 is None:  # sigma == 1.5 and order == 250:
          continue

        rdp_at_q0 = pate.rdp_gaussian(-curr_neglogq0, sigma, order)

        # Data-dependent range. (Successively halve the value of q.)
        logq_dds = (-curr_neglogq0 - np.array(
            [0, np.log(2), np.log(4), np.log(8)]))
        # Check that in q_dds, rdp is decreasing.
        for idx in range(len(logq_dds) - 1):
          self.assertGreater(
              pate.rdp_gaussian(logq_dds[idx], sigma, order),
              pate.rdp_gaussian(logq_dds[idx + 1], sigma, order))

        # Data-independent range.
        q_dids = np.exp(-curr_neglogq0) + np.array([0.1, 0.2, 0.3, 0.4])
        # Check that in q_dids, rdp is constant.
        for q in q_dids:
          self.assertEqual(rdp_at_q0, pate.rdp_gaussian(
              np.log(q), sigma, order))

  def _test_compute_eps_from_delta_value_error(self):
    # Test for ValueError.
    with self.assertRaises(ValueError):
      pate.compute_eps_from_delta([1.1, 2, 3, 4], [1, 2, 3], 0.001)

  def _test_compute_eps_from_delta_monotonicity(self):
    # Test for monotonicity with respect to delta.
    orders = [1.1, 2.5, 250.0]
    sigmas = [1e-3, 1.0, 1e5]
    deltas = [1e-60, 1e-6, 0.1, 0.999]
    for sigma in sigmas:
      list_of_eps = []
      rdps_for_gaussian = np.array(orders) / (2 * sigma**2)
      for delta in deltas:
        list_of_eps.append(
            pate.compute_eps_from_delta(orders, rdps_for_gaussian, delta)[0])

      # Check that in list_of_eps, epsilons are decreasing (as delta increases).
      sorted_list_of_eps = list(list_of_eps)
      sorted_list_of_eps.sort(reverse=True)
      self.assertEqual(list_of_eps, sorted_list_of_eps)

  def _test_compute_q0(self):
    # Stub code to search a logq space and figure out logq0 by eyeballing
    # results. This code does not run with the tests. Remove underscore to run.
    sigma = 15
    order = 250
    logqs = np.arange(-290, -270, 1)
    count = 0
    for logq in logqs:
      count += 1
      sys.stdout.write("\t%0.5g: %0.10g" %
                       (logq, pate.rdp_gaussian(logq, sigma, order)))
      sys.stdout.flush()
      if count % 5 == 0:
        print("")

  def test_rdp_gaussian(self):
    self._test_rdp_gaussian_value_errors()
    self._test_rdp_gaussian_as_function_of_q()

  def test_compute_eps_from_delta(self):
    self._test_compute_eps_from_delta_value_error()
    self._test_compute_eps_from_delta_monotonicity()


if __name__ == "__main__":
  unittest.main()
