"""Tests for robust_mean.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from third_party.robust_mean import robust_mean
from third_party.robust_mean.test_data import random_normal


class RobustMeanTest(absltest.TestCase):

  def testRobustMean(self):
    # To avoid non-determinism in the unit test, we use a pre-generated vector
    # of length 1,000. Each entry is independently sampled from a random normal
    # distribution with mean 2 and standard deviation 1. The maximum value of
    # y is 6.075 (+4.075 sigma from the mean) and the minimum value is -1.54
    # (-3.54 sigma from the mean).
    y = np.array(random_normal.RANDOM_NORMAL)
    self.assertAlmostEqual(np.mean(y), 2.00336615850485)
    self.assertAlmostEqual(np.std(y), 1.01690907798)

    # High cut. No points rejected, so the mean should be the sample mean, and
    # the mean standard deviation should be the sample standard deviation
    # divided by sqrt(1000 - 1).
    mean, mean_stddev, mask = robust_mean.robust_mean(y, cut=5)
    self.assertAlmostEqual(mean, 2.00336615850485)
    self.assertAlmostEqual(mean_stddev, 0.032173579)
    self.assertLen(mask, 1000)
    self.assertEqual(np.sum(mask), 1000)

    # Cut of 3 standard deviations.
    mean, mean_stddev, mask = robust_mean.robust_mean(y, cut=3)
    self.assertAlmostEqual(mean, 2.0059050070632178)
    self.assertAlmostEqual(mean_stddev, 0.03197075302321066)
    # There are exactly 3 points in the sample less than 1 or greater than 5.
    # These have indices 12, 220, 344.
    self.assertLen(mask, 1000)
    self.assertEqual(np.sum(mask), 997)
    self.assertFalse(np.any(mask[[12, 220, 344]]))

    # Add outliers. This corrupts the sample mean to 2.082.
    mean, mean_stddev, mask = robust_mean.robust_mean(
        y=np.concatenate([y, [10] * 10]), cut=5)
    self.assertAlmostEqual(mean, 2.0033661585048681)
    self.assertAlmostEqual(mean_stddev, 0.032013749413590531)
    self.assertLen(mask, 1010)
    self.assertEqual(np.sum(mask), 1000)
    self.assertFalse(np.any(mask[1000:1010]))

    # Add an outlier. This corrupts the mean to 1.002.
    mean, mean_stddev, mask = robust_mean.robust_mean(
        y=np.concatenate([y, [-1000]]), cut=5)
    self.assertAlmostEqual(mean, 2.0033661585048681)
    self.assertAlmostEqual(mean_stddev, 0.032157488597211903)
    self.assertLen(mask, 1001)
    self.assertEqual(np.sum(mask), 1000)
    self.assertFalse(mask[1000])


if __name__ == "__main__":
  absltest.main()
