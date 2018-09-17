# Copyright 2018 The TensorFlow Authors.
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

"""Tests for kepler_io.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from absl import flags
from absl.testing import absltest
import numpy as np

from light_curve_util import kepler_io

FLAGS = flags.FLAGS

_DATA_DIR = "light_curve_util/test_data/"


class KeplerIoTest(absltest.TestCase):

  def setUp(self):
    self.data_dir = os.path.join(FLAGS.test_srcdir, _DATA_DIR)

  def testScrambleLightCurve(self):
    all_flux = [[11, 12], [21], [np.nan, np.nan, 33], [41, 42]]
    all_time = [[101, 102], [201], [301, 302, 303], [401, 402]]
    all_quarters = [3, 4, 7, 14]
    scramble_type = "SCR1"  # New quarters order will be [14,7,3,4].

    scr_time, scr_flux = kepler_io.scramble_light_curve(
        all_time, all_flux, all_quarters, scramble_type)

    # NaNs are not removed in this function.
    gold_flux = [[41, 42], [np.nan, np.nan, 33], [11, 12], [21]]
    gold_time = [[101, 102], [201, 301, 302], [303, 401], [402]]

    self.assertEqual(len(gold_flux), len(scr_flux))
    self.assertEqual(len(gold_time), len(scr_time))

    for i in range(len(gold_flux)):
      np.testing.assert_array_equal(gold_flux[i], scr_flux[i])
      np.testing.assert_array_equal(gold_time[i], scr_time[i])

  def testKeplerFilenames(self):
    # All quarters.
    filenames = kepler_io.kepler_filenames(
        "/my/dir/", 1234567, check_existence=False)
    self.assertItemsEqual([
        "/my/dir/0012/001234567/kplr001234567-2009131105131_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2009166043257_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2009259160929_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2009350155506_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2010078095331_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2010009091648_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2010174085026_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2010265121752_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2010355172524_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2011073133259_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2011177032512_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2011271113734_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2012004120508_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2012088054726_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2012179063303_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2012277125453_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2013011073258_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2013098041711_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2013131215648_llc.fits"
    ], filenames)

    # Subset of quarters.
    filenames = kepler_io.kepler_filenames(
        "/my/dir/", 1234567, quarters=[3, 4], check_existence=False)
    self.assertItemsEqual([
        "/my/dir/0012/001234567/kplr001234567-2009350155506_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2010078095331_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2010009091648_llc.fits"
    ], filenames)

    # Injected group.
    filenames = kepler_io.kepler_filenames(
        "/my/dir/",
        1234567,
        quarters=[3, 4],
        injected_group="inj1",
        check_existence=False)
    # pylint:disable=line-too-long
    self.assertItemsEqual([
        "/my/dir/0012/001234567/kplr001234567-2009350155506_INJECTED-inj1_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2010078095331_INJECTED-inj1_llc.fits",
        "/my/dir/0012/001234567/kplr001234567-2010009091648_INJECTED-inj1_llc.fits"
    ], filenames)
    # pylint:enable=line-too-long

    # Short cadence.
    filenames = kepler_io.kepler_filenames(
        "/my/dir/",
        1234567,
        long_cadence=False,
        quarters=[0, 1],
        check_existence=False)
    self.assertItemsEqual([
        "/my/dir/0012/001234567/kplr001234567-2009131110544_slc.fits",
        "/my/dir/0012/001234567/kplr001234567-2009166044711_slc.fits"
    ], filenames)

    # Check existence.
    filenames = kepler_io.kepler_filenames(
        self.data_dir, 11442793, check_existence=True)
    expected_filenames = [
        os.path.join(self.data_dir,
                     "0114/011442793/kplr011442793-{}_llc.fits".format(q))
        for q in ["2009350155506", "2010009091648", "2010174085026"]
    ]
    self.assertItemsEqual(expected_filenames, filenames)

  def testReadKeplerLightCurve(self):
    filenames = [
        os.path.join(self.data_dir,
                     "0114/011442793/kplr011442793-{}_llc.fits".format(q))
        for q in ["2009350155506", "2010009091648", "2010174085026"]
    ]
    all_time, all_flux = kepler_io.read_kepler_light_curve(filenames)
    self.assertLen(all_time, 3)
    self.assertLen(all_flux, 3)
    self.assertLen(all_time[0], 4134)
    self.assertLen(all_flux[0], 4134)
    self.assertLen(all_time[1], 1008)
    self.assertLen(all_flux[1], 1008)
    self.assertLen(all_time[2], 4486)
    self.assertLen(all_flux[2], 4486)

    for time, flux in zip(all_time, all_flux):
      self.assertTrue(np.isfinite(time).all())
      self.assertTrue(np.isfinite(flux).all())

  def testReadKeplerLightCurveScrambled(self):
    filenames = [
        os.path.join(self.data_dir,
                     "0114/011442793/kplr011442793-{}_llc.fits".format(q))
        for q in ["2009350155506", "2010009091648", "2010174085026"]
    ]
    all_time, all_flux = kepler_io.read_kepler_light_curve(
        filenames, scramble_type="SCR1")
    self.assertLen(all_time, 3)
    self.assertLen(all_flux, 3)

    # Arrays are shorter than above due to separation of time and flux NaNs.
    self.assertLen(all_time[0], 4344)
    self.assertLen(all_flux[0], 4344)
    self.assertLen(all_time[1], 4041)
    self.assertLen(all_flux[1], 4041)
    self.assertLen(all_time[2], 1008)
    self.assertLen(all_flux[2], 1008)

    for time, flux in zip(all_time, all_flux):
      self.assertTrue(np.isfinite(time).all())
      self.assertTrue(np.isfinite(flux).all())

  def testReadKeplerLightCurveScrambledInterpolateMissingTime(self):
    filenames = [
        os.path.join(self.data_dir,
                     "0114/011442793/kplr011442793-{}_llc.fits".format(q))
        for q in ["2009350155506", "2010009091648", "2010174085026"]
    ]
    all_time, all_flux = kepler_io.read_kepler_light_curve(
        filenames, scramble_type="SCR1", interpolate_missing_time=True)
    self.assertLen(all_time, 3)
    self.assertLen(all_flux, 3)

    self.assertLen(all_time[0], 4486)
    self.assertLen(all_flux[0], 4486)
    self.assertLen(all_time[1], 4134)
    self.assertLen(all_flux[1], 4134)
    self.assertLen(all_time[2], 1008)
    self.assertLen(all_flux[2], 1008)

    for time, flux in zip(all_time, all_flux):
      self.assertTrue(np.isfinite(time).all())
      self.assertTrue(np.isfinite(flux).all())


if __name__ == "__main__":
  FLAGS.test_srcdir = ""
  absltest.main()
