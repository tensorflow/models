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

"""Functions for reading Kepler data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from astropy.io import fits
import numpy as np

from tensorflow import gfile

LONG_CADENCE_TIME_DELTA_DAYS = 0.02043422  # Approximately 29.4 minutes.

# Quarter index to filename prefix for long cadence Kepler data.
# Reference: https://archive.stsci.edu/kepler/software/get_kepler.py
LONG_CADENCE_QUARTER_PREFIXES = {
    0: ["2009131105131"],
    1: ["2009166043257"],
    2: ["2009259160929"],
    3: ["2009350155506"],
    4: ["2010078095331", "2010009091648"],
    5: ["2010174085026"],
    6: ["2010265121752"],
    7: ["2010355172524"],
    8: ["2011073133259"],
    9: ["2011177032512"],
    10: ["2011271113734"],
    11: ["2012004120508"],
    12: ["2012088054726"],
    13: ["2012179063303"],
    14: ["2012277125453"],
    15: ["2013011073258"],
    16: ["2013098041711"],
    17: ["2013131215648"]
}

# Quarter index to filename prefix for short cadence Kepler data.
# Reference: https://archive.stsci.edu/kepler/software/get_kepler.py
SHORT_CADENCE_QUARTER_PREFIXES = {
    0: ["2009131110544"],
    1: ["2009166044711"],
    2: ["2009201121230", "2009231120729", "2009259162342"],
    3: ["2009291181958", "2009322144938", "2009350160919"],
    4: ["2010009094841", "2010019161129", "2010049094358", "2010078100744"],
    5: ["2010111051353", "2010140023957", "2010174090439"],
    6: ["2010203174610", "2010234115140", "2010265121752"],
    7: ["2010296114515", "2010326094124", "2010355172524"],
    8: ["2011024051157", "2011053090032", "2011073133259"],
    9: ["2011116030358", "2011145075126", "2011177032512"],
    10: ["2011208035123", "2011240104155", "2011271113734"],
    11: ["2011303113607", "2011334093404", "2012004120508"],
    12: ["2012032013838", "2012060035710", "2012088054726"],
    13: ["2012121044856", "2012151031540", "2012179063303"],
    14: ["2012211050319", "2012242122129", "2012277125453"],
    15: ["2012310112549", "2012341132017", "2013011073258"],
    16: ["2013017113907", "2013065031647", "2013098041711"],
    17: ["2013121191144", "2013131215648"]
}


def kepler_filenames(base_dir,
                     kep_id,
                     long_cadence=True,
                     quarters=None,
                     injected_group=None,
                     check_existence=True):
  """Returns the light curve filenames for a Kepler target star.

  This function assumes the directory structure of the Mikulski Archive for
  Space Telescopes (http://archive.stsci.edu/pub/kepler/lightcurves).
  Specifically, the filenames for a particular Kepler target star have the
  following format:

    ${kep_id:0:4}/${kep_id}/kplr${kep_id}-${quarter_prefix}_${type}.fits,

  where:
    kep_id is the Kepler id left-padded with zeros to length 9;
    quarter_prefix is the filename quarter prefix;
    type is one of "llc" (long cadence light curve) or "slc" (short cadence
        light curve).

  Args:
    base_dir: Base directory containing Kepler data.
    kep_id: Id of the Kepler target star. May be an int or a possibly zero-
        padded string.
    long_cadence: Whether to read a long cadence (~29.4 min / measurement) light
        curve as opposed to a short cadence (~1 min / measurement) light curve.
    quarters: Optional list of integers in [0, 17]; the quarters of the Kepler
        mission to return.
    injected_group: Optional string indicating injected light curves. One of
        "inj1", "inj2", "inj3".
    check_existence: If True, only return filenames corresponding to files that
        exist (not all stars have data for all quarters).

  Returns:
    A list of filenames.
  """
  # Pad the Kepler id with zeros to length 9.
  kep_id = "%.9d" % int(kep_id)

  quarter_prefixes, cadence_suffix = ((LONG_CADENCE_QUARTER_PREFIXES, "llc")
                                      if long_cadence else
                                      (SHORT_CADENCE_QUARTER_PREFIXES, "slc"))

  if quarters is None:
    quarters = quarter_prefixes.keys()

  quarters = sorted(quarters)  # Sort quarters chronologically.

  filenames = []
  base_dir = os.path.join(base_dir, kep_id[0:4], kep_id)
  for quarter in quarters:
    for quarter_prefix in quarter_prefixes[quarter]:
      if injected_group:
        base_name = "kplr%s-%s_INJECTED-%s_%s.fits" % (kep_id, quarter_prefix,
                                                       injected_group,
                                                       cadence_suffix)
      else:
        base_name = "kplr%s-%s_%s.fits" % (kep_id, quarter_prefix,
                                           cadence_suffix)
      filename = os.path.join(base_dir, base_name)
      # Not all stars have data for all quarters.
      if not check_existence or gfile.Exists(filename):
        filenames.append(filename)

  return filenames


def read_kepler_light_curve(filenames,
                            light_curve_extension="LIGHTCURVE",
                            invert=False):
  """Reads time and flux measurements for a Kepler target star.

  Args:
    filenames: A list of .fits files containing time and flux measurements.
    light_curve_extension: Name of the HDU 1 extension containing light curves.
    invert: Whether to invert the flux measurements by multiplying by -1.

  Returns:
    all_time: A list of numpy arrays; the time values of the light curve.
    all_flux: A list of numpy arrays corresponding to the time arrays in
        all_time.
  """
  all_time = []
  all_flux = []

  for filename in filenames:
    with fits.open(gfile.Open(filename, "rb")) as hdu_list:
      light_curve = hdu_list[light_curve_extension].data
      time = light_curve.TIME
      flux = light_curve.PDCSAP_FLUX

    # Remove NaN flux values.
    valid_indices = np.where(np.isfinite(flux))
    time = time[valid_indices]
    flux = flux[valid_indices]

    if invert:
      flux *= -1

    if time.size:
      all_time.append(time)
      all_flux.append(flux)

  return all_time, all_flux
