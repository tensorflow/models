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

r"""Script to preprocesses data from the Kepler space telescope.

This script produces training, validation and test sets of labeled Kepler
Threshold Crossing Events (TCEs). A TCE is a detected periodic event on a
particular Kepler target star that may or may not be a transiting planet. Each
TCE in the output contains local and global views of its light curve; auxiliary
features such as period and duration; and a label indicating whether the TCE is
consistent with being a transiting planet. The data sets produced by this script
can be used to train and evaluate models that classify Kepler TCEs.

The input TCEs and their associated labels are specified by the DR24 TCE Table,
which can be downloaded in CSV format from the NASA Exoplanet Archive at:

  https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=q1_q17_dr24_tce

The downloaded CSV file should contain at least the following column names:
  rowid: Integer ID of the row in the TCE table.
  kepid: Kepler ID of the target star.
  tce_plnt_num: TCE number within the target star.
  tce_period: Orbital period of the detected event, in days.
  tce_time0bk: The time corresponding to the center of the first detected
      traisit in Barycentric Julian Day (BJD) minus a constant offset of
      2,454,833.0 days.
  tce_duration: Duration of the detected transit, in hours.
  av_training_set: Autovetter training set label; one of PC (planet candidate),
      AFP (astrophysical false positive), NTP (non-transiting phenomenon),
      UNK (unknown).

The Kepler light curves can be downloaded from the Mikulski Archive for Space
Telescopes (MAST) at:

  http://archive.stsci.edu/pub/kepler/lightcurves.

The Kepler data is assumed to reside in a directory with the same structure as
the MAST archive. Specifically, the file names for a particular Kepler target
star should have the following format:

    .../${kep_id:0:4}/${kep_id}/kplr${kep_id}-${quarter_prefix}_${type}.fits,

where:
  kep_id is the Kepler id left-padded with zeros to length 9;
  quarter_prefix is the file name quarter prefix;
  type is one of "llc" (long cadence light curve) or "slc" (short cadence light
    curve).

The output TFRecord file contains one serialized tensorflow.train.Example
protocol buffer for each TCE in the input CSV file. Each Example contains the
following light curve representations:
  global_view: Vector of length 2001; the Global View of the TCE.
  local_view: Vector of length 201; the Local View of the TCE.

In addition, each Example contains the value of each column in the input TCE CSV
file. Some of these features may be useful as auxiliary features to the model.
The columns include:
  rowid: Integer ID of the row in the TCE table.
  kepid: Kepler ID of the target star.
  tce_plnt_num: TCE number within the target star.
  av_training_set: Autovetter training set label.
  tce_period: Orbital period of the detected event, in days.
  ...
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from astronet.data import preprocess


parser = argparse.ArgumentParser()

_DR24_TCE_URL = ("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/"
                 "nph-tblView?app=ExoTbls&config=q1_q17_dr24_tce")

parser.add_argument(
    "--input_tce_csv_file",
    type=str,
    required=True,
    help="CSV file containing the Q1-Q17 DR24 Kepler TCE table. Must contain "
    "columns: rowid, kepid, tce_plnt_num, tce_period, tce_duration, "
    "tce_time0bk. Download from: %s" % _DR24_TCE_URL)

parser.add_argument(
    "--kepler_data_dir",
    type=str,
    required=True,
    help="Base folder containing Kepler data.")

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory in which to save the output.")

parser.add_argument(
    "--num_train_shards",
    type=int,
    default=8,
    help="Number of file shards to divide the training set into.")

parser.add_argument(
    "--num_worker_processes",
    type=int,
    default=5,
    help="Number of subprocesses for processing the TCEs in parallel.")

# Name and values of the column in the input CSV file to use as training labels.
_LABEL_COLUMN = "av_training_set"
_ALLOWED_LABELS = {"PC", "AFP", "NTP"}


def _set_float_feature(ex, name, value):
  """Sets the value of a float feature in a tensorflow.train.Example proto."""
  assert name not in ex.features.feature, "Duplicate feature: %s" % name
  ex.features.feature[name].float_list.value.extend([float(v) for v in value])


def _set_bytes_feature(ex, name, value):
  """Sets the value of a bytes feature in a tensorflow.train.Example proto."""
  assert name not in ex.features.feature, "Duplicate feature: %s" % name
  ex.features.feature[name].bytes_list.value.extend([
      str(v).encode("latin-1") for v in value])


def _set_int64_feature(ex, name, value):
  """Sets the value of an int64 feature in a tensorflow.train.Example proto."""
  assert name not in ex.features.feature, "Duplicate feature: %s" % name
  ex.features.feature[name].int64_list.value.extend([int(v) for v in value])


def _process_tce(tce):
  """Processes the light curve for a Kepler TCE and returns an Example proto.

  Args:
    tce: Row of the input TCE table.

  Returns:
    A tensorflow.train.Example proto containing TCE features.

  Raises:
    IOError: If the light curve files for this Kepler ID cannot be found.
  """
  # Read and process the light curve.
  time, flux = preprocess.read_and_process_light_curve(tce.kepid,
                                                       FLAGS.kepler_data_dir)
  time, flux = preprocess.phase_fold_and_sort_light_curve(
      time, flux, tce.tce_period, tce.tce_time0bk)

  # Generate the local and global views.
  global_view = preprocess.global_view(time, flux, tce.tce_period)
  local_view = preprocess.local_view(time, flux, tce.tce_period,
                                     tce.tce_duration)

  # Make output proto.
  ex = tf.train.Example()

  # Set time series features.
  _set_float_feature(ex, "global_view", global_view)
  _set_float_feature(ex, "local_view", local_view)

  # Set other columns.
  for col_name, value in tce.items():
    if np.issubdtype(type(value), np.integer):
      _set_int64_feature(ex, col_name, [value])
    else:
      try:
        _set_float_feature(ex, col_name, [float(value)])
      except ValueError:
        _set_bytes_feature(ex, col_name, [value])

  return ex


def _process_file_shard(tce_table, file_name):
  """Processes a single file shard.

  Args:
    tce_table: A Pandas DateFrame containing the TCEs in the shard.
    file_name: The output TFRecord file.
  """
  process_name = multiprocessing.current_process().name
  shard_name = os.path.basename(file_name)
  shard_size = len(tce_table)
  tf.logging.info("%s: Processing %d items in shard %s", process_name,
                  shard_size, shard_name)

  with tf.python_io.TFRecordWriter(file_name) as writer:
    num_processed = 0
    for _, tce in tce_table.iterrows():
      example = _process_tce(tce)
      if example is not None:
        writer.write(example.SerializeToString())

      num_processed += 1
      if not num_processed % 10:
        tf.logging.info("%s: Processed %d/%d items in shard %s", process_name,
                        num_processed, shard_size, shard_name)

  tf.logging.info("%s: Wrote %d items in shard %s", process_name, shard_size,
                  shard_name)


def main(argv):
  del argv  # Unused.

  # Make the output directory if it doesn't already exist.
  tf.gfile.MakeDirs(FLAGS.output_dir)

  # Read CSV file of Kepler KOIs.
  tce_table = pd.read_csv(
      FLAGS.input_tce_csv_file, index_col="rowid", comment="#")
  tce_table["tce_duration"] /= 24  # Convert hours to days.
  tf.logging.info("Read TCE CSV file with %d rows.", len(tce_table))

  # Filter TCE table to allowed labels.
  allowed_tces = tce_table[_LABEL_COLUMN].apply(lambda l: l in _ALLOWED_LABELS)
  tce_table = tce_table[allowed_tces]
  num_tces = len(tce_table)
  tf.logging.info("Filtered to %d TCEs with labels in %s.", num_tces,
                  list(_ALLOWED_LABELS))

  # Randomly shuffle the TCE table.
  np.random.seed(123)
  tce_table = tce_table.iloc[np.random.permutation(num_tces)]
  tf.logging.info("Randomly shuffled TCEs.")

  # Partition the TCE table as follows:
  #   train_tces = 80% of TCEs
  #   val_tces = 10% of TCEs (for validation during training)
  #   test_tces = 10% of TCEs (for final evaluation)
  train_cutoff = int(0.80 * num_tces)
  val_cutoff = int(0.90 * num_tces)
  train_tces = tce_table[0:train_cutoff]
  val_tces = tce_table[train_cutoff:val_cutoff]
  test_tces = tce_table[val_cutoff:]
  tf.logging.info(
      "Partitioned %d TCEs into training (%d), validation (%d) and test (%d)",
      num_tces, len(train_tces), len(val_tces), len(test_tces))

  # Further split training TCEs into file shards.
  file_shards = []  # List of (tce_table_shard, file_name).
  boundaries = np.linspace(0, len(train_tces),
                           FLAGS.num_train_shards + 1).astype(np.int)
  for i in range(FLAGS.num_train_shards):
    start = boundaries[i]
    end = boundaries[i + 1]
    file_shards.append((train_tces[start:end], os.path.join(
        FLAGS.output_dir, "train-%.5d-of-%.5d" % (i, FLAGS.num_train_shards))))

  # Validation and test sets each have a single shard.
  file_shards.append((val_tces, os.path.join(FLAGS.output_dir,
                                             "val-00000-of-00001")))
  file_shards.append((test_tces, os.path.join(FLAGS.output_dir,
                                              "test-00000-of-00001")))
  num_file_shards = len(file_shards)

  # Launch subprocesses for the file shards.
  num_processes = min(num_file_shards, FLAGS.num_worker_processes)
  tf.logging.info("Launching %d subprocesses for %d total file shards",
                  num_processes, num_file_shards)

  pool = multiprocessing.Pool(processes=num_processes)
  async_results = [
      pool.apply_async(_process_file_shard, file_shard)
      for file_shard in file_shards
  ]
  pool.close()

  # Instead of pool.join(), we call async_result.get() to ensure any exceptions
  # raised by the worker processes are also raised here.
  for async_result in async_results:
    async_result.get()

  tf.logging.info("Finished processing %d total file shards", num_file_shards)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
