# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Preprocesses TIMIT from raw wavfiles to create a set of TFRecords.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import random
import re

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string("raw_timit_dir", None,
                          "Directory containing TIMIT files.")
tf.app.flags.DEFINE_string("out_dir", None,
                          "Output directory for TFRecord files.")
tf.app.flags.DEFINE_float("valid_frac", 0.05,
                          "Fraction of train set to use as valid set. "
                          "Must be between 0.0 and 1.0.")

tf.app.flags.mark_flag_as_required("raw_timit_dir")
tf.app.flags.mark_flag_as_required("out_dir")

FLAGS = tf.app.flags.FLAGS

NUM_TRAIN_FILES = 4620
NUM_TEST_FILES = 1680
SAMPLES_PER_TIMESTEP = 200

# Regexes for reading SPHERE header files.
SAMPLE_COUNT_REGEX = re.compile(r"sample_count -i (\d+)")
SAMPLE_MIN_REGEX = re.compile(r"sample_min -i (-?\d+)")
SAMPLE_MAX_REGEX = re.compile(r"sample_max -i (-?\d+)")


def get_filenames(split):
  """Get all wav filenames from the TIMIT archive."""
  path = os.path.join(FLAGS.raw_timit_dir, "TIMIT", split, "*", "*", "*.WAV")
  # Sort the output by name so the order is deterministic.
  files = sorted(glob.glob(path))
  return files


def load_timit_wav(filename):
  """Loads a TIMIT wavfile into a numpy array.

  TIMIT wavfiles include a SPHERE header, detailed in the TIMIT docs. The first
  line is the header type and the second is the length of the header in bytes.
  After the header, the remaining bytes are actual WAV data.

  The header includes information about the WAV data such as the number of
  samples and minimum and maximum amplitude. This function asserts that the
  loaded wav data matches the header.

  Args:
    filename: The name of the TIMIT wavfile to load.
  Returns:
    wav: A numpy array containing the loaded wav data.
  """
  wav_file = open(filename, "rb")
  header_type = wav_file.readline()
  header_length_str = wav_file.readline()
  # The header length includes the length of the first two lines.
  header_remaining_bytes = (int(header_length_str) - len(header_type) -
                            len(header_length_str))
  header = wav_file.read(header_remaining_bytes)
  # Read the relevant header fields.
  sample_count = int(SAMPLE_COUNT_REGEX.search(header).group(1))
  sample_min = int(SAMPLE_MIN_REGEX.search(header).group(1))
  sample_max = int(SAMPLE_MAX_REGEX.search(header).group(1))
  wav = np.fromstring(wav_file.read(), dtype="int16").astype("float32")
  # Check that the loaded data conforms to the header description.
  assert len(wav) == sample_count
  assert wav.min() == sample_min
  assert wav.max() == sample_max
  return wav


def preprocess(wavs, block_size, mean, std):
  """Normalize the wav data and reshape it into chunks."""
  processed_wavs = []
  for wav in wavs:
    wav = (wav - mean) / std
    wav_length = wav.shape[0]
    if wav_length % block_size != 0:
      pad_width = block_size - (wav_length % block_size)
      wav = np.pad(wav, (0, pad_width), "constant")
    assert wav.shape[0] % block_size == 0
    wav = wav.reshape((-1, block_size))
    processed_wavs.append(wav)
  return processed_wavs


def create_tfrecord_from_wavs(wavs, output_file):
  """Writes processed wav files to disk as sharded TFRecord files."""
  with tf.python_io.TFRecordWriter(output_file) as builder:
    for wav in wavs:
      builder.write(wav.astype(np.float32).tobytes())


def main(unused_argv):
  train_filenames = get_filenames("TRAIN")
  test_filenames = get_filenames("TEST")

  num_train_files = len(train_filenames)
  num_test_files = len(test_filenames)
  num_valid_files = int(num_train_files * FLAGS.valid_frac)
  num_train_files -= num_valid_files

  print("%d train / %d valid / %d test" % (
      num_train_files, num_valid_files, num_test_files))

  random.seed(1234)
  random.shuffle(train_filenames)

  valid_filenames = train_filenames[:num_valid_files]
  train_filenames = train_filenames[num_valid_files:]

  # Make sure there is no overlap in the train, test, and valid sets.
  train_s = set(train_filenames)
  test_s = set(test_filenames)
  valid_s = set(valid_filenames)
  # Disable explicit length testing to make the assertions more readable.
  # pylint: disable=g-explicit-length-test
  assert len(train_s & test_s) == 0
  assert len(train_s & valid_s) == 0
  assert len(valid_s & test_s) == 0
  # pylint: enable=g-explicit-length-test

  train_wavs = [load_timit_wav(f) for f in train_filenames]
  valid_wavs = [load_timit_wav(f) for f in valid_filenames]
  test_wavs = [load_timit_wav(f) for f in test_filenames]
  assert len(train_wavs) + len(valid_wavs) == NUM_TRAIN_FILES
  assert len(test_wavs) == NUM_TEST_FILES

  # Calculate the mean and standard deviation of the train set.
  train_stacked = np.hstack(train_wavs)
  train_mean = np.mean(train_stacked)
  train_std = np.std(train_stacked)
  print("train mean: %f  train std: %f" % (train_mean, train_std))

  # Process all data, normalizing with the train set statistics.
  processed_train_wavs = preprocess(train_wavs, SAMPLES_PER_TIMESTEP,
                                    train_mean, train_std)
  processed_valid_wavs = preprocess(valid_wavs, SAMPLES_PER_TIMESTEP,
                                    train_mean, train_std)
  processed_test_wavs = preprocess(test_wavs, SAMPLES_PER_TIMESTEP, train_mean,
                                   train_std)

  # Write the datasets to disk.
  create_tfrecord_from_wavs(
      processed_train_wavs,
      os.path.join(FLAGS.out_dir, "train"))
  create_tfrecord_from_wavs(
      processed_valid_wavs,
      os.path.join(FLAGS.out_dir, "valid"))
  create_tfrecord_from_wavs(
      processed_test_wavs,
      os.path.join(FLAGS.out_dir, "test"))


if __name__ == "__main__":
  tf.app.run()
