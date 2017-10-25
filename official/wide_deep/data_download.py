# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Download and clean the Census Income Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from six.moves import urllib
import tensorflow as tf

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'adult.test'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='/tmp/census_data',
    help='Directory to download census data')


def _download_and_clean_file(filename, url):
  """Downloads data from url, and makes changes to match the CSV format."""
  temp_file, _ = urllib.request.urlretrieve(url)
  with tf.gfile.Open(temp_file, 'r') as temp_eval_file:
    with tf.gfile.Open(filename, 'w') as eval_file:
      for line in temp_eval_file:
        line = line.strip()
        line = line.replace(', ', ',')
        if not line or ',' not in line:
          continue
        if line[-1] == '.':
          line = line[:-1]
        line += '\n'
        eval_file.write(line)
  tf.gfile.Remove(temp_file)


def main(unused_argv):
  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MkDir(FLAGS.data_dir)

  training_file_path = os.path.join(FLAGS.data_dir, TRAINING_FILE)
  _download_and_clean_file(training_file_path, TRAINING_URL)

  eval_file_path = os.path.join(FLAGS.data_dir, EVAL_FILE)
  _download_and_clean_file(eval_file_path, EVAL_URL)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
