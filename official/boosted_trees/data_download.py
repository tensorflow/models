"""Downloads the UCI HIGGS Dataset and prepares train data.

The details on the dataset are in https://archive.ics.uci.edu/ml/datasets/HIGGS

It takes a while as it needs to download 2.8 GB over the network, process, then
store it into the specified location as a compressed numpy file.

Usage:
$ python data_download.py --data_dir=/tmp/higgs_data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from six.moves import urllib
import tensorflow as tf

URL_ROOT = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280'
INPUT_FILE = 'HIGGS.csv.gz'
NPZ_FILE = 'HIGGS.csv.gz.npz'  # numpy compressed file to contain 'data' array.


def parse_args():
  """Parses arguments and returns a tuple (known_args, unparsed_args)."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir', type=str, default='/tmp/higgs_data',
      help='Directory to download higgs dataset and store training/eval data.')
  return parser.parse_known_args()


def _download_higgs_data_and_save_npz(data_dir):
  """Download higgs data and store as a numpy compressed file."""
  input_url = os.path.join(URL_ROOT, INPUT_FILE)
  np_filename = os.path.join(data_dir, NPZ_FILE)
  if tf.gfile.Exists(np_filename):
    raise ValueError('data_dir already has the processed data file: {}'.format(
        np_filename))
  if not tf.gfile.Exists(data_dir):
    tf.gfile.MkDir(data_dir)
  # 2.8 GB to download.
  try:
    print('Data downloading..')
    temp_filename, _ = urllib.request.urlretrieve(input_url)

    # Reading and parsing 11 million csv lines takes 2~3 minutes.
    print('Data processing.. taking multiple minutes..')
    data = pd.read_csv(
        temp_filename,
        dtype=np.float32,
        names=['c%02d' % i for i in range(29)]  # label + 28 features.
    ).as_matrix()
  finally:
    os.remove(temp_filename)

  # Writing to temporary location then copy to the data_dir (0.8 GB).
  f = tempfile.NamedTemporaryFile()
  np.savez_compressed(f, data=data)
  tf.gfile.Copy(f.name, np_filename)
  print('Data saved to: {}'.format(np_filename))


def main(unused_argv):
  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MkDir(FLAGS.data_dir)
  _download_higgs_data_and_save_npz(FLAGS.data_dir)


if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
