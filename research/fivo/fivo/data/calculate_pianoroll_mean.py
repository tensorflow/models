# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Script to calculate the mean of a pianoroll dataset.

Given a pianoroll pickle file, this script loads the dataset and
calculates the mean of the training set. Then it updates the pickle file
so that the key "train_mean" points to the mean vector.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np

import tensorflow as tf


from datasets import sparse_pianoroll_to_dense

tf.app.flags.DEFINE_string('in_file', None,
                           'Filename of the pickled pianoroll dataset to load.')
tf.app.flags.DEFINE_string('out_file', None,
                           'Name of the output pickle file. Defaults to in_file, '
                           'updating the input pickle file.')
tf.app.flags.mark_flag_as_required('in_file')

FLAGS = tf.app.flags.FLAGS

MIN_NOTE = 21
MAX_NOTE = 108
NUM_NOTES = MAX_NOTE - MIN_NOTE + 1


def main(unused_argv):
  if FLAGS.out_file is None:
    FLAGS.out_file = FLAGS.in_file
  with tf.gfile.Open(FLAGS.in_file, 'r') as f:
    pianorolls = pickle.load(f)
  dense_pianorolls = [sparse_pianoroll_to_dense(p, MIN_NOTE, NUM_NOTES)[0]
                      for p in pianorolls['train']]
  # Concatenate all elements along the time axis.
  concatenated = np.concatenate(dense_pianorolls, axis=0)
  mean = np.mean(concatenated, axis=0)
  pianorolls['train_mean'] = mean
  # Write out the whole pickle file, including the train mean.
  pickle.dump(pianorolls, open(FLAGS.out_file, 'wb'))


if __name__ == '__main__':
  tf.app.run()
