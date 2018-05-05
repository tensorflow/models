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

"""Shuffle samples for human evaluation.

Local launch command:
  python sample_shuffler.py
  --input_ml_path=/tmp/ptb/seq2seq_vd_shareemb_forreal_55_3
  --input_gan_path=/tmp/ptb/MaskGAN_PTB_ari_avg_56.29_v2.0.0
  --output_file_name=/tmp/ptb/shuffled_output.txt

  python sample_shuffler.py
  --input_ml_path=/tmp/generate_samples/MaskGAN_IMDB_Benchmark_87.1_v0.3.0
  --input_gan_path=/tmp/generate_samples/MaskGAN_IMDB_v1.0.1
  --output_file_name=/tmp/imdb/shuffled_output.txt
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Dependency imports
import numpy as np

import tensorflow as tf

tf.app.flags.DEFINE_string('input_ml_path', '/tmp', 'Model output directory.')
tf.app.flags.DEFINE_string('input_gan_path', '/tmp', 'Model output directory.')
tf.app.flags.DEFINE_string('output_file_name', '/tmp/ptb/shuffled_output.txt',
                           'Model output file.')
tf.app.flags.DEFINE_boolean(
    'output_masked_logs', False,
    'Whether to display for human evaluation (show masking).')
tf.app.flags.DEFINE_integer('number_epochs', 1,
                            'The number of epochs to produce.')

FLAGS = tf.app.flags.FLAGS


def shuffle_samples(input_file_1, input_file_2):
  """Shuffle the examples."""
  shuffled = []

  # Set a random seed to keep fixed mask.
  np.random.seed(0)

  for line_1, line_2 in zip(input_file_1, input_file_2):
    rand = np.random.randint(1, 3)
    if rand == 1:
      shuffled.append((rand, line_1, line_2))
    else:
      shuffled.append((rand, line_2, line_1))
  input_file_1.close()
  input_file_2.close()
  return shuffled


def generate_output(shuffled_tuples, output_file_name):
  output_file = tf.gfile.GFile(output_file_name, mode='w')

  for tup in shuffled_tuples:
    formatted_tuple = ('\n{:<1}, {:<1}, {:<1}').format(tup[0], tup[1].rstrip(),
                                                       tup[2].rstrip())
    output_file.write(formatted_tuple)
  output_file.close()


def main(_):
  ml_samples_file = tf.gfile.GFile(
      os.path.join(FLAGS.input_ml_path, 'reviews.txt'), mode='r')
  gan_samples_file = tf.gfile.GFile(
      os.path.join(FLAGS.input_gan_path, 'reviews.txt'), mode='r')

  # Generate shuffled tuples.
  shuffled_tuples = shuffle_samples(ml_samples_file, gan_samples_file)

  # Output to file.
  generate_output(shuffled_tuples, FLAGS.output_file_name)


if __name__ == '__main__':
  tf.app.run()
