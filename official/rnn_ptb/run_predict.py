# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Use a trained model to predict a sequence of words."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

import model_params
import run_training
import util

# Create argument parser with arguments imported from run_training.py
parser = argparse.ArgumentParser(parents=[run_training.parser])
parser.add_argument(
    '--input',
    type=str,
    default='the meaning of life is',
    help='The input sequence')
parser.add_argument(
    '--num_predictions',
    type=int,
    default=100,
    help='Number of words to predict')
FLAGS, unparsed = parser.parse_known_args()


def single_prediction_input_fn(input_sequence, vocab_dict):
  # Convert input sequence to a list of IDs.
  inputs = [vocab_dict[word.lower()] for word in input_sequence.strip().split()]

  # Create a dataset consisting of a single element
  dataset = tf.data.Dataset.from_tensors({'inputs': [inputs],
                                          'reset_state': True})
  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()
  return features


def main(unused_argv):
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide warning messages from c++
  tf.logging.set_verbosity(tf.logging.INFO)  # Show INFO logs

  # Set up prediction parameters
  params = model_params.get_parameters(FLAGS.model)
  params.num_predictions = FLAGS.num_predictions

  vocab_dict = util.build_vocab_id_dict(run_training.TRAIN_FILE)
  reverse_dict = util.build_reverse_vocab_dict(vocab_dict)

  # Set up estimator and predict the words after the input sequence.
  estimator = tf.estimator.Estimator(model_fn=run_training.model_fn,
                                     model_dir=FLAGS.model_dir, params=params)
  predictions = estimator.predict(
      lambda: single_prediction_input_fn(FLAGS.input, vocab_dict))
  predicted_words = next(predictions)

  s = ' '.join([FLAGS.input] + [reverse_dict[i] for i in predicted_words])
  print('Predicted sequence')
  print('=' * 30)
  print(s)


if __name__ == '__main__':
  tf.app.run()
