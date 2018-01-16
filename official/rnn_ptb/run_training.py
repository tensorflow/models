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
"""Train a RNN model on the PTB dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil

import numpy as np
import tensorflow as tf

import model
import model_params
import util

# Create argument parser. Add the help option only when running this file.
parser = argparse.ArgumentParser(add_help=(__name__ == '__main__'))
parser.add_argument(
    '--data_dir',
    type=str,
    default='',
    help='The directory where train, validation, and test data are saved.')
parser.add_argument(
    '--model',
    type=str,
    default='medium',
    help='The model configuration. Options: small, medium(default), large')
parser.add_argument(
    '--model_dir',
    type=str,
    default='/tmp/rnn_ptb_model',
    help='The directory where the model will be saved.')
parser.add_argument(
    '--reset_training',
    action='store_true',
    help='Add this option to clear the model directory prior to training.')
FLAGS, unparsed = parser.parse_known_args()

# Set the train, validation, and test file paths as constants.
TRAIN_FILE = os.path.join(FLAGS.data_dir, 'ptb.train.txt')
VALID_FILE = os.path.join(FLAGS.data_dir, 'ptb.valid.txt')
TEST_FILE = os.path.join(FLAGS.data_dir, 'ptb.test.txt')

# Hard coded number of words in the training file (used to calculate the maximum
# number steps to train the model).
TRAIN_FILE_WORDS = 929589


def input_fn(input_file, vocab_dict, num_epochs=1):
  """Generates batches of data from the input file."""
  def batch_data(data, batch_size, steps_per_epoch, unrolled_count):
    """Batch data so that adjacent batches sequentially traverse the data.

    See the README for details."""
    data = data.reshape([batch_size, steps_per_epoch, unrolled_count])
    return np.swapaxes(data, 0, 1)

  def input_generator(batch_size, unrolled_count):
    """Yield batches of input data."""
    # Read all words from the input file, and append <eos> at every new line.
    with tf.gfile.GFile(input_file, 'r') as f:
      data = f.read().replace(util.NEWLINE, util.EOS).split()

    # Convert each word to the corresponding integer id.
    data = np.array([vocab_dict[word] for word in data])

    # Calculate size that is evenly divisible by batch size and unrolled count
    total_words = len(data)
    steps_per_epoch = util.steps_per_epoch(
        total_words - 1, batch_size, unrolled_count)
    data_size = steps_per_epoch * unrolled_count * batch_size

    # Truncate and batch data
    inputs = data[:data_size]
    inputs = batch_data(inputs, batch_size, steps_per_epoch, unrolled_count)

    labels = data[1:data_size + 1]
    labels = batch_data(labels, batch_size, steps_per_epoch, unrolled_count)

    # Yield a single batch of data at a time
    for i in range(steps_per_epoch):
      reset_state = (i == 0)  # Signal model to reset the state at every epoch
      yield {'inputs': inputs[i], 'reset_state': reset_state}, labels[i]

  def _input_fn(params):
    batch_size, unrolled_count = params.batch_size, params.unrolled_count

    dataset = tf.data.Dataset.from_generator(
        lambda: input_generator(batch_size, unrolled_count),
        ({'inputs': tf.int32, 'reset_state': tf.bool}, tf.int32),
        ({'inputs': tf.TensorShape([batch_size, unrolled_count]),
          'reset_state': tf.TensorShape([])},
         tf.TensorShape([batch_size, unrolled_count])))

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(15)

    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels
  return _input_fn


def model_fn(features, labels, mode, params):
  # Initialize and call PTBModel to obtain the logits.
  m = model.PTBModel.from_params(mode, params)
  logits, state = m(features['inputs'], reset_state=features['reset_state'])

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Get predictions using the output logits and state.
    predictions = m.predict_next(logits[-1], state, params.num_predictions)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  loss = get_loss(labels, logits)

  # Calculate the learning rate and train op (only used when in training mode)
  steps_per_epoch = util.steps_per_epoch(
      TRAIN_FILE_WORDS - 1, params.batch_size, params.unrolled_count)
  learning_rate = get_learning_rate(
      params.initial_learning_rate, params.learning_rate_decay,
      params.epochs_before_decay, steps_per_epoch)
  train_op = get_train_op(loss, learning_rate,
                          params.max_gradient_norm)

  # Calculate the perplexity
  average_loss = loss / params.unrolled_count
  perplexity = tf.exp(average_loss)

  # Save values as tensors, which will be logged by the LoggingTensorHook.
  tf.identity(perplexity, name='train_perplexity')
  tf.identity(learning_rate, name='learning_rate')

  # Save perplexity to summary, which is accessed by Tensorboard.
  tf.summary.scalar('train_perplexity', perplexity)

  metrics = {'perplexity': util.perplexity_metric(average_loss)}

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=average_loss, train_op=train_op, eval_metric_ops=metrics)


def get_loss(labels, logits):
  """Calculate the loss from logits.

  Args:
    labels: The expected labels from the training data.
    logits: The predicted logits outputted from the PTB model.

  Returns:
    The sum of the average loss at each unrolled time step.
  """
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                        logits=logits)

  loss = tf.reduce_mean(loss, 0)  # Calculate average loss at each time step
  return tf.reduce_sum(loss)


def get_learning_rate(initial_lr, lr_decay, epochs_before_decay,
    steps_per_epoch):
  """Generate learning rate that decays after each epoch after a min epoch."""
  steps_before_decay = (epochs_before_decay - 1) * steps_per_epoch
  pseudo_global_step = tf.maximum(
      tf.train.get_or_create_global_step() - steps_before_decay, 0)
  return tf.train.exponential_decay(
      learning_rate=initial_lr,
      global_step=pseudo_global_step,
      decay_steps=steps_per_epoch,
      decay_rate=lr_decay,
      staircase=True)


def get_train_op(loss, learning_rate, max_grad):
  """Returns the op that trains the model."""
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad)
  return optimizer.apply_gradients(zip(grads, tvars),
                                   tf.train.get_or_create_global_step())


def main(unused_argv):
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide warning messages from c++
  tf.logging.set_verbosity(tf.logging.INFO)  # Show info logs

  if FLAGS.reset_training:
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

  # Load word->id dictionary
  vocab_dict = util.build_vocab_id_dict(TRAIN_FILE)

  # Set up estimator
  params = model_params.get_parameters(FLAGS.model)
  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=FLAGS.model_dir, params=params)

  # Set up logging hook that logs the perplexity
  logging_hook = tf.train.LoggingTensorHook(
      tensors={'train_perplexity': 'train_perplexity',
               'learning_rate': 'learning_rate'}, every_n_iter=100)

  # Set up training and evaluation specs.
  max_training_steps = params.max_epochs * util.steps_per_epoch(
      TRAIN_FILE_WORDS - 1, params.batch_size, params.unrolled_count)
  train_spec = tf.estimator.TrainSpec(
      input_fn=input_fn(TRAIN_FILE, vocab_dict, params.epochs_per_eval),
      max_steps=max_training_steps, hooks=[logging_hook])
  valid_spec = tf.estimator.EvalSpec(
      input_fn=input_fn(VALID_FILE, vocab_dict), steps=None)

  # Train and evaluate the estimator until max_training_steps has been reached.
  tf.estimator.train_and_evaluate(estimator, train_spec, valid_spec)

  # Obtain final metrics from evaluating the model on the test dataset.
  print("\nEvaluating model on test file:")
  results = estimator.evaluate(input_fn=input_fn(TEST_FILE, vocab_dict))
  print(results)


if __name__ == '__main__':
  tf.app.run()
