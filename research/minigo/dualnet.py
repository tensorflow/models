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
"""Contains utility and supporting functions for DualNet.

This module provides the model interface, including functions for DualNet model
bootstrap, training, validation, loading and exporting.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf  # pylint: disable=g-bad-import-order

import dualnet_model
import features
import preprocessing
import symmetries


class DualNetRunner(object):
  """The DualNetRunner class for the complete model with graph and weights.

  This class can restore the model from saved files, and provide inference for
  given examples.
  """

  def __init__(self, save_file, params):
    """Initialize the dual network from saved model/checkpoints.

    Args:
      save_file: Path where model parameters were previously saved. For example:
        '/tmp/minigo/models_dir/000000-bootstrap/'
      params: An object with hyperparameters for DualNetRunner
    """
    self.save_file = save_file
    self.hparams = params
    self.inference_input = None
    self.inference_output = None
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(graph=tf.Graph(), config=config)
    self.initialize_graph()

  def initialize_graph(self):
    """Initialize the graph with saved model."""
    with self.sess.graph.as_default():
      input_features, labels = get_inference_input(self.hparams)
      estimator_spec = dualnet_model.model_fn(
          input_features, labels, tf.estimator.ModeKeys.PREDICT, self.hparams)
      self.inference_input = input_features
      self.inference_output = estimator_spec.predictions
      if self.save_file is not None:
        self.initialize_weights(self.save_file)
      else:
        self.sess.run(tf.global_variables_initializer())

  def initialize_weights(self, save_file):
    """Initialize the weights from the given save_file.

    Assumes that the graph has been constructed, and the save_file contains
    weights that match the graph. Used to set the weights to a different version
    of the player without redefining the entire graph.

    Args:
      save_file: Path where model parameters were previously saved.
    """

    tf.train.Saver().restore(self.sess, save_file)

  def run(self, position, use_random_symmetry=True):
    """Compute the policy and value output for a given position.

    Args:
      position: A given go board status
      use_random_symmetry: Apply random symmetry (defined in symmetries.py) to
        the extracted feature (defined in features.py) of the given position

    Returns:
      prob, value: The policy and value output (defined in dualnet_model.py)
    """
    probs, values = self.run_many(
        [position], use_random_symmetry=use_random_symmetry)
    return probs[0], values[0]

  def run_many(self, positions, use_random_symmetry=True):
    """Compute the policy and value output for given positions.

    Args:
      positions: A list of positions for go board status
      use_random_symmetry: Apply random symmetry (defined in symmetries.py) to
        the extracted features (defined in features.py) of the given positions

    Returns:
      probabilities, value: The policy and value outputs (defined in
        dualnet_model.py)
    """
    def _extract_features(positions):
      return features.extract_features(self.hparams.board_size, positions)
    processed = list(map(_extract_features, positions))
    # processed = [
    #  features.extract_features(self.hparams.board_size, p) for p in positions]
    if use_random_symmetry:
      syms_used, processed = symmetries.randomize_symmetries_feat(processed)
    # feed_dict is a dict object to provide the input examples for the step of
    # inference. sess.run() returns the inference predictions (indicated by
    # self.inference_output) of the given input as outputs
    outputs = self.sess.run(
        self.inference_output, feed_dict={self.inference_input: processed})
    probabilities, value = outputs['policy_output'], outputs['value_output']
    if use_random_symmetry:
      probabilities = symmetries.invert_symmetries_pi(
          self.hparams.board_size, syms_used, probabilities)
    return probabilities, value


def get_inference_input(params):
  """Set up placeholders for input features/labels.

  Args:
    params: An object to indicate the hyperparameters of the model.

  Returns:
    The features and output tensors that get passed into model_fn. Check
      dualnet_model.py for more details on the models input and output.
  """
  input_features = tf.placeholder(
      tf.float32, [None, params.board_size, params.board_size,
                   features.NEW_FEATURES_PLANES],
      name='pos_tensor')

  labels = {
      'pi_tensor': tf.placeholder(
          tf.float32, [None, params.board_size * params.board_size + 1]),
      'value_tensor': tf.placeholder(tf.float32, [None])
  }

  return input_features, labels


def bootstrap(working_dir, params):
  """Initialize a tf.Estimator run with random initial weights.

  Args:
    working_dir: The directory where tf.estimator will drop logs,
      checkpoints, and so on
    params: hyperparams of the model.
  """
  # Forge an initial checkpoint with the name that subsequent Estimator will
  # expect to find.
  estimator_initial_checkpoint_name = 'model.ckpt-1'
  save_file = os.path.join(working_dir,
                           estimator_initial_checkpoint_name)
  sess = tf.Session()
  with sess.graph.as_default():
    input_features, labels = get_inference_input(params)
    dualnet_model.model_fn(
        input_features, labels, tf.estimator.ModeKeys.PREDICT, params)
    sess.run(tf.global_variables_initializer())
    tf.train.Saver().save(sess, save_file)


def export_model(working_dir, model_path):
  """Take the latest checkpoint and export it to model_path for selfplay.

  Assumes that all relevant model files are prefixed by the same name.
  (For example, foo.index, foo.meta and foo.data-00000-of-00001).

  Args:
    working_dir: The directory where tf.estimator keeps its checkpoints.
    model_path: Either a local path or a gs:// path to export model to.
  """
  latest_checkpoint = tf.train.latest_checkpoint(working_dir)
  all_checkpoint_files = tf.gfile.Glob(latest_checkpoint + '*')
  for filename in all_checkpoint_files:
    suffix = filename.partition(latest_checkpoint)[2]
    destination_path = model_path + suffix
    tf.gfile.Copy(filename, destination_path)


def train(working_dir, tf_records, generation, params):
  """Train the model for a specific generation.

  Args:
    working_dir: The model working directory to save model parameters,
      drop logs, checkpoints, and so on.
    tf_records: A list of tf_record filenames for training input.
    generation: The generation to be trained.
    params: hyperparams of the model.

  Raises:
    ValueError: if generation is not greater than 0.
  """
  if generation <= 0:
    raise ValueError('Model 0 is random weights')
  estimator = tf.estimator.Estimator(
      dualnet_model.model_fn, model_dir=working_dir, params=params)
  max_steps = (generation * params.examples_per_generation
               // params.batch_size)
  profiler_hook = tf.train.ProfilerHook(output_dir=working_dir, save_secs=600)

  def input_fn():
    return preprocessing.get_input_tensors(
        params, params.batch_size, tf_records)
  estimator.train(
      input_fn, hooks=[profiler_hook], max_steps=max_steps)


def validate(working_dir, tf_records, params):
  """Perform model validation on the hold out data.

  Args:
    working_dir: The model working directory.
    tf_records: A list of tf_records filenames for holdout data.
    params: hyperparams of the model.
  """
  estimator = tf.estimator.Estimator(
      dualnet_model.model_fn, model_dir=working_dir, params=params)
  def input_fn():
    return preprocessing.get_input_tensors(
        params, params.batch_size, tf_records, filter_amount=0.05)
  estimator.evaluate(input_fn, steps=1000)
