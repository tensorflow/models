# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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

# pylint: disable=g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import dualnet_model
import features
import go
import model_utils
import preprocessing
import symmetries


class DualNetwork(object):
  """The DualNetwork class for the complete model with graph and weights.

  This class can restore the model from saved files, and provide inference for
  given examples.
  """

  def __init__(self, save_file, params):
    """Initialize the dual network from saved model/checkpoints.

    Args:
      save_file: Path where model parameters were previously saved.
      params: Hyper parameters for the dual network
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
    with self.sess.graph.as_default():
      input_features, labels = get_inference_input()
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
    """Compute the policy and value output for a given position."""
    probs, values = self.run_many(
        [position], use_random_symmetry=use_random_symmetry)
    return probs[0], values[0]

  def run_many(self, positions, use_random_symmetry=True):
    processed = list(map(features.extract_features, positions))
    if use_random_symmetry:
      syms_used, processed = symmetries.randomize_symmetries_feat(processed)
    outputs = self.sess.run(
        self.inference_output, feed_dict={self.inference_input: processed})
    probabilities, value = outputs['policy_output'], outputs['value_output']
    if use_random_symmetry:
      probabilities = symmetries.invert_symmetries_pi(syms_used, probabilities)
    return probabilities, value


def get_inference_input():
  """Set up placeholders for input features/labels.

  Returns:
    The features and output tensors that get passed into model_fn.
  """
  input_features = tf.placeholder(
      tf.float32, [None, go.N, go.N, features.NEW_FEATURES_PLANES],
      name='pos_tensor')

  labels = {
      'pi_tensor': tf.placeholder(tf.float32, [None, go.N * go.N + 1]),
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
  sess = tf.Session(graph=tf.Graph())
  with sess.graph.as_default():
    input_features, labels = get_inference_input()
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
  estimator = tf.estimator.Estimator(
      dualnet_model.model_fn, model_dir=working_dir, params='ignored')
  latest_checkpoint = estimator.latest_checkpoint()
  all_checkpoint_files = tf.gfile.Glob(latest_checkpoint + '*')
  for filename in all_checkpoint_files:
    suffix = filename.partition(latest_checkpoint)[2]
    destination_path = model_path + suffix
    tf.gfile.Copy(filename, destination_path)


def train(working_dir, tf_records, generation_num, params):
  """Train the model for a specific generation.

  Args:
    working_dir: The model working directory to save model parameters,
      drop logs, checkpoints, and so on.
    tf_records: A list of tf_record filenames for training input.
    generation_num: The generation to be trained.
    params: hyperparams of the model.
  """
  assert generation_num > 0, 'Model 0 is random weights'
  estimator = tf.estimator.Estimator(
      dualnet_model.model_fn, model_dir=working_dir, params=params)
  max_steps = (generation_num * params.examples_per_generation
               // params.batch_size)
  update_ratio_hook = model_utils.UpdateRatioSessionHook(working_dir)

  def input_fn():
    return preprocessing.get_input_tensors(params.batch_size, tf_records)
  estimator.train(
      input_fn, hooks=[update_ratio_hook, tf.train.ProfilerHook(save_secs=600)],
      max_steps=max_steps)


def validate(working_dir, tf_records, checkpoint_name, params):
  """Perform model validation on the hold out data.

  Args:
    working_dir: The model working directory.
    tf_records: A list of tf_records filenames for holdout data.
    checkpoint_name: The checkpoint used to for evaluation.
    params: hyperparams of the model.
  """
  estimator = tf.estimator.Estimator(
      dualnet_model.model_fn, model_dir=working_dir, params=params)
  if checkpoint_name is None:
    checkpoint_name = estimator.latest_checkpoint()

  def input_fn():
    return preprocessing.get_input_tensors(
        params.batch_size, tf_records, shuffle_buffer_size=1000,
        filter_amount=0.05)
  estimator.evaluate(input_fn, steps=1000)


