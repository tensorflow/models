# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Util functions for loading checkpoints.

Especially for loading Tensorflow 1.x
checkpoint to Tensorflow 2.x (keras) model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from absl import logging

import tensorflow as tf


def _build_assignment_map(keras_model,
                          prefix='',
                          skip_variables_regex=None,
                          var_to_shape_map=None):
  """Builds the variable assignment map.

  Compute an assignment mapping for loading older checkpoints into a Keras
  model. Variable names are remapped from the original TPUEstimator model to
  the new Keras name.

  Args:
    keras_model: tf.keras.Model object to provide variables to assign.
    prefix: prefix in the variable name to be remove for alignment with names in
      the checkpoint.
    skip_variables_regex: regular expression to math the names of variables that
      do not need to be assign.
    var_to_shape_map: variable name to shape mapping from the checkpoint.

  Returns:
    The variable assignment map.
  """
  assignment_map = {}

  checkpoint_names = []
  if var_to_shape_map:
    # pylint: disable=g-long-lambda
    checkpoint_names = list(
        filter(
            lambda x: not x.endswith('Momentum') and not x.endswith(
                'global_step'), var_to_shape_map.keys()))
    # pylint: enable=g-long-lambda

  logging.info('Number of variables in the checkpoint %d',
               len(checkpoint_names))

  for var in keras_model.variables:
    var_name = var.name

    if skip_variables_regex and re.match(skip_variables_regex, var_name):
      continue
    # Trim the index of the variable.
    if ':' in var_name:
      var_name = var_name[:var_name.rindex(':')]
    if var_name.startswith(prefix):
      var_name = var_name[len(prefix):]

    if not var_to_shape_map:
      assignment_map[var_name] = var
      continue

    # Match name with variables in the checkpoint.
    # pylint: disable=cell-var-from-loop
    match_names = list(filter(lambda x: x.endswith(var_name), checkpoint_names))
    # pylint: enable=cell-var-from-loop
    try:
      if match_names:
        assert len(match_names) == 1, 'more then on matches for {}: {}'.format(
            var_name, match_names)
        checkpoint_names.remove(match_names[0])
        assignment_map[match_names[0]] = var
      else:
        logging.info('Error not found var name: %s', var_name)
    except Exception as e:
      logging.info('Error removing the match_name: %s', match_names)
      logging.info('Exception: %s', e)
      raise
  logging.info('Found matching variable in checkpoint: %d', len(assignment_map))
  return assignment_map


def _get_checkpoint_map(checkpoint_path):
  reader = tf.train.load_checkpoint(checkpoint_path)
  return reader.get_variable_to_shape_map()


def make_restore_checkpoint_fn(checkpoint_path, prefix='', skip_regex=None):
  """Returns scaffold function to restore parameters from v1 checkpoint.

  Args:
    checkpoint_path: path of the checkpoint folder or file.
      Example 1: '/path/to/model_dir/'
      Example 2: '/path/to/model.ckpt-22500'
    prefix: prefix in the variable name to be remove for alignment with names in
      the checkpoint.
    skip_regex: regular expression to math the names of variables that do not
      need to be assign.

  Returns:
    Callable[tf.kears.Model] -> void. Fn to load v1 checkpoint to keras model.
  """

  def _restore_checkpoint_fn(keras_model):
    """Loads pretrained model through scaffold function."""
    if not checkpoint_path:
      logging.info('checkpoint_path is empty')
      return
    var_prefix = prefix
    if prefix and not prefix.endswith('/'):
      var_prefix += '/'
    var_to_shape_map = _get_checkpoint_map(checkpoint_path)
    assert var_to_shape_map, 'var_to_shape_map should not be empty'
    vars_to_load = _build_assignment_map(
        keras_model,
        prefix=var_prefix,
        skip_variables_regex=skip_regex,
        var_to_shape_map=var_to_shape_map)
    if not vars_to_load:
      raise ValueError('Variables to load is empty.')
    tf.compat.v1.train.init_from_checkpoint(checkpoint_path, vars_to_load)

  return _restore_checkpoint_fn
