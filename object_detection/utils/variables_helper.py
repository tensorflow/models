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

"""Helper functions for manipulating collections of variables during training.
"""
import logging
import re

import tensorflow as tf

slim = tf.contrib.slim


# TODO: Consider replacing with tf.contrib.filter_variables in
# tensorflow/contrib/framework/python/ops/variables.py
def filter_variables(variables, filter_regex_list, invert=False):
  """Filters out the variables matching the filter_regex.

  Filter out the variables whose name matches the any of the regular
  expressions in filter_regex_list and returns the remaining variables.
  Optionally, if invert=True, the complement set is returned.

  Args:
    variables: a list of tensorflow variables.
    filter_regex_list: a list of string regular expressions.
    invert: (boolean).  If True, returns the complement of the filter set; that
      is, all variables matching filter_regex are kept and all others discarded.

  Returns:
    a list of filtered variables.
  """
  kept_vars = []
  variables_to_ignore_patterns = filter(None, filter_regex_list)
  for var in variables:
    add = True
    for pattern in variables_to_ignore_patterns:
      if re.match(pattern, var.op.name):
        add = False
        break
    if add != invert:
      kept_vars.append(var)
  return kept_vars


def multiply_gradients_matching_regex(grads_and_vars, regex_list, multiplier):
  """Multiply gradients whose variable names match a regular expression.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
    regex_list: A list of string regular expressions.
    multiplier: A (float) multiplier to apply to each gradient matching the
      regular expression.

  Returns:
    grads_and_vars: A list of gradient to variable pairs (tuples).
  """
  variables = [pair[1] for pair in grads_and_vars]
  matching_vars = filter_variables(variables, regex_list, invert=True)
  for var in matching_vars:
    logging.info('Applying multiplier %f to variable [%s]',
                 multiplier, var.op.name)
  grad_multipliers = {var: float(multiplier) for var in matching_vars}
  return slim.learning.multiply_gradients(grads_and_vars,
                                          grad_multipliers)


def freeze_gradients_matching_regex(grads_and_vars, regex_list):
  """Freeze gradients whose variable names match a regular expression.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
    regex_list: A list of string regular expressions.

  Returns:
    grads_and_vars: A list of gradient to variable pairs (tuples) that do not
      contain the variables and gradients matching the regex.
  """
  variables = [pair[1] for pair in grads_and_vars]
  matching_vars = filter_variables(variables, regex_list, invert=True)
  kept_grads_and_vars = [pair for pair in grads_and_vars
                         if pair[1] not in matching_vars]
  for var in matching_vars:
    logging.info('Freezing variable [%s]', var.op.name)
  return kept_grads_and_vars


def get_variables_available_in_checkpoint(variables, checkpoint_path):
  """Returns the subset of variables available in the checkpoint.

  Inspects given checkpoint and returns the subset of variables that are
  available in it.

  TODO: force input and output to be a dictionary.

  Args:
    variables: a list or dictionary of variables to find in checkpoint.
    checkpoint_path: path to the checkpoint to restore variables from.

  Returns:
    A list or dictionary of variables.
  Raises:
    ValueError: if `variables` is not a list or dict.
  """
  if isinstance(variables, list):
    variable_names_map = {variable.op.name: variable for variable in variables}
  elif isinstance(variables, dict):
    variable_names_map = variables
  else:
    raise ValueError('`variables` is expected to be a list or dict.')
  ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
  ckpt_vars = ckpt_reader.get_variable_to_shape_map().keys()
  vars_in_ckpt = {}
  for variable_name, variable in sorted(variable_names_map.items()):
    if variable_name in ckpt_vars:
      vars_in_ckpt[variable_name] = variable
    else:
      logging.warning('Variable [%s] not available in checkpoint',
                      variable_name)
  if isinstance(variables, list):
    return vars_in_ckpt.values()
  return vars_in_ckpt
