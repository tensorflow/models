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

"""Contains common flags and functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import locale
import os
from absl import logging
import numpy as np
import tensorflow as tf


def get_seq_middle(seq_length):
  """Returns relative index for the middle frame in sequence."""
  half_offset = int((seq_length - 1) / 2)
  return seq_length - 1 - half_offset


def info(obj):
  """Return info on shape and dtype of a numpy array or TensorFlow tensor."""
  if obj is None:
    return 'None.'
  elif isinstance(obj, list):
    if obj:
      return 'List of %d... %s' % (len(obj), info(obj[0]))
    else:
      return 'Empty list.'
  elif isinstance(obj, tuple):
    if obj:
      return 'Tuple of %d... %s' % (len(obj), info(obj[0]))
    else:
      return 'Empty tuple.'
  else:
    if is_a_numpy_array(obj):
      return 'Array with shape: %s, dtype: %s' % (obj.shape, obj.dtype)
    else:
      return str(obj)


def is_a_numpy_array(obj):
  """Returns true if obj is a numpy array."""
  return type(obj).__module__ == np.__name__


def count_parameters(also_print=True):
  """Cound the number of parameters in the model.

  Args:
    also_print: Boolean.  If True also print the numbers.

  Returns:
    The total number of parameters.
  """
  total = 0
  if also_print:
    logging.info('Model Parameters:')
  for v in get_vars_to_restore():
    shape = v.get_shape()
    if also_print:
      logging.info('%s %s: %s', v.op.name, shape,
                   format_number(shape.num_elements()))
    total += shape.num_elements()
  if also_print:
    logging.info('Total: %s', format_number(total))
  return total


def get_vars_to_restore(ckpt=None):
  """Returns list of variables that should be saved/restored.

  Args:
    ckpt: Path to existing checkpoint.  If present, returns only the subset of
        variables that exist in given checkpoint.

  Returns:
    List of all variables that need to be saved/restored.
  """
  model_vars = tf.trainable_variables()
  # Add batchnorm variables.
  bn_vars = [v for v in tf.global_variables()
             if 'moving_mean' in v.op.name or 'moving_variance' in v.op.name]
  model_vars.extend(bn_vars)
  model_vars = sorted(model_vars, key=lambda x: x.op.name)
  if ckpt is not None:
    ckpt_var_names = tf.contrib.framework.list_variables(ckpt)
    ckpt_var_names = [name for (name, unused_shape) in ckpt_var_names]
    for v in model_vars:
      if v.op.name not in ckpt_var_names:
        logging.warn('Missing var %s in checkpoint: %s', v.op.name,
                     os.path.basename(ckpt))
    model_vars = [v for v in model_vars if v.op.name in ckpt_var_names]
  return model_vars


def format_number(n):
  """Formats number with thousands commas."""
  locale.setlocale(locale.LC_ALL, 'en_US')
  return locale.format('%d', n, grouping=True)


def read_text_lines(filepath):
  with open(filepath, 'r') as f:
    lines = f.readlines()
  lines = [l.rstrip() for l in lines]
  return lines
