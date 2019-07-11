# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Estimator's RunConfig for the object detection model.

This file provides functions for creating a RunConfig for the Estimator that is
used for model training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def create_runconfig(model_dir, runconfig_overrides=None):
  """Returns a RunConfig with default overwritten according to given
  override string.

  Args:
    runconfig_overrides: Optional overrides, represented as a
      string containing comma-separated param_name=value pairs.
  Returns:
    The run config as tf.estimator.RunConfig object.
  """
  def _type_map_from_values(config):
    """Returns a type-map as expected by tf.contrib.training.parse_values
    according to the given sample object.

    Args:
      config: a sample tf.estimator.RunConfig object used to determine
        valid attributes and their types.
    Returns:
      A dict mapping from attribute name to the attribute's type. Attribute
      names and types are computed from vars(config) but only for attributes
      of primitive types (int, float, bool, str). A leading _ is stripped
      from attribute names. For some None-valued attributes default types are
      provided.
    """
    # Note that these default types might need to be adapted if
    # RunConfig's attributes that are None by default changes.
    type_map = {
      'save_checkpoints_steps': int,
      'tf_random_seed': int,
      'protocol': str,
    }
    valid_types = {int, float, str, bool}
    type_map.update({k.lstrip('_'): type(v)
                     for k, v in vars(config).items()
                     if type(v) in valid_types})
    return type_map

  config = tf.estimator.RunConfig(model_dir=model_dir)
  if runconfig_overrides:
    type_map = _type_map_from_values(config)
    overrides = tf.contrib.training.parse_values(runconfig_overrides, type_map)
    config = tf.estimator.RunConfig(model_dir=model_dir, **overrides)
  return config
