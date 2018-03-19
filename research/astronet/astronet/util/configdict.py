# Copyright 2018 The TensorFlow Authors.
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

"""Configuration container for TensorFlow models.

A ConfigDict is simply a dict whose values can be accessed via both dot syntax
(config.key) and dict syntax (config['key']).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _maybe_convert_dict(value):
  if isinstance(value, dict):
    return ConfigDict(value)

  return value


class ConfigDict(dict):
  """Configuration container class."""

  def __init__(self, initial_dictionary=None):
    """Creates an instance of ConfigDict.

    Args:
      initial_dictionary: Optional dictionary or ConfigDict containing initial
        parameters.
    """
    if initial_dictionary:
      for field, value in initial_dictionary.items():
        initial_dictionary[field] = _maybe_convert_dict(value)
    super(ConfigDict, self).__init__(initial_dictionary)

  def __setattr__(self, attribute, value):
    self[attribute] = _maybe_convert_dict(value)

  def __getattr__(self, attribute):
    try:
      return self[attribute]
    except KeyError as e:
      raise AttributeError(e)

  def __delattr__(self, attribute):
    try:
      del self[attribute]
    except KeyError as e:
      raise AttributeError(e)

  def __setitem__(self, key, value):
    super(ConfigDict, self).__setitem__(key, _maybe_convert_dict(value))
