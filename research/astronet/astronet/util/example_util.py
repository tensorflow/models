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

"""Helpers for getting and setting values in tf.Example protocol buffers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def get_feature(ex, name, kind=None, strict=True):
  """Gets a feature value from a tf.train.Example.

  Args:
    ex: A tf.train.Example.
    name: Name of the feature to look up.
    kind: Optional: one of 'bytes_list', 'float_list', 'int64_list'. Inferred if
        not specified.
    strict: Whether to raise a KeyError if there is no such feature.

  Returns:
    A numpy array containing to the values of the specified feature.

  Raises:
    KeyError: If there is no feature with the specified name.
    TypeError: If the feature has a different type to that specified.
  """
  if name not in ex.features.feature:
    if strict:
      raise KeyError(name)
    return np.array([])

  inferred_kind = ex.features.feature[name].WhichOneof("kind")
  if not inferred_kind:
    return np.array([])  # Feature exists, but it's empty.

  if kind and kind != inferred_kind:
    raise TypeError("Requested %s, but Feature has %s" % (kind, inferred_kind))

  return np.array(getattr(ex.features.feature[name], inferred_kind).value)


def get_bytes_feature(ex, name, strict=True):
  """Gets the value of a bytes feature from a tf.train.Example."""
  return get_feature(ex, name, "bytes_list", strict)


def get_float_feature(ex, name, strict=True):
  """Gets the value of a float feature from a tf.train.Example."""
  return get_feature(ex, name, "float_list", strict)


def get_int64_feature(ex, name, strict=True):
  """Gets the value of an int64 feature from a tf.train.Example."""
  return get_feature(ex, name, "int64_list", strict)


def _infer_kind(value):
  """Infers the tf.train.Feature kind from a value."""
  if np.issubdtype(type(value[0]), np.integer):
    return "int64_list"
  try:
    float(value[0])
    return "float_list"
  except ValueError:
    return "bytes_list"


def set_feature(ex, name, value, kind=None, allow_overwrite=False):
  """Sets a feature value in a tf.train.Example.

  Args:
    ex: A tf.train.Example.
    name: Name of the feature to set.
    value: Feature value to set. Must be a sequence.
    kind: Optional: one of 'bytes_list', 'float_list', 'int64_list'. Inferred if
        not specified.
    allow_overwrite: Whether to overwrite the existing value of the feature.

  Raises:
    ValueError: If `allow_overwrite` is False and the feature already exists, or
        if `kind` is unrecognized.
  """
  if name in ex.features.feature:
    if allow_overwrite:
      del ex.features.feature[name]
    else:
      raise ValueError(
          "Attempting to set duplicate feature with name: %s" % name)

  if not kind:
    kind = _infer_kind(value)

  if kind == "bytes_list":
    value = [str(v).encode("latin-1") for v in value]
  elif kind == "float_list":
    value = [float(v) for v in value]
  elif kind == "int64_list":
    value = [int(v) for v in value]
  else:
    raise ValueError("Unrecognized kind: %s" % kind)

  getattr(ex.features.feature[name], kind).value.extend(value)


def set_float_feature(ex, name, value, allow_overwrite=False):
  """Sets the value of a float feature in a tf.train.Example."""
  set_feature(ex, name, value, "float_list", allow_overwrite)


def set_bytes_feature(ex, name, value, allow_overwrite=False):
  """Sets the value of a bytes feature in a tf.train.Example."""
  set_feature(ex, name, value, "bytes_list", allow_overwrite)


def set_int64_feature(ex, name, value, allow_overwrite=False):
  """Sets the value of an int64 feature in a tf.train.Example."""
  set_feature(ex, name, value, "int64_list", allow_overwrite)
