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

"""Builder class for preparing tf.train.Example."""

# https://www.python.org/dev/peps/pep-0563/#enabling-the-future-behavior-in-python-3-7
from __future__ import annotations

from typing import Mapping, Sequence, Union

import numpy as np
import tensorflow as tf

BytesValueType = Union[bytes, Sequence[bytes], str, Sequence[str]]

_to_array = lambda v: [v] if not isinstance(v, (list, np.ndarray)) else v
_to_bytes = lambda v: v.encode() if isinstance(v, str) else v
_to_bytes_array = lambda v: list(map(_to_bytes, _to_array(v)))


class TfExampleBuilder(object):
  """Builder class for preparing tf.train.Example.

  Read API doc at https://www.tensorflow.org/api_docs/python/tf/train/Example.

  Example usage:
    >>> example_builder = TfExampleBuilder()
    >>> example = (
            example_builder.add_bytes_feature('feature_a', 'foobarbaz')
            .add_ints_feature('feature_b', [1, 2, 3])
            .example)
  """

  def __init__(self) -> None:
    self._example = tf.train.Example()

  @property
  def example(self) -> tf.train.Example:
    """Returns a copy of the generated tf.train.Example proto."""
    return self._example

  @property
  def serialized_example(self) -> str:
    """Returns a serialized string of the generated tf.train.Example proto."""
    return self._example.SerializeToString()

  def set(self, example: tf.train.Example) -> TfExampleBuilder:
    """Sets the example."""
    self._example = example
    return self

  def reset(self) -> TfExampleBuilder:
    """Resets the example to an empty proto."""
    self._example = tf.train.Example()
    return self

  ###### Basic APIs for primitive data types ######
  def add_feature_dict(
      self, feature_dict: Mapping[str, tf.train.Feature]) -> TfExampleBuilder:
    """Adds the predefined `feature_dict` to the example.

    Note: Please prefer to using feature-type-specific methods.

    Args:
      feature_dict: A dictionary from tf.Example feature key to
        tf.train.Feature.

    Returns:
      The builder object for subsequent method calls.
    """
    for k, v in feature_dict.items():
      self._example.features.feature[k].CopyFrom(v)
    return self

  def add_feature(self, key: str,
                  feature: tf.train.Feature) -> TfExampleBuilder:
    """Adds predefined `feature` with `key` to the example.

    Args:
      key: String key of the feature.
      feature: The feature to be added to the example.

    Returns:
      The builder object for subsequent method calls.
    """
    self._example.features.feature[key].CopyFrom(feature)
    return self

  def add_bytes_feature(self, key: str,
                        value: BytesValueType) -> TfExampleBuilder:
    """Adds byte(s) or string(s) with `key` to the example.

    Args:
      key: String key of the feature.
      value: The byte(s) or string(s) to be added to the example.

    Returns:
      The builder object for subsequent method calls.
    """
    return self.add_feature(
        key,
        tf.train.Feature(
            bytes_list=tf.train.BytesList(value=_to_bytes_array(value))))

  def add_ints_feature(self, key: str,
                       value: Union[int, Sequence[int]]) -> TfExampleBuilder:
    """Adds integer(s) with `key` to the example.

    Args:
      key: String key of the feature.
      value: The integer(s) to be added to the example.

    Returns:
      The builder object for subsequent method calls.
    """
    return self.add_feature(
        key,
        tf.train.Feature(int64_list=tf.train.Int64List(value=_to_array(value))))

  def add_floats_feature(
      self, key: str, value: Union[float, Sequence[float]]) -> TfExampleBuilder:
    """Adds float(s) with `key` to the example.

    Args:
      key: String key of the feature.
      value: The float(s) to be added to the example.

    Returns:
      The builder object for subsequent method calls.
    """
    return self.add_feature(
        key,
        tf.train.Feature(float_list=tf.train.FloatList(value=_to_array(value))))
