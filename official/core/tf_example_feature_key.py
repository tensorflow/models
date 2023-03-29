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

"""Data classes for tf.Example proto feature keys.

Feature keys are grouped by feature types. Key names follow conventions in
go/tf-example.
"""
import dataclasses
import functools
from typing import Optional

# Disable init function to use the one defined in base class.
dataclass = functools.partial(dataclasses.dataclass(init=False))


@dataclass
class TfExampleFeatureKeyBase:
  """Base dataclass for defining tf.Example proto feature keys.

  This class defines the logic of adding prefix to feature keys. Subclasses
  will define feature keys for a specific feature type in data fields.

  NOTE: Please follow subclass examples in this module to define feature keys
  for a new feature type.
  """

  def __init__(self, prefix: Optional[str] = None):
    """Instantiates the feature key class.

    Adds a string prefix to all fields of a feature key instance if `prefix` is
    not None nor empty.

    Example usage:

    >>> test_key = EncodedImageFeatureKey()
    >>> test_key.encoded
    image/encoded
    >>> test_key = EncodedImageFeatureKey('prefix')
    >>> test_key.encoded
    prefix/image/encoded

    Args:
      prefix: A prefix string that will be added before the feature key string
        with a trailing slash '/'.
    """
    if prefix:
      for field in dataclasses.fields(self):  # pytype: disable=wrong-arg-types  # re-none
        key_name = field.name
        key_value = getattr(self, key_name)
        setattr(self, key_name, f'{prefix}/{key_value}')
