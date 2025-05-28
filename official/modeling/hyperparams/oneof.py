# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Config class that supports oneof functionality."""

from typing import Optional

import dataclasses
from official.modeling.hyperparams import base_config


@dataclasses.dataclass
class OneOfConfig(base_config.Config):
  """Configuration for configs with one of feature.

  Attributes:
    type: 'str', name of the field to select.
  """
  type: Optional[str] = None

  def as_dict(self):
    """Returns a dict representation of OneOfConfig.

    For the nested base_config.Config, a nested dict will be returned.
    """
    if self.type is None:
      return {'type': None}
    elif self.__dict__['type'] not in self.__dict__:
      raise ValueError('type: {!r} is not a valid key!'.format(
          self.__dict__['type']))
    else:
      chosen_type = self.type
      chosen_value = self.__dict__[chosen_type]
      return {'type': self.type, chosen_type: self._export_config(chosen_value)}

  def get(self):
    """Returns selected config based on the value of type.

    If type is not set (None), None is returned.
    """
    chosen_type = self.type
    if chosen_type is None:
      return None
    if chosen_type not in self.__dict__:
      raise ValueError('type: {!r} is not a valid key!'.format(self.type))
    return self.__dict__[chosen_type]
