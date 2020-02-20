# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Base configurations to standardize experiments."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import copy
from typing import Any, List, Mapping, Optional

import dataclasses
import tensorflow as tf
import yaml

from official.modeling.hyperparams import params_dict


@dataclasses.dataclass
class Config(params_dict.ParamsDict):
  """The base configuration class that supports YAML/JSON based overrides."""
  default_params: dataclasses.InitVar[Mapping[str, Any]] = None
  restrictions: dataclasses.InitVar[List[str]] = None

  def __post_init__(self, default_params, restrictions, *args, **kwargs):
    super().__init__(default_params=default_params,
                     restrictions=restrictions,
                     *args,
                     **kwargs)

  def _set(self, k, v):
    if isinstance(v, dict):
      if k not in self.__dict__:
        self.__dict__[k] = params_dict.ParamsDict(v, [])
      else:
        self.__dict__[k].override(v)
    else:
      self.__dict__[k] = copy.deepcopy(v)

  def __setattr__(self, k, v):
    if k in params_dict.ParamsDict.RESERVED_ATTR:
      # Set the essential private ParamsDict attributes.
      self.__dict__[k] = copy.deepcopy(v)
    else:
      self._set(k, v)

  def replace(self, **kwargs):
    """Like `override`, but returns a copy with the current config unchanged."""
    params = self.__class__(self)
    params.override(kwargs, is_strict=True)
    return params

  @classmethod
  def from_yaml(cls, file_path: str):
    # Note: This only works if the Config has all default values.
    with tf.io.gfile.GFile(file_path, 'r') as f:
      loaded = yaml.load(f)
      config = cls()
      config.override(loaded)
      return config

  @classmethod
  def from_json(cls, file_path: str):
    """Wrapper for `from_yaml`."""
    return cls.from_yaml(file_path)

  @classmethod
  def from_args(cls, *args, **kwargs):
    """Builds a config from the given list of arguments."""
    attributes = list(cls.__annotations__.keys())
    default_params = {a: p for a, p in zip(attributes, args)}
    default_params.update(kwargs)
    return cls(default_params)
