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

import copy
import functools
from typing import Any, List, Mapping, Optional, Type

import dataclasses
import tensorflow as tf
import yaml

from official.modeling.hyperparams import params_dict


@dataclasses.dataclass
class Config(params_dict.ParamsDict):
  """The base configuration class that supports YAML/JSON based overrides.

  * It recursively enforces a whitelist of basic types and container types, so
    it avoids surprises with copy and reuse caused by unanticipated types.
  * It converts dict to Config even within sequences,
    e.g. for config = Config({'key': [([{'a': 42}],)]),
         type(config.key[0][0][0]) is Config rather than dict.
  """

  # It's safe to add bytes and other immutable types here.
  IMMUTABLE_TYPES = (str, int, float, bool, type(None))
  # It's safe to add set, frozenset and other collections here.
  SEQUENCE_TYPES = (list, tuple)

  default_params: dataclasses.InitVar[Optional[Mapping[str, Any]]] = None
  restrictions: dataclasses.InitVar[Optional[List[str]]] = None

  @classmethod
  def _isvalidsequence(cls, v):
    """Check if the input values are valid sequences.

    Args:
      v: Input sequence.

    Returns:
      True if the sequence is valid. Valid sequence includes the sequence
      type in cls.SEQUENCE_TYPES and element type is in cls.IMMUTABLE_TYPES or
      is dict or ParamsDict.
    """
    if not isinstance(v, cls.SEQUENCE_TYPES):
      return False
    return (all(isinstance(e, cls.IMMUTABLE_TYPES) for e in v) or
            all(isinstance(e, dict) for e in v) or
            all(isinstance(e, params_dict.ParamsDict) for e in v))

  @classmethod
  def _import_config(cls, v, subconfig_type):
    """Returns v with dicts converted to Configs, recursively."""
    if not issubclass(subconfig_type, params_dict.ParamsDict):
      raise TypeError(
          'Subconfig_type should be subclass of ParamsDict, found {!r}'.format(
              subconfig_type))
    if isinstance(v, cls.IMMUTABLE_TYPES):
      return v
    elif isinstance(v, cls.SEQUENCE_TYPES):
      # Only support one layer of sequence.
      if not cls._isvalidsequence(v):
        raise TypeError(
            'Invalid sequence: only supports single level {!r} of {!r} or '
            'dict or ParamsDict found: {!r}'.format(cls.SEQUENCE_TYPES,
                                                    cls.IMMUTABLE_TYPES, v))
      import_fn = functools.partial(
          cls._import_config, subconfig_type=subconfig_type)
      return type(v)(map(import_fn, v))
    elif isinstance(v, params_dict.ParamsDict):
      # Deepcopy here is a temporary solution for preserving type in nested
      # Config object.
      return copy.deepcopy(v)
    elif isinstance(v, dict):
      return subconfig_type(v)
    else:
      raise TypeError('Unknown type: {!r}'.format(type(v)))

  @classmethod
  def _export_config(cls, v):
    """Returns v with Configs converted to dicts, recursively."""
    if isinstance(v, cls.IMMUTABLE_TYPES):
      return v
    elif isinstance(v, cls.SEQUENCE_TYPES):
      return type(v)(map(cls._export_config, v))
    elif isinstance(v, params_dict.ParamsDict):
      return v.as_dict()
    elif isinstance(v, dict):
      raise TypeError('dict value not supported in converting.')
    else:
      raise TypeError('Unknown type: {!r}'.format(type(v)))

  @classmethod
  def _get_subconfig_type(cls, k) -> Type[params_dict.ParamsDict]:
    """Get element type by the field name.

    Args:
      k: the key/name of the field.

    Returns:
      Config as default. If a type annotation is found for `k`,
      1) returns the type of the annotation if it is subtype of ParamsDict;
      2) returns the element type if the annotation of `k` is List[SubType]
         or Tuple[SubType].
    """
    subconfig_type = Config
    if k in cls.__annotations__:
      # Directly Config subtype.
      type_annotation = cls.__annotations__[k]  # pytype: disable=invalid-annotation
      if (isinstance(type_annotation, type) and
          issubclass(type_annotation, Config)):
        subconfig_type = cls.__annotations__[k]  # pytype: disable=invalid-annotation
      else:
        # Check if the field is a sequence of subtypes.
        field_type = getattr(type_annotation, '__origin__', type(None))
        if (isinstance(field_type, type) and
            issubclass(field_type, cls.SEQUENCE_TYPES)):
          element_type = getattr(type_annotation, '__args__', [type(None)])[0]
          subconfig_type = (
              element_type if issubclass(element_type, params_dict.ParamsDict)
              else subconfig_type)
    return subconfig_type

  def __post_init__(self, default_params, restrictions, *args, **kwargs):
    super().__init__(
        default_params=default_params,
        restrictions=restrictions,
        *args,
        **kwargs)

  def _set(self, k, v):
    """Overrides same method in ParamsDict.

    Also called by ParamsDict methods.

    Args:
      k: key to set.
      v: value.

    Raises:
      RuntimeError
    """
    subconfig_type = self._get_subconfig_type(k)
    if isinstance(v, dict):
      if k not in self.__dict__ or not self.__dict__[k]:
        # If the key not exist or the value is None, a new Config-family object
        # sould be created for the key.
        self.__dict__[k] = subconfig_type(v)
      else:
        self.__dict__[k].override(v)
    else:
      self.__dict__[k] = self._import_config(v, subconfig_type)

  def __setattr__(self, k, v):
    if k not in self.RESERVED_ATTR:
      if getattr(self, '_locked', False):
        raise ValueError('The Config has been locked. ' 'No change is allowed.')
    self._set(k, v)

  def _override(self, override_dict, is_strict=True):
    """Overrides same method in ParamsDict.

    Also called by ParamsDict methods.

    Args:
      override_dict: dictionary to write to .
      is_strict: If True, not allows to add new keys.

    Raises:
      KeyError: overriding reserved keys or keys not exist (is_strict=True).
    """
    for k, v in sorted(override_dict.items()):
      if k in self.RESERVED_ATTR:
        raise KeyError('The key {!r} is internally reserved. '
                       'Can not be overridden.'.format(k))
      if k not in self.__dict__:
        if is_strict:
          raise KeyError('The key {!r} does not exist in {!r}. '
                         'To extend the existing keys, use '
                         '`override` with `is_strict` = False.'.format(
                             k, type(self)))
        else:
          self._set(k, v)
      else:
        if isinstance(v, dict) and self.__dict__[k]:
          self.__dict__[k]._override(v, is_strict)  # pylint: disable=protected-access
        elif isinstance(v, params_dict.ParamsDict) and self.__dict__[k]:
          self.__dict__[k]._override(v.as_dict(), is_strict)  # pylint: disable=protected-access
        else:
          self._set(k, v)

  def as_dict(self):
    """Returns a dict representation of params_dict.ParamsDict.

    For the nested params_dict.ParamsDict, a nested dict will be returned.
    """
    return {
        k: self._export_config(v)
        for k, v in self.__dict__.items()
        if k not in self.RESERVED_ATTR
    }

  def replace(self, **kwargs):
    """Overrides/returns a unlocked copy with the current config unchanged."""
    # pylint: disable=protected-access
    params = copy.deepcopy(self)
    params._locked = False
    params._override(kwargs, is_strict=True)
    # pylint: enable=protected-access
    return params

  @classmethod
  def from_yaml(cls, file_path: str):
    # Note: This only works if the Config has all default values.
    with tf.io.gfile.GFile(file_path, 'r') as f:
      loaded = yaml.load(f, Loader=yaml.FullLoader)
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
