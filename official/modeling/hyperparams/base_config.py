# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Base configurations to standardize experiments."""

import copy
import dataclasses
import functools
import inspect
import types
import typing
from typing import Any, List, Mapping, Optional, Type, Union

from absl import logging
import tensorflow as tf, tf_keras
import yaml

from official.modeling.hyperparams import params_dict


_BOUND = set()


def bind(config_cls):
  """Bind a class to config cls."""
  if not inspect.isclass(config_cls):
    raise ValueError('The bind decorator is supposed to apply on the class '
                     f'attribute. Received {config_cls}, not a class.')

  def decorator(builder):
    if config_cls in _BOUND:
      raise ValueError('Inside a program, we should not bind the config with a'
                       ' class twice.')
    if inspect.isclass(builder):
      config_cls._BUILDER = builder  # pylint: disable=protected-access
    elif inspect.isfunction(builder):

      def _wrapper(self, *args, **kwargs):  # pylint: disable=unused-argument
        return builder(*args, **kwargs)

      config_cls._BUILDER = _wrapper  # pylint: disable=protected-access
    else:
      raise ValueError(f'The `BUILDER` type is not supported: {builder}')
    _BOUND.add(config_cls)
    return builder

  return decorator


def _is_optional(field):
  # Two styles of annotating optional fields:
  #   Optional[T] -> typing.Union
  #   T | None    -> types.UnionType
  is_union = typing.get_origin(field) in (Union, types.UnionType)
  # An optional field is a union of a type, and NoneType.
  args = typing.get_args(field)
  return is_union and len(args) == 2 and type(None) in args


@dataclasses.dataclass
class Config(params_dict.ParamsDict):
  """The base configuration class that supports YAML/JSON based overrides.

  Because of YAML/JSON serialization limitations, some semantics of dataclass
  are not supported:
  * It recursively enforces a allowlist of basic types and container types, so
    it avoids surprises with copy and reuse caused by unanticipated types.
  * Warning: it converts Dict to `Config` even within sequences,
    e.g. for config = Config({'key': [([{'a': 42}],)]),
         type(config.key[0][0][0]) is Config rather than dict.
    If you define/annotate some field as Dict, the field will convert to a
    `Config` instance and lose the dictionary type.
  """
  # The class or method to bind with the params class.
  _BUILDER = None
  # It's safe to add bytes and other immutable types here.
  IMMUTABLE_TYPES = (str, int, float, bool, type(None))
  # It's safe to add set, frozenset and other collections here.
  SEQUENCE_TYPES = (list, tuple)

  default_params: dataclasses.InitVar[Optional[Mapping[str, Any]]] = None
  restrictions: dataclasses.InitVar[Optional[List[str]]] = None

  def __post_init__(self, default_params, restrictions):
    super().__init__(
        default_params=default_params,
        restrictions=restrictions)

  @property
  def BUILDER(self):
    return self._BUILDER

  @classmethod
  def _get_annotations(cls):
    """Returns valid annotations.

    Note: this is similar to dataclasses.__annotations__ except it also includes
      annotations from its parent classes.
    """
    all_annotations = typing.get_type_hints(cls)
    # Removes Config class annotation from the value, e.g., default_params,
    # restrictions, etc.
    for k in Config.__annotations__:
      del all_annotations[k]
    return all_annotations

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
  def _get_subconfig_type(
      cls, k, subconfig_type=None
  ) -> Type[params_dict.ParamsDict]:
    """Get element type by the field name.

    Args:
      k: the key/name of the field.
      subconfig_type: default subconfig_type. If None, it is set to
        Config.

    Returns:
      Config as default. If a type annotation is found for `k`,
      1) returns the type of the annotation if it is subtype of ParamsDict;
      2) returns the element type if the annotation of `k` is List[SubType]
         or Tuple[SubType].
    """
    if not subconfig_type:
      subconfig_type = Config

    def _is_subtype(x, target) -> bool:
      return isinstance(x, type) and issubclass(x, target)

    annotations = cls._get_annotations()
    if k in annotations:
      # Directly Config subtype.
      type_annotation = annotations[k]
      i = 0
      # Loop for striping the Optional annotation.
      traverse_in = True
      while traverse_in:
        i += 1
        if _is_subtype(type_annotation, Config):
          subconfig_type = type_annotation
          break
        else:
          # If the field is a sequence of sub-config types or an Optional
          # sub-config, then strip the container and process the sub-config.
          is_sequence = _is_subtype(
              typing.get_origin(type_annotation), cls.SEQUENCE_TYPES
          )
          if is_sequence or _is_optional(type_annotation):
            type_annotation = typing.get_args(type_annotation)[0]
            continue
        traverse_in = False
    return subconfig_type

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

    def is_null(k):
      if k not in self.__dict__ or not self.__dict__[k]:
        return True
      return False

    if isinstance(v, dict):
      if is_null(k):
        # If the key not exist or the value is None, a new Config-family object
        # sould be created for the key.
        self.__dict__[k] = subconfig_type(v)
      else:
        self.__dict__[k].override(v)
    elif not is_null(k) and isinstance(v, self.SEQUENCE_TYPES) and all(
        [not isinstance(e, self.IMMUTABLE_TYPES) for e in v]):
      if len(self.__dict__[k]) == len(v):
        for i in range(len(v)):
          self.__dict__[k][i].override(v[i])
      elif not all([isinstance(e, self.IMMUTABLE_TYPES) for e in v]):
        logging.warning(
            "The list/tuple don't match the value dictionaries provided. Thus, "
            'the list/tuple is determined by the type annotation and '
            'values provided. This is error-prone.')
        self.__dict__[k] = self._import_config(v, subconfig_type)
      else:
        self.__dict__[k] = self._import_config(v, subconfig_type)
    else:
      self.__dict__[k] = self._import_config(v, subconfig_type)

  def __setattr__(self, k, v):
    if k == 'BUILDER' or k == '_BUILDER':
      raise AttributeError('`BUILDER` is a property and `_BUILDER` is the '
                           'reserved class attribute. We should only assign '
                           '`_BUILDER` at the class level.')

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
    # Note we intend to keep `__annotations__` instead of `_get_annotations`.
    # Assuming a parent class of (a, b) with the sub-class of (c, d), the
    # sub-class will take (c, d) for args, rather than starting from (a, b).
    attributes = list(cls.__annotations__.keys())
    default_params = {a: p for a, p in zip(attributes, args)}
    default_params.update(kwargs)
    return cls(default_params=default_params)
