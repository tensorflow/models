# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""A parameter dictionary class which supports the nest structure."""

import collections
import copy
import re

import six
import tensorflow as tf
import yaml

# regex pattern that matches on key-value pairs in a comma-separated
# key-value pair string. It splits each k-v pair on the = sign, and
# matches on values that are within single quotes, double quotes, single
# values (e.g. floats, ints, etc.), and a lists within brackets.
_PARAM_RE = re.compile(
    r"""
  (?P<name>[a-zA-Z][\w\.]*)    # variable name: "var" or "x"
  \s*=\s*
  ((?P<val>\'(.*?)\'           # single quote
  |
  \"(.*?)\"                    # double quote
  |
  [^,\[]*                      # single value
  |
  \[[^\]]*\]))                 # list of values
  ($|,\s*)""", re.VERBOSE)

_CONST_VALUE_RE = re.compile(r'(\d.*|-\d.*|None)')

# Yaml loader with an implicit resolver to parse float decimal and exponential
# format. The regular experission parse the following cases:
# 1- Decimal number with an optional exponential term.
# 2- Integer number with an exponential term.
# 3- Decimal number with an optional exponential term.
# 4- Decimal number.

LOADER = yaml.SafeLoader
LOADER.add_implicit_resolver(
    'tag:yaml.org,2002:float',
    re.compile(r'''
    ^(?:[-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |
    [-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |
    \\.[0-9_]+(?:[eE][-+][0-9]+)?
    |
    [-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*)$''', re.X),
    list('-+0123456789.'))


class ParamsDict(object):
  """A hyperparameter container class."""

  RESERVED_ATTR = ['_locked', '_restrictions']

  def __init__(self, default_params=None, restrictions=None):
    """Instantiate a ParamsDict.

    Instantiate a ParamsDict given a set of default parameters and a list of
    restrictions. Upon initialization, it validates itself by checking all the
    defined restrictions, and raise error if it finds inconsistency.

    Args:
      default_params: a Python dict or another ParamsDict object including the
        default parameters to initialize.
      restrictions: a list of strings, which define a list of restrictions to
        ensure the consistency of different parameters internally. Each
        restriction string is defined as a binary relation with a set of
        operators, including {'==', '!=',  '<', '<=', '>', '>='}.
    """
    self._locked = False
    self._restrictions = []
    if restrictions:
      self._restrictions = restrictions
    if default_params is None:
      default_params = {}
    self.override(default_params, is_strict=False)

  def _set(self, k, v):
    if isinstance(v, dict):
      self.__dict__[k] = ParamsDict(v)
    else:
      self.__dict__[k] = copy.deepcopy(v)

  def __setattr__(self, k, v):
    """Sets the value of the existing key.

    Note that this does not allow directly defining a new key. Use the
    `override` method with `is_strict=False` instead.

    Args:
      k: the key string.
      v: the value to be used to set the key `k`.

    Raises:
      KeyError: if k is not defined in the ParamsDict.
    """
    if k not in ParamsDict.RESERVED_ATTR:
      if k not in self.__dict__.keys():
        raise KeyError('The key `%{}` does not exist. '
                       'To extend the existing keys, use '
                       '`override` with `is_strict` = True.'.format(k))
      if self._locked:
        raise ValueError('The ParamsDict has been locked. '
                         'No change is allowed.')
    self._set(k, v)

  def __getattr__(self, k):
    """Gets the value of the existing key.

    Args:
      k: the key string.

    Returns:
      the value of the key.

    Raises:
      AttributeError: if k is not defined in the ParamsDict.
    """
    if k not in self.__dict__.keys():
      raise AttributeError('The key `{}` does not exist. '.format(k))
    return self.__dict__[k]

  def __contains__(self, key):
    """Implements the membership test operator."""
    return key in self.__dict__

  def get(self, key, value=None):
    """Accesses through built-in dictionary get method."""
    return self.__dict__.get(key, value)

  def __delattr__(self, k):
    """Deletes the key and removes its values.

    Args:
      k: the key string.

    Raises:
      AttributeError: if k is reserverd or not defined in the ParamsDict.
      ValueError: if the ParamsDict instance has been locked.
    """
    if k in ParamsDict.RESERVED_ATTR:
      raise AttributeError(
          'The key `{}` is reserved. No change is allowes. '.format(k))
    if k not in self.__dict__.keys():
      raise AttributeError('The key `{}` does not exist. '.format(k))
    if self._locked:
      raise ValueError('The ParamsDict has been locked. No change is allowed.')
    del self.__dict__[k]

  def override(self, override_params, is_strict=True):
    """Override the ParamsDict with a set of given params.

    Args:
      override_params: a dict or a ParamsDict specifying the parameters to be
        overridden.
      is_strict: a boolean specifying whether override is strict or not. If
        True, keys in `override_params` must be present in the ParamsDict. If
        False, keys in `override_params` can be different from what is currently
        defined in the ParamsDict. In this case, the ParamsDict will be extended
        to include the new keys.
    """
    if self._locked:
      raise ValueError('The ParamsDict has been locked. No change is allowed.')
    if isinstance(override_params, ParamsDict):
      override_params = override_params.as_dict()
    self._override(override_params, is_strict)  # pylint: disable=protected-access

  def _override(self, override_dict, is_strict=True):
    """The implementation of `override`."""
    for k, v in six.iteritems(override_dict):
      if k in ParamsDict.RESERVED_ATTR:
        raise KeyError('The key `%{}` is internally reserved. '
                       'Can not be overridden.')
      if k not in self.__dict__.keys():
        if is_strict:
          raise KeyError('The key `{}` does not exist. '
                         'To extend the existing keys, use '
                         '`override` with `is_strict` = False.'.format(k))
        else:
          self._set(k, v)
      else:
        if isinstance(v, dict):
          self.__dict__[k]._override(v, is_strict)  # pylint: disable=protected-access
        elif isinstance(v, ParamsDict):
          self.__dict__[k]._override(v.as_dict(), is_strict)  # pylint: disable=protected-access
        else:
          self.__dict__[k] = copy.deepcopy(v)

  def lock(self):
    """Makes the ParamsDict immutable."""
    self._locked = True

  def as_dict(self):
    """Returns a dict representation of ParamsDict.

    For the nested ParamsDict, a nested dict will be returned.
    """
    params_dict = {}
    for k, v in six.iteritems(self.__dict__):
      if k not in ParamsDict.RESERVED_ATTR:
        if isinstance(v, ParamsDict):
          params_dict[k] = v.as_dict()
        else:
          params_dict[k] = copy.deepcopy(v)
    return params_dict

  def validate(self):
    """Validate the parameters consistency based on the restrictions.

    This method validates the internal consistency using the pre-defined list of
    restrictions. A restriction is defined as a string which specfiies a binary
    operation. The supported binary operations are {'==', '!=', '<', '<=', '>',
    '>='}. Note that the meaning of these operators are consistent with the
    underlying Python immplementation. Users should make sure the define
    restrictions on their type make sense.

    For example, for a ParamsDict like the following
    ```
    a:
      a1: 1
      a2: 2
    b:
      bb:
        bb1: 10
        bb2: 20
      ccc:
        a1: 1
        a3: 3
    ```
    one can define two restrictions like this
    ['a.a1 == b.ccc.a1', 'a.a2 <= b.bb.bb2']

    What it enforces are:
     - a.a1 = 1 == b.ccc.a1 = 1
     - a.a2 = 2 <= b.bb.bb2 = 20

    Raises:
      KeyError: if any of the following happens
        (1) any of parameters in any of restrictions is not defined in
            ParamsDict,
        (2) any inconsistency violating the restriction is found.
      ValueError: if the restriction defined in the string is not supported.
    """

    def _get_kv(dotted_string, params_dict):
      """Get keys and values indicated by dotted_string."""
      if _CONST_VALUE_RE.match(dotted_string) is not None:
        const_str = dotted_string
        if const_str == 'None':
          constant = None
        else:
          constant = float(const_str)
        return None, constant
      else:
        tokenized_params = dotted_string.split('.')
        v = params_dict
        for t in tokenized_params:
          v = v[t]
        return tokenized_params[-1], v

    def _get_kvs(tokens, params_dict):
      if len(tokens) != 2:
        raise ValueError('Only support binary relation in restriction.')
      stripped_tokens = [t.strip() for t in tokens]
      left_k, left_v = _get_kv(stripped_tokens[0], params_dict)
      right_k, right_v = _get_kv(stripped_tokens[1], params_dict)
      return left_k, left_v, right_k, right_v

    params_dict = self.as_dict()
    for restriction in self._restrictions:
      if '==' in restriction:
        tokens = restriction.split('==')
        _, left_v, _, right_v = _get_kvs(tokens, params_dict)
        if left_v != right_v:
          raise KeyError(
              'Found inconsistncy between key `{}` and key `{}`.'.format(
                  tokens[0], tokens[1]))
      elif '!=' in restriction:
        tokens = restriction.split('!=')
        _, left_v, _, right_v = _get_kvs(tokens, params_dict)
        if left_v == right_v:
          raise KeyError(
              'Found inconsistncy between key `{}` and key `{}`.'.format(
                  tokens[0], tokens[1]))
      elif '<' in restriction:
        tokens = restriction.split('<')
        _, left_v, _, right_v = _get_kvs(tokens, params_dict)
        if left_v >= right_v:
          raise KeyError(
              'Found inconsistncy between key `{}` and key `{}`.'.format(
                  tokens[0], tokens[1]))
      elif '<=' in restriction:
        tokens = restriction.split('<=')
        _, left_v, _, right_v = _get_kvs(tokens, params_dict)
        if left_v > right_v:
          raise KeyError(
              'Found inconsistncy between key `{}` and key `{}`.'.format(
                  tokens[0], tokens[1]))
      elif '>' in restriction:
        tokens = restriction.split('>')
        _, left_v, _, right_v = _get_kvs(tokens, params_dict)
        if left_v <= right_v:
          raise KeyError(
              'Found inconsistncy between key `{}` and key `{}`.'.format(
                  tokens[0], tokens[1]))
      elif '>=' in restriction:
        tokens = restriction.split('>=')
        _, left_v, _, right_v = _get_kvs(tokens, params_dict)
        if left_v < right_v:
          raise KeyError(
              'Found inconsistncy between key `{}` and key `{}`.'.format(
                  tokens[0], tokens[1]))
      else:
        raise ValueError('Unsupported relation in restriction.')


def read_yaml_to_params_dict(file_path: str):
  """Reads a YAML file to a ParamsDict."""
  with tf.io.gfile.GFile(file_path, 'r') as f:
    params_dict = yaml.load(f, Loader=LOADER)
    return ParamsDict(params_dict)


def save_params_dict_to_yaml(params, file_path):
  """Saves the input ParamsDict to a YAML file."""
  with tf.io.gfile.GFile(file_path, 'w') as f:

    def _my_list_rep(dumper, data):
      # u'tag:yaml.org,2002:seq' is the YAML internal tag for sequence.
      return dumper.represent_sequence(
          u'tag:yaml.org,2002:seq', data, flow_style=True)

    yaml.add_representer(list, _my_list_rep)
    yaml.dump(params.as_dict(), f, default_flow_style=False)


def nested_csv_str_to_json_str(csv_str):
  """Converts a nested (using '.') comma-separated k=v string to a JSON string.

  Converts a comma-separated string of key/value pairs that supports
  nesting of keys to a JSON string. Nesting is implemented using
  '.' between levels for a given key.

  Spacing between commas and = is supported (e.g. there is no difference between
  "a=1,b=2", "a = 1, b = 2", or "a=1, b=2") but there should be no spaces before
  keys or after values (e.g. " a=1,b=2" and "a=1,b=2 " are not supported).

  Note that this will only support values supported by CSV, meaning
  values such as nested lists (e.g. "a=[[1,2,3],[4,5,6]]") are not
  supported. Strings are supported as well, e.g. "a='hello'".

  An example conversion would be:

  "a=1, b=2, c.a=2, c.b=3, d.a.a=5"

  to

  "{ a: 1, b : 2, c: {a : 2, b : 3}, d: {a: {a : 5}}}"

  Args:
    csv_str: the comma separated string.

  Returns:
    the converted JSON string.

  Raises:
    ValueError: If csv_str is not in a comma separated string or
      if the string is formatted incorrectly.
  """
  if not csv_str:
    return ''

  formatted_entries = []
  nested_map = collections.defaultdict(list)
  pos = 0
  while pos < len(csv_str):
    m = _PARAM_RE.match(csv_str, pos)
    if not m:
      raise ValueError('Malformed hyperparameter value while parsing '
                       'CSV string: %s' % csv_str[pos:])
    pos = m.end()
    # Parse the values.
    m_dict = m.groupdict()
    name = m_dict['name']
    v = m_dict['val']

    # If a GCS path (e.g. gs://...) is provided, wrap this in quotes
    # as yaml.load would otherwise throw an exception
    if re.match(r'(?=[^\"\'])(?=[gs://])', v):
      v = '\'{}\''.format(v)

    name_nested = name.split('.')
    if len(name_nested) > 1:
      grouping = name_nested[0]
      value = '.'.join(name_nested[1:]) + '=' + v
      nested_map[grouping].append(value)
    else:
      formatted_entries.append('%s : %s' % (name, v))

  for grouping, value in nested_map.items():
    value = ','.join(value)
    value = nested_csv_str_to_json_str(value)
    formatted_entries.append('%s : %s' % (grouping, value))
  return '{' + ', '.join(formatted_entries) + '}'


def override_params_dict(params, dict_or_string_or_yaml_file, is_strict):
  """Override a given ParamsDict using a dict, JSON/YAML/CSV string or YAML file.

  The logic of the function is outlined below:
  1. Test that the input is a dict. If not, proceed to 2.
  2. Tests that the input is a string. If not, raise unknown ValueError
  2.1. Test if the string is in a CSV format. If so, parse.
  If not, proceed to 2.2.
  2.2. Try loading the string as a YAML/JSON. If successful, parse to
  dict and use it to override. If not, proceed to 2.3.
  2.3. Try using the string as a file path and load the YAML file.

  Args:
    params: a ParamsDict object to be overridden.
    dict_or_string_or_yaml_file: a Python dict, JSON/YAML/CSV string or path to
      a YAML file specifying the parameters to be overridden.
    is_strict: a boolean specifying whether override is strict or not.

  Returns:
    params: the overridden ParamsDict object.

  Raises:
    ValueError: if failed to override the parameters.
  """
  if not dict_or_string_or_yaml_file:
    return params
  if isinstance(dict_or_string_or_yaml_file, dict):
    params.override(dict_or_string_or_yaml_file, is_strict)
  elif isinstance(dict_or_string_or_yaml_file, six.string_types):
    try:
      dict_or_string_or_yaml_file = (
          nested_csv_str_to_json_str(dict_or_string_or_yaml_file))
    except ValueError:
      pass
    params_dict = yaml.load(dict_or_string_or_yaml_file, Loader=LOADER)
    if isinstance(params_dict, dict):
      params.override(params_dict, is_strict)
    else:
      with tf.io.gfile.GFile(dict_or_string_or_yaml_file) as f:
        params.override(yaml.load(f, Loader=yaml.FullLoader), is_strict)
  else:
    raise ValueError('Unknown input type to parse.')
  return params
