# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""General utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import six
from utils.luatables import T
import tensorflow as tf
import yaml
from yaml.constructor import ConstructorError
# pylint: disable=invalid-name


def GetFilesRecursively(topdir):
  """Gets all records recursively for some topdir.

  Args:
    topdir: String, path to top directory.
  Returns:
    allpaths: List of Strings, full paths to all leaf records.
  Raises:
    ValueError: If there are no files found for this directory.
  """
  assert topdir
  topdir = os.path.expanduser(topdir)
  allpaths = []
  for path, _, leaffiles in tf.gfile.Walk(topdir):
    if leaffiles:
      allpaths.extend([os.path.join(path, i) for i in leaffiles])
  if not allpaths:
    raise ValueError('No files found for top directory %s' % topdir)
  return allpaths


def NoDuplicatesConstructor(loader, node, deep=False):
  """Check for duplicate keys."""
  mapping = {}
  for key_node, value_node in node.value:
    key = loader.construct_object(key_node, deep=deep)
    value = loader.construct_object(value_node, deep=deep)
    if key in mapping:
      raise ConstructorError('while constructing a mapping', node.start_mark,
                             'found duplicate key (%s)' % key,
                             key_node.start_mark)
    mapping[key] = value
  return loader.construct_mapping(node, deep)


def WriteConfigAsYaml(config, logdir, filename):
  """Writes a config dict as yaml to logdir/experiment.yml."""
  if not tf.gfile.Exists(logdir):
    tf.gfile.MakeDirs(logdir)
  config_filename = os.path.join(logdir, filename)
  with tf.gfile.GFile(config_filename, 'w') as f:
    f.write(yaml.dump(config))
  tf.logging.info('wrote config to %s', config_filename)


def LoadConfigDict(config_paths, model_params):
  """Loads config dictionary from specified yaml files or command line yaml."""

  # Ensure that no duplicate keys can be loaded (causing pain).
  yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                       NoDuplicatesConstructor)

  # Handle either ',' or '#' separated config lists, since borg will only
  # accept '#'.
  sep = ',' if ',' in config_paths else '#'

  # Load flags from config file.
  final_config = {}
  if config_paths:
    for config_path in config_paths.split(sep):
      config_path = config_path.strip()
      if not config_path:
        continue
      config_path = os.path.abspath(config_path)
      tf.logging.info('Loading config from %s', config_path)
      with tf.gfile.GFile(config_path.strip()) as config_file:
        config_flags = yaml.load(config_file)
        final_config = DeepMergeDict(final_config, config_flags)
  if model_params:
    model_params = MaybeLoadYaml(model_params)
    final_config = DeepMergeDict(final_config, model_params)
  tf.logging.info('Final Config:\n%s', yaml.dump(final_config))
  return final_config


def MaybeLoadYaml(item):
  """Parses item if it's a string. If it's a dictionary it's returned as-is."""
  if isinstance(item, six.string_types):
    return yaml.load(item)
  elif isinstance(item, dict):
    return item
  else:
    raise ValueError('Got {}, expected YAML string or dict', type(item))


def DeepMergeDict(dict_x, dict_y, path=None):
  """Recursively merges dict_y into dict_x."""
  if path is None: path = []
  for key in dict_y:
    if key in dict_x:
      if isinstance(dict_x[key], dict) and isinstance(dict_y[key], dict):
        DeepMergeDict(dict_x[key], dict_y[key], path + [str(key)])
      elif dict_x[key] == dict_y[key]:
        pass  # same leaf value
      else:
        dict_x[key] = dict_y[key]
    else:
      dict_x[key] = dict_y[key]
  return dict_x


def ParseConfigsToLuaTable(config_paths, extra_model_params=None,
                           save=False, save_name='final_training_config.yml',
                           logdir=None):
  """Maps config_paths and extra_model_params to a Luatable-like object."""
  # Parse config dict from yaml config files / command line flags.
  config = LoadConfigDict(config_paths, extra_model_params)
  if save:
    WriteConfigAsYaml(config, logdir, save_name)
  # Convert config dictionary to T object with dot notation.
  config = RecursivelyConvertToLuatable(config)
  return config


def SetNestedValue(d, keys, value):
  """Sets a value in a nested dictionary.

  Example:
    d = {}, keys = ['data','augmentation','minscale'], value = 1.0.
    returns {'data': {'augmentation' : {'minscale': 1.0 }}}

  Args:
    d: A dictionary to set a nested value in.
    keys: list of dict keys nesting left to right.
    value: the nested value to set.
  Returns:
    None
  """
  for key in keys[:-1]:
    d = d.setdefault(key, {})
  d[keys[-1]] = value


def RecursivelyConvertToLuatable(yaml_dict):
  """Converts a dictionary to a LuaTable-like T object."""
  if isinstance(yaml_dict, dict):
    yaml_dict = T(yaml_dict)
  for key, item in yaml_dict.iteritems():
    if isinstance(item, dict):
      yaml_dict[key] = RecursivelyConvertToLuatable(item)
  return yaml_dict


def KNNIds(query_vec, target_seq, k=1):
  """Gets the knn ids to the query vec from the target sequence."""
  sorted_distances = KNNIdsWithDistances(query_vec, target_seq, k)
  return [i[0] for i in sorted_distances]


def KNNIdsWithDistances(query_vec, target_seq, k=1):
  """Gets the knn ids to the query vec from the target sequence."""
  if not isinstance(np.array(target_seq), np.ndarray):
    target_seq = np.array(target_seq)
  assert np.shape(query_vec) == np.shape(target_seq[0])
  distances = [(i, np.linalg.norm(query_vec-target_vec)) for (
      i, target_vec) in enumerate(target_seq)]
  sorted_distances = sorted(distances, key=lambda x: x[1])
  return sorted_distances[:k]


def CopyLocalConfigsToCNS(outdir, configs, gfs_user):
  """Copies experiment yaml config files to the job_logdir on /cns."""
  assert configs
  assert outdir
  conf_files = configs.split(',')
  for conf_file in conf_files:
    copy_command = 'fileutil --gfs_user %s cp -f %s %s' % (
        gfs_user, conf_file, outdir)
    tf.logging.info(copy_command)
    os.system(copy_command)


def pairwise_distances(feature, squared=True):
  """Computes the pairwise distance matrix in numpy.

  Args:
    feature: 2-D numpy array of size [number of data, feature dimension]
    squared: Boolean. If true, output is the pairwise squared euclidean
      distance matrix; else, output is the pairwise euclidean distance matrix.

  Returns:
    pdists: 2-D numpy array of size
      [number of data, number of data].
  """
  triu = np.triu_indices(feature.shape[0], 1)
  upper_tri_pdists = np.linalg.norm(feature[triu[1]] - feature[triu[0]], axis=1)
  if squared:
    upper_tri_pdists **= 2.
  num_data = feature.shape[0]
  pdists = np.zeros((num_data, num_data))
  pdists[np.triu_indices(num_data, 1)] = upper_tri_pdists
  # Make symmetrical.
  pdists = pdists + pdists.T - np.diag(
      pdists.diagonal())
  return pdists


def is_tfrecord_input(inp):
  """Checks if input is a TFRecord or list of TFRecords."""
  def _is_tfrecord(inp):
    if not isinstance(inp, str):
      return False
    _, extension = os.path.splitext(inp)
    return extension == '.tfrecord'
  if isinstance(inp, str):
    return _is_tfrecord(inp)
  if isinstance(inp, list):
    return all(map(_is_tfrecord, inp))
  return False


def is_np_array(inp):
  if isinstance(inp, np.ndarray):
    return True
  if isinstance(inp, list):
    return all([isinstance(i, np.ndarray) for i in inp])
  return False
