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

"""Utility functions for configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path

import tensorflow as tf


def parse_json(json_string_or_file):
  """Parses values from a JSON string or JSON file.

  This function is useful for command line flags containing configuration
  overrides. Using this function, the flag can be passed either as a JSON string
  (e.g. '{"learning_rate": 1.0}') or the path to a JSON configuration file.

  Args:
    json_string_or_file: A JSON serialized string OR the path to a JSON file.

  Returns:
    A dictionary; the parsed JSON.

  Raises:
    ValueError: If the JSON could not be parsed.
  """
  # First, attempt to parse the string as a JSON dict.
  try:
    json_dict = json.loads(json_string_or_file)
  except ValueError as literal_json_parsing_error:
    try:
      # Otherwise, try to use it as a path to a JSON file.
      with tf.gfile.Open(json_string_or_file) as f:
        json_dict = json.load(f)
    except ValueError as json_file_parsing_error:
      raise ValueError("Unable to parse the content of the json file {}. "
                       "Parsing error: {}.".format(
                           json_string_or_file,
                           json_file_parsing_error.message))
    except tf.gfile.FileError:
      message = ("Unable to parse the input parameter neither as literal "
                 "JSON nor as the name of a file that exists.\n"
                 "JSON parsing error: {}\n\n Input parameter:\n{}.".format(
                     literal_json_parsing_error.message, json_string_or_file))
      raise ValueError(message)

  return json_dict


def to_json(config):
  """Converts a JSON-serializable configuration object to a JSON string."""
  if hasattr(config, "to_json") and callable(config.to_json):
    return config.to_json(indent=2)
  else:
    return json.dumps(config, indent=2)


def log_and_save_config(config, output_dir):
  """Logs and writes a JSON-serializable configuration object.

  Args:
    config: A JSON-serializable object.
    output_dir: Destination directory.
  """
  config_json = to_json(config)
  tf.logging.info("config: %s", config_json)

  tf.gfile.MakeDirs(output_dir)
  with tf.gfile.Open(os.path.join(output_dir, "config.json"), "w") as f:
    f.write(config_json)


def unflatten(flat_config):
  """Transforms a flat configuration dictionary into a nested dictionary.

  Example:
    {
      "a": 1,
      "b.c": 2,
      "b.d.e": 3,
      "b.d.f": 4,
    }
  would be transformed to:
    {
      "a": 1,
      "b": {
        "c": 2,
        "d": {
          "e": 3,
          "f": 4,
        }
      }
    }

  Args:
    flat_config: A dictionary with strings as keys where nested configuration
      parameters are represented with period-separated names.

  Returns:
    A dictionary nested according to the keys of the input dictionary.
  """
  config = {}
  for path, value in flat_config.items():
    path = path.split(".")
    final_key = path.pop()
    nested_config = config
    for key in path:
      nested_config = nested_config.setdefault(key, {})
    nested_config[final_key] = value
  return config
