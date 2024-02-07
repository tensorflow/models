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

"""Functions used to convert a TF checkpoint into a dictionary."""

import numpy as np
import tensorflow as tf, tf_keras


def update_weights_dict(weights_dict, variable_key, value):
  """Inserts weight value into a weight dictionary.

  This function inserts a weight value into a weights_dict based on the
  variable key. It is designed to organize TF checkpoint weights by organizing
  them by submodules.

  Args:
    weights_dict: Dictionary to store weights.
    variable_key: String, name of the variable assocaited with the value.
    value: An ndarray that stores the weights assocaited to the variable key.
  """
  current_dict = weights_dict
  variable_key_list = variable_key.split("/")

  key = variable_key_list.pop(0)
  # pylint: disable=g-explicit-length-test
  while len(variable_key_list):
    if variable_key_list[0] == ".ATTRIBUTES":
      current_dict[key] = value
      return

    if key not in current_dict.keys():
      current_dict[key] = {}
    current_dict = current_dict[key]
    key = variable_key_list.pop(0)


def get_ckpt_weights_as_dict(ckpt_path):
  """Converts a TF checkpoint into a nested dictionary of weights.

  Args:
    ckpt_path: String, indicating filepath of the TF checkpoint

  Returns:
    Dictionary where the checkpoint weights are stored
    Number of weights read
  """
  print("\nConverting model checkpoint from {} to weights dictionary\n".format(
      ckpt_path))
  reader = tf.train.load_checkpoint(ckpt_path)
  shape_from_key = reader.get_variable_to_shape_map()
  # dtype_from_key = reader.get_variable_to_dtype_map()

  variable_keys = shape_from_key.keys()
  weights_dict = {}
  n_read = 0

  for key in variable_keys:
    # shape = shape_from_key[key]
    # dtype = dtype_from_key[key]
    value = reader.get_tensor(key)
    n_read += tf.size(value)
    update_weights_dict(weights_dict, key, value)

  print("Successfully read {} checkpoint weights\n".format(n_read))
  return weights_dict, n_read


def write_dict_as_tree(dictionary, filename, spaces=0):
  """Writes nested dictionary keys to a file.

  Given a dictionary that contains nested dictionaries, this function
  writes the name of the keys recursively to a specified file as a tree

  Args:
    dictionary: Desired dictionary to write to a file
    filename: String, name of file to write dictionary to
    spaces: Optional; Number of spaces to insert before writing
      the dictionary key names
  """
  if isinstance(dictionary, dict):
    mode = "w" if spaces == 0 else "a"
    for key in dictionary.keys():
      with open(filename, mode) as fp:
        fp.write(" " * spaces + key + "\n")
      mode = "a"
      write_dict_as_tree(dictionary[key], filename, spaces + 2)


def print_layer_weights_and_shape(layer):
  """Prints variables information corresponding to a Keras layer.

  This function prints the name and the shape of its associated weights
  of all variables (trainable and untrainable) in a Keras layer.

  Args:
    layer: A Keras.layer.Layer object
  """
  weights = layer.get_weights()
  variables = layer.variables

  for i in range(len(weights)):
    tf.print(np.shape(weights[i]), variables[i].name)
