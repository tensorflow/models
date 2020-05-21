# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""Utility functions for manipulating Keras models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def extract_submodel(model, inputs, outputs, name=None):
  """Extracts a section of a Keras model into a new model.

  This method walks an existing model from the specified outputs back to the
  specified inputs in order to construct a new model containing only a portion
  of the old model, while sharing the layers and weights with the original
  model.

  WARNING: This method does not work for submodels containing layers that have
  been used multiple times in the original model, or in other models beyond
  the original model. (E.g. does not work for submodels that contain layers that
  use shared weights). This also means that multiple overlapping submodels
  cannot be extracted from the same model.

  It also relies on recursion and will hit python's recursion limit for large
  submodels.

  Args:
    model: The existing Keras model this method extracts a submodel from.
    inputs: The layer inputs in the existing model that start the submodel
    outputs: The layer outputs in the existing model that should be output by
      the submodel
    name: The name for the extracted model

  Returns:
    The extracted submodel specified by the given inputs and outputs
  """
  output_to_layer = {}
  output_to_layer_input = {}
  for layer in model.layers:
    layer_output = layer.output
    layer_inputs = layer.input
    output_to_layer[layer_output.ref()] = layer
    output_to_layer_input[layer_output.ref()] = layer_inputs

  model_inputs_dict = {}
  memoized_results = {}

  # Relies on recursion, very low limit in python
  def _recurse_in_model(tensor):
    """Walk the existing model recursively to copy a submodel."""
    if tensor.ref() in memoized_results:
      return memoized_results[tensor.ref()]
    if (tensor.ref() == inputs.ref()) or (
        isinstance(inputs, list) and tensor in inputs):
      if tensor.ref() not in model_inputs_dict:
        model_inputs_dict[tensor.ref()] = tf.keras.layers.Input(tensor=tensor)
      out = model_inputs_dict[tensor.ref()]
    else:
      cur_inputs = output_to_layer_input[tensor.ref()]
      cur_layer = output_to_layer[tensor.ref()]
      if isinstance(cur_inputs, list):
        out = cur_layer([_recurse_in_model(inp) for inp in cur_inputs])
      else:
        out = cur_layer(_recurse_in_model(cur_inputs))
    memoized_results[tensor.ref()] = out
    return out

  if isinstance(outputs, list):
    model_outputs = [_recurse_in_model(tensor) for tensor in outputs]
  else:
    model_outputs = _recurse_in_model(outputs)

  if isinstance(inputs, list):
    model_inputs = [model_inputs_dict[tensor.ref()] for tensor in inputs]
  else:
    model_inputs = model_inputs_dict[inputs.ref()]

  return tf.keras.Model(inputs=model_inputs, outputs=model_outputs, name=name)
