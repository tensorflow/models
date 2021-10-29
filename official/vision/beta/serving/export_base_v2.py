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

"""Base class for model export."""

from typing import Dict, Optional, Text, Callable, Any, Union

import tensorflow as tf

from official.core import export_base


class ExportModule(export_base.ExportModule):
  """Base Export Module."""

  def __init__(self,
               params,
               model: tf.keras.Model,
               input_signature: Union[tf.TensorSpec, Dict[str, tf.TensorSpec]],
               preprocessor: Optional[Callable[..., Any]] = None,
               inference_step: Optional[Callable[..., Any]] = None,
               postprocessor: Optional[Callable[..., Any]] = None):
    """Initializes a module for export.

    Args:
      params: A dataclass for parameters to the module.
      model: A tf.keras.Model instance to be exported.
      input_signature: tf.TensorSpec, e.g.
        tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.uint8)
      preprocessor: An optional callable function to preprocess the inputs.
      inference_step: An optional callable function to forward-pass the model.
      postprocessor: An optional callable function to postprocess the model
        outputs.
    """
    super().__init__(params, model=model, inference_step=inference_step)
    self.preprocessor = preprocessor
    self.postprocessor = postprocessor
    self.input_signature = input_signature

  @tf.function
  def serve(self, inputs):
    x = self.preprocessor(inputs=inputs) if self.preprocessor else inputs
    x = self.inference_step(x)
    x = self.postprocessor(x) if self.postprocessor else x
    return x

  def get_inference_signatures(self, function_keys: Dict[Text, Text]):
    """Gets defined function signatures.

    Args:
      function_keys: A dictionary with keys as the function to create signature
        for and values as the signature keys when returns.

    Returns:
      A dictionary with key as signature key and value as concrete functions
        that can be used for tf.saved_model.save.
    """
    signatures = {}
    for _, def_name in function_keys.items():
      signatures[def_name] = self.serve.get_concrete_function(
          self.input_signature)
    return signatures
