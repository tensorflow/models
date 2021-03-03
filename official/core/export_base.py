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

import abc
import functools
from typing import Any, Callable, Dict, Mapping, List, Optional, Text, Union

import tensorflow as tf
from tensorflow.python.saved_model.model_utils import export_utils


class ExportModule(tf.Module, metaclass=abc.ABCMeta):
  """Base Export Module."""

  def __init__(self,
               params,
               model: Union[tf.Module, tf.keras.Model],
               inference_step: Optional[Callable[..., Any]] = None):
    """Instantiates an ExportModel.

    Args:
      params: A dataclass for parameters to the module.
      model: A model instance which contains weights and forward computation.
      inference_step: An optional callable to define how the model is called.
    """
    super().__init__(name=None)
    self.model = model
    self.params = params

    if inference_step is not None:
      self.inference_step = functools.partial(inference_step, model=self.model)
    else:
      self.inference_step = functools.partial(
          self.model.__call__, training=False)

  @abc.abstractmethod
  def serve(self) -> Mapping[Text, tf.Tensor]:
    """The bare inference function which should run on all devices.

    Expecting tensors are passed in through keyword arguments. Returns a
    dictionary of tensors, when the keys will be used inside the SignatureDef.
    """

  @abc.abstractmethod
  def get_inference_signatures(
      self, function_keys: Dict[Text, Text]) -> Mapping[Text, Any]:
    """Get defined function signatures."""


def export(export_module: ExportModule,
           function_keys: Union[List[Text], Dict[Text, Text]],
           export_savedmodel_dir: Text,
           checkpoint_path: Optional[Text] = None,
           timestamped: bool = True,
           save_options: Optional[tf.saved_model.SaveOptions] = None) -> Text:
  """Exports to SavedModel format.

  Args:
    export_module: a ExportModule with the keras Model and serving tf.functions.
    function_keys: a list of string keys to retrieve pre-defined serving
      signatures. The signaute keys will be set with defaults. If a dictionary
      is provided, the values will be used as signature keys.
    export_savedmodel_dir: Output saved model directory.
    checkpoint_path: Object-based checkpoint path or directory.
    timestamped: Whether to export the savedmodel to a timestamped directory.
    save_options: `SaveOptions` for `tf.saved_model.save`.

  Returns:
    The savedmodel directory path.
  """
  ckpt_dir_or_file = checkpoint_path
  if tf.io.gfile.isdir(ckpt_dir_or_file):
    ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
  if ckpt_dir_or_file:
    checkpoint = tf.train.Checkpoint(model=export_module.model)
    checkpoint.read(
        ckpt_dir_or_file).assert_existing_objects_matched().expect_partial()
  if isinstance(function_keys, list):
    if len(function_keys) == 1:
      function_keys = {
          function_keys[0]: tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
      }
    else:
      raise ValueError(
          "If the function_keys is a list, it must contain a single element. %s"
          % function_keys)

  signatures = export_module.get_inference_signatures(function_keys)
  if timestamped:
    export_dir = export_utils.get_timestamped_export_dir(
        export_savedmodel_dir).decode("utf-8")
  else:
    export_dir = export_savedmodel_dir
  tf.saved_model.save(
      export_module, export_dir, signatures=signatures, options=save_options)
  return export_dir
