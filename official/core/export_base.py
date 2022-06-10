# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
import time
from typing import Any, Callable, Dict, Mapping, List, Optional, Text, Union

from absl import logging
import tensorflow as tf

MAX_DIRECTORY_CREATION_ATTEMPTS = 10


class ExportModule(tf.Module, metaclass=abc.ABCMeta):
  """Base Export Module."""

  def __init__(self,
               params,
               model: Union[tf.Module, tf.keras.Model],
               inference_step: Optional[Callable[..., Any]] = None,
               *,
               preprocessor: Optional[Callable[..., Any]] = None,
               postprocessor: Optional[Callable[..., Any]] = None):
    """Instantiates an ExportModel.

    Examples:

    `inference_step` must be a function that has `model` as an kwarg or the
    second positional argument.
    ```
    def _inference_step(inputs, model=None):
      return model(inputs, training=False)

    module = ExportModule(params, model, inference_step=_inference_step)
    ```

    `preprocessor` and `postprocessor` could be either functions or `tf.Module`.
    The usages of preprocessor and postprocessor are managed by the
    implementation of `serve()` method.

    Args:
      params: A dataclass for parameters to the module.
      model: A model instance which contains weights and forward computation.
      inference_step: An optional callable to forward-pass the model. If not
        specified, it creates a parital function with `model` as an required
        kwarg.
      preprocessor: An optional callable to preprocess the inputs.
      postprocessor: An optional callable to postprocess the model outputs.
    """
    super().__init__(name=None)
    self.model = model
    self.params = params

    if inference_step is not None:
      self.inference_step = functools.partial(inference_step, model=self.model)
    else:
      if issubclass(type(model), tf.keras.Model):
        # Default to self.model.call instead of self.model.__call__ to avoid
        # keras tracing logic designed for training.
        # Since most of Model Garden's call doesn't not have training kwargs
        # or the default is False, we don't pass anything here.
        # Please pass custom inference step if your model has training=True as
        # default.
        self.inference_step = self.model.call
      else:
        self.inference_step = functools.partial(
            self.model.__call__, training=False)
    self.preprocessor = preprocessor
    self.postprocessor = postprocessor

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
           save_options: Optional[tf.saved_model.SaveOptions] = None,
           checkpoint: Optional[tf.train.Checkpoint] = None) -> Text:
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
    checkpoint: An optional tf.train.Checkpoint. If provided, the export module
      will use it to read the weights.

  Returns:
    The savedmodel directory path.
  """
  ckpt_dir_or_file = checkpoint_path
  if ckpt_dir_or_file is not None and tf.io.gfile.isdir(ckpt_dir_or_file):
    ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
  if ckpt_dir_or_file:
    if checkpoint is None:
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
          'If the function_keys is a list, it must contain a single element. %s'
          % function_keys)

  signatures = export_module.get_inference_signatures(function_keys)
  if timestamped:
    export_dir = get_timestamped_export_dir(export_savedmodel_dir).decode(
        'utf-8')
  else:
    export_dir = export_savedmodel_dir
  tf.saved_model.save(
      export_module, export_dir, signatures=signatures, options=save_options)
  return export_dir


def get_timestamped_export_dir(export_dir_base):
  """Builds a path to a new subdirectory within the base directory.

  Args:
    export_dir_base: A string containing a directory to write the exported graph
      and checkpoints.

  Returns:
    The full path of the new subdirectory (which is not actually created yet).

  Raises:
    RuntimeError: if repeated attempts fail to obtain a unique timestamped
      directory name.
  """
  attempts = 0
  while attempts < MAX_DIRECTORY_CREATION_ATTEMPTS:
    timestamp = int(time.time())

    result_dir = tf.io.gfile.join(
        tf.compat.as_bytes(export_dir_base), tf.compat.as_bytes(str(timestamp)))
    if not tf.io.gfile.exists(result_dir):
      # Collisions are still possible (though extremely unlikely): this
      # directory is not actually created yet, but it will be almost
      # instantly on return from this function.
      return result_dir
    time.sleep(1)
    attempts += 1
    logging.warning('Directory %s already exists; retrying (attempt %s/%s)',
                    str(result_dir), attempts, MAX_DIRECTORY_CREATION_ATTEMPTS)
  raise RuntimeError('Failed to obtain a unique export directory name after '
                     f'{MAX_DIRECTORY_CREATION_ATTEMPTS} attempts.')
