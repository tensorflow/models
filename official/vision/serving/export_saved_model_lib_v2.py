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

r"""Vision models export utility function for serving/inference."""

import os
from typing import Optional, List, Union, Text, Dict

import tensorflow as tf, tf_keras

from official.core import config_definitions as cfg
from official.core import export_base
from official.core import train_utils
from official.vision.serving import export_module_factory


def export(
    input_type: str,
    batch_size: Optional[int],
    input_image_size: List[int],
    params: cfg.ExperimentConfig,
    checkpoint_path: str,
    export_dir: str,
    num_channels: Optional[int] = 3,
    export_module: Optional[export_base.ExportModule] = None,
    export_checkpoint_subdir: Optional[str] = None,
    export_saved_model_subdir: Optional[str] = None,
    function_keys: Optional[Union[List[Text], Dict[Text, Text]]] = None,
    save_options: Optional[tf.saved_model.SaveOptions] = None):
  """Exports the model specified in the exp config.

  Saved model is stored at export_dir/saved_model, checkpoint is saved
  at export_dir/checkpoint, and params is saved at export_dir/params.yaml.

  Args:
    input_type: One of `image_tensor`, `image_bytes`, `tf_example`.
    batch_size: 'int', or None.
    input_image_size: List or Tuple of height and width.
    params: Experiment params.
    checkpoint_path: Trained checkpoint path or directory.
    export_dir: Export directory path.
    num_channels: The number of input image channels.
    export_module: Optional export module to be used instead of using params
      to create one. If None, the params will be used to create an export
      module.
    export_checkpoint_subdir: Optional subdirectory under export_dir
      to store checkpoint.
    export_saved_model_subdir: Optional subdirectory under export_dir
      to store saved model.
    function_keys: a list of string keys to retrieve pre-defined serving
      signatures. The signaute keys will be set with defaults. If a dictionary
      is provided, the values will be used as signature keys.
    save_options: `SaveOptions` for `tf.saved_model.save`.
  """

  if export_checkpoint_subdir:
    output_checkpoint_directory = os.path.join(
        export_dir, export_checkpoint_subdir)
  else:
    output_checkpoint_directory = None

  if export_saved_model_subdir:
    output_saved_model_directory = os.path.join(
        export_dir, export_saved_model_subdir)
  else:
    output_saved_model_directory = export_dir

  export_module = export_module_factory.get_export_module(
      params,
      input_type=input_type,
      batch_size=batch_size,
      input_image_size=input_image_size,
      num_channels=num_channels)

  export_base.export(
      export_module,
      function_keys=function_keys if function_keys else [input_type],
      export_savedmodel_dir=output_saved_model_directory,
      checkpoint_path=checkpoint_path,
      timestamped=False,
      save_options=save_options)

  if output_checkpoint_directory:
    ckpt = tf.train.Checkpoint(model=export_module.model)
    ckpt.save(os.path.join(output_checkpoint_directory, 'ckpt'))
  train_utils.serialize_config(params, export_dir)
