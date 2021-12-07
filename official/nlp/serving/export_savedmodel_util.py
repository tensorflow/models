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

"""Common library to export a SavedModel from the export module."""
from typing import Dict, List, Optional, Text, Union

import tensorflow as tf

from official.core import export_base

get_timestamped_export_dir = export_base.get_timestamped_export_dir


def export(export_module: export_base.ExportModule,
           function_keys: Union[List[Text], Dict[Text, Text]],
           export_savedmodel_dir: Text,
           checkpoint_path: Optional[Text] = None,
           timestamped: bool = True) -> Text:
  """Exports to SavedModel format.

  Args:
    export_module: a ExportModule with the keras Model and serving tf.functions.
    function_keys: a list of string keys to retrieve pre-defined serving
      signatures. The signaute keys will be set with defaults. If a dictionary
      is provided, the values will be used as signature keys.
    export_savedmodel_dir: Output saved model directory.
    checkpoint_path: Object-based checkpoint path or directory.
    timestamped: Whether to export the savedmodel to a timestamped directory.

  Returns:
    The savedmodel directory path.
  """
  save_options = tf.saved_model.SaveOptions(function_aliases={
      'tpu_candidate': export_module.serve,
  })
  return export_base.export(export_module, function_keys, export_savedmodel_dir,
                            checkpoint_path, timestamped, save_options)
