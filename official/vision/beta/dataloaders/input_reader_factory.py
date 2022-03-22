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

"""Factory for getting TF-Vision input readers."""

from official.common import dataset_fn as dataset_fn_util
from official.core import config_definitions as cfg
from official.core import input_reader as core_input_reader

from official.vision.beta.dataloaders import input_reader as vision_input_reader


def input_reader_generator(params: cfg.DataConfig,
                           **kwargs) -> core_input_reader.InputReader:
  """Instantiates an input reader class according to the params.

  Args:
    params: A config_definitions.DataConfig object.
    **kwargs: Additional arguments passed to input reader initialization.

  Returns:
    An InputReader object.

  """
  if params.is_training and params.get('pseudo_label_data', False):
    return vision_input_reader.CombinationDatasetInputReader(
        params,
        pseudo_label_dataset_fn=dataset_fn_util.pick_dataset_fn(
            params.pseudo_label_data.file_type),
        **kwargs)
  else:
    return core_input_reader.InputReader(params, **kwargs)
