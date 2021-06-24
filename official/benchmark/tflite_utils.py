# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""TFLite utils."""
import orbit
from official.core import base_task
from official.core import base_trainer
from official.core import config_definitions


def train_and_evaluate(
    params: config_definitions.ExperimentConfig,
    task: base_task.Task,
    trainer: base_trainer.Trainer,
    controller: orbit.Controller):
  """Train and evaluate on TFLite."""
  raise NotImplementedError('train_and_evaluate on tflite_utils is not '
                            'implemented yet.')
