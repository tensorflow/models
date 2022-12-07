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

"""Fine-tuning configuration definition."""

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.nlp.configs import finetuning_experiments
from official.projects.qat.nlp.tasks import question_answering


@exp_factory.register_config_factory('bert/squad_qat')
def bert_squad() -> cfg.ExperimentConfig:
  """BERT Squad V1/V2 with QAT."""
  config = finetuning_experiments.bert_squad()
  task = question_answering.QuantizedModelQAConfig.from_args(
      **config.task.as_dict())

  # Copy QADataConfig objects.
  task.train_data = config.task.train_data
  task.validation_data = config.task.validation_data
  config.task = task

  return config
