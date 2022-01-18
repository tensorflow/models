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

"""Configuration definitions for multi-task training."""
from typing import Optional, Tuple

import dataclasses

from official.core import config_definitions as cfg
from official.modeling import hyperparams


@dataclasses.dataclass
class TaskRoutine(hyperparams.Config):
  # TODO(hongkuny): deprecate the task_name once we migrated client code.
  task_name: str = ""
  task_config: cfg.TaskConfig = None
  eval_steps: Optional[int] = None
  task_weight: Optional[float] = 1.0


@dataclasses.dataclass
class MultiTaskConfig(hyperparams.Config):
  init_checkpoint: str = ""
  model: hyperparams.Config = None
  task_routines: Tuple[TaskRoutine, ...] = ()


@dataclasses.dataclass
class ProportionalSampleConfig(hyperparams.Config):
  alpha: float = 1.0


@dataclasses.dataclass
class AnnealingSampleConfig(hyperparams.Config):
  steps_per_epoch: int = 5
  total_steps: int = 20


@dataclasses.dataclass
class TaskSamplingConfig(hyperparams.OneOfConfig):
  type: str = ""
  uniform: hyperparams.Config = hyperparams.Config()
  proportional: ProportionalSampleConfig = ProportionalSampleConfig()
  annealing: AnnealingSampleConfig = AnnealingSampleConfig()


@dataclasses.dataclass
class MultiTaskTrainerConfig(cfg.TrainerConfig):
  trainer_type: str = "interleaving"
  task_sampler: TaskSamplingConfig = TaskSamplingConfig(type="proportional")


@dataclasses.dataclass
class MultiTaskExperimentConfig(hyperparams.Config):
  """An experiment config for multi-task training and multi-task evaluation."""
  task: MultiTaskConfig = MultiTaskConfig()
  trainer: MultiTaskTrainerConfig = MultiTaskTrainerConfig()
  runtime: cfg.RuntimeConfig = cfg.RuntimeConfig()


@dataclasses.dataclass
class MultiEvalExperimentConfig(cfg.ExperimentConfig):
  """An experiment config for single-task training and multi-task evaluation.

  Attributes:
    eval_tasks: individual evaluation tasks.
  """
  eval_tasks: Tuple[TaskRoutine, ...] = ()
