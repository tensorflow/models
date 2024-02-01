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

"""Image classification configuration definition."""
import dataclasses

from typing import Optional, Tuple

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.vision.configs import image_classification


@dataclasses.dataclass
class PruningConfig(hyperparams.Config):
  """Pruning parameters.

  Attributes:
    pretrained_original_checkpoint: The pretrained checkpoint location of the
      original model.
    pruning_schedule: A string that indicates the name of `PruningSchedule`
      object that controls pruning rate throughout training. Current available
      options are: `PolynomialDecay` and `ConstantSparsity`.
    begin_step: Step at which to begin pruning.
    end_step: Step at which to end pruning.
    initial_sparsity: Sparsity ratio at which pruning begins.
    final_sparsity: Sparsity ratio at which pruning ends.
    frequency: Number of training steps between sparsity adjustment.
    sparsity_m_by_n: Structured sparsity specification. It specifies m zeros
      over n consecutive weight elements.
  """
  pretrained_original_checkpoint: Optional[str] = None
  pruning_schedule: str = 'PolynomialDecay'
  begin_step: int = 0
  end_step: int = 1000
  initial_sparsity: float = 0.0
  final_sparsity: float = 0.1
  frequency: int = 100
  sparsity_m_by_n: Optional[Tuple[int, int]] = None


@dataclasses.dataclass
class ImageClassificationTask(image_classification.ImageClassificationTask):
  pruning: Optional[PruningConfig] = None


@exp_factory.register_config_factory('resnet_imagenet_pruning')
def image_classification_imagenet() -> cfg.ExperimentConfig:
  """Builds an image classification config for the resnet with pruning."""
  config = image_classification.image_classification_imagenet()
  task = ImageClassificationTask.from_args(
      pruning=PruningConfig(), **config.task.as_dict())
  config.task = task
  runtime = cfg.RuntimeConfig(enable_xla=False)
  config.runtime = runtime

  return config


@exp_factory.register_config_factory('mobilenet_imagenet_pruning')
def image_classification_imagenet_mobilenet() -> cfg.ExperimentConfig:
  """Builds an image classification config for the mobilenetV2 with pruning."""
  config = image_classification.image_classification_imagenet_mobilenet()
  task = ImageClassificationTask.from_args(
      pruning=PruningConfig(), **config.task.as_dict())
  config.task = task

  return config
