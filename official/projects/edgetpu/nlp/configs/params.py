# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Datastructures for all the configurations for MobileBERT-EdgeTPU training."""
import dataclasses
from typing import Optional

from official.modeling import optimization
from official.modeling.hyperparams import base_config
from official.nlp.configs import bert
from official.nlp.data import pretrain_dataloader

DatasetParams = pretrain_dataloader.BertPretrainDataConfig
PretrainerModelParams = bert.PretrainerConfig


@dataclasses.dataclass
class OrbitParams(base_config.Config):
  """Parameters that setup Orbit training/evaluation pipeline.

  Attributes:
    mode: Orbit controller mode, can be 'train', 'train_and_evaluate', or
      'evaluate'.
    steps_per_loop: The number of steps to run in each inner loop of training.
    total_steps: The global step count to train up to.
    eval_steps: The number of steps to run during an evaluation. If -1, this
      method will evaluate over the entire evaluation dataset.
    eval_interval: The number of training steps to run between evaluations. If
      set, training will always stop every `eval_interval` steps, even if this
      results in a shorter inner loop than specified by `steps_per_loop`
      setting. If None, evaluation will only be performed after training is
      complete.
  """
  mode: str = 'train'
  steps_per_loop: int = 1000
  total_steps: int = 1000000
  eval_steps: int = -1
  eval_interval: Optional[int] = None


@dataclasses.dataclass
class OptimizerParams(optimization.OptimizationConfig):
  """Optimizer parameters for MobileBERT-EdgeTPU."""
  optimizer: optimization.OptimizerConfig = dataclasses.field(
      default_factory=lambda: optimization.OptimizerConfig(  # pylint: disable=g-long-lambda
          type='adamw',
          adamw=optimization.AdamWeightDecayConfig(
              weight_decay_rate=0.01,
              exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'],
          ),
      )
  )
  learning_rate: optimization.LrConfig = dataclasses.field(
      default_factory=lambda: optimization.LrConfig(  # pylint: disable=g-long-lambda
          type='polynomial',
          polynomial=optimization.PolynomialLrConfig(
              initial_learning_rate=1e-4,
              decay_steps=1000000,
              end_learning_rate=0.0,
          ),
      )
  )
  warmup: optimization.WarmupConfig = dataclasses.field(
      default_factory=lambda: optimization.WarmupConfig(  # pylint: disable=g-long-lambda
          type='polynomial',
          polynomial=optimization.PolynomialWarmupConfig(warmup_steps=10000),
      )
  )


@dataclasses.dataclass
class RuntimeParams(base_config.Config):
  """Parameters that set up the training runtime.

  TODO(longy): Can reuse the Runtime Config in:
  official/core/config_definitions.py

  Attributes
    distribution_strategy: Keras distribution strategy
    use_gpu: Whether to use GPU
    use_tpu: Whether to use TPU
    num_gpus: Number of gpus to use for training
    num_workers: Number of parallel workers
    tpu_address: The bns address of the TPU to use.
  """
  distribution_strategy: str = 'off'
  num_gpus: Optional[int] = 0
  all_reduce_alg: Optional[str] = None
  num_workers: int = 1
  tpu_address: str = ''
  use_gpu: Optional[bool] = None
  use_tpu: Optional[bool] = None


@dataclasses.dataclass
class LayerWiseDistillationParams(base_config.Config):
  """Define the behavior of layer-wise distillation.

  Layer-wise distillation is an optional step where the knowledge is transferred
  layerwisely for all the transformer layers. The end-to-end distillation is
  performed after layer-wise distillation if layer-wise distillation steps is
  not zero.
  """
  num_steps: int = 10000
  warmup_steps: int = 10000
  initial_learning_rate: float = 1.5e-3
  end_learning_rate: float = 1.5e-3
  decay_steps: int = 10000
  hidden_distill_factor: float = 100.0
  beta_distill_factor: float = 5000.0
  gamma_distill_factor: float = 5.0
  attention_distill_factor: float = 1.0


@dataclasses.dataclass
class EndToEndDistillationParams(base_config.Config):
  """Define the behavior of end2end pretrainer distillation."""
  num_steps: int = 580000
  warmup_steps: int = 20000
  initial_learning_rate: float = 1.5e-3
  end_learning_rate: float = 1.5e-7
  decay_steps: int = 580000
  distill_ground_truth_ratio: float = 0.5


@dataclasses.dataclass
class EdgeTPUBERTCustomParams(base_config.Config):
  """EdgeTPU-BERT custom params.

  Attributes:
    train_dataset: An instance of the DatasetParams.
    eval_dataset: An instance of the DatasetParams.
    teacher_model: An instance of the PretrainerModelParams. If None, then the
      student model is trained independently without distillation.
    student_model: An instance of the PretrainerModelParams
    teacher_model_init_checkpoint: Path for the teacher model init checkpoint.
    student_model_init_checkpoint: Path for the student model init checkpoint.
    layer_wise_distillation: Distillation config for the layer-wise step.
    end_to_end_distillation: Distillation config for the end2end step.
    optimizer: An instance of the OptimizerParams.
    runtime: An instance of the RuntimeParams.
    learning_rate: An instance of the LearningRateParams.
    orbit_config: An instance of the OrbitParams.
    distill_ground_truth_ratio: A float number representing the ratio between
      distillation output and ground truth.
  """
  train_datasest: DatasetParams = dataclasses.field(
      default_factory=DatasetParams
  )
  eval_dataset: DatasetParams = dataclasses.field(default_factory=DatasetParams)
  teacher_model: Optional[PretrainerModelParams] = dataclasses.field(
      default_factory=PretrainerModelParams
  )
  student_model: PretrainerModelParams = dataclasses.field(
      default_factory=PretrainerModelParams
  )
  teacher_model_init_checkpoint: str = ''
  student_model_init_checkpoint: str = ''
  layer_wise_distillation: LayerWiseDistillationParams = dataclasses.field(
      default_factory=LayerWiseDistillationParams
  )
  end_to_end_distillation: EndToEndDistillationParams = dataclasses.field(
      default_factory=EndToEndDistillationParams
  )
  optimizer: OptimizerParams = dataclasses.field(
      default_factory=OptimizerParams
  )
  runtime: RuntimeParams = dataclasses.field(default_factory=RuntimeParams)
  orbit_config: OrbitParams = dataclasses.field(default_factory=OrbitParams)
