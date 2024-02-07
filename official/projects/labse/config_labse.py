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

# pylint: disable=g-doc-return-or-yield,line-too-long
"""LaBSE configurations."""
import dataclasses
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.nlp.data import dual_encoder_dataloader
from official.nlp.tasks import dual_encoder

AdamWeightDecay = optimization.AdamWeightDecayConfig
PolynomialLr = optimization.PolynomialLrConfig
PolynomialWarmupConfig = optimization.PolynomialWarmupConfig


@dataclasses.dataclass
class LaBSEOptimizationConfig(optimization.OptimizationConfig):
  """Bert optimization config."""
  optimizer: optimization.OptimizerConfig = dataclasses.field(
      default_factory=lambda: optimization.OptimizerConfig(  # pylint: disable=g-long-lambda
          type="adamw", adamw=AdamWeightDecay()
      )
  )
  learning_rate: optimization.LrConfig = dataclasses.field(
      default_factory=lambda: optimization.LrConfig(  # pylint: disable=g-long-lambda
          type="polynomial",
          polynomial=PolynomialLr(
              initial_learning_rate=1e-4,
              decay_steps=1000000,
              end_learning_rate=0.0,
          ),
      )
  )
  warmup: optimization.WarmupConfig = dataclasses.field(
      default_factory=lambda: optimization.WarmupConfig(  # pylint: disable=g-long-lambda
          type="polynomial",
          polynomial=PolynomialWarmupConfig(warmup_steps=10000),
      )
  )


@exp_factory.register_config_factory("labse/train")
def labse_train() -> cfg.ExperimentConfig:
  r"""Language-agnostic bert sentence embedding.

  *Note*: this experiment does not use cross-accelerator global softmax so it
  does not reproduce the exact LABSE training.
  """
  config = cfg.ExperimentConfig(
      task=dual_encoder.DualEncoderConfig(
          train_data=dual_encoder_dataloader.DualEncoderDataConfig(),
          validation_data=dual_encoder_dataloader.DualEncoderDataConfig(
              is_training=False, drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          optimizer_config=LaBSEOptimizationConfig(
              learning_rate=optimization.LrConfig(
                  type="polynomial",
                  polynomial=PolynomialLr(
                      initial_learning_rate=3e-5, end_learning_rate=0.0)),
              warmup=optimization.WarmupConfig(
                  type="polynomial", polynomial=PolynomialWarmupConfig()))),
      restrictions=[
          "task.train_data.is_training != None",
          "task.validation_data.is_training != None"
      ])
  return config
