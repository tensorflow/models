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

# pylint: disable=g-doc-return-or-yield,line-too-long
"""TEAMS experiments."""
import dataclasses

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader
from official.nlp.data import question_answering_dataloader
from official.nlp.data import sentence_prediction_dataloader
from official.nlp.tasks import question_answering
from official.nlp.tasks import sentence_prediction
from official.projects.teams import teams
from official.projects.teams import teams_task

AdamWeightDecay = optimization.AdamWeightDecayConfig
PolynomialLr = optimization.PolynomialLrConfig
PolynomialWarmupConfig = optimization.PolynomialWarmupConfig


@dataclasses.dataclass
class TeamsOptimizationConfig(optimization.OptimizationConfig):
  """TEAMS optimization config."""
  optimizer: optimization.OptimizerConfig = optimization.OptimizerConfig(
      type="adamw",
      adamw=AdamWeightDecay(
          weight_decay_rate=0.01,
          exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
          epsilon=1e-6))
  learning_rate: optimization.LrConfig = optimization.LrConfig(
      type="polynomial",
      polynomial=PolynomialLr(
          initial_learning_rate=1e-4,
          decay_steps=1000000,
          end_learning_rate=0.0))
  warmup: optimization.WarmupConfig = optimization.WarmupConfig(
      type="polynomial", polynomial=PolynomialWarmupConfig(warmup_steps=10000))


@exp_factory.register_config_factory("teams/pretraining")
def teams_pretrain() -> cfg.ExperimentConfig:
  """TEAMS pretraining."""
  config = cfg.ExperimentConfig(
      task=teams_task.TeamsPretrainTaskConfig(
          train_data=pretrain_dataloader.BertPretrainDataConfig(),
          validation_data=pretrain_dataloader.BertPretrainDataConfig(
              is_training=False)),
      trainer=cfg.TrainerConfig(
          optimizer_config=TeamsOptimizationConfig(), train_steps=1000000),
      restrictions=[
          "task.train_data.is_training != None",
          "task.validation_data.is_training != None"
      ])
  return config


@exp_factory.register_config_factory("teams/sentence_prediction")
def teams_sentence_prediction() -> cfg.ExperimentConfig:
  r"""Teams GLUE."""
  config = cfg.ExperimentConfig(
      task=sentence_prediction.SentencePredictionConfig(
          model=sentence_prediction.ModelConfig(
              encoder=encoders.EncoderConfig(
                  type="any", any=teams.TeamsEncoderConfig(num_layers=1))),
          train_data=sentence_prediction_dataloader
          .SentencePredictionDataConfig(),
          validation_data=sentence_prediction_dataloader
          .SentencePredictionDataConfig(
              is_training=False, drop_remainder=False)),
      trainer=cfg.TrainerConfig(optimizer_config=TeamsOptimizationConfig()),
      restrictions=[
          "task.train_data.is_training != None",
          "task.validation_data.is_training != None"
      ])
  return config


@exp_factory.register_config_factory("teams/squad")
def teams_squad() -> cfg.ExperimentConfig:
  """Teams Squad V1/V2."""
  config = cfg.ExperimentConfig(
      task=question_answering.QuestionAnsweringConfig(
          model=question_answering.ModelConfig(
              encoder=encoders.EncoderConfig(
                  type="any", any=teams.TeamsEncoderConfig(num_layers=1))),
          train_data=question_answering_dataloader.QADataConfig(),
          validation_data=question_answering_dataloader.QADataConfig()),
      trainer=cfg.TrainerConfig(optimizer_config=TeamsOptimizationConfig()),
      restrictions=[
          "task.train_data.is_training != None",
          "task.validation_data.is_training != None"
      ])
  return config
