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

"""Task for the Ranking model."""

import math
from typing import Dict, List, Optional, Union

import tensorflow as tf
import tensorflow_recommenders as tfrs

from official.core import base_task
from official.core import config_definitions
from official.recommendation.ranking import common
from official.recommendation.ranking.configs import config
from official.recommendation.ranking.data import data_pipeline

RuntimeConfig = config_definitions.RuntimeConfig


def _get_tpu_embedding_feature_config(
    vocab_sizes: List[int],
    embedding_dim: Union[int, List[int]],
    table_name_prefix: str = 'embedding_table'
) -> Dict[str, tf.tpu.experimental.embedding.FeatureConfig]:
  """Returns TPU embedding feature config.

  i'th table config will have vocab size of vocab_sizes[i] and embedding
  dimension of embedding_dim if embedding_dim is an int or embedding_dim[i] if
  embedding_dim is a list).
  Args:
    vocab_sizes: List of sizes of categories/id's in the table.
    embedding_dim: An integer or a list of embedding table dimensions.
    table_name_prefix: a prefix for embedding tables.
  Returns:
    A dictionary of feature_name, FeatureConfig pairs.
  """
  if isinstance(embedding_dim, List):
    if len(vocab_sizes) != len(embedding_dim):
      raise ValueError(
          f'length of vocab_sizes: {len(vocab_sizes)} is not equal to the '
          f'length of embedding_dim: {len(embedding_dim)}')
  elif isinstance(embedding_dim, int):
    embedding_dim = [embedding_dim] * len(vocab_sizes)
  else:
    raise ValueError('embedding_dim is not either a list or an int, got '
                     f'{type(embedding_dim)}')

  feature_config = {}

  for i, vocab_size in enumerate(vocab_sizes):
    table_config = tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=vocab_size,
        dim=embedding_dim[i],
        combiner='mean',
        initializer=tf.initializers.TruncatedNormal(
            mean=0.0, stddev=1 / math.sqrt(embedding_dim[i])),
        name=table_name_prefix + '_%s' % i)
    feature_config[str(i)] = tf.tpu.experimental.embedding.FeatureConfig(
        table=table_config)

  return feature_config


class RankingTask(base_task.Task):
  """A task for Ranking Model."""

  def __init__(self,
               params: config.Task,
               optimizer_config: config.OptimizationConfig,
               logging_dir: Optional[str] = None,
               steps_per_execution: int = 1,
               name: Optional[str] = None):
    """Task initialization.

    Args:
      params: the RankingModel task configuration instance.
      optimizer_config: Optimizer configuration instance.
      logging_dir: a string pointing to where the model, summaries etc. will be
        saved.
      steps_per_execution: Int. Defaults to 1. The number of batches to run
        during each `tf.function` call. It's used for compile/fit API.
      name: the task name.
    """
    super().__init__(params, logging_dir, name=name)
    self._optimizer_config = optimizer_config
    self._steps_per_execution = steps_per_execution

  def build_inputs(self, params, input_context=None):
    """Builds classification input."""

    dataset = data_pipeline.CriteoTsvReader(
        file_pattern=params.input_path,
        params=params,
        vocab_sizes=self.task_config.model.vocab_sizes,
        num_dense_features=self.task_config.model.num_dense_features,
        use_synthetic_data=self.task_config.use_synthetic_data)

    return dataset(input_context)

  @classmethod
  def create_optimizer(cls, optimizer_config: config.OptimizationConfig,
                       runtime_config: Optional[RuntimeConfig] = None) -> None:
    """See base class. Return None, optimizer is set in `build_model`."""
    return None

  def build_model(self) -> tf.keras.Model:
    """Creates Ranking model architecture and Optimizers.

    The RankingModel uses different optimizers/learning rates for embedding
    variables and dense variables.

    Returns:
      A Ranking model instance.
    """
    lr_config = self.optimizer_config.lr_config
    lr_callable = common.WarmUpAndPolyDecay(
        batch_size=self.task_config.train_data.global_batch_size,
        decay_exp=lr_config.decay_exp,
        learning_rate=lr_config.learning_rate,
        warmup_steps=lr_config.warmup_steps,
        decay_steps=lr_config.decay_steps,
        decay_start_steps=lr_config.decay_start_steps)

    dense_optimizer = tf.keras.optimizers.Adam()
    embedding_optimizer = tf.keras.optimizers.get(
        self.optimizer_config.embedding_optimizer)
    embedding_optimizer.learning_rate = lr_callable

    feature_config = _get_tpu_embedding_feature_config(
        embedding_dim=self.task_config.model.embedding_dim,
        vocab_sizes=self.task_config.model.vocab_sizes)

    embedding_layer = tfrs.experimental.layers.embedding.PartialTPUEmbedding(
        feature_config=feature_config,
        optimizer=embedding_optimizer,
        size_threshold=self.task_config.model.size_threshold)

    if self.task_config.model.interaction == 'dot':
      feature_interaction = tfrs.layers.feature_interaction.DotInteraction(
          skip_gather=True)
    elif self.task_config.model.interaction == 'cross':
      feature_interaction = tf.keras.Sequential([
          tf.keras.layers.Concatenate(),
          tfrs.layers.feature_interaction.Cross()
      ])
    else:
      raise ValueError(
          f'params.task.model.interaction {self.task_config.model.interaction} '
          f'is not supported it must be either \'dot\' or \'cross\'.')

    model = tfrs.experimental.models.Ranking(
        embedding_layer=embedding_layer,
        bottom_stack=tfrs.layers.blocks.MLP(
            units=self.task_config.model.bottom_mlp, final_activation='relu'),
        feature_interaction=feature_interaction,
        top_stack=tfrs.layers.blocks.MLP(
            units=self.task_config.model.top_mlp, final_activation='sigmoid'),
    )
    optimizer = tfrs.experimental.optimizers.CompositeOptimizer([
        (embedding_optimizer, lambda: model.embedding_trainable_variables),
        (dense_optimizer, lambda: model.dense_trainable_variables),
    ])

    model.compile(optimizer, steps_per_execution=self._steps_per_execution)
    return model

  def train_step(
      self,
      inputs: Dict[str, tf.Tensor],
      model: tf.keras.Model,
      optimizer: tf.keras.optimizers.Optimizer,
      metrics: Optional[List[tf.keras.metrics.Metric]] = None) -> tf.Tensor:
    """See base class."""
    # All metrics need to be passed through the RankingModel.
    assert metrics == model.metrics
    return model.train_step(inputs)

  def validation_step(
      self,
      inputs: Dict[str, tf.Tensor],
      model: tf.keras.Model,
      metrics: Optional[List[tf.keras.metrics.Metric]] = None) -> tf.Tensor:
    """See base class."""
    # All metrics need to be passed through the RankingModel.
    assert metrics == model.metrics
    return model.test_step(inputs)

  @property
  def optimizer_config(self) -> config.OptimizationConfig:
    return self._optimizer_config


