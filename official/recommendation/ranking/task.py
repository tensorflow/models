# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
from typing import Dict, List, Optional, Union, Tuple

import tensorflow as tf, tf_keras
import tensorflow_recommenders as tfrs

from official.core import base_task
from official.core import config_definitions
from official.recommendation.ranking import common
from official.recommendation.ranking.configs import config
from official.recommendation.ranking.data import data_pipeline
from official.recommendation.ranking.data import data_pipeline_multi_hot


RuntimeConfig = config_definitions.RuntimeConfig


def _get_tpu_embedding_feature_config(
    vocab_sizes: List[int],
    embedding_dim: Union[int, List[int]],
    table_name_prefix: str = 'embedding_table',
    batch_size: Optional[int] = None,
    max_ids_per_chip_per_sample: Optional[int] = None,
    max_ids_per_table: Optional[Union[int, List[int]]] = None,
    max_unique_ids_per_table: Optional[Union[int, List[int]]] = None,
    allow_id_dropping: bool = False,
    initialize_tables_on_host: bool = False,
) -> Tuple[
    Dict[str, tf.tpu.experimental.embedding.FeatureConfig],
    Optional[tf.tpu.experimental.embedding.SparseCoreEmbeddingConfig],
]:
  """Returns TPU embedding feature config.

  i'th table config will have vocab size of vocab_sizes[i] and embedding
  dimension of embedding_dim if embedding_dim is an int or embedding_dim[i] if
  embedding_dim is a list).
  Args:
    vocab_sizes: List of sizes of categories/id's in the table.
    embedding_dim: An integer or a list of embedding table dimensions.
    table_name_prefix: a prefix for embedding tables.
    batch_size: Per-replica batch size.
    max_ids_per_chip_per_sample: Maximum number of embedding ids per chip per
      sample.
    max_ids_per_table: Maximum number of embedding ids per table.
    max_unique_ids_per_table: Maximum number of unique embedding ids per table.
    allow_id_dropping: bool to allow id dropping.
    initialize_tables_on_host: bool : if the embedding table size is more than 
      what HBM can handle, this flag will help initialize the full embedding
      tables on host and then copy shards to HBM.

  Returns:
    A dictionary of feature_name, FeatureConfig pairs.
  """
  if isinstance(embedding_dim, List):
    if len(vocab_sizes) != len(embedding_dim):
      raise ValueError(
          f'length of vocab_sizes: {len(vocab_sizes)} is not equal to the '
          f'length of embedding_dim: {len(embedding_dim)}'
      )
  elif isinstance(embedding_dim, int):
    embedding_dim = [embedding_dim] * len(vocab_sizes)
  else:
    raise ValueError(
        'embedding_dim is not either a list or an int, got '
        f'{type(embedding_dim)}'
    )

  if isinstance(max_ids_per_table, List):
    if len(vocab_sizes) != len(max_ids_per_table):
      raise ValueError(
          f'length of vocab_sizes: {len(vocab_sizes)} is not equal to the '
          f'length of max_ids_per_table: {len(max_ids_per_table)}'
      )
  elif isinstance(max_ids_per_table, int):
    max_ids_per_table = [max_ids_per_table] * len(vocab_sizes)
  elif max_ids_per_table is not None:
    raise ValueError(
        'max_ids_per_table is not either a list or an int or None, got '
        f'{type(max_ids_per_table)}'
    )

  if isinstance(max_unique_ids_per_table, List):
    if len(vocab_sizes) != len(max_unique_ids_per_table):
      raise ValueError(
          f'length of vocab_sizes: {len(vocab_sizes)} is not equal to the '
          'length of max_unique_ids_per_table: '
          f'{len(max_unique_ids_per_table)}'
      )
  elif isinstance(max_unique_ids_per_table, int):
    max_unique_ids_per_table = [max_unique_ids_per_table] * len(vocab_sizes)
  elif max_unique_ids_per_table is not None:
    raise ValueError(
        'max_unique_ids_per_table is not either a list or an int or None, '
        f'got {type(max_unique_ids_per_table)}'
    )

  feature_config = {}
  sparsecore_config = None
  max_ids_per_table_dict = {}
  max_unique_ids_per_table_dict = {}

  for i, vocab_size in enumerate(vocab_sizes):
    table_config = tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=vocab_size,
        dim=embedding_dim[i],
        combiner='mean',
        initializer=tf.initializers.TruncatedNormal(
            mean=0.0, stddev=1 / math.sqrt(embedding_dim[i])
        ),
        name=table_name_prefix + '_%02d' % i,
    )
    feature_config[str(i)] = tf.tpu.experimental.embedding.FeatureConfig(
        name=str(i),
        table=table_config,
        output_shape=[batch_size] if batch_size else None,
    )
    if max_ids_per_table:
      max_ids_per_table_dict[str(table_name_prefix + '_%02d' % i)] = (
          max_ids_per_table[i]
      )
    if max_unique_ids_per_table:
      max_unique_ids_per_table_dict[str(table_name_prefix + '_%02d' % i)] = (
          max_unique_ids_per_table[i]
      )

  if all((max_ids_per_chip_per_sample, max_ids_per_table,
          max_unique_ids_per_table)):
    sparsecore_config = tf.tpu.experimental.embedding.SparseCoreEmbeddingConfig(
        disable_table_stacking=False,
        max_ids_per_chip_per_sample=max_ids_per_chip_per_sample,
        max_ids_per_table=max_ids_per_table_dict,
        max_unique_ids_per_table=max_unique_ids_per_table_dict,
        allow_id_dropping=allow_id_dropping,
        initialize_tables_on_host=initialize_tables_on_host,
    )

  return feature_config, sparsecore_config


class RankingTask(base_task.Task):
  """A task for Ranking Model."""

  def __init__(self,
               params: config.Task,
               trainer_config: config.TrainerConfig,
               logging_dir: Optional[str] = None,
               steps_per_execution: int = 1,
               name: Optional[str] = None):
    """Task initialization.

    Args:
      params: the RankingModel task configuration instance.
      trainer_config: Trainer configuration instance.
      logging_dir: a string pointing to where the model, summaries etc. will be
        saved.
      steps_per_execution: Int. Defaults to 1. The number of batches to run
        during each `tf.function` call. It's used for compile/fit API.
      name: the task name.
    """
    super().__init__(params, logging_dir, name=name)
    self._trainer_config = trainer_config
    self._optimizer_config = trainer_config.optimizer_config
    self._steps_per_execution = steps_per_execution

  def build_inputs(self, params, input_context=None):
    """Builds classification input."""
    if self.task_config.model.use_multi_hot:
      if self.task_config.use_tf_record_reader:
        dataset = data_pipeline_multi_hot.CriteoTFRecordReader(
            file_pattern=params.input_path,
            params=params,
            vocab_sizes=self.task_config.model.vocab_sizes,
            multi_hot_sizes=self.task_config.model.multi_hot_sizes,
            num_dense_features=self.task_config.model.num_dense_features)
      else:
        dataset = data_pipeline_multi_hot.CriteoTsvReaderMultiHot(
            file_pattern=params.input_path,
            params=params,
            vocab_sizes=self.task_config.model.vocab_sizes,
            multi_hot_sizes=self.task_config.model.multi_hot_sizes,
            num_dense_features=self.task_config.model.num_dense_features,
            use_synthetic_data=self.task_config.use_synthetic_data)
    else:
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

  def build_model(self) -> tf_keras.Model:
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
    embedding_optimizer = tf_keras.optimizers.get(
        self.optimizer_config.embedding_optimizer, use_legacy_optimizer=True)
    embedding_optimizer.learning_rate = lr_callable

    dense_optimizer = tf_keras.optimizers.get(
        self.optimizer_config.dense_optimizer, use_legacy_optimizer=True)
    if self.optimizer_config.dense_optimizer == 'SGD':
      dense_lr_config = self.optimizer_config.dense_sgd_config
      dense_lr_callable = common.WarmUpAndPolyDecay(
          batch_size=self.task_config.train_data.global_batch_size,
          decay_exp=dense_lr_config.decay_exp,
          learning_rate=dense_lr_config.learning_rate,
          warmup_steps=dense_lr_config.warmup_steps,
          decay_steps=dense_lr_config.decay_steps,
          decay_start_steps=dense_lr_config.decay_start_steps)
      dense_optimizer.learning_rate = dense_lr_callable

    feature_config, sparse_core_embedding_config = (
        _get_tpu_embedding_feature_config(
            embedding_dim=self.task_config.model.embedding_dim,
            vocab_sizes=self.task_config.model.vocab_sizes,
            batch_size=self.task_config.train_data.global_batch_size
            // tf.distribute.get_strategy().num_replicas_in_sync,
            max_ids_per_chip_per_sample=self.task_config.model.max_ids_per_chip_per_sample,
            max_ids_per_table=self.task_config.model.max_ids_per_table,
            max_unique_ids_per_table=self.task_config.model.max_unique_ids_per_table,
            allow_id_dropping=self.task_config.model.allow_id_dropping,
            initialize_tables_on_host=self.task_config.model.initialize_tables_on_host,
        )
    )

    # to work around PartialTPUEmbedding issue in v5p and to enable multi hot
    # features
    if self.task_config.model.use_partial_tpu_embedding:
      embedding_layer = tfrs.experimental.layers.embedding.PartialTPUEmbedding(
          feature_config=feature_config,
          optimizer=embedding_optimizer,
          pipeline_execution_with_tensor_core=self.trainer_config.pipeline_sparse_and_dense_execution,
          size_threshold=self.task_config.model.size_threshold,
      )
    else:
      embedding_layer = tfrs.layers.embedding.tpu_embedding_layer.TPUEmbedding(
          feature_config=feature_config,
          optimizer=embedding_optimizer,
          pipeline_execution_with_tensor_core=self.trainer_config.pipeline_sparse_and_dense_execution,
          sparse_core_embedding_config=sparse_core_embedding_config,
      )

    if self.task_config.model.interaction == 'dot':
      feature_interaction = tfrs.layers.feature_interaction.DotInteraction(
          skip_gather=True)
    elif self.task_config.model.interaction == 'cross':
      feature_interaction = tf_keras.Sequential([
          tf_keras.layers.Concatenate(),
          tfrs.layers.feature_interaction.Cross()
      ])
    elif self.task_config.model.interaction == 'multi_layer_dcn':
      feature_interaction = tf_keras.Sequential([
          tf_keras.layers.Concatenate(),
          tfrs.layers.feature_interaction.MultiLayerDCN(
              projection_dim=self.task_config.model.dcn_low_rank_dim,
              num_layers=self.task_config.model.dcn_num_layers,
              use_bias=self.task_config.model.dcn_use_bias,
              kernel_initializer=self.task_config.model.dcn_kernel_initializer,
              bias_initializer=self.task_config.model.dcn_bias_initializer,
          ),
      ])
    else:
      raise ValueError(
          f' {self.task_config.model.interaction} is not supported it must be'
          " either 'dot' or 'cross' or 'multi_layer_dcn'."
      )

    model = tfrs.experimental.models.Ranking(
        embedding_layer=embedding_layer,
        bottom_stack=tfrs.layers.blocks.MLP(
            units=self.task_config.model.bottom_mlp, final_activation='relu'
        ),
        feature_interaction=feature_interaction,
        top_stack=tfrs.layers.blocks.MLP(
            units=self.task_config.model.top_mlp, final_activation='sigmoid'
        ),
        concat_dense=self.task_config.model.concat_dense,
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
      model: tf_keras.Model,
      optimizer: tf_keras.optimizers.Optimizer,
      metrics: Optional[List[tf_keras.metrics.Metric]] = None) -> tf.Tensor:
    """See base class."""
    # All metrics need to be passed through the RankingModel.
    assert metrics == model.metrics
    return model.train_step(inputs)

  def validation_step(
      self,
      inputs: Dict[str, tf.Tensor],
      model: tf_keras.Model,
      metrics: Optional[List[tf_keras.metrics.Metric]] = None) -> tf.Tensor:
    """See base class."""
    # All metrics need to be passed through the RankingModel.
    assert metrics == model.metrics
    return model.test_step(inputs)

  @property
  def trainer_config(self) -> config.TrainerConfig:
    return self._trainer_config

  @property
  def optimizer_config(self) -> config.OptimizationConfig:
    return self._optimizer_config
