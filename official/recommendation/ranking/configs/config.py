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

"""Ranking Model configuration definition."""
import dataclasses
from typing import List, Optional, Union
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams


@dataclasses.dataclass
class CallbacksConfig(hyperparams.Config):
  """Configuration for Callbacks.

  Attributes:
    enable_checkpoint_and_export: Whether or not to enable checkpoints as a
      Callback. Defaults to True.
    enable_backup_and_restore: Whether or not to add BackupAndRestore
      callback. Defaults to True.
    enable_tensorboard: Whether or not to enable TensorBoard as a Callback.
      Defaults to True.
    enable_time_history: Whether or not to enable TimeHistory Callbacks.
      Defaults to True.
  """
  enable_checkpoint_and_export: bool = True
  enable_backup_and_restore: bool = False
  enable_tensorboard: bool = True
  enable_time_history: bool = True


@dataclasses.dataclass
class LearningRateConfig(hyperparams.Config):
  """Learning rate scheduler config."""
  learning_rate: float = 1.25
  warmup_steps: int = 8000
  decay_steps: int = 30000
  decay_start_steps: int = 70000
  decay_exp: float = 2


@dataclasses.dataclass
class OptimizationConfig(hyperparams.Config):
  """Embedding and dense optimizer configs."""
  lr_config: LearningRateConfig = dataclasses.field(
      default_factory=LearningRateConfig
  )
  dense_sgd_config: LearningRateConfig = dataclasses.field(
      default_factory=lambda: LearningRateConfig(warmup_steps=0)
  )
  embedding_optimizer: str = 'SGD'
  dense_optimizer: str = 'Adam'


@dataclasses.dataclass
class DataConfig(hyperparams.Config):
  """Dataset config for training and evaluation."""
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = True
  dtype: str = 'float32'
  shuffle_buffer_size: int = 10000
  cycle_length: int = 10
  sharding: bool = True
  num_shards_per_host: int = 8
  use_cached_data: bool = False


@dataclasses.dataclass
class ModelConfig(hyperparams.Config):
  """Configuration for training.

  Attributes:
    num_dense_features: Number of dense features.
    vocab_sizes: Vocab sizes for each of the sparse features. The order agrees
      with the order of the input data.
    use_multi_hot: Flag to determine if enabling multi-hot data loading and 
      training used for DLRM V2
    multi_hot_sizes: List to pass in the multi hot size of each sparse embedding
      feature
    embedding_dim: An integer or a list of embedding table dimensions.
      If it's an integer then all tables will have the same embedding dimension.
      If it's a list then the length should match with `vocab_sizes`.
    size_threshold: A threshold for table sizes below which a keras
        embedding layer is used, and above which a TPU embedding layer is used.
        If it's -1 then only keras embedding layer will be used for all tables,
        if 0 only then only TPU embedding layer will be used.
    bottom_mlp: The sizes of hidden layers for bottom MLP applied to dense
      features.
    top_mlp: The sizes of hidden layers for top MLP.
    interaction: Interaction can be on of the following:
     'dot', 'cross', 'multi_layer_dcn'.
    concat_dense: Weather to concatenate output from interaction module with 
      dense output again
    dcn_num_layers: Number of Stacked DCN layers used in the dcn interaction 
      module
    dcn_low_rank_dim: Project dimension in stacked DCN layers for dcn
      interaction module
    dcn_kernel_initializer: the kernel initializer used for the dcn interaction
      module
    dcn_bias_initializer: the bias initializer used for the dcn interaction
      module
    dcn_use_bias: Flag to determine whether to use bias for the dcn interaction
      module
    use_partial_tpu_embedding: Flag to determine whether to use partial tpu
      embedding layer or not.
    max_ids_per_chip_per_sample: Maximum number of ids per chip per sample.
    max_ids_per_table: Maximum number of ids per table.
    max_unique_ids_per_table: Maximum number of unique ids per table.
  """
  num_dense_features: int = 13
  vocab_sizes: List[int] = dataclasses.field(default_factory=list)
  use_multi_hot: bool = False
  multi_hot_sizes: List[int] = dataclasses.field(default_factory=list)
  embedding_dim: Union[int, List[int]] = 8
  size_threshold: int = 50_000
  bottom_mlp: List[int] = dataclasses.field(default_factory=list)
  top_mlp: List[int] = dataclasses.field(default_factory=list)
  interaction: str = 'dot'
  concat_dense: bool = True
  dcn_num_layers: int = 3
  dcn_low_rank_dim: int = 512
  dcn_kernel_initializer: str = 'truncated_normal'
  dcn_bias_initializer: str = 'zeros'
  dcn_use_bias: bool = True
  use_partial_tpu_embedding: bool = True
  max_ids_per_chip_per_sample: int | None = None
  max_ids_per_table: Union[int, List[int]] | None = None
  max_unique_ids_per_table: Union[int, List[int]] | None = None
  allow_id_dropping: bool = False
  initialize_tables_on_host: bool = False


@dataclasses.dataclass
class Loss(hyperparams.Config):
  """Configuration for Loss.

  Attributes:
    label_smoothing: Whether or not to apply label smoothing to the
    Binary Crossentropy loss.
  """
  label_smoothing: float = 0.0


@dataclasses.dataclass
class Task(hyperparams.Config):
  """The model config."""
  init_checkpoint: str = ''
  model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
  train_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(is_training=True)
  )
  validation_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(is_training=False)
  )
  loss: Loss = dataclasses.field(default_factory=Loss)
  use_synthetic_data: bool = False
  use_tf_record_reader: bool = False


@dataclasses.dataclass
class TimeHistoryConfig(hyperparams.Config):
  """Configuration for the TimeHistory callback.

  Attributes:
    log_steps: Interval of steps between logging of batch level stats.
  """
  log_steps: Optional[int] = None


@dataclasses.dataclass
class TrainerConfig(cfg.TrainerConfig):
  """Configuration for training.

  Attributes:
    train_steps: The number of steps used to train.
    validation_steps: The number of steps used to eval.
    validation_interval: The Number of training steps to run between
      evaluations.
    callbacks: An instance of CallbacksConfig.
    use_orbit: Whether to use orbit library with custom training loop or
      compile/fit API.
    enable_metrics_in_training: Whether to enable metrics during training.
    time_history: Config of TimeHistory callback.
    optimizer_config: An `OptimizerConfig` instance for embedding optimizer.
       Defaults to None.
    pipeline_sparse_and_dense_exeuction: Whether to pipeline embedding and
      dense execution. This is a performance optimization.
  """
  train_steps: int = 0
  # Sets validation steps to be -1 to evaluate the entire dataset.
  validation_steps: int = -1
  validation_interval: int = 70000
  callbacks: CallbacksConfig = dataclasses.field(
      default_factory=CallbacksConfig
  )
  use_orbit: bool = False
  enable_metrics_in_training: bool = True
  time_history: TimeHistoryConfig = dataclasses.field(
      default_factory=lambda: TimeHistoryConfig(log_steps=5000)
  )
  optimizer_config: OptimizationConfig = dataclasses.field(
      default_factory=OptimizationConfig
  )
  pipeline_sparse_and_dense_execution: bool = False


NUM_TRAIN_EXAMPLES = 4195197692
NUM_EVAL_EXAMPLES = 89137318

train_batch_size = 16384
eval_batch_size = 16384
steps_per_epoch = NUM_TRAIN_EXAMPLES // train_batch_size
vocab_sizes = [
    39884406,
    39043,
    17289,
    7420,
    20263,
    3,
    7120,
    1543,
    63,
    38532951,
    2953546,
    403346,
    10,
    2208,
    11938,
    155,
    4,
    976,
    14,
    39979771,
    25641295,
    39664984,
    585935,
    12972,
    108,
    36,
]
multi_hot_sizes = [
    3,
    2,
    1,
    2,
    6,
    1,
    1,
    1,
    1,
    7,
    3,
    8,
    1,
    6,
    9,
    5,
    1,
    1,
    1,
    12,
    100,
    27,
    10,
    3,
    1,
    1,
]


@dataclasses.dataclass
class Config(hyperparams.Config):
  """Configuration to train the RankingModel.

  By default it configures DLRM model on criteo dataset.

  Attributes:
    runtime: A `RuntimeConfig` instance.
    task: `Task` instance.
    trainer: A `TrainerConfig` instance.
  """
  runtime: cfg.RuntimeConfig = dataclasses.field(
      default_factory=cfg.RuntimeConfig
  )
  task: Task = dataclasses.field(
      default_factory=lambda: Task(  # pylint: disable=g-long-lambda
          model=ModelConfig(
              embedding_dim=8,
              vocab_sizes=vocab_sizes,
              bottom_mlp=[64, 32, 8],
              top_mlp=[64, 32, 1],
          ),
          loss=Loss(label_smoothing=0.0),
          train_data=DataConfig(
              is_training=True, global_batch_size=train_batch_size
          ),
          validation_data=DataConfig(
              is_training=False, global_batch_size=eval_batch_size
          ),
      )
  )
  trainer: TrainerConfig = dataclasses.field(
      default_factory=lambda: TrainerConfig(  # pylint: disable=g-long-lambda
          train_steps=2 * steps_per_epoch,
          validation_interval=steps_per_epoch,
          validation_steps=NUM_EVAL_EXAMPLES // eval_batch_size,
          enable_metrics_in_training=True,
          optimizer_config=OptimizationConfig(),
      )
  )
  restrictions: dataclasses.InitVar[Optional[List[str]]] = None


def default_config() -> Config:
  return Config(
      runtime=cfg.RuntimeConfig(),
      task=Task(
          model=ModelConfig(
              embedding_dim=8,
              vocab_sizes=vocab_sizes,
              bottom_mlp=[64, 32, 4],
              top_mlp=[64, 32, 1]),
          loss=Loss(label_smoothing=0.0),
          train_data=DataConfig(
              global_batch_size=train_batch_size,
              is_training=True,
              sharding=True),
          validation_data=DataConfig(
              global_batch_size=eval_batch_size,
              is_training=False,
              sharding=False)),
      trainer=TrainerConfig(
          train_steps=2 * steps_per_epoch,
          validation_interval=steps_per_epoch,
          validation_steps=NUM_EVAL_EXAMPLES // eval_batch_size,
          enable_metrics_in_training=True,
          optimizer_config=OptimizationConfig()),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ])


@exp_factory.register_config_factory('dlrm_criteo')
def dlrm_criteo_tb_config() -> Config:
  return Config(
      runtime=cfg.RuntimeConfig(),
      task=Task(
          model=ModelConfig(
              num_dense_features=13,
              vocab_sizes=vocab_sizes,
              bottom_mlp=[512, 256, 64],
              embedding_dim=64,
              top_mlp=[1024, 1024, 512, 256, 1],
              interaction='dot'),
          loss=Loss(label_smoothing=0.0),
          train_data=DataConfig(
              global_batch_size=train_batch_size,
              is_training=True,
              sharding=True),
          validation_data=DataConfig(
              global_batch_size=eval_batch_size,
              is_training=False,
              sharding=False)),
      trainer=TrainerConfig(
          train_steps=steps_per_epoch,
          validation_interval=steps_per_epoch // 2,
          validation_steps=NUM_EVAL_EXAMPLES // eval_batch_size,
          enable_metrics_in_training=True,
          optimizer_config=OptimizationConfig()),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ])


@exp_factory.register_config_factory('dcn_criteo')
def dcn_criteo_tb_config() -> Config:
  return Config(
      runtime=cfg.RuntimeConfig(),
      task=Task(
          model=ModelConfig(
              num_dense_features=13,
              vocab_sizes=vocab_sizes,
              bottom_mlp=[512, 256, 64],
              embedding_dim=64,
              top_mlp=[1024, 1024, 512, 256, 1],
              interaction='cross'),
          loss=Loss(label_smoothing=0.0),
          train_data=DataConfig(
              global_batch_size=train_batch_size,
              is_training=True,
              sharding=True),
          validation_data=DataConfig(
              global_batch_size=eval_batch_size,
              is_training=False,
              sharding=False)),
      trainer=TrainerConfig(
          train_steps=steps_per_epoch,
          validation_interval=steps_per_epoch // 2,
          validation_steps=NUM_EVAL_EXAMPLES // eval_batch_size,
          enable_metrics_in_training=True,
          optimizer_config=OptimizationConfig()),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ])


@exp_factory.register_config_factory('dlrm_dcn_v2_criteo')
def dlrm_dcn_v2_criteo_tb_config() -> Config:
  return Config(
      runtime=cfg.RuntimeConfig(),
      task=Task(
          model=ModelConfig(
              num_dense_features=13,
              vocab_sizes=vocab_sizes,
              bottom_mlp=[512, 256, 64],
              embedding_dim=64,
              top_mlp=[1024, 1024, 512, 256, 1],
              interaction='multi_layer_dcn',
              dcn_num_layers=3,
              dcn_low_rank_dim=512,
              dcn_use_bias=True,
              concat_dense=False,
              use_multi_hot=True,
              use_partial_tpu_embedding=False,
              multi_hot_sizes=multi_hot_sizes,
              ),
          loss=Loss(label_smoothing=0.0),
          train_data=DataConfig(
              global_batch_size=train_batch_size,
              is_training=True,
              sharding=True),
          validation_data=DataConfig(
              global_batch_size=eval_batch_size,
              is_training=False,
              sharding=False)),
      trainer=TrainerConfig(
          train_steps=steps_per_epoch,
          validation_interval=steps_per_epoch // 2,
          validation_steps=NUM_EVAL_EXAMPLES // eval_batch_size,
          enable_metrics_in_training=True,
          optimizer_config=OptimizationConfig()),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ])
