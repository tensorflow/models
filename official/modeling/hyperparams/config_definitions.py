# Lint as: python3
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
"""Common configuration settings."""
from typing import Optional

import dataclasses

from official.modeling import optimization
from official.modeling.hyperparams import base_config
from official.utils import registry

OptimizationConfig = optimization.OptimizationConfig


@dataclasses.dataclass
class DataConfig(base_config.Config):
  """The base configuration for building datasets.

  Attributes:
    input_path: The path to the input. It can be either (1) a file pattern, or
      (2) multiple file patterns separated by comma.
    global_batch_size: The global batch size across all replicas.
    is_training: Whether this data is used for training or not.
    drop_remainder: Whether the last batch should be dropped in the case it has
      fewer than `global_batch_size` elements.
    shuffle_buffer_size: The buffer size used for shuffling training data.
    cache: Whether to cache dataset examples. Can be used to avoid re-reading
      from disk on the second epoch. Requires significant memory overhead.
    cycle_length: The number of files that will be processed concurrently when
      interleaving files.
    sharding: Whether sharding is used in the input pipeline.
    examples_consume: An `integer` specifying the number of examples it will
      produce. If positive, it only takes this number of examples and raises
      tf.error.OutOfRangeError after that. Default is -1, meaning it will
      exhaust all the examples in the dataset.
  """
  input_path: str = ""
  global_batch_size: int = 0
  is_training: bool = None
  drop_remainder: bool = True
  shuffle_buffer_size: int = 100
  cache: bool = False
  cycle_length: int = 8
  sharding: bool = True
  examples_consume: int = -1


@dataclasses.dataclass
class RuntimeConfig(base_config.Config):
  """High-level configurations for Runtime.

  These include parameters that are not directly related to the experiment,
  e.g. directories, accelerator type, etc.

  Attributes:
    distribution_strategy: e.g. 'mirrored', 'tpu', etc.
    enable_xla: Whether or not to enable XLA.
    per_gpu_thread_count: thread count per GPU.
    gpu_thread_mode: Whether and how the GPU device uses its own threadpool.
    dataset_num_private_threads: Number of threads for a private threadpool
      created for all datasets computation.
    tpu: The address of the TPU to use, if any.
    num_gpus: The number of GPUs to use, if any.
    worker_hosts: comma-separated list of worker ip:port pairs for running
      multi-worker models with DistributionStrategy.
    task_index: If multi-worker training, the task index of this worker.
    all_reduce_alg: Defines the algorithm for performing all-reduce.
    num_packs: Sets `num_packs` in the cross device ops used in
      MirroredStrategy.  For details, see tf.distribute.NcclAllReduce.
    loss_scale: The type of loss scale. This is used when setting the mixed
      precision policy.
    run_eagerly: Whether or not to run the experiment eagerly.
    batchnorm_spatial_persistent: Whether or not to enable the spatial
      persistent mode for CuDNN batch norm kernel for improved GPU performance.
  """
  distribution_strategy: str = "mirrored"
  enable_xla: bool = False
  gpu_thread_mode: Optional[str] = None
  dataset_num_private_threads: Optional[int] = None
  per_gpu_thread_count: int = 0
  tpu: Optional[str] = None
  num_gpus: int = 0
  worker_hosts: Optional[str] = None
  task_index: int = -1
  all_reduce_alg: Optional[str] = None
  num_packs: int = 1
  loss_scale: Optional[str] = None
  run_eagerly: bool = False
  batchnorm_spatial_persistent: bool = False


@dataclasses.dataclass
class TensorboardConfig(base_config.Config):
  """Configuration for Tensorboard.

  Attributes:
    track_lr: Whether or not to track the learning rate in Tensorboard. Defaults
      to True.
    write_model_weights: Whether or not to write the model weights as images in
      Tensorboard. Defaults to False.
  """
  track_lr: bool = True
  write_model_weights: bool = False


@dataclasses.dataclass
class CallbacksConfig(base_config.Config):
  """Configuration for Callbacks.

  Attributes:
    enable_checkpoint_and_export: Whether or not to enable checkpoints as a
      Callback. Defaults to True.
    enable_tensorboard: Whether or not to enable Tensorboard as a Callback.
      Defaults to True.
    enable_time_history: Whether or not to enable TimeHistory Callbacks.
      Defaults to True.
  """
  enable_checkpoint_and_export: bool = True
  enable_tensorboard: bool = True
  enable_time_history: bool = True


@dataclasses.dataclass
class TrainerConfig(base_config.Config):
  optimizer_config: OptimizationConfig = OptimizationConfig()
  train_tf_while_loop: bool = True
  train_tf_function: bool = True
  eval_tf_function: bool = True
  steps_per_loop: int = 1000
  summary_interval: int = 1000
  checkpoint_interval: int = 1000
  max_to_keep: int = 5


@dataclasses.dataclass
class TaskConfig(base_config.Config):
  network: base_config.Config = None
  train_data: DataConfig = DataConfig()
  validation_data: DataConfig = DataConfig()


@dataclasses.dataclass
class ExperimentConfig(base_config.Config):
  """Top-level configuration."""
  mode: str = "train"  # train, eval, train_and_eval.
  task: TaskConfig = TaskConfig()
  trainer: TrainerConfig = TrainerConfig()
  runtime: RuntimeConfig = RuntimeConfig()
  train_steps: int = 0
  validation_steps: Optional[int] = None
  validation_interval: int = 100


_REGISTERED_CONFIGS = {}


def register_config_factory(name):
  """Register ExperimentConfig factory method."""
  return registry.register(_REGISTERED_CONFIGS, name)


def get_exp_config_creater(exp_name: str):
  """Looks up ExperimentConfig factory methods."""
  exp_creater = registry.lookup(_REGISTERED_CONFIGS, exp_name)
  return exp_creater
