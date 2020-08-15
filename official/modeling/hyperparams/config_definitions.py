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

from typing import Optional, Union

import dataclasses

from official.modeling.hyperparams import base_config
from official.modeling.optimization.configs import optimization_config

OptimizationConfig = optimization_config.OptimizationConfig


@dataclasses.dataclass
class DataConfig(base_config.Config):
  """The base configuration for building datasets.

  Attributes:
    input_path: The path to the input. It can be either (1) a file pattern, or
      (2) multiple file patterns separated by comma. It should not be specified
      when the following `tfds_name` is specified.
    tfds_name: The name of the tensorflow dataset (TFDS). It should not be
      specified when the above `input_path` is specified.
    tfds_split: A str indicating which split of the data to load from TFDS. It
      is required when above `tfds_name` is specified.
    global_batch_size: The global batch size across all replicas.
    is_training: Whether this data is used for training or not.
    drop_remainder: Whether the last batch should be dropped in the case it has
      fewer than `global_batch_size` elements.
    shuffle_buffer_size: The buffer size used for shuffling training data.
    cache: Whether to cache dataset examples. Can be used to avoid re-reading
      from disk on the second epoch. Requires significant memory overhead.
    cycle_length: The number of files that will be processed concurrently when
      interleaving files.
    block_length: The number of consecutive elements to produce from each input
      element before cycling to another input element when interleaving files.
    sharding: Whether sharding is used in the input pipeline.
    examples_consume: An `integer` specifying the number of examples it will
      produce. If positive, it only takes this number of examples and raises
      tf.error.OutOfRangeError after that. Default is -1, meaning it will
      exhaust all the examples in the dataset.
    tfds_data_dir: A str specifying the directory to read/write TFDS data.
    tfds_download: A bool to indicate whether to download data using TFDS.
    tfds_as_supervised: A bool. When loading dataset from TFDS, if True, the
      returned tf.data.Dataset will have a 2-tuple structure (input, label)
      according to builder.info.supervised_keys; if False, the default, the
      returned tf.data.Dataset will have a dictionary with all the features.
    tfds_skip_decoding_feature: A str to indicate which features are skipped for
      decoding when loading dataset from TFDS. Use comma to separate multiple
      features. The main use case is to skip the image/video decoding for better
      performance.
  """
  input_path: str = ""
  tfds_name: str = ""
  tfds_split: str = ""
  global_batch_size: int = 0
  is_training: bool = None
  drop_remainder: bool = True
  shuffle_buffer_size: int = 100
  cache: bool = False
  cycle_length: int = 8
  block_length: int = 1
  sharding: bool = True
  examples_consume: int = -1
  tfds_data_dir: str = ""
  tfds_download: bool = False
  tfds_as_supervised: bool = False
  tfds_skip_decoding_feature: str = ""


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
    mixed_precision_dtype: dtype of mixed precision policy. It can be 'float32',
      'float16', or 'bfloat16'.
    loss_scale: The type of loss scale, or 'float' value. This is used when
      setting the mixed precision policy.
    run_eagerly: Whether or not to run the experiment eagerly.
    batchnorm_spatial_persistent: Whether or not to enable the spatial
      persistent mode for CuDNN batch norm kernel for improved GPU performance.
    allow_tpu_summary: Whether to allow summary happen inside the XLA program
      runs on TPU through automatic outside compilation.
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
  mixed_precision_dtype: Optional[str] = None
  loss_scale: Optional[Union[str, float]] = None
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
  """Configuration for trainer.

  Attributes:
    optimizer_config: optimizer config, it includes optimizer, learning rate,
      and warmup schedule configs.
    train_tf_while_loop: whether or not to use tf while loop.
    train_tf_function: whether or not to use tf_function for training loop.
    eval_tf_function: whether or not to use tf_function for eval.
    steps_per_loop: number of steps per loop.
    summary_interval: number of steps between each summary.
    checkpoint_interval: number of steps between checkpoints.
    max_to_keep: max checkpoints to keep.
    continuous_eval_timeout: maximum number of seconds to wait between
      checkpoints, if set to None, continuous eval will wait indefinitely. This
      is only used continuous_train_and_eval and continuous_eval modes.
    train_steps: number of train steps.
    validation_steps: number of eval steps. If `None`, the entire eval dataset
      is used.
    validation_interval: number of training steps to run between evaluations.
  """
  optimizer_config: OptimizationConfig = OptimizationConfig()
  # Orbit settings.
  train_tf_while_loop: bool = True
  train_tf_function: bool = True
  eval_tf_function: bool = True
  allow_tpu_summary: bool = False
  # Trainer intervals.
  steps_per_loop: int = 1000
  summary_interval: int = 1000
  checkpoint_interval: int = 1000
  # Checkpoint manager.
  max_to_keep: int = 5
  continuous_eval_timeout: Optional[int] = None
  # Train/Eval routines.
  train_steps: int = 0
  validation_steps: Optional[int] = None
  validation_interval: int = 1000


@dataclasses.dataclass
class TaskConfig(base_config.Config):
  init_checkpoint: str = ""
  model: base_config.Config = None
  train_data: DataConfig = DataConfig()
  validation_data: DataConfig = DataConfig()


@dataclasses.dataclass
class ExperimentConfig(base_config.Config):
  """Top-level configuration."""
  task: TaskConfig = TaskConfig()
  trainer: TrainerConfig = TrainerConfig()
  runtime: RuntimeConfig = RuntimeConfig()
