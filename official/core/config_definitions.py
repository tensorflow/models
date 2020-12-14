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

from typing import Optional, Sequence, Union

import dataclasses

from official.modeling.hyperparams import base_config
from official.modeling.optimization.configs import optimization_config

OptimizationConfig = optimization_config.OptimizationConfig


@dataclasses.dataclass
class DataConfig(base_config.Config):
  """The base configuration for building datasets.

  Attributes:
    input_path: The path to the input. It can be either (1) a str indicating
      a file path/pattern, or (2) a str indicating multiple file paths/patterns
      separated by comma (e.g "a, b, c" or no spaces "a,b,c"), or
      (3) a list of str, each of which is a file path/pattern or multiple file
      paths/patterns separated by comma.
      It should not be specified when the following `tfds_name` is specified.
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
    deterministic: A boolean controlling whether determinism should be enforced.
    sharding: Whether sharding is used in the input pipeline.
    enable_tf_data_service: A boolean indicating whether to enable tf.data
      service for the input pipeline.
    tf_data_service_address: The URI of a tf.data service to offload
      preprocessing onto during training. The URI should be in the format
      "protocol://address", e.g. "grpc://tf-data-service:5050". It can be
      overridden by `FLAGS.tf_data_service` flag in the binary.
    tf_data_service_job_name: The name of the tf.data service job. This
      argument makes it possible for multiple datasets to share the same job.
      The default behavior is that the dataset creates anonymous, exclusively
      owned jobs.
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
  input_path: Union[Sequence[str], str] = ""
  tfds_name: str = ""
  tfds_split: str = ""
  global_batch_size: int = 0
  is_training: bool = None
  drop_remainder: bool = True
  shuffle_buffer_size: int = 100
  cache: bool = False
  cycle_length: Optional[int] = None
  block_length: int = 1
  deterministic: Optional[bool] = None
  sharding: bool = True
  enable_tf_data_service: bool = False
  tf_data_service_address: Optional[str] = None
  tf_data_service_job_name: Optional[str] = None
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

  # Global model parallelism configurations.
  num_cores_per_replica: int = 1
  default_shard_dim: int = -1

  def model_parallelism(self):
    return dict(
        num_cores_per_replica=self.num_cores_per_replica,
        default_shard_dim=self.default_shard_dim)


@dataclasses.dataclass
class TrainerConfig(base_config.Config):
  """Configuration for trainer.

  Attributes:
    optimizer_config: optimizer config, it includes optimizer, learning rate,
      and warmup schedule configs.
    train_tf_while_loop: whether or not to use tf while loop.
    train_tf_function: whether or not to use tf_function for training loop.
    eval_tf_function: whether or not to use tf_function for eval.
    allow_tpu_summary: Whether to allow summary happen inside the XLA program
      runs on TPU through automatic outside compilation.
    steps_per_loop: number of steps per loop.
    summary_interval: number of steps between each summary.
    checkpoint_interval: number of steps between checkpoints.
    max_to_keep: max checkpoints to keep.
    continuous_eval_timeout: maximum number of seconds to wait between
      checkpoints, if set to None, continuous eval will wait indefinitely. This
      is only used continuous_train_and_eval and continuous_eval modes. Default
      value is 1 hrs.
    train_steps: number of train steps.
    validation_steps: number of eval steps. If `None`, the entire eval dataset
      is used.
    validation_interval: number of training steps to run between evaluations.
    best_checkpoint_export_subdir: if set, the trainer will keep track of the
      best evaluation metric, and export the corresponding best checkpoint under
      `model_dir/best_checkpoint_export_subdir`. Note that this only works if
      mode contains eval (such as `train_and_eval`, `continuous_eval`, and
      `continuous_train_and_eval`).
    best_checkpoint_eval_metric: for exporting the best checkpoint, which
      evaluation metric the trainer should monitor. This can be any evaluation
      metric appears on tensorboard.
    best_checkpoint_metric_comp: for exporting the best checkpoint, how the
      trainer should compare the evaluation metrics. This can be either `higher`
      (higher the better) or `lower` (lower the better).
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
  continuous_eval_timeout: int = 60 * 60
  # Train/Eval routines.
  train_steps: int = 0
  # Sets validation steps to be -1 to evaluate the entire dataset.
  validation_steps: int = -1
  validation_interval: int = 1000
  # Best checkpoint export.
  best_checkpoint_export_subdir: str = ""
  best_checkpoint_eval_metric: str = ""
  best_checkpoint_metric_comp: str = "higher"
  # Blowup recovery.
  loss_upper_bound: float = 1e6
  recovery_begin_steps: int = 0  # Enforcing the loss bound after these steps.
  # When max trials < 0, no recovery module; max trials = 0, we will check
  # the condition and fail the job if the condition happens; max trials > 0,
  # we will retore the model states.
  recovery_max_trials: int = 0


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
