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

"""Common configuration settings."""

import dataclasses
from typing import Optional, Sequence, Union

from official.modeling.hyperparams import base_config
from official.modeling.optimization.configs import optimization_config
from official.modeling.privacy import configs as dp_configs

OptimizationConfig = optimization_config.OptimizationConfig


@dataclasses.dataclass
class DataConfig(base_config.Config):
  """The base configuration for building datasets.

  Attributes:
    input_path: The path to the input. It can be either (1) a str indicating a
      file path/pattern, or (2) a str indicating multiple file paths/patterns
      separated by comma (e.g "a, b, c" or no spaces "a,b,c"), or (3) a list of
      str, each of which is a file path/pattern or multiple file paths/patterns
      separated by comma, or (4) a dictionary of the previous three approaches
      for more advanced data mixing using named access. It should not be
      specified when the following `tfds_name` is specified.
    tfds_name: The name of the tensorflow dataset (TFDS). It should not be
      specified when the above `input_path` is specified.
    tfds_split: A str indicating which split of the data to load from TFDS. It
      is required when above `tfds_name` is specified.
    global_batch_size: The global batch size across all replicas.
    is_training: Whether this data is used for training or not. This flag is
      useful for consumers of this object to determine whether the data should
      be repeated or shuffled.
    drop_remainder: Whether the last batch should be dropped in the case it has
      fewer than `global_batch_size` elements.
    shuffle_buffer_size: The buffer size used for shuffling training data.
    cache: Whether to cache dataset examples. If `True`, we will cache the
      dataset after applying the decode_fn and parse_fn. It can be used to avoid
      re-reading from disk, re-decoding and re-parsing the example on the second
      epoch, but it requires significant memory overhead.
    cycle_length: The number of files that will be processed concurrently when
      interleaving files.
    block_length: The number of consecutive elements to produce from each input
      element before cycling to another input element when interleaving files.
    ram_budget: RAM budget for tf.data service in GB. If None, tf.data will use
      50% of the available host RAM.
    deterministic: A boolean controlling whether determinism should be enforced.
    sharding: Whether sharding is used in the input pipeline.
    enable_tf_data_service: A boolean indicating whether to enable tf.data
      service for the input pipeline.
    tf_data_service_address: The URI of a tf.data service to offload
      preprocessing onto during training. The URI should be in the format
      "protocol://address", e.g. "grpc://tf-data-service:5050". It can be
      overridden by `FLAGS.tf_data_service` flag in the binary.
    tf_data_service_job_name: The name of the tf.data service job. This argument
      makes it possible for multiple datasets to share the same job. The default
      behavior is that the dataset creates anonymous, exclusively owned jobs.
    tfds_data_dir: A str specifying the directory to read/write TFDS data.
    tfds_as_supervised: A bool. When loading dataset from TFDS, if True, the
      returned tf.data.Dataset will have a 2-tuple structure (input, label)
      according to builder.info.supervised_keys; if False, the default, the
      returned tf.data.Dataset will have a dictionary with all the features.
    tfds_skip_decoding_feature: A str to indicate which features are skipped for
      decoding when loading dataset from TFDS. Use comma to separate multiple
      features. The main use case is to skip the image/video decoding for better
      performance.
    enable_shared_tf_data_service_between_parallel_trainers: A bool. When set to
      true, only a single tf.data service will be started, and it will be shared
      between all the trainer run simultaneously, e.g. using vizier to tune
      hyperparameters. This will save CPU and RAM resources compared to running
      separate tf.data service for each trainer. Notice that if batch size is
      different for different trainers, the field
      apply_tf_data_service_before_batching also needs to be true so that only a
      single tf.data service instance will be created. In this case, tf.data
      service will be applied before batching operation. So make sure to not
      apply any processing steps after batching (e.g. in postprocess_fn) since
      they wouldn't be paralleled by tf.data service and may slow down your
      tf.data pipeline. When using shared tf.data service, the tf.data dataset
      must be infinite, and slow trainer may skip certain training examples.
      More details about shared tf.data service can be found at:
      https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers.
    apply_tf_data_service_before_batching: A bool. If set to True, tf.data
      service will be applied before batching operation. This is useful to make
      sure only a single tf.data service instance is created when
      enable_shared_tf_data_service_between_parallel_trainers is true and batch
      size is changing between parallel trainers.
    trainer_id: A string. The id of the trainer if there are multiple parallel
      trainer running at the same time, e.g. in vizier tuning case. It will be
      automatically set if this field is needed. Users does not need to set it
      when creating experiment configs.
    seed: An optional seed to use for deterministic shuffling/preprocessing.
    prefetch_buffer_size: An int specifying the buffer size of prefetch
      datasets. If None, the buffer size is autotuned. Specifying this is useful
      in case autotuning uses up too much memory by making the buffer size too
      high.
    autotune_algorithm: If specified, use this algorithm for AUTOTUNE. See:
      https://www.tensorflow.org/api_docs/python/tf/data/experimental/AutotuneAlgorithm
  """
  input_path: Union[Sequence[str], str, base_config.Config] = ""
  tfds_name: Union[str, base_config.Config] = ""
  tfds_split: str = ""
  global_batch_size: int = 0
  is_training: Optional[bool] = None
  drop_remainder: bool = True
  shuffle_buffer_size: int = 100
  cache: bool = False
  cycle_length: Optional[int] = None
  block_length: int = 1
  ram_budget: Optional[int] = None
  deterministic: Optional[bool] = None
  sharding: bool = True
  enable_tf_data_service: bool = False
  tf_data_service_address: Optional[str] = None
  tf_data_service_job_name: Optional[str] = None
  tfds_data_dir: str = ""
  tfds_as_supervised: bool = False
  tfds_skip_decoding_feature: str = ""
  enable_shared_tf_data_service_between_parallel_trainers: bool = False
  apply_tf_data_service_before_batching: bool = False
  trainer_id: Optional[str] = None
  seed: Optional[int] = None
  prefetch_buffer_size: Optional[int] = None
  autotune_algorithm: Optional[str] = None


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

  # XLA runtime params.
  # XLA params are only applied to the train_step.
  # These augments can improve training speed. They can also improve eval, but
  # may reduce usability and users would need to make changes to code.

  # Whether to enable XLA dynamic padder
  # infrastructure to handle dynamic shapes inputs inside XLA. True by
  # default. Disabling this may cause correctness issues with dynamic shapes
  # inputs, as XLA will just assume the inputs are with padded shapes. However
  # users can optionally set it to False to improve device time if masking is
  # already handled in the user side.
  # If None, will respect XLA default.
  tpu_enable_xla_dynamic_padder: Optional[bool] = None

  # Global model parallelism configurations.
  num_cores_per_replica: int = 1
  default_shard_dim: int = -1
  use_tpu_mp_strategy: bool = False

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
    eval_tf_while_loop: whether or not to use tf while loop for eval.
    allow_tpu_summary: Whether to allow summary happen inside the XLA program
      runs on TPU through automatic outside compilation.
    steps_per_loop: number of steps per loop to report training metrics. This
      can also be used to reduce host worker communication in a TPU setup.
    summary_interval: number of steps between each summary.
    checkpoint_interval: number of steps between checkpoints.
    max_to_keep: max checkpoints to keep.
    continuous_eval_timeout: maximum number of seconds to wait between
      checkpoints, if set to None, continuous eval will wait indefinitely. This
      is only used continuous_train_and_eval and continuous_eval modes. Default
      value is 1 hrs.
    train_steps: number of train steps.
    validation_steps: number of eval steps. If -1, the entire eval dataset is
      used.
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
    validation_summary_subdir: A 'str', sub directory for saving eval summary.
    preemption_on_demand_checkpoint: whether or not to save on-demand
      checkpoints after a preemption.
  """
  optimizer_config: OptimizationConfig = dataclasses.field(
      default_factory=OptimizationConfig
  )
  # Orbit settings.
  train_tf_while_loop: bool = True
  train_tf_function: bool = True
  eval_tf_function: bool = True
  eval_tf_while_loop: bool = False
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
  validation_summary_subdir: str = "validation"
  # Preemption on-demand checkpoint.
  preemption_on_demand_checkpoint: bool = True  # copybara-replace


@dataclasses.dataclass
class TaskConfig(base_config.Config):
  """Config passed to task."""
  init_checkpoint: str = ""
  model: Optional[base_config.Config] = None
  train_data: DataConfig = dataclasses.field(default_factory=DataConfig)
  validation_data: DataConfig = dataclasses.field(default_factory=DataConfig)
  name: Optional[str] = None
  # Configs for differential privacy
  # These configs are only effective if you use create_optimizer in
  # tensorflow_models/official/core/base_task.py
  # DEPRECATED b/264611883
  differential_privacy_config: Optional[
      dp_configs.DifferentialPrivacyConfig] = None
  # Whether to show image summary. Useful to visualize model predictions. Only
  # work for vision tasks.
  allow_image_summary: bool = False


@dataclasses.dataclass
class ExperimentConfig(base_config.Config):
  """Top-level configuration."""
  task: TaskConfig = dataclasses.field(default_factory=TaskConfig)
  trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)
  runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)
