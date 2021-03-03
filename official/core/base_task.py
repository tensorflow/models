# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Defines the base task abstraction."""
import abc
from typing import Optional

from absl import logging
import tensorflow as tf

from official.core import config_definitions
from official.modeling import optimization
from official.modeling import performance

OptimizationConfig = optimization.OptimizationConfig
RuntimeConfig = config_definitions.RuntimeConfig


class Task(tf.Module, metaclass=abc.ABCMeta):
  """A single-replica view of training procedure.

  Tasks provide artifacts for training/validation procedures, including
  loading/iterating over Datasets, training/validation steps, calculating the
  loss and customized metrics with reduction.
  """

  # Special keys in train/validate step returned logs.
  loss = "loss"

  def __init__(self, params, logging_dir: str = None, name: str = None):
    """Task initialization.

    Args:
      params: the task configuration instance, which can be any of dataclass,
        ConfigDict, namedtuple, etc.
      logging_dir: a string pointing to where the model, summaries etc. will be
        saved. You can also write additional stuff in this directory.
      name: the task name.
    """
    super().__init__(name=name)
    self._task_config = params
    self._logging_dir = logging_dir

  @property
  def task_config(self):
    return self._task_config

  @property
  def logging_dir(self) -> str:
    return self._logging_dir

  @classmethod
  def create_optimizer(cls, optimizer_config: OptimizationConfig,
                       runtime_config: Optional[RuntimeConfig] = None):
    """Creates an TF optimizer from configurations.

    Args:
      optimizer_config: the parameters of the Optimization settings.
      runtime_config: the parameters of the runtime.

    Returns:
      A tf.optimizers.Optimizer object.
    """
    opt_factory = optimization.OptimizerFactory(optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    # Configuring optimizer when loss_scale is set in runtime config. This helps
    # avoiding overflow/underflow for float16 computations.
    if runtime_config and runtime_config.loss_scale:
      optimizer = performance.configure_optimizer(
          optimizer,
          use_float16=runtime_config.mixed_precision_dtype == "float16",
          loss_scale=runtime_config.loss_scale,
          use_experimental_api=True)

    return optimizer

  def initialize(self, model: tf.keras.Model):
    """[Optional] A callback function used as CheckpointManager's init_fn.

    This function will be called when no checkpoint is found for the model.
    If there is a checkpoint, the checkpoint will be loaded and this function
    will not be called. You can use this callback function to load a pretrained
    checkpoint, saved under a directory other than the model_dir.

    Args:
      model: The keras.Model built or used by this task.
    """
    ckpt_dir_or_file = self.task_config.init_checkpoint
    logging.info("Trying to load pretrained checkpoint from %s",
                 ckpt_dir_or_file)
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
    if not ckpt_dir_or_file:
      return

    if hasattr(model, "checkpoint_items"):
      checkpoint_items = model.checkpoint_items
    else:
      checkpoint_items = dict(model=model)
    ckpt = tf.train.Checkpoint(**checkpoint_items)
    status = ckpt.read(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()
    logging.info("Finished loading pretrained checkpoint from %s",
                 ckpt_dir_or_file)

  def build_model(self) -> tf.keras.Model:
    """[Optional] Creates model architecture.

    Returns:
      A model instance.
    """

  @abc.abstractmethod
  def build_inputs(self,
                   params,
                   input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a dataset or a nested structure of dataset functions.

    Dataset functions define per-host datasets with the per-replica batch size.
    With distributed training, this method runs on remote hosts.

    Args:
      params: hyperparams to create input pipelines, which can be any of
        dataclass, ConfigDict, namedtuple, etc.
      input_context: optional distribution input pipeline context.

    Returns:
      A nested structure of per-replica input functions.
    """

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    """Standard interface to compute losses.

    Args:
      labels: optional label tensors.
      model_outputs: a nested structure of output tensors.
      aux_losses: auxiliary loss tensors, i.e. `losses` in keras.Model.

    Returns:
      The total loss tensor.
    """
    del model_outputs, labels

    if aux_losses is None:
      losses = [tf.constant(0.0, dtype=tf.float32)]
    else:
      losses = aux_losses
    total_loss = tf.add_n(losses)
    return total_loss

  def build_metrics(self, training: bool = True):
    """Gets streaming metrics for training/validation."""
    del training
    return []

  def process_metrics(self, metrics, labels, model_outputs):
    """Process and update metrics.

    Called when using custom training loop API.

    Args:
      metrics: a nested structure of metrics objects. The return of function
        self.build_metrics.
      labels: a tensor or a nested structure of tensors.
      model_outputs: a tensor or a nested structure of tensors. For example,
        output of the keras model built by self.build_model.
    """
    for metric in metrics:
      metric.update_state(labels, model_outputs)

  def process_compiled_metrics(self, compiled_metrics, labels, model_outputs):
    """Process and update compiled_metrics.

    call when using compile/fit API.

    Args:
      compiled_metrics: the compiled metrics (model.compiled_metrics).
      labels: a tensor or a nested structure of tensors.
      model_outputs: a tensor or a nested structure of tensors. For example,
        output of the keras model built by self.build_model.
    """
    compiled_metrics.update_state(labels, model_outputs)

  def train_step(self,
                 inputs,
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 metrics=None):
    """Does forward and backward.

    With distribution strategies, this method runs on devices.

    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    if isinstance(inputs, tuple) and len(inputs) == 2:
      features, labels = inputs
    else:
      features, labels = inputs, inputs
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      # Computes per-replica loss.
      loss = self.build_losses(
          labels=labels, model_outputs=outputs, aux_losses=model.losses)
      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = loss / tf.distribute.get_strategy().num_replicas_in_sync

      # For mixed precision, when a LossScaleOptimizer is used, the loss is
      # scaled to avoid numeric underflow.
      if isinstance(optimizer,
                    tf.keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)

    if isinstance(optimizer,
                  tf.keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))
    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
    if model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics or []})
      logs.update({m.name: m.result() for m in model.metrics})
    return logs

  def validation_step(self, inputs, model: tf.keras.Model, metrics=None):
    """Validation step.

    With distribution strategies, this method runs on devices.

    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    if isinstance(inputs, tuple) and len(inputs) == 2:
      features, labels = inputs
    else:
      features, labels = inputs, inputs
    outputs = self.inference_step(features, model)
    loss = self.build_losses(
        labels=labels, model_outputs=outputs, aux_losses=model.losses)
    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
    if model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics or []})
      logs.update({m.name: m.result() for m in model.metrics})
    return logs

  def inference_step(self, inputs, model: tf.keras.Model):
    """Performs the forward step.

    With distribution strategies, this method runs on devices.

    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.

    Returns:
      Model outputs.
    """
    return model(inputs, training=False)

  def aggregate_logs(self, state, step_logs):
    """Optional aggregation over logs returned from a validation step."""
    pass

  def reduce_aggregated_logs(self,
                             aggregated_logs,
                             global_step: Optional[tf.Tensor] = None):
    """Optional reduce of aggregated logs over validation steps."""
    return {}
