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

"""BASNet task definition."""
from typing import Optional

from absl import logging
import tensorflow as tf, tf_keras

from official.common import dataset_fn
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.projects.basnet.configs import basnet as exp_cfg
from official.projects.basnet.evaluation import metrics as basnet_metrics
from official.projects.basnet.losses import basnet_losses
from official.projects.basnet.modeling import basnet_model
from official.projects.basnet.modeling import refunet
from official.vision.dataloaders import segmentation_input


def build_basnet_model(
    input_specs: tf_keras.layers.InputSpec,
    model_config: exp_cfg.BASNetModel,
    l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None):
  """Builds BASNet model."""
  norm_activation_config = model_config.norm_activation
  backbone = basnet_model.BASNetEncoder(
      input_specs=input_specs,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      use_bias=model_config.use_bias,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)

  decoder = basnet_model.BASNetDecoder(
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      use_bias=model_config.use_bias,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)

  refinement = refunet.RefUnet(
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      use_bias=model_config.use_bias,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)

  model = basnet_model.BASNetModel(backbone, decoder, refinement)
  return model


@task_factory.register_task_cls(exp_cfg.BASNetTask)
class BASNetTask(base_task.Task):
  """A task for basnet."""

  def build_model(self):
    """Builds basnet model."""
    input_specs = tf_keras.layers.InputSpec(
        shape=[None] + self.task_config.model.input_size)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf_keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = build_basnet_model(
        input_specs=input_specs,
        model_config=self.task_config.model,
        l2_regularizer=l2_regularizer)
    return model

  def initialize(self, model: tf_keras.Model):
    """Loads pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      return

    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    # Restoring checkpoint.
    if 'all' in self.task_config.init_checkpoint_modules:
      ckpt = tf.train.Checkpoint(**model.checkpoint_items)
      status = ckpt.restore(ckpt_dir_or_file)
      status.assert_consumed()
    else:
      ckpt_items = {}
      if 'backbone' in self.task_config.init_checkpoint_modules:
        ckpt_items.update(backbone=model.backbone)
      if 'decoder' in self.task_config.init_checkpoint_modules:
        ckpt_items.update(decoder=model.decoder)

      ckpt = tf.train.Checkpoint(**ckpt_items)
      status = ckpt.restore(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def build_inputs(self,
                   params: exp_cfg.DataConfig,
                   input_context: Optional[tf.distribute.InputContext] = None):
    """Builds BASNet input."""

    ignore_label = self.task_config.losses.ignore_label

    decoder = segmentation_input.Decoder()
    parser = segmentation_input.Parser(
        output_size=params.output_size,
        crop_size=params.crop_size,
        ignore_label=ignore_label,
        aug_rand_hflip=params.aug_rand_hflip,
        dtype=params.dtype)

    reader = input_reader.InputReader(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self, label, model_outputs, aux_losses=None):
    """Hybrid loss proposed in BASNet.

    Args:
      label: label.
      model_outputs: Output logits of the classifier.
      aux_losses: auxiliarly loss tensors, i.e. `losses` in keras.Model.

    Returns:
      The total loss tensor.
    """
    basnet_loss_fn = basnet_losses.BASNetLoss()
    total_loss = basnet_loss_fn(model_outputs, label['masks'])

    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    return total_loss

  def build_metrics(self, training=False):
    """Gets streaming metrics for training/validation."""
    evaluations = []

    if training:
      evaluations = []
    else:
      self.mae_metric = basnet_metrics.MAE()
      self.maxf_metric = basnet_metrics.MaxFscore()
      self.relaxf_metric = basnet_metrics.RelaxedFscore()

    return evaluations

  def train_step(self, inputs, model, optimizer, metrics=None):
    """Does forward and backward.

    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features, labels = inputs
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      loss = self.build_losses(
          model_outputs=outputs, label=labels, aux_losses=model.losses)

      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = loss / num_replicas

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)

    # Scales back gradient before apply_gradients when LossScaleOptimizer is
    # used.
    if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)

    # Apply gradient clipping.
    if self.task_config.gradient_clip_norm > 0:
      grads, _ = tf.clip_by_global_norm(
          grads, self.task_config.gradient_clip_norm)
    optimizer.apply_gradients(list(zip(grads, tvars)))
    logs = {self.loss: loss}
    return logs

  def validation_step(self, inputs, model, metrics=None):
    """Validatation step.

    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.
    Returns:
      A dictionary of logs.
    """
    features, labels = inputs

    outputs = self.inference_step(features, model)
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

    loss = 0
    logs = {self.loss: loss}

    levels = sorted(outputs.keys())

    logs.update(
        {self.mae_metric.name: (labels['masks'], outputs[levels[-1]])})
    logs.update(
        {self.maxf_metric.name: (labels['masks'], outputs[levels[-1]])})
    logs.update(
        {self.relaxf_metric.name: (labels['masks'], outputs[levels[-1]])})
    return logs

  def inference_step(self, inputs, model):
    """Performs the forward step."""
    return model(inputs, training=False)

  def aggregate_logs(self, state=None, step_outputs=None):
    if state is None:
      self.mae_metric.reset_states()
      self.maxf_metric.reset_states()
      self.relaxf_metric.reset_states()
      state = self.mae_metric
    self.mae_metric.update_state(
        step_outputs[self.mae_metric.name][0],
        step_outputs[self.mae_metric.name][1])
    self.maxf_metric.update_state(
        step_outputs[self.maxf_metric.name][0],
        step_outputs[self.maxf_metric.name][1])
    self.relaxf_metric.update_state(
        step_outputs[self.relaxf_metric.name][0],
        step_outputs[self.relaxf_metric.name][1])
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    result = {}
    result['MAE'] = self.mae_metric.result()
    result['maxF'] = self.maxf_metric.result()
    result['relaxF'] = self.relaxf_metric.result()
    return result
