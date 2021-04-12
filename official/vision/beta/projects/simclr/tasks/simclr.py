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
"""Image SimCLR task definition.

SimCLR training two different modes:
- pretrain
- fine-tuning

For the above two different modes, the following components are different in
the task definition:
- training data format
- training loss
- projection_head and/or supervised_head
"""

from typing import Dict, Optional

from absl import logging
import tensorflow as tf

from official.core import base_task
from official.core import config_definitions
from official.core import input_reader
from official.core import task_factory
from official.modeling import optimization
from official.modeling import performance
from official.modeling import tf_utils
from official.vision.beta.modeling import backbones
from official.vision.beta.projects.simclr.configs import simclr as exp_cfg
from official.vision.beta.projects.simclr.dataloaders import simclr_input
from official.vision.beta.projects.simclr.heads import simclr_head
from official.vision.beta.projects.simclr.losses import contrastive_losses
from official.vision.beta.projects.simclr.modeling import simclr_model

OptimizationConfig = optimization.OptimizationConfig
RuntimeConfig = config_definitions.RuntimeConfig


@task_factory.register_task_cls(exp_cfg.SimCLRPretrainTask)
class SimCLRPretrainTask(base_task.Task):
  """A task for image classification."""

  def create_optimizer(self, optimizer_config: OptimizationConfig,
                       runtime_config: Optional[RuntimeConfig] = None):
    """Creates an TF optimizer from configurations.

    Args:
      optimizer_config: the parameters of the Optimization settings.
      runtime_config: the parameters of the runtime.

    Returns:
      A tf.optimizers.Optimizer object.
    """
    if (optimizer_config.optimizer.type == 'lars'
        and self.task_config.loss.l2_weight_decay > 0.0):
      raise ValueError('The l2_weight_decay cannot be used together with lars '
                       'optimizer. Please set it to 0.')

    opt_factory = optimization.OptimizerFactory(optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    # Configuring optimizer when loss_scale is set in runtime config. This helps
    # avoiding overflow/underflow for float16 computations.
    if runtime_config and runtime_config.loss_scale:
      optimizer = performance.configure_optimizer(
          optimizer,
          use_float16=runtime_config.mixed_precision_dtype == 'float16',
          loss_scale=runtime_config.loss_scale)

    return optimizer

  def build_model(self):
    model_config = self.task_config.model
    input_specs = tf.keras.layers.InputSpec(
        shape=[None] + model_config.input_size)

    l2_weight_decay = self.task_config.loss.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf.keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    # Build backbone
    backbone = backbones.factory.build_backbone(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)

    # Build projection head
    norm_activation_config = model_config.norm_activation
    projection_head_config = model_config.projection_head
    projection_head = simclr_head.ProjectionHead(
        proj_output_dim=projection_head_config.proj_output_dim,
        num_proj_layers=projection_head_config.num_proj_layers,
        ft_proj_idx=projection_head_config.ft_proj_idx,
        kernel_regularizer=l2_regularizer,
        use_sync_bn=norm_activation_config.use_sync_bn,
        norm_momentum=norm_activation_config.norm_momentum,
        norm_epsilon=norm_activation_config.norm_epsilon)

    # Build supervised head
    supervised_head_config = model_config.supervised_head
    if supervised_head_config:
      if supervised_head_config.zero_init:
        s_kernel_initializer = 'zeros'
      else:
        s_kernel_initializer = 'random_uniform'
      supervised_head = simclr_head.ClassificationHead(
          num_classes=supervised_head_config.num_classes,
          kernel_initializer=s_kernel_initializer,
          kernel_regularizer=l2_regularizer)
    else:
      supervised_head = None

    model = simclr_model.SimCLRModel(
        input_specs=input_specs,
        backbone=backbone,
        projection_head=projection_head,
        supervised_head=supervised_head,
        mode=model_config.mode,
        backbone_trainable=model_config.backbone_trainable)

    logging.info(model.get_config())

    return model

  def initialize(self, model: tf.keras.Model):
    """Loading pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      return

    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    # Restoring checkpoint.
    if self.task_config.init_checkpoint_modules == 'all':
      ckpt = tf.train.Checkpoint(**model.checkpoint_items)
      status = ckpt.restore(ckpt_dir_or_file)
      status.assert_consumed()
    elif self.task_config.init_checkpoint_modules == 'backbone':
      ckpt = tf.train.Checkpoint(backbone=model.backbone)
      status = ckpt.restore(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    else:
      assert "Only 'all' or 'backbone' can be used to initialize the model."

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def build_inputs(self, params, input_context=None):
    input_size = self.task_config.model.input_size

    if params.tfds_name:
      decoder = simclr_input.TFDSDecoder(params.decoder.decode_label)
    else:
      decoder = simclr_input.Decoder(params.decoder.decode_label)

    parser = simclr_input.Parser(
        output_size=input_size[:2],
        aug_rand_crop=params.parser.aug_rand_crop,
        aug_rand_hflip=params.parser.aug_rand_hflip,
        aug_color_distort=params.parser.aug_color_distort,
        aug_color_jitter_strength=params.parser.aug_color_jitter_strength,
        aug_color_jitter_impl=params.parser.aug_color_jitter_impl,
        aug_rand_blur=params.parser.aug_rand_blur,
        parse_label=params.parser.parse_label,
        test_crop=params.parser.test_crop,
        mode=params.parser.mode,
        dtype=params.dtype)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self,
                   labels,
                   model_outputs,
                   aux_losses=None) -> Dict[str, tf.Tensor]:
    # Compute contrastive relative loss
    con_losses_obj = contrastive_losses.ContrastiveLoss(
        projection_norm=self.task_config.loss.projection_norm,
        temperature=self.task_config.loss.temperature)
    # The projection outputs from model has the size of
    # (2 * bsz, project_dim)
    projection_outputs = model_outputs[simclr_model.PROJECTION_OUTPUT_KEY]
    projection1, projection2 = tf.split(projection_outputs, 2, 0)
    contrast_loss, (contrast_logits, contrast_labels) = con_losses_obj(
        projection1=projection1,
        projection2=projection2)

    contrast_accuracy = tf.equal(
        tf.argmax(contrast_labels, axis=1), tf.argmax(contrast_logits, axis=1))
    contrast_accuracy = tf.reduce_mean(tf.cast(contrast_accuracy, tf.float32))

    contrast_prob = tf.nn.softmax(contrast_logits)
    contrast_entropy = -tf.reduce_mean(
        tf.reduce_sum(contrast_prob * tf.math.log(contrast_prob + 1e-8), -1))

    model_loss = contrast_loss

    losses = {
        'contrast_loss': contrast_loss,
        'contrast_accuracy': contrast_accuracy,
        'contrast_entropy': contrast_entropy
    }

    if self.task_config.model.supervised_head is not None:
      outputs = model_outputs[simclr_model.SUPERVISED_OUTPUT_KEY]
      labels = tf.concat([labels, labels], 0)

      if self.task_config.evaluation.one_hot:
        sup_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels,
                                                                        outputs)
      else:
        sup_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels,
                                                                        outputs)
      sup_loss = tf.reduce_mean(sup_loss)

      label_acc = tf.equal(tf.argmax(labels, axis=1),
                           tf.argmax(outputs, axis=1))
      label_acc = tf.reduce_mean(tf.cast(label_acc, tf.float32))

      model_loss = contrast_loss + sup_loss

      losses.update({
          'accuracy': label_acc,
          'supervised_loss': sup_loss,
      })

    total_loss = model_loss
    if aux_losses:
      reg_loss = tf.reduce_sum(aux_losses)
      total_loss = model_loss + reg_loss

    losses['total_loss'] = total_loss

    return losses

  def build_metrics(self, training=True):

    if training:
      metrics = []
      metric_names = [
          'total_loss',
          'contrast_loss',
          'contrast_accuracy',
          'contrast_entropy'
      ]
      if self.task_config.model.supervised_head:
        metric_names.extend(['supervised_loss', 'accuracy'])
      for name in metric_names:
        metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))
    else:
      k = self.task_config.evaluation.top_k
      if self.task_config.evaluation.one_hot:
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=k, name='top_{}_accuracy'.format(k))]
      else:
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=k, name='top_{}_accuracy'.format(k))]
    return metrics

  def train_step(self, inputs, model, optimizer, metrics=None):
    features, labels = inputs
    if (self.task_config.model.supervised_head is not None
        and self.task_config.evaluation.one_hot):
      num_classes = self.task_config.model.supervised_head.num_classes
      labels = tf.one_hot(labels, num_classes)

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      losses = self.build_losses(
          model_outputs=outputs, labels=labels, aux_losses=model.losses)

      scaled_loss = losses['total_loss'] / num_replicas
      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    logging.info('Trainable variables:')
    for var in tvars:
      logging.info(var.name)
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient when LossScaleOptimizer is used.
    if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {self.loss: losses['total_loss']}

    for m in metrics:
      m.update_state(losses[m.name])
      logs.update({m.name: m.result()})

    return logs

  def validation_step(self, inputs, model, metrics=None):
    if self.task_config.model.supervised_head is None:
      assert 'Skipping eval during pretraining without supervised head.'

    features, labels = inputs
    if self.task_config.evaluation.one_hot:
      num_classes = self.task_config.model.supervised_head.num_classes
      labels = tf.one_hot(labels, num_classes)

    outputs = model(
        features, training=False)[simclr_model.SUPERVISED_OUTPUT_KEY]
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

    logs = {self.loss: 0}

    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in model.metrics})

    return logs


@task_factory.register_task_cls(exp_cfg.SimCLRFinetuneTask)
class SimCLRFinetuneTask(base_task.Task):
  """A task for image classification."""

  def create_optimizer(self, optimizer_config: OptimizationConfig,
                       runtime_config: Optional[RuntimeConfig] = None):
    """Creates an TF optimizer from configurations.

    Args:
      optimizer_config: the parameters of the Optimization settings.
      runtime_config: the parameters of the runtime.

    Returns:
      A tf.optimizers.Optimizer object.
    """
    if (optimizer_config.optimizer.type == 'lars'
        and self.task_config.loss.l2_weight_decay > 0.0):
      raise ValueError('The l2_weight_decay cannot be used together with lars '
                       'optimizer. Please set it to 0.')

    opt_factory = optimization.OptimizerFactory(optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    # Configuring optimizer when loss_scale is set in runtime config. This helps
    # avoiding overflow/underflow for float16 computations.
    if runtime_config and runtime_config.loss_scale:
      optimizer = performance.configure_optimizer(
          optimizer,
          use_float16=runtime_config.mixed_precision_dtype == 'float16',
          loss_scale=runtime_config.loss_scale)

    return optimizer

  def build_model(self):
    model_config = self.task_config.model
    input_specs = tf.keras.layers.InputSpec(
        shape=[None] + model_config.input_size)

    l2_weight_decay = self.task_config.loss.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf.keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    backbone = backbones.factory.build_backbone(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)

    norm_activation_config = model_config.norm_activation
    projection_head_config = model_config.projection_head
    projection_head = simclr_head.ProjectionHead(
        proj_output_dim=projection_head_config.proj_output_dim,
        num_proj_layers=projection_head_config.num_proj_layers,
        ft_proj_idx=projection_head_config.ft_proj_idx,
        kernel_regularizer=l2_regularizer,
        use_sync_bn=norm_activation_config.use_sync_bn,
        norm_momentum=norm_activation_config.norm_momentum,
        norm_epsilon=norm_activation_config.norm_epsilon)

    supervised_head_config = model_config.supervised_head
    if supervised_head_config.zero_init:
      s_kernel_initializer = 'zeros'
    else:
      s_kernel_initializer = 'random_uniform'
    supervised_head = simclr_head.ClassificationHead(
        num_classes=supervised_head_config.num_classes,
        kernel_initializer=s_kernel_initializer,
        kernel_regularizer=l2_regularizer)

    model = simclr_model.SimCLRModel(
        input_specs=input_specs,
        backbone=backbone,
        projection_head=projection_head,
        supervised_head=supervised_head,
        mode=model_config.mode,
        backbone_trainable=model_config.backbone_trainable)

    logging.info(model.get_config())

    return model

  def initialize(self, model: tf.keras.Model):
    """Loading pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      return

    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    # Restoring checkpoint.
    if self.task_config.init_checkpoint_modules == 'all':
      ckpt = tf.train.Checkpoint(**model.checkpoint_items)
      status = ckpt.restore(ckpt_dir_or_file)
      status.assert_consumed()
    elif self.task_config.init_checkpoint_modules == 'backbone_projection':
      ckpt = tf.train.Checkpoint(backbone=model.backbone,
                                 projection_head=model.projection_head)
      status = ckpt.restore(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    elif self.task_config.init_checkpoint_modules == 'backbone':
      ckpt = tf.train.Checkpoint(backbone=model.backbone)
      status = ckpt.restore(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    else:
      assert "Only 'all' or 'backbone' can be used to initialize the model."

    # If the checkpoint is from pretraining, reset the following parameters
    model.backbone_trainable = self.task_config.model.backbone_trainable
    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def build_inputs(self, params, input_context=None):
    input_size = self.task_config.model.input_size

    if params.tfds_name:
      decoder = simclr_input.TFDSDecoder(params.decoder.decode_label)
    else:
      decoder = simclr_input.Decoder(params.decoder.decode_label)
    parser = simclr_input.Parser(
        output_size=input_size[:2],
        parse_label=params.parser.parse_label,
        test_crop=params.parser.test_crop,
        mode=params.parser.mode,
        dtype=params.dtype)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self, labels, model_outputs, aux_losses=None):
    """Sparse categorical cross entropy loss.

    Args:
      labels: labels.
      model_outputs: Output logits of the classifier.
      aux_losses: auxiliarly loss tensors, i.e. `losses` in keras.Model.

    Returns:
      The total loss tensor.
    """
    losses_config = self.task_config.loss
    if losses_config.one_hot:
      total_loss = tf.keras.losses.categorical_crossentropy(
          labels,
          model_outputs,
          from_logits=True,
          label_smoothing=losses_config.label_smoothing)
    else:
      total_loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, model_outputs, from_logits=True)

    total_loss = tf_utils.safe_mean(total_loss)
    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    return total_loss

  def build_metrics(self, training=True):
    """Gets streaming metrics for training/validation."""
    k = self.task_config.evaluation.top_k
    if self.task_config.evaluation.one_hot:
      metrics = [
          tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
          tf.keras.metrics.TopKCategoricalAccuracy(
              k=k, name='top_{}_accuracy'.format(k))]
    else:
      metrics = [
          tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
          tf.keras.metrics.SparseTopKCategoricalAccuracy(
              k=k, name='top_{}_accuracy'.format(k))]
    return metrics

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
    if self.task_config.loss.one_hot:
      num_classes = self.task_config.model.supervised_head.num_classes
      labels = tf.one_hot(labels, num_classes)

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(
          features, training=True)[simclr_model.SUPERVISED_OUTPUT_KEY]
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      loss = self.build_losses(
          model_outputs=outputs,
          labels=labels, aux_losses=model.losses)
      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = loss / num_replicas

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(
          optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    logging.info('Trainable variables:')
    for var in tvars:
      logging.info(var.name)
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient before apply_gradients when LossScaleOptimizer is
    # used.
    if isinstance(
        optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in model.metrics})
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
    if self.task_config.loss.one_hot:
      num_classes = self.task_config.model.supervised_head.num_classes
      labels = tf.one_hot(labels, num_classes)

    outputs = self.inference_step(
        features, model)[simclr_model.SUPERVISED_OUTPUT_KEY]
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)
    loss = self.build_losses(model_outputs=outputs,
                             labels=labels, aux_losses=model.losses)

    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in model.metrics})
    return logs
