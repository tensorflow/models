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

"""PointPillars task definition."""

import functools
from typing import Any, List, Mapping, Optional, Tuple

from absl import logging
import tensorflow as tf, tf_keras

from official.core import base_task
from official.core import task_factory
from official.projects.pointpillars.configs import pointpillars as cfg
from official.projects.pointpillars.dataloaders import decoders
from official.projects.pointpillars.dataloaders import parsers
from official.projects.pointpillars.modeling import factory
from official.projects.pointpillars.utils import utils
from official.vision.dataloaders import input_reader_factory
from official.vision.losses import focal_loss
from official.vision.losses import loss_utils


def pick_dataset_fn(file_type: str) -> Any:
  if file_type == 'tfrecord':
    return tf.data.TFRecordDataset
  if file_type == 'tfrecord_compressed':
    return functools.partial(tf.data.TFRecordDataset, compression_type='GZIP')
  raise ValueError('Unrecognized file_type: {}'.format(file_type))


def get_batch_size_per_replica(global_batch_size: int) -> int:
  """Get batch size per accelerator replica."""
  num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
  if global_batch_size < num_replicas:
    logging.warning('Global batch size is smaller than num replicas. '
                    'Set batch size per replica to 1.')
    return 1
  if global_batch_size % num_replicas != 0:
    raise ValueError(
        'global_batch_size {} is not a multiple of num_replicas {}'
        .format(global_batch_size, num_replicas))
  batch_size = int(global_batch_size / num_replicas)
  return batch_size


@task_factory.register_task_cls(cfg.PointPillarsTask)
class PointPillarsTask(base_task.Task):
  """A single-replica view of training procedure."""

  def __init__(self,
               params: cfg.PointPillarsTask,
               logging_dir: Optional[str] = None,
               name: Optional[str] = None):
    super().__init__(params, logging_dir, name)
    self._model = None
    self._attribute_heads = self.task_config.model.head.attribute_heads

  def build_model(self) -> tf_keras.Model:
    # Create only one model instance if this function is called multiple times.
    if self._model is not None:
      return self._model

    pillars_config = self.task_config.model.pillars
    input_specs = {
        'pillars':
            tf_keras.layers.InputSpec(
                shape=(None, pillars_config.num_pillars,
                       pillars_config.num_points_per_pillar,
                       pillars_config.num_features_per_point)),
        'indices':
            tf_keras.layers.InputSpec(
                shape=(None, pillars_config.num_pillars, 2), dtype='int32'),
    }

    train_batch_size = get_batch_size_per_replica(
        self.task_config.train_data.global_batch_size)
    eval_batch_size = get_batch_size_per_replica(
        self.task_config.validation_data.global_batch_size)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    l2_regularizer = (tf_keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    self._model = factory.build_pointpillars(
        input_specs=input_specs,
        model_config=self.task_config.model,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        l2_regularizer=l2_regularizer)
    return self._model

  def initialize(self, model: tf_keras.Model):
    """Loading pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      return

    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    if self.task_config.init_checkpoint_modules == 'all':
      ckpt = tf.train.Checkpoint(**model.checkpoint_items)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    else:
      ckpt_items = {}
      if 'backbone' in self.task_config.init_checkpoint_modules:
        ckpt_items.update(backbone=model.backbone)
      if 'decoder' in self.task_config.init_checkpoint_modules:
        ckpt_items.update(decoder=model.decoder)

      ckpt = tf.train.Checkpoint(**ckpt_items)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def build_inputs(
      self,
      params: cfg.DataConfig,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Build input dataset."""
    model_config = self.task_config.model
    if (model_config.classes != 'all' and
        model_config.num_classes != 2):
      raise ValueError('Model num_classes must be 2 when not for all classes.')

    decoder = decoders.ExampleDecoder(model_config.image, model_config.pillars)

    image_size = [model_config.image.height, model_config.image.width]
    anchor_sizes = [(a.length, a.width) for a in model_config.anchors]
    anchor_labeler_config = model_config.anchor_labeler
    parser = parsers.Parser(
        classes=model_config.classes,
        min_level=model_config.min_level,
        max_level=model_config.max_level,
        image_size=image_size,
        anchor_sizes=anchor_sizes,
        match_threshold=anchor_labeler_config.match_threshold,
        unmatched_threshold=anchor_labeler_config.unmatched_threshold,
        max_num_detections=model_config.detection_generator
        .max_num_detections,
        dtype=params.dtype,
    )
    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))
    dataset = reader.read(input_context=input_context)

    return dataset

  def compute_attribute_losses(
      self,
      outputs: Mapping[str, Any],
      labels: Mapping[str, Any],
      box_sample_weight: tf.Tensor) -> Mapping[str, float]:
    """Computes attribute loss."""
    att_loss_fn = tf_keras.losses.Huber(
        self.task_config.losses.huber_loss_delta,
        reduction=tf_keras.losses.Reduction.SUM)

    losses = {}
    total_loss = 0.0
    for head in self._attribute_heads:
      if head.type != 'regression':
        raise ValueError(f'Attribute type {head.type} not supported.')

      y_true_att = loss_utils.multi_level_flatten(
          labels['attribute_targets'][head.name], last_dim=head.size)
      y_pred_att = loss_utils.multi_level_flatten(
          outputs['attribute_outputs'][head.name], last_dim=head.size)
      if head.name == 'heading':
        # Direction aware loss, wrap the delta angle to [-pi, pi].
        # Otherwise for a loss that is symmetric to direction (i.e., heading 0
        # and pi are the same), we use a tf.sin transform.
        delta = utils.wrap_angle_rad(y_pred_att - y_true_att)
        loss = att_loss_fn(
            y_true=tf.zeros_like(delta),
            y_pred=delta,
            sample_weight=box_sample_weight)
      else:
        loss = att_loss_fn(
            y_true=y_true_att,
            y_pred=y_pred_att,
            sample_weight=box_sample_weight)
      total_loss += loss
      losses[head.name] = loss
    losses['total'] = total_loss
    return losses

  def compute_losses(
      self,
      outputs: Mapping[str, Any],
      labels: Mapping[str, Any],
      aux_losses: Optional[Any] = None) -> Mapping[str, float]:
    """Build losses."""
    params = self.task_config

    cls_loss_fn = focal_loss.FocalLoss(
        alpha=params.losses.focal_loss_alpha,
        gamma=params.losses.focal_loss_gamma,
        reduction=tf_keras.losses.Reduction.SUM)
    box_loss_fn = tf_keras.losses.Huber(
        params.losses.huber_loss_delta,
        reduction=tf_keras.losses.Reduction.SUM)

    # Sums all positives in a batch for normalization and avoids zero
    # num_positives_sum, which would lead to inf loss during training
    cls_sample_weight = labels['cls_weights']
    box_sample_weight = labels['box_weights']
    num_positives = tf.reduce_sum(box_sample_weight) + 1.0
    cls_sample_weight = cls_sample_weight / num_positives
    box_sample_weight = box_sample_weight / num_positives

    y_true_cls = loss_utils.multi_level_flatten(
        labels['cls_targets'], last_dim=None)
    y_true_cls = tf.one_hot(y_true_cls, params.model.num_classes)
    y_pred_cls = loss_utils.multi_level_flatten(
        outputs['cls_outputs'], last_dim=params.model.num_classes)
    y_true_box = loss_utils.multi_level_flatten(
        labels['box_targets'], last_dim=4)
    y_pred_box = loss_utils.multi_level_flatten(
        outputs['box_outputs'], last_dim=4)

    cls_loss = cls_loss_fn(
        y_true=y_true_cls, y_pred=y_pred_cls, sample_weight=cls_sample_weight)
    box_loss = box_loss_fn(
        y_true=y_true_box, y_pred=y_pred_box, sample_weight=box_sample_weight)
    attribute_losses = self.compute_attribute_losses(outputs, labels,
                                                     box_sample_weight)
    model_loss = (
        cls_loss + box_loss * params.losses.box_loss_weight +
        attribute_losses['total'] * params.losses.attribute_loss_weight)

    total_loss = model_loss
    if aux_losses:
      reg_loss = tf.reduce_sum(aux_losses)
      total_loss += reg_loss
    total_loss = params.losses.loss_weight * total_loss

    losses = {
        'class_loss': cls_loss,
        'box_loss': box_loss,
        'attribute_loss': attribute_losses['total'],
        'model_loss': model_loss,
        'total_loss': total_loss,
    }
    for head in self._attribute_heads:
      losses[head.name + '_loss'] = attribute_losses[head.name]
    return losses

  def build_metrics(self, training: bool = True) -> List[tf.metrics.Metric]:
    """Define metrics and how to calculate them."""
    # train/validation loss metrics
    loss_names = [
        'class_loss', 'box_loss', 'attribute_loss', 'model_loss', 'total_loss'
    ]
    for head in self._attribute_heads:
      loss_names.append(head.name + '_loss')
    metrics = []
    for name in loss_names:
      metrics.append(tf_keras.metrics.Mean(name, dtype=tf.float32))

    # Use a separate metric for WOD validation.
    if not training:
      if self.task_config.use_wod_metrics:
        # To use Waymo open dataset metrics, please install one of the pip
        # package `waymo-open-dataset-tf-*` from
        # https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md#use-pre-compiled-pippip3-packages-for-linux
        # Note that the package is built with specific tensorflow version and
        # will produce error if it does not match the tf version that is
        # currently used.
        try:
          from official.projects.pointpillars.utils import wod_detection_evaluator  # pylint: disable=g-import-not-at-top
        except ModuleNotFoundError:
          logging.error('waymo-open-dataset should be installed to enable Waymo'
                        ' evaluator.')
          raise
        self._wod_metric = wod_detection_evaluator.create_evaluator(
            self.task_config.model)
    return metrics

  def train_step(
      self,
      inputs: Tuple[Any, Any],
      model: tf_keras.Model,
      optimizer: tf_keras.optimizers.Optimizer,
      metrics: Optional[List[tf.metrics.Metric]] = None) -> Mapping[str, Any]:
    """Does forward and backward."""
    features, labels = inputs
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(pillars=features['pillars'],
                      indices=features['indices'],
                      training=True)
      losses = self.compute_losses(
          outputs=outputs, labels=labels, aux_losses=model.losses)

      # Computes per-replica loss.
      scaled_loss = losses['total_loss'] / num_replicas

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient when LossScaleOptimizer is used.
    if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    # For updating trainer.train_loss
    logs = {self.loss: losses['total_loss']}
    # For updating trainer.train_metrics
    if metrics:
      for m in metrics:
        m.update_state(losses[m.name])
    return logs

  def validation_step(
      self,
      inputs: Tuple[Any, Any],
      model: tf_keras.Model,
      metrics: Optional[List[tf.metrics.Metric]] = None) -> Mapping[str, Any]:
    """Validatation step."""
    features, labels = inputs
    outputs = model(pillars=features['pillars'],
                    indices=features['indices'],
                    image_shape=labels['image_shape'],
                    anchor_boxes=labels['anchor_boxes'],
                    training=False)
    losses = self.compute_losses(
        outputs=outputs, labels=labels, aux_losses=model.losses)

    # For updating trainer.validation_loss
    logs = {self.loss: losses['total_loss']}
    # For updating trainer.validation_metrics
    if metrics:
      for m in metrics:
        m.update_state(losses[m.name])
    if self.task_config.use_wod_metrics:
      logs.update(
          {self._wod_metric.name: (labels['groundtruths'], outputs)})
    return logs

  def aggregate_logs(self,
                     state: Any = None,
                     step_outputs: Any = None) -> Any:
    """Called after each validation_step to update metrics."""
    logging.log_every_n(logging.INFO,
                        'Aggregating metrics after one evaluation step.', 1000)
    if self.task_config.use_wod_metrics:
      if state is None:
        self._wod_metric.reset_states()
      self._wod_metric.update_state(step_outputs[self._wod_metric.name][0],
                                    step_outputs[self._wod_metric.name][1])
    if state is None:
      state = True
    return state

  def reduce_aggregated_logs(self,
                             aggregated_logs: Any,
                             global_step: Optional[tf.Tensor] = None) -> Any:
    """Called after eval_end to calculate metrics."""
    logging.info('Reducing aggregated metrics after one evaluation cycle.')
    logs = {}
    if self.task_config.use_wod_metrics:
      logs.update(self._wod_metric.result())
    return logs
