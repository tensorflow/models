# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Panoptic Deeplab task definition."""
from typing import Any, Dict, List, Mapping, Optional, Tuple

from absl import logging
import tensorflow as tf

from official.common import dataset_fn
from official.core import base_task
from official.core import task_factory
from official.vision.beta.projects.panoptic_maskrcnn.configs import panoptic_deeplab as exp_cfg
from official.vision.beta.projects.panoptic_maskrcnn.dataloaders import panoptic_deeplab_input
from official.vision.beta.projects.panoptic_maskrcnn.losses import panoptic_deeplab_losses
from official.vision.beta.projects.panoptic_maskrcnn.modeling import factory
from official.vision.dataloaders import input_reader_factory
from official.vision.evaluation import panoptic_quality_evaluator
from official.vision.evaluation import segmentation_metrics


@task_factory.register_task_cls(exp_cfg.PanopticDeeplabTask)
class PanopticDeeplabTask(base_task.Task):
  """A task for Panoptic Deeplab."""

  def build_model(self):
    """Builds panoptic deeplab model."""
    input_specs = tf.keras.layers.InputSpec(
        shape=[None] + self.task_config.model.input_size)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf.keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = factory.build_panoptic_deeplab(
        input_specs=input_specs,
        model_config=self.task_config.model,
        l2_regularizer=l2_regularizer)
    return model

  def initialize(self, model: tf.keras.Model):
    """Loads pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      return

    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    # Restoring checkpoint.
    if 'all' in self.task_config.init_checkpoint_modules:
      ckpt = tf.train.Checkpoint(**model.checkpoint_items)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    else:
      ckpt_items = {}
      if 'backbone' in self.task_config.init_checkpoint_modules:
        ckpt_items.update(backbone=model.backbone)
      if 'decoder' in self.task_config.init_checkpoint_modules:
        ckpt_items.update(semantic_decoder=model.semantic_decoder)
        if not self.task_config.model.shared_decoder:
          ckpt_items.update(instance_decoder=model.instance_decoder)

      ckpt = tf.train.Checkpoint(**ckpt_items)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def build_inputs(self,
                   params: exp_cfg.DataConfig,
                   input_context: Optional[tf.distribute.InputContext] = None):
    """Builds panoptic deeplab input."""
    decoder_cfg = params.decoder.get()

    if params.decoder.type == 'simple_decoder':
      decoder = panoptic_deeplab_input.TfExampleDecoder(
          regenerate_source_id=decoder_cfg.regenerate_source_id,
          panoptic_category_mask_key=decoder_cfg.panoptic_category_mask_key,
          panoptic_instance_mask_key=decoder_cfg.panoptic_instance_mask_key)
    else:
      raise ValueError('Unknown decoder type: {}!'.format(params.decoder.type))

    parser = panoptic_deeplab_input.Parser(
        output_size=self.task_config.model.input_size[:2],
        ignore_label=params.parser.ignore_label,
        resize_eval_groundtruth=params.parser.resize_eval_groundtruth,
        groundtruth_padded_size=params.parser.groundtruth_padded_size,
        aug_scale_min=params.parser.aug_scale_min,
        aug_scale_max=params.parser.aug_scale_max,
        aug_rand_hflip=params.parser.aug_rand_hflip,
        aug_type=params.parser.aug_type,
        sigma=params.parser.sigma,
        dtype=params.parser.dtype)

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self,
                   labels: Mapping[str, tf.Tensor],
                   model_outputs: Mapping[str, tf.Tensor],
                   aux_losses: Optional[Any] = None):
    """Panoptic deeplab losses.

    Args:
      labels: labels.
      model_outputs: Output logits from panoptic deeplab.
      aux_losses: auxiliarly loss tensors, i.e. `losses` in keras.Model.

    Returns:
      The total loss tensor.
    """
    loss_config = self._task_config.losses
    segmentation_loss_fn = panoptic_deeplab_losses.WeightedBootstrappedCrossEntropyLoss(
        loss_config.label_smoothing,
        loss_config.class_weights,
        loss_config.ignore_label,
        top_k_percent_pixels=loss_config.top_k_percent_pixels)
    instance_center_heatmap_loss_fn = panoptic_deeplab_losses.CenterHeatmapLoss(
    )
    instance_center_offset_loss_fn = panoptic_deeplab_losses.CenterOffsetLoss()

    semantic_weights = tf.cast(
        labels['semantic_weights'],
        dtype=model_outputs['instance_centers_heatmap'].dtype)
    things_mask = tf.cast(
        tf.squeeze(labels['things_mask'], axis=3),
        dtype=model_outputs['instance_centers_heatmap'].dtype)
    valid_mask = tf.cast(
        tf.squeeze(labels['valid_mask'], axis=3),
        dtype=model_outputs['instance_centers_heatmap'].dtype)

    segmentation_loss = segmentation_loss_fn(
        model_outputs['segmentation_outputs'],
        labels['category_mask'],
        sample_weight=semantic_weights)
    instance_center_heatmap_loss = instance_center_heatmap_loss_fn(
        model_outputs['instance_centers_heatmap'],
        labels['instance_centers_heatmap'],
        sample_weight=valid_mask)
    instance_center_offset_loss = instance_center_offset_loss_fn(
        model_outputs['instance_centers_offset'],
        labels['instance_centers_offset'],
        sample_weight=things_mask)

    model_loss = (
        loss_config.segmentation_loss_weight * segmentation_loss +
        loss_config.center_heatmap_loss_weight * instance_center_heatmap_loss +
        loss_config.center_offset_loss_weight * instance_center_offset_loss)

    total_loss = model_loss
    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    losses = {
        'total_loss': total_loss,
        'model_loss': model_loss,
        'segmentation_loss': segmentation_loss,
        'instance_center_heatmap_loss': instance_center_heatmap_loss,
        'instance_center_offset_loss': instance_center_offset_loss
    }

    return losses

  def build_metrics(self, training: bool = True) -> List[
      tf.keras.metrics.Metric]:
    """Build metrics."""
    eval_config = self.task_config.evaluation
    metrics = []
    if training:
      metric_names = [
          'total_loss',
          'segmentation_loss',
          'instance_center_heatmap_loss',
          'instance_center_offset_loss',
          'model_loss']
      for name in metric_names:
        metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))

      if eval_config.report_train_mean_iou:
        self.train_mean_iou = segmentation_metrics.MeanIoU(
            name='train_mean_iou',
            num_classes=self.task_config.model.num_classes,
            rescale_predictions=False,
            dtype=tf.float32)
    else:
      rescale_predictions = (not self.task_config.validation_data.parser
                             .resize_eval_groundtruth)
      self.perclass_iou_metric = segmentation_metrics.PerClassIoU(
          name='per_class_iou',
          num_classes=self.task_config.model.num_classes,
          rescale_predictions=rescale_predictions,
          dtype=tf.float32)

      if isinstance(tf.distribute.get_strategy(), tf.distribute.TPUStrategy):
        self._process_iou_metric_on_cpu = True
      else:
        self._process_iou_metric_on_cpu = False

      if self.task_config.model.generate_panoptic_masks:
        self.panoptic_quality_metric = panoptic_quality_evaluator.PanopticQualityEvaluator(
            num_categories=self.task_config.model.num_classes,
            ignored_label=eval_config.ignored_label,
            max_instances_per_category=eval_config.max_instances_per_category,
            offset=eval_config.offset,
            is_thing=eval_config.is_thing,
            rescale_predictions=eval_config.rescale_predictions)

    # Update state on CPU if TPUStrategy due to dynamic resizing.
    self._process_iou_metric_on_cpu = isinstance(
        tf.distribute.get_strategy(),
        tf.distribute.TPUStrategy)

    return metrics

  def train_step(
      self,
      inputs: Tuple[Any, Any],
      model: tf.keras.Model,
      optimizer: tf.keras.optimizers.Optimizer,
      metrics: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Does forward and backward.

    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    images, labels = inputs
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

    with tf.GradientTape() as tape:
      outputs = model(
          inputs=images,
          image_info=labels['image_info'],
          training=True)
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      losses = self.build_losses(
          labels=labels,
          model_outputs=outputs,
          aux_losses=model.losses)
      scaled_loss = losses['total_loss'] / num_replicas

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient when LossScaleOptimizer is used.
    if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {self.loss: losses['total_loss']}

    if metrics:
      for m in metrics:
        m.update_state(losses[m.name])

    if self.task_config.evaluation.report_train_mean_iou:
      segmentation_labels = {
          'masks': labels['category_mask'],
          'valid_masks': labels['valid_mask'],
          'image_info': labels['image_info']
      }
      self.process_metrics(
          metrics=[self.train_mean_iou],
          labels=segmentation_labels,
          model_outputs=outputs['segmentation_outputs'])
      logs.update({
          self.train_mean_iou.name:
              self.train_mean_iou.result()
      })

    return logs

  def validation_step(
      self,
      inputs: Tuple[Any, Any],
      model: tf.keras.Model,
      metrics: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Validatation step.

    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    images, labels = inputs

    outputs = model(
        inputs=images,
        image_info=labels['image_info'],
        training=False)

    logs = {self.loss: 0}
    segmentation_labels = {
        'masks': labels['category_mask'],
        'valid_masks': labels['valid_mask'],
        'image_info': labels['image_info']
    }

    if self._process_iou_metric_on_cpu:
      logs.update({
          self.perclass_iou_metric.name:
              (segmentation_labels, outputs['segmentation_outputs'])
      })
    else:
      self.perclass_iou_metric.update_state(
          segmentation_labels,
          outputs['segmentation_outputs'])

    if self.task_config.model.generate_panoptic_masks:
      pq_metric_labels = {
          'category_mask':
              tf.squeeze(labels['category_mask'], axis=3),
          'instance_mask':
              tf.squeeze(labels['instance_mask'], axis=3),
          'image_info': labels['image_info']
      }
      panoptic_outputs = {
          'category_mask':
              outputs['category_mask'],
          'instance_mask':
              outputs['instance_mask'],
      }
      logs.update({
          self.panoptic_quality_metric.name:
              (pq_metric_labels, panoptic_outputs)})
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    if state is None:
      self.perclass_iou_metric.reset_states()
      state = [self.perclass_iou_metric]
      if self.task_config.model.generate_panoptic_masks:
        state += [self.panoptic_quality_metric]

    if self._process_iou_metric_on_cpu:
      self.perclass_iou_metric.update_state(
          step_outputs[self.perclass_iou_metric.name][0],
          step_outputs[self.perclass_iou_metric.name][1])

    if self.task_config.model.generate_panoptic_masks:
      self.panoptic_quality_metric.update_state(
          step_outputs[self.panoptic_quality_metric.name][0],
          step_outputs[self.panoptic_quality_metric.name][1])

    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    result = {}
    ious = self.perclass_iou_metric.result()
    if self.task_config.evaluation.report_per_class_iou:
      for i, value in enumerate(ious.numpy()):
        result.update({'segmentation_iou/class_{}'.format(i): value})

    # Computes mean IoU
    result.update({'segmentation_mean_iou': tf.reduce_mean(ious).numpy()})

    if self.task_config.model.generate_panoptic_masks:
      panoptic_quality_results = self.panoptic_quality_metric.result()
      for k, value in panoptic_quality_results.items():
        if k.endswith('per_class'):
          if self.task_config.evaluation.report_per_class_pq:
            for i, per_class_value in enumerate(value):
              metric_key = 'panoptic_quality/{}/class_{}'.format(k, i)
              result[metric_key] = per_class_value
          else:
            continue
        else:
          result['panoptic_quality/{}'.format(k)] = value

    return result
