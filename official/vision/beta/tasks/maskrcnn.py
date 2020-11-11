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
"""RetinaNet task definition."""

from absl import logging
import tensorflow as tf
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.vision.beta.configs import maskrcnn as exp_cfg
from official.vision.beta.dataloaders import maskrcnn_input
from official.vision.beta.dataloaders import tf_example_decoder
from official.vision.beta.dataloaders import tf_example_label_map_decoder
from official.vision.beta.evaluation import coco_evaluator
from official.vision.beta.losses import maskrcnn_losses
from official.vision.beta.modeling import factory


def zero_out_disallowed_class_ids(batch_class_ids, allowed_class_ids):
  """Zero out IDs of classes not in allowed_class_ids.

  Args:
    batch_class_ids: A [batch_size, num_instances] int tensor of input
      class IDs.
    allowed_class_ids: A python list of class IDs which we want to allow.

  Returns:
      filtered_class_ids: A [batch_size, num_instances] int tensor with any
        class ID not in allowed_class_ids set to 0.
  """

  allowed_class_ids = tf.constant(allowed_class_ids,
                                  dtype=batch_class_ids.dtype)

  match_ids = (batch_class_ids[:, :, tf.newaxis] ==
               allowed_class_ids[tf.newaxis, tf.newaxis, :])

  match_ids = tf.reduce_any(match_ids, axis=2)
  return tf.where(match_ids, batch_class_ids, tf.zeros_like(batch_class_ids))


@task_factory.register_task_cls(exp_cfg.MaskRCNNTask)
class MaskRCNNTask(base_task.Task):
  """A single-replica view of training procedure.

  Mask R-CNN task provides artifacts for training/evalution procedures,
  including loading/iterating over Datasets, initializing the model, calculating
  the loss, post-processing, and customized metrics with reduction.
  """

  def build_model(self):
    """Build Mask R-CNN model."""

    input_specs = tf.keras.layers.InputSpec(
        shape=[None] + self.task_config.model.input_size)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf.keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = factory.build_maskrcnn(
        input_specs=input_specs,
        model_config=self.task_config.model,
        l2_regularizer=l2_regularizer)
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
    """Build input dataset."""
    decoder_cfg = params.decoder.get()
    if params.decoder.type == 'simple_decoder':
      decoder = tf_example_decoder.TfExampleDecoder(
          include_mask=self._task_config.model.include_mask,
          regenerate_source_id=decoder_cfg.regenerate_source_id)
    elif params.decoder.type == 'label_map_decoder':
      decoder = tf_example_label_map_decoder.TfExampleDecoderLabelMap(
          label_map=decoder_cfg.label_map,
          include_mask=self._task_config.model.include_mask,
          regenerate_source_id=decoder_cfg.regenerate_source_id)
    else:
      raise ValueError('Unknown decoder type: {}!'.format(params.decoder.type))

    parser = maskrcnn_input.Parser(
        output_size=self.task_config.model.input_size[:2],
        min_level=self.task_config.model.min_level,
        max_level=self.task_config.model.max_level,
        num_scales=self.task_config.model.anchor.num_scales,
        aspect_ratios=self.task_config.model.anchor.aspect_ratios,
        anchor_size=self.task_config.model.anchor.anchor_size,
        dtype=params.dtype,
        rpn_match_threshold=params.parser.rpn_match_threshold,
        rpn_unmatched_threshold=params.parser.rpn_unmatched_threshold,
        rpn_batch_size_per_im=params.parser.rpn_batch_size_per_im,
        rpn_fg_fraction=params.parser.rpn_fg_fraction,
        aug_rand_hflip=params.parser.aug_rand_hflip,
        aug_scale_min=params.parser.aug_scale_min,
        aug_scale_max=params.parser.aug_scale_max,
        skip_crowd_during_training=params.parser.skip_crowd_during_training,
        max_num_instances=params.parser.max_num_instances,
        include_mask=self._task_config.model.include_mask,
        mask_crop_size=params.parser.mask_crop_size)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))
    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self, outputs, labels, aux_losses=None):
    """Build Mask R-CNN losses."""
    params = self.task_config

    rpn_score_loss_fn = maskrcnn_losses.RpnScoreLoss(
        tf.shape(outputs['box_outputs'])[1])
    rpn_box_loss_fn = maskrcnn_losses.RpnBoxLoss(
        params.losses.rpn_huber_loss_delta)
    rpn_score_loss = tf.reduce_mean(
        rpn_score_loss_fn(
            outputs['rpn_scores'], labels['rpn_score_targets']))
    rpn_box_loss = tf.reduce_mean(
        rpn_box_loss_fn(
            outputs['rpn_boxes'], labels['rpn_box_targets']))

    frcnn_cls_loss_fn = maskrcnn_losses.FastrcnnClassLoss()
    frcnn_box_loss_fn = maskrcnn_losses.FastrcnnBoxLoss(
        params.losses.frcnn_huber_loss_delta)
    frcnn_cls_loss = tf.reduce_mean(
        frcnn_cls_loss_fn(
            outputs['class_outputs'], outputs['class_targets']))
    frcnn_box_loss = tf.reduce_mean(
        frcnn_box_loss_fn(
            outputs['box_outputs'],
            outputs['class_targets'],
            outputs['box_targets']))

    if params.model.include_mask:
      mask_loss_fn = maskrcnn_losses.MaskrcnnLoss()
      mask_class_targets = outputs['mask_class_targets']
      if self._task_config.allowed_mask_class_ids is not None:
        # Classes with ID=0 are ignored by mask_loss_fn in loss computation.
        mask_class_targets = zero_out_disallowed_class_ids(
            mask_class_targets, self._task_config.allowed_mask_class_ids)

      mask_loss = tf.reduce_mean(
          mask_loss_fn(
              outputs['mask_outputs'],
              outputs['mask_targets'],
              mask_class_targets))
    else:
      mask_loss = 0.0

    model_loss = (
        params.losses.rpn_score_weight * rpn_score_loss +
        params.losses.rpn_box_weight * rpn_box_loss +
        params.losses.frcnn_class_weight * frcnn_cls_loss +
        params.losses.frcnn_box_weight * frcnn_box_loss +
        params.losses.mask_weight * mask_loss)

    total_loss = model_loss
    if aux_losses:
      reg_loss = tf.reduce_sum(aux_losses)
      total_loss = model_loss + reg_loss

    losses = {
        'total_loss': total_loss,
        'rpn_score_loss': rpn_score_loss,
        'rpn_box_loss': rpn_box_loss,
        'frcnn_cls_loss': frcnn_cls_loss,
        'frcnn_box_loss': frcnn_box_loss,
        'mask_loss': mask_loss,
        'model_loss': model_loss,
    }
    return losses

  def build_metrics(self, training=True):
    """Build detection metrics."""
    metrics = []
    if training:
      metric_names = [
          'total_loss',
          'rpn_score_loss',
          'rpn_box_loss',
          'frcnn_cls_loss',
          'frcnn_box_loss',
          'mask_loss',
          'model_loss'
      ]
      for name in metric_names:
        metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))

    else:
      self.coco_metric = coco_evaluator.COCOEvaluator(
          annotation_file=self._task_config.annotation_file,
          include_mask=self._task_config.model.include_mask,
          per_category_metrics=self._task_config.per_category_metrics)

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
    images, labels = inputs
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(
          images,
          image_shape=labels['image_info'][:, 1, :],
          anchor_boxes=labels['anchor_boxes'],
          gt_boxes=labels['gt_boxes'],
          gt_classes=labels['gt_classes'],
          gt_masks=(labels['gt_masks'] if self.task_config.model.include_mask
                    else None),
          training=True)
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      losses = self.build_losses(
          outputs=outputs, labels=labels, aux_losses=model.losses)
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
        logs.update({m.name: m.result()})

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
    images, labels = inputs

    outputs = model(
        images,
        anchor_boxes=labels['anchor_boxes'],
        image_shape=labels['image_info'][:, 1, :],
        training=False)

    logs = {self.loss: 0}
    coco_model_outputs = {
        'detection_boxes': outputs['detection_boxes'],
        'detection_scores': outputs['detection_scores'],
        'detection_classes': outputs['detection_classes'],
        'num_detections': outputs['num_detections'],
        'source_id': labels['groundtruths']['source_id'],
        'image_info': labels['image_info']
    }
    if self.task_config.model.include_mask:
      coco_model_outputs.update({
          'detection_masks': outputs['detection_masks'],
      })
    logs.update({
        self.coco_metric.name: (labels['groundtruths'], coco_model_outputs)
    })
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    if state is None:
      self.coco_metric.reset_states()
      state = self.coco_metric
    self.coco_metric.update_state(
        step_outputs[self.coco_metric.name][0],
        step_outputs[self.coco_metric.name][1])
    return state

  def reduce_aggregated_logs(self, aggregated_logs):
    return self.coco_metric.result()
