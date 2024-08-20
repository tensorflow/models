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

"""Panoptic MaskRCNN task definition."""
from typing import Any, Dict, List, Mapping, Optional, Tuple

import tensorflow as tf, tf_keras

from official.common import dataset_fn
from official.core import task_factory
from official.projects.centernet.ops import loss_ops
from official.projects.maskconver.configs import maskconver as exp_cfg
from official.projects.maskconver.dataloaders import maskconver_segmentation_input
from official.projects.maskconver.dataloaders import panoptic_maskrcnn_input
from official.projects.maskconver.losses import maskconver_losses
from official.projects.maskconver.modeling import factory
from official.projects.volumetric_models.losses import segmentation_losses as volumeteric_segmentation_losses
from official.vision.dataloaders import input_reader_factory
from official.vision.dataloaders import segmentation_input
from official.vision.evaluation import panoptic_quality_evaluator
from official.vision.evaluation import segmentation_metrics
from official.vision.tasks import maskrcnn
from official.vision.tasks import semantic_segmentation


@task_factory.register_task_cls(exp_cfg.MaskConverTask)
class PanopticMaskRCNNTask(maskrcnn.MaskRCNNTask):

  """A single-replica view of training procedure.

  Panoptic Mask R-CNN task provides artifacts for training/evalution procedures,
  including loading/iterating over Datasets, initializing the model, calculating
  the loss, post-processing, and customized metrics with reduction.
  """

  def build_model(self) -> tf_keras.Model:
    """Build Panoptic Mask R-CNN model."""

    tf_keras.utils.set_random_seed(0)
    tf.config.experimental.enable_op_determinism()
    input_specs = tf_keras.layers.InputSpec(
        shape=[None] + self.task_config.model.input_size)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf_keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = factory.build_maskconver_model(
        input_specs=input_specs,
        model_config=self.task_config.model,
        l2_regularizer=l2_regularizer)
    return model

  def build_inputs(
      self,
      params: exp_cfg.DataConfig,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Build input dataset."""
    decoder_cfg = params.decoder.get()
    if params.decoder.type == 'simple_decoder':
      decoder = panoptic_maskrcnn_input.TfExampleDecoder(
          regenerate_source_id=decoder_cfg.regenerate_source_id,
          mask_binarize_threshold=decoder_cfg.mask_binarize_threshold,
          include_panoptic_masks=decoder_cfg.include_panoptic_masks,
          panoptic_category_mask_key=decoder_cfg.panoptic_category_mask_key,
          panoptic_instance_mask_key=decoder_cfg.panoptic_instance_mask_key)
    else:
      raise ValueError('Unknown decoder type: {}!'.format(params.decoder.type))

    parser = panoptic_maskrcnn_input.Parser(
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
        max_num_instances=self.task_config.model.num_instances,
        mask_crop_size=params.parser.mask_crop_size,
        segmentation_resize_eval_groundtruth=params.parser
        .segmentation_resize_eval_groundtruth,
        segmentation_groundtruth_padded_size=params.parser
        .segmentation_groundtruth_padded_size,
        segmentation_ignore_label=params.parser.segmentation_ignore_label,
        panoptic_ignore_label=params.parser.panoptic_ignore_label,
        include_panoptic_masks=params.parser.include_panoptic_masks,
        num_panoptic_categories=self.task_config.model.num_classes,
        num_thing_categories=self.task_config.model.num_thing_classes,
        level=self.task_config.model.level,
        gaussian_iou=params.parser.gaussaian_iou,
        aug_type=params.parser.aug_type,
        max_num_stuff_centers=params.parser.max_num_stuff_centers)

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))
    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self,
                   outputs: Mapping[str, Any],
                   labels: Mapping[str, Any],
                   aux_losses: Optional[Any] = None,
                   step=None) -> Dict[str, tf.Tensor]:
    """Build Panoptic Mask R-CNN losses."""
    loss_params = self._task_config.losses

    batch_size = tf.cast(tf.shape(labels['num_instances'])[0], tf.float32)
    center_loss_fn = maskconver_losses.PenaltyReducedLogisticFocalLoss(
        alpha=loss_params.alpha, beta=loss_params.beta)
    mask_loss_fn = maskconver_losses.PenaltyReducedLogisticFocalLoss()

    # Calculate center heatmap loss
    # TODO(arashwan): add valid weights.
    # output_unpad_image_shapes = labels['image_info'][:, 0, :]
    # valid_anchor_weights = loss_ops.get_valid_anchor_weights_in_flattened_image(  # pylint: disable=line-too-long
    #     output_unpad_image_shapes, h, w)
    # valid_anchor_weights = tf.expand_dims(valid_anchor_weights, 2)

    true_flattened_ct_heatmap = loss_ops.flatten_spatial_dimensions(
        labels['panoptic_heatmaps'])
    true_flattened_ct_heatmap = tf.cast(true_flattened_ct_heatmap, tf.float32)

    pred_flattened_ct_heatmap = loss_ops.flatten_spatial_dimensions(
        outputs['class_heatmaps'])
    pred_flattened_ct_heatmap = tf.cast(pred_flattened_ct_heatmap, tf.float32)
    center_padding_mask = 1 - labels['panoptic_padding_mask'][:, :, :, None]
    center_padding_mask = tf.image.resize(
        center_padding_mask, tf.shape(
            labels['panoptic_heatmaps'])[1:3], method='nearest')
    center_padding_mask = tf.maximum(center_padding_mask, 0.0)
    center_padding_mask = center_padding_mask * tf.ones_like(labels['panoptic_heatmaps'])
    weights_flattened_mask = loss_ops.flatten_spatial_dimensions(
        center_padding_mask)
    center_loss = center_loss_fn(
        target_tensor=true_flattened_ct_heatmap,
        prediction_tensor=pred_flattened_ct_heatmap,
        weights=weights_flattened_mask)

    center_loss = tf.reduce_sum(
        center_loss / (labels['num_instances'][:, None, None] + 1.0))  / batch_size

    gt_masks = labels['panoptic_masks']
    gt_mask_weights = labels['panoptic_mask_weights'][:, None, None, :] * tf.ones_like(gt_masks)
    panoptic_padding_mask = labels['panoptic_padding_mask'][:, :, :, None] * tf.ones_like(gt_masks)

    true_flattened_masks = loss_ops.flatten_spatial_dimensions(
        gt_masks)
    true_flattened_ct_heatmap = tf.cast(true_flattened_ct_heatmap, tf.float32)
    predicted_masks = tf.cast(outputs['mask_proposal_logits'], tf.float32)
    predicted_masks = tf.image.resize(
        predicted_masks, tf.shape(gt_masks)[1:3], method='bilinear')
    pred_flattened_masks = loss_ops.flatten_spatial_dimensions(predicted_masks)
    mask_loss = tf.cast(0.0, tf.float32)
    mask_loss_fn = tf_keras.losses.BinaryCrossentropy(
        from_logits=True,
        label_smoothing=0.0,
        axis=-1,
        reduction=tf_keras.losses.Reduction.NONE,
        name='binary_crossentropy')
    mask_weights = tf.reshape(
        tf.cast(true_flattened_masks >= 0, tf.float32),
        [-1, 1]) * tf.reshape(gt_mask_weights, [-1, 1])  * tf.reshape(
            (1 - panoptic_padding_mask), [-1, 1])
    mask_loss = mask_loss_fn(
        tf.reshape(gt_masks, [-1, 1]),
        tf.reshape(pred_flattened_masks, [-1, 1]),
        sample_weight=mask_weights)
    mask_loss = tf.reduce_sum(mask_loss) / (tf.reduce_sum(mask_weights) + 1.0)

    # Dice loss
    _, h, w, _ = gt_masks.get_shape().as_list()
    masked_predictions = tf.sigmoid(predicted_masks) * gt_mask_weights * (1 - panoptic_padding_mask)
    masked_gt_masks = gt_masks * gt_mask_weights * (1 - panoptic_padding_mask)

    masked_predictions = tf.transpose(masked_predictions, [0, 3, 1, 2])
    masked_predictions = tf.reshape(masked_predictions, [-1, h, w, 1])
    masked_gt_masks = tf.transpose(masked_gt_masks, [0, 3, 1, 2])
    masked_gt_masks = tf.reshape(masked_gt_masks, [-1, h, w, 1])

    dice_loss_fn = volumeteric_segmentation_losses.SegmentationLossDiceScore(
        metric_type='adaptive', axis=(2, 3))
    dice_loss = dice_loss_fn(logits=masked_predictions, labels=masked_gt_masks)

    total_loss = (center_loss + loss_params.mask_weight * mask_loss + loss_params.mask_weight * dice_loss)

    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    total_loss = loss_params.loss_weight * total_loss

    losses = {'total_loss': total_loss,
              'mask_loss': mask_loss,
              'center_loss': center_loss,
              'dice_loss': dice_loss}
    return losses

  def build_metrics(self, training: bool = True) -> List[
      tf_keras.metrics.Metric]:
    """Build detection metrics."""
    metrics = []
    if training:
      metric_names = [
          'total_loss',
          'center_loss',
          'mask_loss',
          'dice_loss',
      ]
      for name in metric_names:
        metrics.append(tf_keras.metrics.Mean(name, dtype=tf.float32))

    else:
      pq_config = self.task_config.panoptic_quality_evaluator
      self.panoptic_quality_metric = (
          panoptic_quality_evaluator.PanopticQualityEvaluator(
              num_categories=pq_config.num_categories,
              ignored_label=pq_config.ignored_label,
              max_instances_per_category=pq_config.max_instances_per_category,
              offset=pq_config.offset,
              is_thing=pq_config.is_thing,
              rescale_predictions=pq_config.rescale_predictions))
    return metrics

  def train_step(self,
                 inputs: Tuple[Any, Any],
                 model: tf_keras.Model,
                 optimizer: tf_keras.optimizers.Optimizer,
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
          images,
          box_indices=labels['panoptic_box_indices'],
          classes=labels['panoptic_classes'],
          training=True)
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      losses = self.build_losses(
          outputs=outputs,
          labels=labels,
          aux_losses=model.losses,
          step=optimizer.iterations)
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

    logs = {self.loss: losses['total_loss']}

    if metrics:
      for m in metrics:
        m.update_state(losses[m.name])

    return logs

  def validation_step(self,
                      inputs: Tuple[Any, Any],
                      model: tf_keras.Model,
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
        images,
        image_info=labels['image_info'],
        training=False)

    logs = {self.loss: 0}

    pq_metric_labels = {
        'category_mask':
            labels['groundtruths']['gt_panoptic_category_mask'],
        'instance_mask':
            labels['groundtruths']['gt_panoptic_instance_mask'],
        'image_info': labels['image_info']
    }
    logs.update({
        self.panoptic_quality_metric.name:
            (pq_metric_labels, outputs['panoptic_outputs'])})
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    if state is None:
      self.panoptic_quality_metric.reset_states()
      state = [self.panoptic_quality_metric]

    self.panoptic_quality_metric.update_state(
        step_outputs[self.panoptic_quality_metric.name][0],
        step_outputs[self.panoptic_quality_metric.name][1])

    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    result = {}

    report_per_class_metrics = (
        self.task_config.panoptic_quality_evaluator.report_per_class_metrics)
    panoptic_quality_results = self.panoptic_quality_metric.result()
    for k, value in panoptic_quality_results.items():
      if k.endswith('per_class'):
        if report_per_class_metrics:
          for i, per_class_value in enumerate(value):
            metric_key = 'panoptic_quality/{}/class_{}'.format(k, i)
            result[metric_key] = per_class_value
        else:
          continue
      else:
        result['panoptic_quality/{}'.format(k)] = value
    return result


@task_factory.register_task_cls(exp_cfg.MaskConverSegTask)
class MaskConverSegmentation(semantic_segmentation.SemanticSegmentationTask):

  """A single-replica view of training procedure.

  MaskConver task provides artifacts for training/evalution procedures,
  including loading/iterating over Datasets, initializing the model, calculating
  the loss, post-processing, and customized metrics with reduction.
  """

  def build_model(self) -> tf_keras.Model:
    """Build maskconver model."""

    tf_keras.utils.set_random_seed(0)
    tf.config.experimental.enable_op_determinism()
    input_specs = tf_keras.layers.InputSpec(
        shape=[None] + self.task_config.model.input_size)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf_keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = factory.build_maskconver_model(
        input_specs=input_specs,
        model_config=self.task_config.model,
        l2_regularizer=l2_regularizer,
        segmentation_inference=True)
    return model

  def build_inputs(self,
                   params: exp_cfg.DataConfig,
                   input_context: Optional[tf.distribute.InputContext] = None):
    """Builds classification input."""

    ignore_label = self.task_config.losses.ignore_label

    decoder = segmentation_input.Decoder()

    parser = maskconver_segmentation_input.Parser(
        output_size=params.output_size,
        num_classes=self.task_config.model.num_classes,
        crop_size=params.crop_size,
        ignore_label=ignore_label,
        resize_eval_groundtruth=params.resize_eval_groundtruth,
        groundtruth_padded_size=params.groundtruth_padded_size,
        aug_scale_min=params.aug_scale_min,
        aug_scale_max=params.aug_scale_max,
        aug_rand_hflip=params.aug_rand_hflip,
        preserve_aspect_ratio=params.preserve_aspect_ratio,
        level=self.task_config.model.level,
        aug_type=params.aug_type,
        max_num_stuff_centers=params.max_num_stuff_centers,
        dtype=params.dtype)

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self,
                   outputs: Mapping[str, Any],
                   labels: Mapping[str, Any],
                   aux_losses: Optional[Any] = None,
                   step=None) -> Dict[str, tf.Tensor]:
    """Build Panoptic Mask R-CNN losses."""
    loss_params = self._task_config.losses

    # b, h, w, c = outputs['class_heatmaps'].get_shape().as_list()
    batch_size = tf.cast(tf.shape(labels['num_instances'])[0], tf.float32)
    center_loss_fn = maskconver_losses.PenaltyReducedLogisticFocalLoss(
        alpha=loss_params.alpha, beta=loss_params.beta)
    mask_loss_fn = maskconver_losses.PenaltyReducedLogisticFocalLoss()

    true_flattened_ct_heatmap = loss_ops.flatten_spatial_dimensions(
        labels['seg_ct_heatmaps'])
    true_flattened_ct_heatmap = tf.cast(true_flattened_ct_heatmap, tf.float32)

    pred_flattened_ct_heatmap = loss_ops.flatten_spatial_dimensions(
        outputs['class_heatmaps'])
    pred_flattened_ct_heatmap = tf.cast(pred_flattened_ct_heatmap, tf.float32)
    center_valid_mask = labels['seg_valid_mask'][:, :, :, None]
    center_valid_mask = tf.image.resize(
        center_valid_mask, tf.shape(
            labels['seg_ct_heatmaps'])[1:3], method='nearest')
    center_valid_mask = tf.maximum(center_valid_mask, 0.0)
    center_valid_mask = center_valid_mask * tf.ones_like(
        labels['seg_ct_heatmaps'])
    weights_flattened_mask = loss_ops.flatten_spatial_dimensions(
        center_valid_mask)
    center_loss = center_loss_fn(
        target_tensor=true_flattened_ct_heatmap,
        prediction_tensor=pred_flattened_ct_heatmap,
        weights=weights_flattened_mask)

    center_loss = tf.reduce_sum(
        center_loss /
        (labels['num_instances'][:, None, None] + 1.0)) / batch_size

    gt_masks = labels['seg_masks']
    gt_mask_weights = labels['seg_mask_weights'][:, None,
                                                 None, :] * tf.ones_like(
                                                     gt_masks)
    valid_mask = labels['seg_valid_mask'][:, :, :,
                                          None] * tf.ones_like(gt_masks)

    true_flattened_masks = loss_ops.flatten_spatial_dimensions(gt_masks)
    true_flattened_ct_heatmap = tf.cast(true_flattened_ct_heatmap, tf.float32)
    predicted_masks = tf.cast(outputs['mask_proposal_logits'], tf.float32)
    predicted_masks = tf.image.resize(
        predicted_masks, tf.shape(gt_masks)[1:3], method='bilinear')
    pred_flattened_masks = loss_ops.flatten_spatial_dimensions(predicted_masks)
    mask_loss = tf.cast(0.0, tf.float32)

    mask_loss_fn = tf_keras.losses.BinaryCrossentropy(
        from_logits=True,
        label_smoothing=0.0,
        axis=-1,
        reduction=tf_keras.losses.Reduction.NONE,
        name='binary_crossentropy')
    mask_weights = tf.reshape(
        tf.cast(true_flattened_masks >= 0, tf.float32),
        [-1, 1]) * tf.reshape(gt_mask_weights, [-1, 1])  * tf.reshape(
            (valid_mask), [-1, 1])
    mask_loss = mask_loss_fn(
        tf.reshape(gt_masks, [-1, 1]),
        tf.reshape(pred_flattened_masks, [-1, 1]),
        sample_weight=mask_weights)

    mask_loss = tf.reduce_sum(mask_loss) / (tf.reduce_sum(mask_weights) + 1.0)

    total_loss = (center_loss + loss_params.mask_weight * mask_loss)

    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    total_loss = loss_params.loss_weight * total_loss

    losses = {'total_loss': total_loss,
              'mask_loss': mask_loss,
              'center_loss': center_loss}
    return losses

  def build_metrics(self, training: bool = True) -> List[
      tf_keras.metrics.Metric]:
    """Build detection metrics."""
    metrics = []
    if training:
      metric_names = [
          'total_loss',
          'center_loss',
          'mask_loss',
      ]
      for name in metric_names:
        metrics.append(tf_keras.metrics.Mean(name, dtype=tf.float32))
    else:
      self.iou_metric = segmentation_metrics.PerClassIoU(
          name='per_class_iou',
          num_classes=self.task_config.model.num_classes,
          rescale_predictions=False,
          dtype=tf.float32)

    return metrics

  def train_step(self,
                 inputs: Tuple[Any, Any],
                 model: tf_keras.Model,
                 optimizer: tf_keras.optimizers.Optimizer,
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
          images,
          box_indices=labels['seg_box_indices'],
          classes=labels['seg_classes'],
          training=True)
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      losses = self.build_losses(
          outputs=outputs,
          labels=labels,
          aux_losses=model.losses,
          step=optimizer.iterations)
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

    logs = {self.loss: losses['total_loss']}

    if metrics:
      for m in metrics:
        m.update_state(losses[m.name])

    return logs

  def validation_step(self,
                      inputs: Tuple[Any, Any],
                      model: tf_keras.Model,
                      metrics: Optional[List[Any]] = None):
    """Validatation step.

    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features, labels = inputs

    outputs = model(
        features,
        image_info=labels['image_info'],
        training=False)
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

    logs = {self.loss: 0}
    outputs = tf.one_hot(
        tf.cast(outputs['panoptic_outputs']['category_mask'], tf.int32),
        self.task_config.model.num_classes)

    self.iou_metric.update_state(labels, tf.cast(outputs, tf.float32))
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    if state is None:
      self.iou_metric.reset_states()
      state = self.iou_metric
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    result = {}
    ious = self.iou_metric.result()
    for i, value in enumerate(ious.numpy()):
      result.update({'iou/{}'.format(i): value})
    # Computes mean IoU
    result.update({'mean_iou': tf.reduce_mean(ious).numpy()})
    return result
