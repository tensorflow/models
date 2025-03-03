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

from absl import logging
import tensorflow as tf, tf_keras

from official.common import dataset_fn
from official.core import task_factory
from official.projects.panoptic.configs import panoptic_maskrcnn as exp_cfg
from official.projects.panoptic.dataloaders import panoptic_maskrcnn_input
from official.projects.panoptic.modeling import factory
from official.vision.dataloaders import input_reader
from official.vision.dataloaders import input_reader_factory
from official.vision.evaluation import panoptic_quality
from official.vision.evaluation import segmentation_metrics
from official.vision.losses import segmentation_losses
from official.vision.tasks import maskrcnn


@task_factory.register_task_cls(exp_cfg.PanopticMaskRCNNTask)
class PanopticMaskRCNNTask(maskrcnn.MaskRCNNTask):

  """A single-replica view of training procedure.

  Panoptic Mask R-CNN task provides artifacts for training/evalution procedures,
  including loading/iterating over Datasets, initializing the model, calculating
  the loss, post-processing, and customized metrics with reduction.
  """

  def __init__(self,
               params,
               logging_dir: Optional[str] = None,
               name: Optional[str] = None):
    super().__init__(params, logging_dir=logging_dir, name=name)
    self.segmentation_train_mean_iou = None
    self.segmentation_perclass_iou_metric = None
    self.panoptic_quality_metric = None

  def build_model(self) -> tf_keras.Model:
    """Builds Panoptic Mask R-CNN model."""

    input_specs = tf_keras.layers.InputSpec(
        shape=[None] + self.task_config.model.input_size)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf_keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = factory.build_panoptic_maskrcnn(
        input_specs=input_specs,
        model_config=self.task_config.model,
        l2_regularizer=l2_regularizer)

    if self.task_config.freeze_backbone:
      model.backbone.trainable = False

    # Builds the model through warm-up call.
    dummy_images = tf_keras.Input(self.task_config.model.input_size)
    # Note that image_info is always in the shape of [4, 2].
    dummy_image_info = tf_keras.layers.Input([4, 2])
    _ = model(dummy_images, image_info=dummy_image_info, training=False)

    return model

  def initialize(self, model: tf_keras.Model) -> None:
    """Loads pretrained checkpoint."""

    if not self.task_config.init_checkpoint:
      return

    def _get_checkpoint_path(checkpoint_dir_or_file):
      checkpoint_path = checkpoint_dir_or_file
      if tf.io.gfile.isdir(checkpoint_dir_or_file):
        checkpoint_path = tf.train.latest_checkpoint(
            checkpoint_dir_or_file)
      return checkpoint_path

    for init_module in self.task_config.init_checkpoint_modules:
      # Restoring checkpoint.
      if init_module == 'all':
        checkpoint_path = _get_checkpoint_path(
            self.task_config.init_checkpoint)
        ckpt = tf.train.Checkpoint(**model.checkpoint_items)
        status = ckpt.read(checkpoint_path)
        status.expect_partial().assert_existing_objects_matched()

      elif init_module == 'backbone':
        checkpoint_path = _get_checkpoint_path(
            self.task_config.init_checkpoint)

        if self.task_config.model.backbone.type == 'uvit':
          model.backbone.load_checkpoint(ckpt_filepath=checkpoint_path)
        else:
          ckpt = tf.train.Checkpoint(backbone=model.backbone)
          status = ckpt.read(checkpoint_path)
          status.expect_partial().assert_existing_objects_matched()

      elif init_module == 'decoder':
        checkpoint_path = _get_checkpoint_path(
            self.task_config.init_checkpoint)
        ckpt = tf.train.Checkpoint(decoder=model.decoder)
        status = ckpt.read(checkpoint_path)
        status.expect_partial().assert_existing_objects_matched()

      elif init_module == 'segmentation_backbone':
        checkpoint_path = _get_checkpoint_path(
            self.task_config.segmentation_init_checkpoint)
        ckpt = tf.train.Checkpoint(
            segmentation_backbone=model.segmentation_backbone)
        status = ckpt.read(checkpoint_path)
        status.expect_partial().assert_existing_objects_matched()

      elif init_module == 'segmentation_decoder':
        checkpoint_path = _get_checkpoint_path(
            self.task_config.segmentation_init_checkpoint)
        ckpt = tf.train.Checkpoint(
            segmentation_decoder=model.segmentation_decoder)
        status = ckpt.read(checkpoint_path)
        status.expect_partial().assert_existing_objects_matched()

      else:
        raise ValueError(
            "Only 'all', 'backbone', 'decoder', 'segmentation_backbone' and/or "
            "'segmentation_decoder' can be used to initialize the model, but "
            "got {}".format(init_module))
      logging.info('Finished loading pretrained checkpoint from %s for %s',
                   checkpoint_path, init_module)

  def build_inputs(
      self,
      params: exp_cfg.DataConfig,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Builds input dataset."""
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
        rpn_match_threshold=params.parser.rpn_match_threshold,
        rpn_unmatched_threshold=params.parser.rpn_unmatched_threshold,
        rpn_batch_size_per_im=params.parser.rpn_batch_size_per_im,
        rpn_fg_fraction=params.parser.rpn_fg_fraction,
        aug_rand_hflip=params.parser.aug_rand_hflip,
        aug_rand_vflip=params.parser.aug_rand_vflip,
        aug_scale_min=params.parser.aug_scale_min,
        aug_scale_max=params.parser.aug_scale_max,
        aug_type=params.parser.aug_type,
        skip_crowd_during_training=params.parser.skip_crowd_during_training,
        max_num_instances=params.parser.max_num_instances,
        outer_boxes_scale=self.task_config.model.outer_boxes_scale,
        mask_crop_size=params.parser.mask_crop_size,
        segmentation_resize_eval_groundtruth=params.parser
        .segmentation_resize_eval_groundtruth,
        segmentation_groundtruth_padded_size=params.parser
        .segmentation_groundtruth_padded_size,
        segmentation_ignore_label=params.parser.segmentation_ignore_label,
        panoptic_ignore_label=params.parser.panoptic_ignore_label,
        include_panoptic_masks=params.parser.include_panoptic_masks,
        dtype=params.dtype,
    )

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        combine_fn=input_reader.create_combine_fn(params),
        parser_fn=parser.parse_fn(params.is_training),
    )
    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self,
                   outputs: Mapping[str, Any],
                   labels: Mapping[str, Any],
                   aux_losses: Optional[Any] = None) -> Dict[str, tf.Tensor]:
    """Builds Panoptic Mask R-CNN losses."""
    params = self.task_config.losses

    use_groundtruth_dimension = (
        params.semantic_segmentation_use_groundtruth_dimension)

    segmentation_loss_fn = segmentation_losses.SegmentationLoss(
        label_smoothing=params.semantic_segmentation_label_smoothing,
        class_weights=params.semantic_segmentation_class_weights,
        ignore_label=params.semantic_segmentation_ignore_label,
        gt_is_matting_map=params.semantic_segmentation_gt_is_matting_map,
        use_groundtruth_dimension=use_groundtruth_dimension,
        use_binary_cross_entropy=params
        .semantic_segmentation_use_binary_cross_entropy,
        top_k_percent_pixels=params.semantic_segmentation_top_k_percent_pixels)

    instance_segmentation_weight = params.instance_segmentation_weight
    semantic_segmentation_weight = params.semantic_segmentation_weight

    losses = super().build_losses(
        outputs=outputs,
        labels=labels,
        aux_losses=None)
    maskrcnn_loss = losses['model_loss']
    segmentation_loss = segmentation_loss_fn(
        outputs['segmentation_outputs'],
        labels['gt_segmentation_mask'])

    model_loss = (
        instance_segmentation_weight * maskrcnn_loss +
        semantic_segmentation_weight * segmentation_loss)

    total_loss = model_loss
    if aux_losses:
      reg_loss = tf.reduce_sum(aux_losses)
      total_loss = model_loss + reg_loss

    losses.update({
        'total_loss': total_loss,
        'maskrcnn_loss': maskrcnn_loss,
        'segmentation_loss': segmentation_loss,
        'model_loss': model_loss,
    })
    return losses

  def build_metrics(
      self, training: bool = True
  ) -> List[tf_keras.metrics.Metric]:
    """Builds detection metrics."""
    metrics = super().build_metrics(training)

    if training:
      metric_names = ['maskrcnn_loss', 'segmentation_loss']
      for name in metric_names:
        metrics.append(tf_keras.metrics.Mean(name, dtype=tf.float32))

      if self.task_config.segmentation_evaluation.report_train_mean_iou:
        self.segmentation_train_mean_iou = segmentation_metrics.MeanIoU(
            name='train_mean_iou',
            num_classes=self.task_config.model.segmentation_model.num_classes,
            rescale_predictions=False,
            dtype=tf.float32,
        )
    else:
      rescale_predictions = (
          not self.task_config.validation_data.parser.segmentation_resize_eval_groundtruth
      )
      self.segmentation_perclass_iou_metric = segmentation_metrics.PerClassIoU(
          name='per_class_iou',
          num_classes=self.task_config.model.segmentation_model.num_classes,
          rescale_predictions=rescale_predictions,
          dtype=tf.float32,
      )

      if (
          self.task_config.model.generate_panoptic_masks
          and self.task_config.panoptic_quality_evaluator is not None
      ):
        if not self.task_config.validation_data.parser.include_panoptic_masks:
          raise ValueError(
              '`include_panoptic_masks` should be set to True when'
              ' computing panoptic quality.'
          )
        pq_config = self.task_config.panoptic_quality_evaluator
        self.panoptic_quality_metric = panoptic_quality.PanopticQualityV2(
            num_categories=pq_config.num_categories,
            is_thing=pq_config.is_thing,
            ignored_label=pq_config.ignored_label,
            rescale_predictions=pq_config.rescale_predictions,
        )

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
      model_kwargs = {
          'image_info': labels['image_info'],
          'anchor_boxes': labels['anchor_boxes'],
          'gt_boxes': labels['gt_boxes'],
          'gt_classes': labels['gt_classes'],
          'training': True,
      }
      if self.task_config.model.include_mask:
        model_kwargs['gt_masks'] = labels['gt_masks']
        if self.task_config.model.outer_boxes_scale > 1.0:
          model_kwargs['gt_outer_boxes'] = labels['gt_outer_boxes']
      outputs = model(images, **model_kwargs)
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      losses = self.build_losses(
          outputs=outputs, labels=labels, aux_losses=model.losses)
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

    if (self.task_config.segmentation_evaluation.report_train_mean_iou and
        self.segmentation_train_mean_iou is not None):
      segmentation_labels = {
          'masks': labels['gt_segmentation_mask'],
          'valid_masks': labels['gt_segmentation_valid_mask'],
          'image_info': labels['image_info']
      }
      self.process_metrics(
          metrics=[self.segmentation_train_mean_iou],
          labels=segmentation_labels,
          model_outputs=outputs['segmentation_outputs'])
      logs.update({
          self.segmentation_train_mean_iou.name:
              self.segmentation_train_mean_iou.result()
      })

    return logs

  def _update_metrics(self, labels, outputs, logs):
    super()._update_metrics(labels, outputs, logs)

    if self.segmentation_perclass_iou_metric is not None:
      segmentation_labels = {
          'masks': labels['groundtruths']['gt_segmentation_mask'],
          'valid_masks': labels['groundtruths']['gt_segmentation_valid_mask'],
          'image_info': labels['image_info'],
      }
      self.segmentation_perclass_iou_metric.update_state(
          segmentation_labels, outputs['segmentation_outputs']
      )

    if self.panoptic_quality_metric is not None:
      pq_metric_labels = {
          'category_mask': labels['groundtruths']['gt_panoptic_category_mask'],
          'instance_mask': labels['groundtruths']['gt_panoptic_instance_mask'],
          'image_info': labels['image_info'],
      }
      self.panoptic_quality_metric.update_state(
          pq_metric_labels, outputs['panoptic_outputs']
      )

  def validation_step(
      self,
      inputs: Tuple[Any, Any],
      model: tf_keras.Model,
      metrics: Optional[List[Any]] = None,
  ) -> Dict[str, Any]:
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
        image_info=labels['image_info'],
        training=False,
    )

    logs = {self.loss: 0}
    self._update_metrics(labels, outputs, logs)
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    is_first_step = not state
    super().aggregate_logs(state, step_outputs)

    if is_first_step:
      if not isinstance(state, list):
        state = []
      if self.segmentation_perclass_iou_metric is not None:
        state.append(self.segmentation_perclass_iou_metric)
      if self.panoptic_quality_metric is not None:
        state.append(self.panoptic_quality_metric)

    if not state:
      # Create an arbitrary state to indicate it's not the first step in the
      # following calls to this function.
      state = True
    return state

  def _reduce_semantic_metrics(self, logs: Dict[str, Any]):
    """Updates the per class and mean semantic metrics in the logs."""
    assert self.segmentation_perclass_iou_metric is not None
    ious = self.segmentation_perclass_iou_metric.result()
    if self.task_config.segmentation_evaluation.report_per_class_iou:
      for i, value in enumerate(ious.numpy()):
        logs.update({'segmentation_iou/class_{}'.format(i): value})
    logs.update({'segmentation_mean_iou': tf.reduce_mean(ious)})

  def _reduce_panoptic_metrics(self, logs: Dict[str, Any]):
    """Updates the per class and mean panoptic metrics in the logs."""
    assert self.panoptic_quality_metric is not None
    result = self.panoptic_quality_metric.result()
    valid_thing_classes = result['valid_thing_classes']
    valid_stuff_classes = result['valid_stuff_classes']
    valid_classes = valid_stuff_classes | valid_thing_classes
    num_categories = tf.math.count_nonzero(valid_classes, dtype=tf.float32)
    num_thing_categories = tf.math.count_nonzero(
        valid_thing_classes, dtype=tf.float32
    )
    num_stuff_categories = tf.math.count_nonzero(
        valid_stuff_classes, dtype=tf.float32
    )
    valid_thing_classes = tf.cast(valid_thing_classes, dtype=tf.float32)
    valid_stuff_classes = tf.cast(valid_stuff_classes, dtype=tf.float32)

    logs['panoptic_quality/All_num_categories'] = num_categories
    logs['panoptic_quality/Things_num_categories'] = num_thing_categories
    logs['panoptic_quality/Stuff_num_categories'] = num_stuff_categories
    for metric in ['pq', 'sq', 'rq']:
      metric_per_class = result[f'{metric}_per_class']
      logs[f'panoptic_quality/All_{metric}'] = tf.math.divide_no_nan(
          tf.reduce_sum(metric_per_class), num_categories
      )
      logs[f'panoptic_quality/Things_{metric}'] = tf.math.divide_no_nan(
          tf.reduce_sum(metric_per_class * valid_thing_classes),
          num_thing_categories,
      )
      logs[f'panoptic_quality/Stuff_{metric}'] = tf.math.divide_no_nan(
          tf.reduce_sum(metric_per_class * valid_stuff_classes),
          num_stuff_categories,
      )
      if self.task_config.panoptic_quality_evaluator.report_per_class_metrics:
        for i, is_valid in enumerate(valid_classes.numpy()):
          if is_valid:
            logs[f'panoptic_quality/{metric}/class_{i}'] = metric_per_class[i]

  def reduce_aggregated_logs(
      self,
      aggregated_logs: Dict[str, Any],
      global_step: Optional[tf.Tensor] = None,
  ) -> Dict[str, tf.Tensor]:
    """Optional reduce of aggregated logs over validation steps."""
    logs = super().reduce_aggregated_logs(aggregated_logs, global_step)

    if self.segmentation_perclass_iou_metric is not None:
      self._reduce_semantic_metrics(logs)
      self.segmentation_perclass_iou_metric.reset_state()
    if self.panoptic_quality_metric is not None:
      self._reduce_panoptic_metrics(logs)
      self.panoptic_quality_metric.reset_state()

    return logs
