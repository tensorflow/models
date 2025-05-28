# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Centernet task definition."""

from typing import Any, List, Optional, Tuple

from absl import logging
import tensorflow as tf, tf_keras

from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.projects.centernet.configs import centernet as exp_cfg
from official.projects.centernet.dataloaders import centernet_input
from official.projects.centernet.losses import centernet_losses
from official.projects.centernet.modeling import centernet_model
from official.projects.centernet.modeling.heads import centernet_head
from official.projects.centernet.modeling.layers import detection_generator
from official.projects.centernet.ops import loss_ops
from official.projects.centernet.ops import target_assigner
from official.vision.dataloaders import tf_example_decoder
from official.vision.dataloaders import tfds_factory
from official.vision.dataloaders import tf_example_label_map_decoder
from official.vision.evaluation import coco_evaluator
from official.vision.modeling.backbones import factory


@task_factory.register_task_cls(exp_cfg.CenterNetTask)
class CenterNetTask(base_task.Task):
  """Task definition for centernet."""

  def build_inputs(self,
                   params: exp_cfg.DataConfig,
                   input_context: Optional[tf.distribute.InputContext] = None):
    """Build input dataset."""
    if params.tfds_name:
      decoder = tfds_factory.get_detection_decoder(params.tfds_name)
    else:
      decoder_cfg = params.decoder.get()
      if params.decoder.type == 'simple_decoder':
        decoder = tf_example_decoder.TfExampleDecoder(
            regenerate_source_id=decoder_cfg.regenerate_source_id)
      elif params.decoder.type == 'label_map_decoder':
        decoder = tf_example_label_map_decoder.TfExampleDecoderLabelMap(
            label_map=decoder_cfg.label_map,
            regenerate_source_id=decoder_cfg.regenerate_source_id)
      else:
        raise ValueError('Unknown decoder type: {}!'.format(
            params.decoder.type))

    parser = centernet_input.CenterNetParser(
        output_height=self.task_config.model.input_size[0],
        output_width=self.task_config.model.input_size[1],
        max_num_instances=self.task_config.model.max_num_instances,
        bgr_ordering=params.parser.bgr_ordering,
        channel_means=params.parser.channel_means,
        channel_stds=params.parser.channel_stds,
        aug_rand_hflip=params.parser.aug_rand_hflip,
        aug_scale_min=params.parser.aug_scale_min,
        aug_scale_max=params.parser.aug_scale_max,
        aug_rand_hue=params.parser.aug_rand_hue,
        aug_rand_brightness=params.parser.aug_rand_brightness,
        aug_rand_contrast=params.parser.aug_rand_contrast,
        aug_rand_saturation=params.parser.aug_rand_saturation,
        odapi_augmentation=params.parser.odapi_augmentation,
        dtype=params.dtype)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_model(self):
    """get an instance of CenterNet."""
    model_config = self.task_config.model
    input_specs = tf_keras.layers.InputSpec(
        shape=[None] + model_config.input_size)

    l2_weight_decay = self.task_config.weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf_keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    backbone = factory.build_backbone(
        input_specs=input_specs,
        backbone_config=model_config.backbone,
        norm_activation_config=model_config.norm_activation,
        l2_regularizer=l2_regularizer)

    task_outputs = self.task_config.get_output_length_dict()
    head_config = model_config.head
    head = centernet_head.CenterNetHead(
        input_specs=backbone.output_specs,
        task_outputs=task_outputs,
        input_levels=head_config.input_levels,
        heatmap_bias=head_config.heatmap_bias)

    # output_specs is a dict
    backbone_output_spec = backbone.output_specs[head_config.input_levels[-1]]
    if len(backbone_output_spec) == 4:
      bb_output_height = backbone_output_spec[1]
    elif len(backbone_output_spec) == 3:
      bb_output_height = backbone_output_spec[0]
    else:
      raise ValueError
    self._net_down_scale = int(model_config.input_size[0] / bb_output_height)
    dg_config = model_config.detection_generator
    detect_generator_obj = detection_generator.CenterNetDetectionGenerator(
        max_detections=dg_config.max_detections,
        peak_error=dg_config.peak_error,
        peak_extract_kernel_size=dg_config.peak_extract_kernel_size,
        class_offset=dg_config.class_offset,
        net_down_scale=self._net_down_scale,
        input_image_dims=(
            model_config.input_size[0],
            model_config.input_size[1],
        ),
        use_nms=dg_config.use_nms,
        nms_pre_thresh=dg_config.nms_pre_thresh,
        nms_thresh=dg_config.nms_thresh)

    model = centernet_model.CenterNetModel(
        backbone=backbone,
        head=head,
        detection_generator=detect_generator_obj)

    return model

  def initialize(self, model: tf_keras.Model):
    """Loading pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      return

    ckpt_dir_or_file = self.task_config.init_checkpoint

    # Restoring checkpoint.
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    if self.task_config.init_checkpoint_modules == 'all':
      ckpt = tf.train.Checkpoint(**model.checkpoint_items)
      status = ckpt.restore(ckpt_dir_or_file)
      status.assert_consumed()
    elif self.task_config.init_checkpoint_modules == 'backbone':
      ckpt = tf.train.Checkpoint(backbone=model.backbone)
      status = ckpt.restore(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    else:
      raise ValueError(
          "Only 'all' or 'backbone' can be used to initialize the model.")

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def build_losses(self,
                   outputs,
                   labels,
                   aux_losses=None):
    """Build losses."""
    input_size = self.task_config.model.input_size[0:2]
    output_size = outputs['ct_heatmaps'][0].get_shape().as_list()[1:3]

    gt_label = tf.map_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x: target_assigner.assign_centernet_targets(
            labels=x,
            input_size=input_size,
            output_size=output_size,
            num_classes=self.task_config.model.num_classes,
            max_num_instances=self.task_config.model.max_num_instances,
            gaussian_iou=self.task_config.losses.gaussian_iou,
            class_offset=self.task_config.losses.class_offset),
        elems=labels,
        fn_output_signature={
            'ct_heatmaps': tf.TensorSpec(
                shape=[output_size[0], output_size[1],
                       self.task_config.model.num_classes],
                dtype=tf.float32),
            'ct_offset': tf.TensorSpec(
                shape=[self.task_config.model.max_num_instances, 2],
                dtype=tf.float32),
            'size': tf.TensorSpec(
                shape=[self.task_config.model.max_num_instances, 2],
                dtype=tf.float32),
            'box_mask': tf.TensorSpec(
                shape=[self.task_config.model.max_num_instances],
                dtype=tf.int32),
            'box_indices': tf.TensorSpec(
                shape=[self.task_config.model.max_num_instances, 2],
                dtype=tf.int32),
        }
    )

    losses = {}

    # Create loss functions
    object_center_loss_fn = centernet_losses.PenaltyReducedLogisticFocalLoss()
    localization_loss_fn = centernet_losses.L1LocalizationLoss()

    # Set up box indices so that they have a batch element as well
    box_indices = loss_ops.add_batch_to_indices(gt_label['box_indices'])

    box_mask = tf.cast(gt_label['box_mask'], dtype=tf.float32)
    num_boxes = tf.cast(
        loss_ops.get_num_instances_from_weights(gt_label['box_mask']),
        dtype=tf.float32)

    # Calculate center heatmap loss
    output_unpad_image_shapes = tf.math.ceil(
        tf.cast(labels['unpad_image_shapes'],
                tf.float32) / self._net_down_scale)
    valid_anchor_weights = loss_ops.get_valid_anchor_weights_in_flattened_image(
        output_unpad_image_shapes, output_size[0], output_size[1])
    valid_anchor_weights = tf.expand_dims(valid_anchor_weights, 2)

    pred_ct_heatmap_list = outputs['ct_heatmaps']
    true_flattened_ct_heatmap = loss_ops.flatten_spatial_dimensions(
        gt_label['ct_heatmaps'])
    true_flattened_ct_heatmap = tf.cast(true_flattened_ct_heatmap, tf.float32)

    total_center_loss = 0.0
    for ct_heatmap in pred_ct_heatmap_list:
      pred_flattened_ct_heatmap = loss_ops.flatten_spatial_dimensions(
          ct_heatmap)
      pred_flattened_ct_heatmap = tf.cast(pred_flattened_ct_heatmap, tf.float32)
      total_center_loss += object_center_loss_fn(
          target_tensor=true_flattened_ct_heatmap,
          prediction_tensor=pred_flattened_ct_heatmap,
          weights=valid_anchor_weights)

    center_loss = tf.reduce_sum(total_center_loss) / float(
        len(pred_ct_heatmap_list) * num_boxes)
    losses['ct_loss'] = center_loss

    # Calculate scale loss
    pred_scale_list = outputs['ct_size']
    true_scale = tf.cast(gt_label['size'], tf.float32)

    total_scale_loss = 0.0
    for scale_map in pred_scale_list:
      pred_scale = loss_ops.get_batch_predictions_from_indices(scale_map,
                                                               box_indices)
      pred_scale = tf.cast(pred_scale, tf.float32)
      # Only apply loss for boxes that appear in the ground truth
      total_scale_loss += tf.reduce_sum(
          localization_loss_fn(target_tensor=true_scale,
                               prediction_tensor=pred_scale),
          axis=-1) * box_mask

    scale_loss = tf.reduce_sum(total_scale_loss) / float(
        len(pred_scale_list) * num_boxes)
    losses['scale_loss'] = scale_loss

    # Calculate offset loss
    pred_offset_list = outputs['ct_offset']
    true_offset = tf.cast(gt_label['ct_offset'], tf.float32)

    total_offset_loss = 0.0
    for offset_map in pred_offset_list:
      pred_offset = loss_ops.get_batch_predictions_from_indices(offset_map,
                                                                box_indices)
      pred_offset = tf.cast(pred_offset, tf.float32)
      # Only apply loss for boxes that appear in the ground truth
      total_offset_loss += tf.reduce_sum(
          localization_loss_fn(target_tensor=true_offset,
                               prediction_tensor=pred_offset),
          axis=-1) * box_mask

    offset_loss = tf.reduce_sum(total_offset_loss) / float(
        len(pred_offset_list) * num_boxes)
    losses['ct_offset_loss'] = offset_loss

    # Aggregate and finalize loss
    loss_weights = self.task_config.losses.detection
    total_loss = (loss_weights.object_center_weight * center_loss +
                  loss_weights.scale_weight * scale_loss +
                  loss_weights.offset_weight * offset_loss)

    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    losses['total_loss'] = total_loss
    return losses

  def build_metrics(self, training=True):
    metrics = []
    metric_names = ['total_loss', 'ct_loss', 'scale_loss', 'ct_offset_loss']
    for name in metric_names:
      metrics.append(tf_keras.metrics.Mean(name, dtype=tf.float32))

    if not training:
      if (self.task_config.validation_data.tfds_name
          and self.task_config.annotation_file):
        raise ValueError(
            "Can't evaluate using annotation file when TFDS is used.")
      self.coco_metric = coco_evaluator.COCOEvaluator(
          annotation_file=self.task_config.annotation_file,
          include_mask=False,
          per_category_metrics=self.task_config.per_category_metrics)

    return metrics

  def train_step(self,
                 inputs: Tuple[Any, Any],
                 model: tf_keras.Model,
                 optimizer: tf_keras.optimizers.Optimizer,
                 metrics: Optional[List[Any]] = None):
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
      outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

      losses = self.build_losses(outputs['raw_output'], labels)

      scaled_loss = losses['total_loss'] / num_replicas
      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    # compute the gradient
    tvars = model.trainable_variables
    gradients = tape.gradient(scaled_loss, tvars)

    # get unscaled loss if the scaled loss was used
    if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
      gradients = optimizer.get_unscaled_gradients(gradients)

    if self.task_config.gradient_clip_norm > 0.0:
      gradients, _ = tf.clip_by_global_norm(gradients,
                                            self.task_config.gradient_clip_norm)

    optimizer.apply_gradients(list(zip(gradients, tvars)))

    logs = {self.loss: losses['total_loss']}

    if metrics:
      for m in metrics:
        m.update_state(losses[m.name])
        logs.update({m.name: m.result()})

    return logs

  def validation_step(self,
                      inputs: Tuple[Any, Any],
                      model: tf_keras.Model,
                      metrics: Optional[List[Any]] = None):
    """Validation step.

    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features, labels = inputs

    outputs = model(features, training=False)
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

    losses = self.build_losses(outputs['raw_output'], labels)

    logs = {self.loss: losses['total_loss']}

    coco_model_outputs = {
        'detection_boxes': outputs['boxes'],
        'detection_scores': outputs['confidence'],
        'detection_classes': outputs['classes'],
        'num_detections': outputs['num_detections'],
        'source_id': labels['groundtruths']['source_id'],
        'image_info': labels['image_info']
    }

    logs.update({self.coco_metric.name: (labels['groundtruths'],
                                         coco_model_outputs)})

    if metrics:
      for m in metrics:
        m.update_state(losses[m.name])
        logs.update({m.name: m.result()})
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    if state is None:
      self.coco_metric.reset_states()
      state = self.coco_metric
    self.coco_metric.update_state(step_outputs[self.coco_metric.name][0],
                                  step_outputs[self.coco_metric.name][1])
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    return self.coco_metric.result()
