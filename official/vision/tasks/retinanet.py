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

"""RetinaNet task definition."""
from typing import Any, List, Mapping, Optional, Tuple

from absl import logging
import tensorflow as tf, tf_keras

from official.common import dataset_fn
from official.core import base_task
from official.core import task_factory
from official.vision.configs import retinanet as exp_cfg
from official.vision.dataloaders import input_reader
from official.vision.dataloaders import input_reader_factory
from official.vision.dataloaders import retinanet_input
from official.vision.dataloaders import tf_example_decoder
from official.vision.dataloaders import tfds_factory
from official.vision.dataloaders import tf_example_label_map_decoder
from official.vision.evaluation import coco_evaluator
from official.vision.losses import focal_loss
from official.vision.losses import loss_utils
from official.vision.modeling import factory
from official.vision.utils.object_detection import visualization_utils


@task_factory.register_task_cls(exp_cfg.RetinaNetTask)
class RetinaNetTask(base_task.Task):
  """A single-replica view of training procedure.

  RetinaNet task provides artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss,
  post-processing, and customized metrics with reduction.
  """

  def build_model(self):
    """Build RetinaNet model."""

    input_specs = tf_keras.layers.InputSpec(
        shape=[None] + self.task_config.model.input_size)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf_keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = factory.build_retinanet(
        input_specs=input_specs,
        model_config=self.task_config.model,
        l2_regularizer=l2_regularizer)

    if self.task_config.freeze_backbone:
      model.backbone.trainable = False

    return model

  def initialize(self, model: tf_keras.Model):
    """Loading pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      return

    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    # Restoring checkpoint.
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
            regenerate_source_id=decoder_cfg.regenerate_source_id,
            attribute_names=decoder_cfg.attribute_names,
        )
      elif params.decoder.type == 'label_map_decoder':
        decoder = tf_example_label_map_decoder.TfExampleDecoderLabelMap(
            label_map=decoder_cfg.label_map,
            regenerate_source_id=decoder_cfg.regenerate_source_id)
      else:
        raise ValueError('Unknown decoder type: {}!'.format(
            params.decoder.type))

    parser = retinanet_input.Parser(
        output_size=self.task_config.model.input_size[:2],
        min_level=self.task_config.model.min_level,
        max_level=self.task_config.model.max_level,
        num_scales=self.task_config.model.anchor.num_scales,
        aspect_ratios=self.task_config.model.anchor.aspect_ratios,
        anchor_size=self.task_config.model.anchor.anchor_size,
        dtype=params.dtype,
        match_threshold=params.parser.match_threshold,
        unmatched_threshold=params.parser.unmatched_threshold,
        box_coder_weights=(
            self.task_config.model.detection_generator.box_coder_weights
        ),
        aug_type=params.parser.aug_type,
        aug_rand_hflip=params.parser.aug_rand_hflip,
        aug_scale_min=params.parser.aug_scale_min,
        aug_scale_max=params.parser.aug_scale_max,
        skip_crowd_during_training=params.parser.skip_crowd_during_training,
        max_num_instances=params.parser.max_num_instances,
        pad=params.parser.pad,
        keep_aspect_ratio=params.parser.keep_aspect_ratio,
    )

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        combine_fn=input_reader.create_combine_fn(params),
        parser_fn=parser.parse_fn(params.is_training))
    dataset = reader.read(input_context=input_context)

    return dataset

  def build_attribute_loss(self,
                           attribute_heads: List[exp_cfg.AttributeHead],
                           outputs: Mapping[str, Any],
                           labels: Mapping[str, Any],
                           box_sample_weight: tf.Tensor) -> float:
    """Computes attribute loss.

    Args:
      attribute_heads: a list of attribute head configs.
      outputs: RetinaNet model outputs.
      labels: RetinaNet labels.
      box_sample_weight: normalized bounding box sample weights.

    Returns:
      Attribute loss of all attribute heads.
    """
    params = self.task_config
    attribute_loss = 0.0
    for head in attribute_heads:
      if head.name not in labels['attribute_targets']:
        raise ValueError(f'Attribute {head.name} not found in label targets.')
      if head.name not in outputs['attribute_outputs']:
        raise ValueError(f'Attribute {head.name} not found in model outputs.')

      if head.type == 'regression':
        y_true_att = loss_utils.multi_level_flatten(
            labels['attribute_targets'][head.name], last_dim=head.size
        )
        y_pred_att = loss_utils.multi_level_flatten(
            outputs['attribute_outputs'][head.name], last_dim=head.size
        )
        att_loss_fn = tf_keras.losses.Huber(
            1.0, reduction=tf_keras.losses.Reduction.SUM)
        att_loss = att_loss_fn(
            y_true=y_true_att,
            y_pred=y_pred_att,
            sample_weight=box_sample_weight)
      elif head.type == 'classification':
        y_true_att = loss_utils.multi_level_flatten(
            labels['attribute_targets'][head.name], last_dim=None
        )
        y_true_att = tf.one_hot(y_true_att, head.size)
        y_pred_att = loss_utils.multi_level_flatten(
            outputs['attribute_outputs'][head.name], last_dim=head.size
        )
        cls_loss_fn = focal_loss.FocalLoss(
            alpha=params.losses.focal_loss_alpha,
            gamma=params.losses.focal_loss_gamma,
            reduction=tf_keras.losses.Reduction.SUM,
        )
        att_loss = cls_loss_fn(
            y_true=y_true_att,
            y_pred=y_pred_att,
            sample_weight=box_sample_weight,
        )
      else:
        raise ValueError(f'Attribute type {head.type} not supported.')
      attribute_loss += att_loss

    return attribute_loss

  def build_losses(
      self,
      outputs: Mapping[str, Any],
      labels: Mapping[str, Any],
      aux_losses: Optional[Any] = None,
  ):
    """Build RetinaNet losses."""
    params = self.task_config
    attribute_heads = self.task_config.model.head.attribute_heads

    cls_loss_fn = focal_loss.FocalLoss(
        alpha=params.losses.focal_loss_alpha,
        gamma=params.losses.focal_loss_gamma,
        reduction=tf_keras.losses.Reduction.SUM)
    box_loss_fn = tf_keras.losses.Huber(
        params.losses.huber_loss_delta, reduction=tf_keras.losses.Reduction.SUM)

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

    model_loss = cls_loss + params.losses.box_loss_weight * box_loss

    if attribute_heads:
      model_loss += self.build_attribute_loss(attribute_heads, outputs, labels,
                                              box_sample_weight)

    total_loss = model_loss
    if aux_losses:
      reg_loss = tf.reduce_sum(aux_losses)
      total_loss = model_loss + reg_loss

    total_loss = params.losses.loss_weight * total_loss

    return total_loss, cls_loss, box_loss, model_loss

  def build_metrics(self, training: bool = True):
    """Build detection metrics."""
    metrics = []
    metric_names = ['total_loss', 'cls_loss', 'box_loss', 'model_loss']
    for name in metric_names:
      metrics.append(tf_keras.metrics.Mean(name, dtype=tf.float32))

    if not training:
      if (
          self.task_config.validation_data.tfds_name
          and self.task_config.annotation_file
      ):
        raise ValueError(
            "Can't evaluate using annotation file when TFDS is used."
        )
      if self._task_config.use_coco_metrics:
        self.coco_metric = coco_evaluator.COCOEvaluator(
            annotation_file=self.task_config.annotation_file,
            include_mask=False,
            per_category_metrics=self.task_config.per_category_metrics,
            max_num_eval_detections=self.task_config.max_num_eval_detections,
        )
      if self._task_config.use_wod_metrics:
        # To use Waymo open dataset metrics, please install one of the pip
        # package `waymo-open-dataset-tf-*` from
        # https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md#use-pre-compiled-pippip3-packages-for-linux
        # Note that the package is built with specific tensorflow version and
        # will produce error if it does not match the tf version that is
        # currently used.
        try:
          from official.vision.evaluation import wod_detection_evaluator  # pylint: disable=g-import-not-at-top
        except ModuleNotFoundError:
          logging.error('waymo-open-dataset should be installed to enable Waymo'
                        ' evaluator.')
          raise
        self.wod_metric = wod_detection_evaluator.WOD2dDetectionEvaluator()

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
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      loss, cls_loss, box_loss, model_loss = self.build_losses(
          outputs=outputs, labels=labels, aux_losses=model.losses
      )
      scaled_loss = loss / num_replicas

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

    logs = {self.loss: loss}

    all_losses = {
        'total_loss': loss,
        'cls_loss': cls_loss,
        'box_loss': box_loss,
        'model_loss': model_loss,
    }
    if metrics:
      for m in metrics:
        m.update_state(all_losses[m.name])
        logs.update({m.name: m.result()})

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

    outputs = model(features, anchor_boxes=labels['anchor_boxes'],
                    image_shape=labels['image_info'][:, 1, :],
                    training=False)
    loss, cls_loss, box_loss, model_loss = self.build_losses(
        outputs=outputs, labels=labels, aux_losses=model.losses
    )
    logs = {self.loss: loss}

    all_losses = {
        'total_loss': loss,
        'cls_loss': cls_loss,
        'box_loss': box_loss,
        'model_loss': model_loss,
    }

    if self._task_config.use_coco_metrics:
      coco_model_outputs = {
          'detection_boxes': outputs['detection_boxes'],
          'detection_scores': outputs['detection_scores'],
          'detection_classes': outputs['detection_classes'],
          'num_detections': outputs['num_detections'],
          'source_id': labels['groundtruths']['source_id'],
          'image_info': labels['image_info']
      }
      logs.update(
          {self.coco_metric.name: (labels['groundtruths'], coco_model_outputs)})
    if self.task_config.use_wod_metrics:
      wod_model_outputs = {
          'detection_boxes': outputs['detection_boxes'],
          'detection_scores': outputs['detection_scores'],
          'detection_classes': outputs['detection_classes'],
          'num_detections': outputs['num_detections'],
          'source_id': labels['groundtruths']['source_id'],
          'image_info': labels['image_info']
      }
      logs.update(
          {self.wod_metric.name: (labels['groundtruths'], wod_model_outputs)})

    if metrics:
      for m in metrics:
        m.update_state(all_losses[m.name])
        logs.update({m.name: m.result()})

    if (
        hasattr(self.task_config, 'allow_image_summary')
        and self.task_config.allow_image_summary
    ):
      logs.update(
          {'visualization': (tf.cast(features, dtype=tf.float32), outputs)}
      )
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    if self._task_config.use_coco_metrics:
      if state is None:
        self.coco_metric.reset_states()
      self.coco_metric.update_state(step_outputs[self.coco_metric.name][0],
                                    step_outputs[self.coco_metric.name][1])
    if self._task_config.use_wod_metrics:
      if state is None:
        self.wod_metric.reset_states()
      self.wod_metric.update_state(step_outputs[self.wod_metric.name][0],
                                   step_outputs[self.wod_metric.name][1])

    if 'visualization' in step_outputs:
      # Update detection state for writing summary if there are artifacts for
      # visualization.
      if state is None:
        state = {}
      state.update(visualization_utils.update_detection_state(step_outputs))

    if state is None:
      # Create an arbitrary state to indicate it's not the first step in the
      # following calls to this function.
      state = True

    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    logs = {}
    if self._task_config.use_coco_metrics:
      logs.update(self.coco_metric.result())
    if self._task_config.use_wod_metrics:
      logs.update(self.wod_metric.result())

    # Add visualization for summary.
    if isinstance(aggregated_logs, dict) and 'image' in aggregated_logs:
      validation_outputs = visualization_utils.visualize_outputs(
          logs=aggregated_logs, task_config=self.task_config
      )
      logs.update(validation_outputs)

    return logs
