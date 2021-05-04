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

"""RetinaNet task definition."""
from typing import Any, Optional, List, Tuple, Mapping

from absl import logging
import tensorflow as tf
from official.common import dataset_fn
from official.core import base_task
from official.core import task_factory
from official.vision import keras_cv
from official.vision.beta.configs import retinanet as exp_cfg
from official.vision.beta.dataloaders import input_reader_factory
from official.vision.beta.dataloaders import retinanet_input
from official.vision.beta.dataloaders import tf_example_decoder
from official.vision.beta.dataloaders import tfds_detection_decoders
from official.vision.beta.dataloaders import tf_example_label_map_decoder
from official.vision.beta.evaluation import coco_evaluator
from official.vision.beta.modeling import factory


@task_factory.register_task_cls(exp_cfg.RetinaNetTask)
class RetinaNetTask(base_task.Task):
  """A single-replica view of training procedure.

  RetinaNet task provides artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss,
  post-processing, and customized metrics with reduction.
  """

  def build_model(self):
    """Build RetinaNet model."""

    input_specs = tf.keras.layers.InputSpec(
        shape=[None] + self.task_config.model.input_size)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf.keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = factory.build_retinanet(
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
      raise ValueError(
          "Only 'all' or 'backbone' can be used to initialize the model.")

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def build_inputs(self,
                   params: exp_cfg.DataConfig,
                   input_context: Optional[tf.distribute.InputContext] = None):
    """Build input dataset."""

    if params.tfds_name:
      if params.tfds_name in tfds_detection_decoders.TFDS_ID_TO_DECODER_MAP:
        decoder = tfds_detection_decoders.TFDS_ID_TO_DECODER_MAP[
            params.tfds_name]()
      else:
        raise ValueError('TFDS {} is not supported'.format(params.tfds_name))
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
        aug_rand_hflip=params.parser.aug_rand_hflip,
        aug_scale_min=params.parser.aug_scale_min,
        aug_scale_max=params.parser.aug_scale_max,
        skip_crowd_during_training=params.parser.skip_crowd_during_training,
        max_num_instances=params.parser.max_num_instances)

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
                   aux_losses: Optional[Any] = None):
    """Build RetinaNet losses."""
    params = self.task_config
    cls_loss_fn = keras_cv.losses.FocalLoss(
        alpha=params.losses.focal_loss_alpha,
        gamma=params.losses.focal_loss_gamma,
        reduction=tf.keras.losses.Reduction.SUM)
    box_loss_fn = tf.keras.losses.Huber(
        params.losses.huber_loss_delta, reduction=tf.keras.losses.Reduction.SUM)

    # Sums all positives in a batch for normalization and avoids zero
    # num_positives_sum, which would lead to inf loss during training
    cls_sample_weight = labels['cls_weights']
    box_sample_weight = labels['box_weights']
    num_positives = tf.reduce_sum(box_sample_weight) + 1.0
    cls_sample_weight = cls_sample_weight / num_positives
    box_sample_weight = box_sample_weight / num_positives
    y_true_cls = keras_cv.losses.multi_level_flatten(
        labels['cls_targets'], last_dim=None)
    y_true_cls = tf.one_hot(y_true_cls, params.model.num_classes)
    y_pred_cls = keras_cv.losses.multi_level_flatten(
        outputs['cls_outputs'], last_dim=params.model.num_classes)
    y_true_box = keras_cv.losses.multi_level_flatten(
        labels['box_targets'], last_dim=4)
    y_pred_box = keras_cv.losses.multi_level_flatten(
        outputs['box_outputs'], last_dim=4)

    cls_loss = cls_loss_fn(
        y_true=y_true_cls, y_pred=y_pred_cls, sample_weight=cls_sample_weight)
    box_loss = box_loss_fn(
        y_true=y_true_box, y_pred=y_pred_box, sample_weight=box_sample_weight)

    model_loss = cls_loss + params.losses.box_loss_weight * box_loss

    total_loss = model_loss
    if aux_losses:
      reg_loss = tf.reduce_sum(aux_losses)
      total_loss = model_loss + reg_loss

    return total_loss, cls_loss, box_loss, model_loss

  def build_metrics(self, training: bool = True):
    """Build detection metrics."""
    metrics = []
    metric_names = ['total_loss', 'cls_loss', 'box_loss', 'model_loss']
    for name in metric_names:
      metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))

    if not training:
      if self.task_config.validation_data.tfds_name and self.task_config.annotation_file:
        raise ValueError(
            "Can't evaluate using annotation file when TFDS is used.")
      self.coco_metric = coco_evaluator.COCOEvaluator(
          annotation_file=self.task_config.annotation_file,
          include_mask=False,
          per_category_metrics=self.task_config.per_category_metrics)

    return metrics

  def train_step(self,
                 inputs: Tuple[Any, Any],
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
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
          outputs=outputs, labels=labels, aux_losses=model.losses)
      scaled_loss = loss / num_replicas

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
                      model: tf.keras.Model,
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
        outputs=outputs, labels=labels, aux_losses=model.losses)
    logs = {self.loss: loss}

    all_losses = {
        'total_loss': loss,
        'cls_loss': cls_loss,
        'box_loss': box_loss,
        'model_loss': model_loss,
    }

    coco_model_outputs = {
        'detection_boxes': outputs['detection_boxes'],
        'detection_scores': outputs['detection_scores'],
        'detection_classes': outputs['detection_classes'],
        'num_detections': outputs['num_detections'],
        'source_id': labels['groundtruths']['source_id'],
        'image_info': labels['image_info']
    }
    logs.update({self.coco_metric.name: (labels['groundtruths'],
                                         coco_model_outputs)})
    if metrics:
      for m in metrics:
        m.update_state(all_losses[m.name])
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
