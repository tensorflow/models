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

"""DETR detection task definition."""

import tensorflow as tf

from official.core import base_task
from official.core import task_factory
from official.projects.detr.configs import detr as detr_cfg
from official.projects.detr.dataloaders import coco
from official.projects.detr.modeling import detr
from official.projects.detr.ops import matchers
from official.vision.evaluation import coco_evaluator
from official.vision.ops import box_ops


@task_factory.register_task_cls(detr_cfg.DetectionConfig)
class DectectionTask(base_task.Task):
  """A single-replica view of training procedure.

  DETR task provides artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss,
  post-processing, and customized metrics with reduction.
  """

  def build_model(self):
    """Build DETR model."""
    model = detr.DETR(
        self._task_config.num_queries,
        self._task_config.num_hidden,
        self._task_config.num_classes,
        self._task_config.num_encoder_layers,
        self._task_config.num_decoder_layers)
    return model

  def initialize(self, model: tf.keras.Model):
    """Loading pretrained checkpoint."""
    ckpt = tf.train.Checkpoint(backbone=model.backbone)
    status = ckpt.read(self._task_config.init_ckpt)
    status.expect_partial().assert_existing_objects_matched()

  def build_inputs(self, params, input_context=None):
    """Build input dataset."""
    return coco.COCODataLoader(params).load(input_context)

  def _compute_cost(self, cls_outputs, box_outputs, cls_targets, box_targets):
    # Approximate classification cost with 1 - prob[target class].
    # The 1 is a constant that doesn't change the matching, it can be ommitted.
    # background: 0
    cls_cost = self._task_config.lambda_cls * tf.gather(
        -tf.nn.softmax(cls_outputs), cls_targets, batch_dims=1, axis=-1)

    # Compute the L1 cost between boxes,
    paired_differences = self._task_config.lambda_box * tf.abs(
        tf.expand_dims(box_outputs, 2) - tf.expand_dims(box_targets, 1))
    box_cost = tf.reduce_sum(paired_differences, axis=-1)

    # Compute the giou cost betwen boxes
    giou_cost = self._task_config.lambda_giou * -box_ops.bbox_generalized_overlap(
        box_ops.cycxhw_to_yxyx(box_outputs),
        box_ops.cycxhw_to_yxyx(box_targets))

    total_cost = cls_cost + box_cost + giou_cost

    max_cost = (
        self._task_config.lambda_cls * 0.0 + self._task_config.lambda_box * 4. +
        self._task_config.lambda_giou * 0.0)

    # Set pads to large constant
    valid = tf.expand_dims(
        tf.cast(tf.not_equal(cls_targets, 0), dtype=total_cost.dtype), axis=1)
    total_cost = (1 - valid) * max_cost + valid * total_cost

    # Set inf of nan to large constant
    total_cost = tf.where(
        tf.logical_or(tf.math.is_nan(total_cost), tf.math.is_inf(total_cost)),
        max_cost * tf.ones_like(total_cost, dtype=total_cost.dtype),
        total_cost)

    return total_cost

  def build_losses(self, outputs, labels, aux_losses=None):
    """Build DETR losses."""
    cls_outputs = outputs['cls_outputs']
    box_outputs = outputs['box_outputs']
    cls_targets = labels['classes']
    box_targets = labels['boxes']

    cost = self._compute_cost(
        cls_outputs, box_outputs, cls_targets, box_targets)

    _, indices = matchers.hungarian_matching(cost)
    indices = tf.stop_gradient(indices)

    target_index = tf.math.argmax(indices, axis=1)
    cls_assigned = tf.gather(cls_outputs, target_index, batch_dims=1, axis=1)
    box_assigned = tf.gather(box_outputs, target_index, batch_dims=1, axis=1)

    background = tf.equal(cls_targets, 0)
    num_boxes = tf.reduce_sum(
        tf.cast(tf.logical_not(background), tf.float32), axis=-1)

    # Down-weight background to account for class imbalance.
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=cls_targets, logits=cls_assigned)
    cls_loss = self._task_config.lambda_cls * tf.where(
        background,
        self._task_config.background_cls_weight * xentropy,
        xentropy
        )
    cls_weights = tf.where(
        background,
        self._task_config.background_cls_weight * tf.ones_like(cls_loss),
        tf.ones_like(cls_loss)
        )

    # Box loss is only calculated on non-background class.
    l_1 = tf.reduce_sum(tf.abs(box_assigned - box_targets), axis=-1)
    box_loss = self._task_config.lambda_box * tf.where(
        background,
        tf.zeros_like(l_1),
        l_1
        )

    # Giou loss is only calculated on non-background class.
    giou = tf.linalg.diag_part(1.0 - box_ops.bbox_generalized_overlap(
        box_ops.cycxhw_to_yxyx(box_assigned),
        box_ops.cycxhw_to_yxyx(box_targets)
        ))
    giou_loss = self._task_config.lambda_giou * tf.where(
        background,
        tf.zeros_like(giou),
        giou
        )

    # Consider doing all reduce once in train_step to speed up.
    num_boxes_per_replica = tf.reduce_sum(num_boxes)
    cls_weights_per_replica = tf.reduce_sum(cls_weights)
    replica_context = tf.distribute.get_replica_context()
    num_boxes_sum, cls_weights_sum = replica_context.all_reduce(
        tf.distribute.ReduceOp.SUM,
        [num_boxes_per_replica, cls_weights_per_replica])
    cls_loss = tf.math.divide_no_nan(
        tf.reduce_sum(cls_loss), cls_weights_sum)
    box_loss = tf.math.divide_no_nan(
        tf.reduce_sum(box_loss), num_boxes_sum)
    giou_loss = tf.math.divide_no_nan(
        tf.reduce_sum(giou_loss), num_boxes_sum)

    aux_losses = tf.add_n(aux_losses) if aux_losses else 0.0
    total_loss = cls_loss + box_loss + giou_loss + aux_losses
    return total_loss, cls_loss, box_loss, giou_loss

  def build_metrics(self, training=True):
    """Build detection metrics."""
    metrics = []
    metric_names = ['cls_loss', 'box_loss', 'giou_loss']
    for name in metric_names:
      metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))

    if not training:
      self.coco_metric = coco_evaluator.COCOEvaluator(
          annotation_file='',
          include_mask=False,
          need_rescale_bboxes=True,
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
    features, labels = inputs
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)

      loss = 0.0
      cls_loss = 0.0
      box_loss = 0.0
      giou_loss = 0.0

      for output in outputs:
        # Computes per-replica loss.
        layer_loss, layer_cls_loss, layer_box_loss, layer_giou_loss = self.build_losses(
            outputs=output, labels=labels, aux_losses=model.losses)
        loss += layer_loss
        cls_loss += layer_cls_loss
        box_loss += layer_box_loss
        giou_loss += layer_giou_loss

      # Consider moving scaling logic from build_losses to here.
      scaled_loss = loss
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

    # Multiply for logging.
    # Since we expect the gradient replica sum to happen in the optimizer,
    # the loss is scaled with global num_boxes and weights.
    # To have it more interpretable/comparable we scale it back when logging.
    num_replicas_in_sync = tf.distribute.get_strategy().num_replicas_in_sync
    loss *= num_replicas_in_sync
    cls_loss *= num_replicas_in_sync
    box_loss *= num_replicas_in_sync
    giou_loss *= num_replicas_in_sync

    # Trainer class handles loss metric for you.
    logs = {self.loss: loss}

    all_losses = {
        'cls_loss': cls_loss,
        'box_loss': box_loss,
        'giou_loss': giou_loss,
    }

    # Metric results will be added to logs for you.
    if metrics:
      for m in metrics:
        m.update_state(all_losses[m.name])
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

    outputs = model(features, training=False)[-1]
    loss, cls_loss, box_loss, giou_loss = self.build_losses(
        outputs=outputs, labels=labels, aux_losses=model.losses)

    # Multiply for logging.
    # Since we expect the gradient replica sum to happen in the optimizer,
    # the loss is scaled with global num_boxes and weights.
    # To have it more interpretable/comparable we scale it back when logging.
    num_replicas_in_sync = tf.distribute.get_strategy().num_replicas_in_sync
    loss *= num_replicas_in_sync
    cls_loss *= num_replicas_in_sync
    box_loss *= num_replicas_in_sync
    giou_loss *= num_replicas_in_sync

    # Evaluator class handles loss metric for you.
    logs = {self.loss: loss}

    predictions = {
        'detection_boxes':
                box_ops.cycxhw_to_yxyx(outputs['box_outputs'])
                * tf.expand_dims(
                    tf.concat([
                        labels['image_info'][:, 1:2, 0],
                        labels['image_info'][:, 1:2, 1],
                        labels['image_info'][:, 1:2, 0],
                        labels['image_info'][:, 1:2, 1]
                    ],
                              axis=1),
                    axis=1),
        'detection_scores':
            tf.math.reduce_max(
                tf.nn.softmax(outputs['cls_outputs'])[:, :, 1:], axis=-1),
        'detection_classes':
            tf.math.argmax(outputs['cls_outputs'][:, :, 1:], axis=-1) + 1,
        # Fix this. It's not being used at the moment.
        'num_detections': tf.reduce_sum(
            tf.cast(
                tf.math.greater(tf.math.reduce_max(
                    outputs['cls_outputs'], axis=-1), 0), tf.int32), axis=-1),
        'source_id': labels['id'],
        'image_info': labels['image_info']
    }
    ground_truths = {
        'source_id': labels['id'],
        'height': labels['image_info'][:, 0:1, 0],
        'width': labels['image_info'][:, 0:1, 1],
        'num_detections': tf.reduce_sum(
            tf.cast(tf.math.greater(labels['classes'], 0), tf.int32), axis=-1),
        'boxes': labels['gt_boxes'],
        'classes': labels['classes'],
        'is_crowds': labels['is_crowd']
    }
    logs.update({'predictions': predictions,
                 'ground_truths': ground_truths})

    all_losses = {
        'cls_loss': cls_loss,
        'box_loss': box_loss,
        'giou_loss': giou_loss,
    }

    # Metric results will be added to logs for you.
    if metrics:
      for m in metrics:
        m.update_state(all_losses[m.name])
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    if state is None:
      self.coco_metric.reset_states()
      state = self.coco_metric

    state.update_state(
        step_outputs['ground_truths'],
        step_outputs['predictions'])
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    return aggregated_logs.result()
