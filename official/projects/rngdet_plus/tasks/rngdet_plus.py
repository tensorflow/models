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

"""RNGDet task definition."""
import numpy as np
from typing import Optional
from absl import logging
import tensorflow as tf
from PIL import Image, ImageDraw
from official.common import dataset_fn
from official.core import base_task
from official.core import task_factory
from official.projects.rngdet_plus.configs import rngdet as rngdet_cfg
from official.projects.rngdet_plus.dataloaders import rngdet_input
from official.projects.rngdet_plus.modeling import rngdet_plus
from official.projects.detr.ops import matchers
from official.vision.dataloaders import input_reader_factory
from official.vision.evaluation import coco_evaluator
from official.vision.modeling import backbones
from official.vision.modeling import decoders
from official.vision.ops import box_ops
from PIL import Image, ImageDraw

@task_factory.register_task_cls(rngdet_cfg.RngdetTask)
class RNGDetTask(base_task.Task):
  """A single-replica view of training procedure.
  RNGDet task provides artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss,
  post-processing, and customized metrics with reduction.
  """

  def build_model(self):
    """Build RNGDet model."""

    input_specs = tf.keras.layers.InputSpec(shape=[None] +
                                            self._task_config.model.input_size)

    backbone = backbones.factory.build_backbone(
        input_specs=input_specs,
        backbone_config=self._task_config.model.backbone,
        norm_activation_config=self._task_config.model.norm_activation)

    backbone_history = backbones.factory.build_backbone(
        input_specs=input_specs,
        backbone_config=self._task_config.model.backbone,
        norm_activation_config=self._task_config.model.norm_activation)

    segment_fpn = decoders.factory.build_decoder(
        input_specs=backbone.output_specs,
        model_config=self._task_config.model)

    keypoint_fpn = decoders.factory.build_decoder(
        input_specs=backbone.output_specs,
        model_config=self._task_config.model)

    transformer = rngdet_plus.DETRTransformer(
        hidden_size=self._task_config.model.hidden_size,
        num_encoder_layers=self._task_config.model.num_encoder_layers,
        num_decoder_layers=self._task_config.model.num_decoder_layers)
    
    multi_scale = rngdet_plus.MultiScale( 
        transformer, 
        dim=transformer._hidden_size, 
        nheads=transformer._num_heads, 
        fpn_dims= [2048, 1024, 512, 256], 
        output_size = 128 )

    model = rngdet_plus.RNGDet(backbone,
                      backbone_history,
                      self._task_config.model.backbone_endpoint_name,
                      segment_fpn,
                      keypoint_fpn,
                      transformer,
                      multi_scale,
                      self._task_config.model.num_queries,
                      self._task_config.model.hidden_size,
                      self._task_config.model.num_classes)        

    # Builds the model through warm-up call.
    dummy_images = tf.keras.Input(self.task_config.model.input_size)
    dummy_history = tf.keras.Input(self.task_config.model.input_size[:2] + [1])
    _ = model(dummy_images, dummy_history, training=False)

    return model

  def initialize(self, model: tf.keras.Model):
    """Loading pretrained checkpoint."""
    if not self._task_config.init_checkpoint:
      return

    ckpt_dir_or_file = self._task_config.init_checkpoint

    # Restoring checkpoint.
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    if self._task_config.init_checkpoint_modules == 'all':
      ckpt = tf.train.Checkpoint(**model.checkpoint_items)
      status = ckpt.read(ckpt_dir_or_file)
      status.assert_consumed()
    elif self._task_config.init_checkpoint_modules == 'backbone':
      ckpt = tf.train.Checkpoint(backbone=model.backbone)
      status = ckpt.restore(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
      ckpt = tf.train.Checkpoint(backbone=model.backbone_history)
      status = ckpt.restore(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    logging.info('Finished loading pretrained checkpoint from %s', ckpt_dir_or_file)

  def build_inputs(self,
                   params,
                   input_context: Optional[tf.distribute.InputContext] = None):
    """Build input dataset."""

    decoder = rngdet_input.Decoder()

    parser = rngdet_input.Parser(
        roi_size=self._task_config.model.roi_size,
        num_queries=self._task_config.model.num_queries,
        dtype=params.dtype,
    )

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))
    dataset = reader.read(input_context=input_context)

    return dataset

  def _compute_cost(self, cls_outputs, box_outputs, cls_targets, box_targets):
    # Approximate classification cost with 1 - prob[target class].
    # The 1 is a constant that doesn't change the matching, it can be ommitted.
    
    # background: 0
    # (gunho) background : 1 in RNGDet
    background = 1
    cls_cost = self._task_config.losses.lambda_cls * tf.gather(-tf.nn.softmax(cls_outputs), cls_targets, batch_dims=1, axis=-1)

    # Compute the L1 cost between boxes,
    paired_differences = self._task_config.losses.lambda_box * tf.abs( tf.expand_dims(box_outputs, 2) - tf.expand_dims(box_targets, 1))
    box_cost = tf.reduce_sum(paired_differences, axis=-1)
    
    # Compute instacne segmenation loss 
    total_cost = cls_cost + box_cost 
    max_cost = (
        self._task_config.losses.lambda_cls * 0.0 +
        self._task_config.losses.lambda_box * 4.0 )

    # Set pads to large constant
    valid = tf.expand_dims( tf.cast(tf.not_equal(cls_targets, background), dtype=total_cost.dtype), axis=1)
    total_cost = (1 - valid) * max_cost + valid * total_cost

    # Set inf of nan to large constant
    total_cost = tf.where(
        tf.logical_or(tf.math.is_nan(total_cost), tf.math.is_inf(total_cost)),
        max_cost * tf.ones_like(total_cost, dtype=total_cost.dtype),
        total_cost)

    return total_cost

  def segmentation_loss(self, pred_segment, pred_keypoint, labels):
    """Builds segmentation losses for RNGDet."""
    gt_segment = labels['label_masks_roi'][:,:,:,0] 
    gt_keypoint = labels['label_masks_roi'][:,:,:,1]
    focal_loss_fn = tf.keras.losses.BinaryCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

    batch_size = tf.shape(pred_segment)[0]
    pred_segment = tf.reshape(pred_segment, [batch_size, -1, 1])
    gt_segment = tf.reshape(gt_segment, [batch_size, -1, 1])
    valid_segment = tf.where(
        tf.equal(gt_segment, 1), tf.ones_like(gt_segment)*3, tf.ones_like(gt_segment))
    segment_weights_per_replica = tf.reduce_sum(valid_segment)

    pred_keypoint = tf.reshape(pred_keypoint, [batch_size, -1, 1])
    gt_keypoint = tf.reshape(gt_keypoint, [batch_size, -1, 1])
    valid_keypoint = tf.where(
        tf.equal(gt_keypoint, 1), tf.ones_like(gt_keypoint)*6, tf.ones_like(gt_keypoint))
    keypoint_weights_per_replica = tf.reduce_sum(valid_keypoint)


    replica_context = tf.distribute.get_replica_context()
    segment_weights_sum, keypoint_weights_sum = replica_context.all_reduce(
        tf.distribute.ReduceOp.SUM,
        [segment_weights_per_replica, keypoint_weights_per_replica])

    segment_loss = tf.math.divide_no_nan(
        focal_loss_fn(gt_segment, pred_segment, sample_weight=valid_segment),
        tf.cast(segment_weights_sum, tf.float32))
    keypoint_loss = tf.math.divide_no_nan(
        focal_loss_fn(gt_keypoint, pred_keypoint, sample_weight=valid_keypoint),
        tf.cast(keypoint_weights_sum, tf.float32))

    loss = segment_loss + keypoint_loss
    return loss

  def build_losses(self, outputs, labels, aux_losses=None):
    """Builds RNGDet losses."""
    cls_outputs = outputs['cls_outputs']
    box_outputs = outputs['box_outputs']
    instance_outputs = outputs['pred_instance_masks']    

    cls_targets = labels['gt_probs']
    box_targets = labels['gt_coords']
    instance_targets = tf.transpose( labels['gt_masks'], perm=(0, 3, 1, 2)) # (B, size, size, Q) -> (B, Q, size, size)

    cost = self._compute_cost(
        cls_outputs, box_outputs,  
        cls_targets, box_targets)

    _, indices = matchers.hungarian_matching(cost)
    indices = tf.stop_gradient(indices)
    
    target_index = tf.math.argmax(indices, axis=1)
    cls_assigned = tf.gather(cls_outputs, target_index, batch_dims=1, axis=1)
    box_assigned = tf.gather(box_outputs, target_index, batch_dims=1, axis=1)
    instance_assigned =  tf.gather(instance_outputs, target_index, batch_dims=1, axis=1)

    # (gunho) background (eos in RNGDet) is assigned to 1
    background = tf.equal(cls_targets, 1)
    num_boxes = tf.reduce_sum( tf.cast(tf.logical_not(background), tf.float32), axis=-1)

    # Down-weight background to account for class imbalance.
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits( labels=cls_targets, logits=cls_assigned)
    cls_loss = self._task_config.losses.lambda_cls * tf.where( background, self._task_config.losses.background_cls_weight * xentropy, xentropy)
    cls_weights = tf.where( background, self._task_config.losses.background_cls_weight * tf.ones_like(cls_loss), tf.ones_like(cls_loss))

    # Box loss is only calculated on non-background class.
    l_1 = tf.reduce_sum(tf.abs(box_assigned - box_targets), axis=-1)
    box_loss = self._task_config.losses.lambda_box * tf.where( background, tf.zeros_like(l_1), l_1)

    # BCE loss is only calculated on non-background class 
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)  
    ins_loss_before_reduce = self._task_config.losses.lambda_ins * bce(instance_targets, instance_assigned, sample_weight = tf.cast(tf.logical_not(background), tf.float32) )
    ins_loss =  tf.reduce_sum ( ins_loss_before_reduce, axis=-1)  

    # Consider doing all reduce once in train_step to speed up.
    num_boxes_per_replica = tf.reduce_sum(num_boxes)
    cls_weights_per_replica = tf.reduce_sum(cls_weights)
    ins_loss_per_replica = tf.reduce_sum(ins_loss)
    replica_context = tf.distribute.get_replica_context()
    
    num_boxes_sum, cls_weights_sum, ins_loss_sum = replica_context.all_reduce(
        tf.distribute.ReduceOp.SUM,
        [num_boxes_per_replica, cls_weights_per_replica, ins_loss_per_replica])
    
    cls_loss = tf.math.divide_no_nan(
        tf.reduce_sum(cls_loss), cls_weights_sum)

    box_loss = tf.math.divide_no_nan(
        tf.reduce_sum(box_loss), num_boxes_sum)

    ins_loss = tf.math.divide_no_nan(
        tf.cast( tf.reduce_sum(ins_loss) , tf.float32 ), num_boxes_sum)
    
    aux_losses = tf.add_n(aux_losses) if aux_losses else 0.0

    total_loss = cls_loss + box_loss + aux_losses + ins_loss
    return total_loss, cls_loss, box_loss, ins_loss

  def build_metrics(self, training=True):
    """Builds detection metrics."""
    metrics = []
    metric_names = ['cls_loss', 'box_loss', 'seg_loss', 'ins_loss']
    for name in metric_names:
      metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))

    if not training:
      self.coco_metric = coco_evaluator.COCOEvaluator(
          annotation_file=self._task_config.annotation_file,
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
      outputs, pred_segment, pred_keypoint = model(features['sat_roi'],
                                                   features['historical_roi'],
                                                   training=True)
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)
      pred_segment = tf.cast(pred_segment, tf.float32)
      pred_keypoint = tf.cast(pred_keypoint, tf.float32)

      loss = 0.0
      cls_loss = 0.0
      box_loss = 0.0
      seg_loss = 0.0
      ins_loss = 0.0

      seg_loss = self.segmentation_loss(pred_segment, pred_keypoint, labels)
      loss += seg_loss

      # Computes per-replica loss.
      layer_loss, layer_cls_loss, layer_box_loss, layer_ins_loss = self.build_losses( outputs=outputs, labels=labels, aux_losses=model.losses)
      loss += layer_loss
      cls_loss += layer_cls_loss
      box_loss += layer_box_loss
      ins_loss += layer_ins_loss

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
    seg_loss *= num_replicas_in_sync
    ins_loss *= num_replicas_in_sync


    # Trainer class handles loss metric for you.
    logs = {self.loss: loss}


    all_losses = {
        'cls_loss': cls_loss,
        'box_loss': box_loss,
        'seg_loss': seg_loss,
        'ins_loss': ins_loss
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

    roi_size =128 
    sat_roi = features['sat_roi']
    historical_roi =  features['historical_roi'] 
    label_masks_roi =  labels['label_masks_roi'] 
    gt_coords =  labels['gt_coords'] 
    gt_masks = labels['gt_masks']

    rot_index = np.random.randint(0, 4)
    cos_theta = 0 if ( rot_index%2==1 ) else (1 if (rot_index==0) else -1)
    sin_theta = 0 if ( rot_index%2==0 ) else (1 if (rot_index==1) else -1)

    R = tf.constant([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=tf.float32)

    gt_coords = tf.reverse(gt_coords, axis=[1])
    gt_coords = tf.transpose(tf.linalg.matmul(R, gt_coords, transpose_b=True) ) 
    gt_coords = tf.reverse(gt_coords, axis=[1])

    label_masks_roi = tf.image.rot90(label_masks_roi, rot_index) #counter clock wise
    historical_roi = tf.image.rot90(historical_roi, rot_index) #counter clock wise 
    sat_roi = tf.image.rot90(sat_roi, rot_index)
    gt_masks = tf.image.rot90(gt_masks, rot_index)

    #===============================================

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

    # This is for backward compatibility.
    if 'detection_boxes' not in outputs:
      detection_boxes = box_ops.cycxhw_to_yxyx(
          outputs['box_outputs']) * tf.expand_dims(
              tf.concat([
                  labels['image_info'][:, 1:2, 0], labels['image_info'][:, 1:2, 1],
                  labels['image_info'][:, 1:2, 0], labels['image_info'][:, 1:2, 1]
              ],
                        axis=1),
              axis=1)
    else:
      detection_boxes = outputs['detection_boxes']

    detection_scores = tf.math.reduce_max(
        tf.nn.softmax(outputs['cls_outputs'])[:, :, 1:], axis=-1
    ) if 'detection_scores' not in outputs else outputs['detection_scores']

    if 'detection_classes' not in outputs:
      detection_classes = tf.math.argmax(
          outputs['cls_outputs'][:, :, 1:], axis=-1) + 1
    else:
      detection_classes = outputs['detection_classes']

    if 'num_detections' not in outputs:
      num_detections = tf.reduce_sum(
          tf.cast(
              tf.math.greater(
                  tf.math.reduce_max(outputs['cls_outputs'], axis=-1), 0),
              tf.int32),
          axis=-1)
    else:
      num_detections = outputs['num_detections']

    predictions = {
        'detection_boxes': detection_boxes,
        'detection_scores': detection_scores,
        'detection_classes': detection_classes,
        'num_detections': num_detections,
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