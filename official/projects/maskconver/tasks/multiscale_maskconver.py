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

"""Panoptic Multi-scale MaskConver task definition."""
from typing import Any, Dict, List, Mapping, Optional, Tuple
import tensorflow as tf, tf_keras

from official.common import dataset_fn
from official.core import task_factory
from official.projects.maskconver.configs import multiscale_maskconver as exp_cfg
from official.projects.maskconver.dataloaders import multiscale_maskconver_input
from official.projects.maskconver.losses import maskconver_losses
from official.projects.maskconver.modeling import factory
from official.projects.maskconver.modeling.layers import copypaste
from official.projects.maskconver.tasks import maskconver
from official.projects.volumetric_models.losses import segmentation_losses as volumeteric_segmentation_losses
from official.vision.dataloaders import input_reader_factory


@task_factory.register_task_cls(exp_cfg.MultiScaleMaskConverTask)
class PanopticMultiScaleMaskConverTask(maskconver.PanopticMaskRCNNTask):

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

    model = factory.build_multiscale_maskconver_model(
        input_specs=input_specs,
        model_config=self.task_config.model,
        l2_regularizer=l2_regularizer)

    # Get images and labels with batch size of 1.
    images, labels = next(
        iter(self.build_inputs(self.task_config.validation_data)))
    images = tf.nest.map_structure(lambda x: x[0:1, ...], images)
    labels = tf.nest.map_structure(lambda x: x[0:1, ...], labels)
    _ = model(
        images,
        image_info=labels['image_info'],
        training=False)
    return model

  def build_inputs(
      self,
      params: exp_cfg.DataConfig,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Build input dataset."""
    decoder_cfg = params.decoder.get()

    if params.decoder.type == 'simple_decoder':
      decoder = multiscale_maskconver_input.TfExampleDecoder(
          regenerate_source_id=decoder_cfg.regenerate_source_id,
          mask_binarize_threshold=decoder_cfg.mask_binarize_threshold,
          include_panoptic_masks=decoder_cfg.include_panoptic_masks,
          panoptic_category_mask_key=decoder_cfg.panoptic_category_mask_key,
          panoptic_instance_mask_key=decoder_cfg.panoptic_instance_mask_key)
    else:
      raise ValueError('Unknown decoder type: {}!'.format(params.decoder.type))

    if params.parser.copypaste:
      sample_fn = copypaste.CopyPaste(
          self.task_config.model.input_size[:2],
          copypaste_frequency=params.parser.copypaste.copypaste_frequency,
          copypaste_aug_scale_max=params.parser.copypaste.copypaste_aug_scale_max,
          copypaste_aug_scale_min=params.parser.copypaste.copypaste_aug_scale_min,
          aug_scale_min=params.parser.copypaste.aug_scale_min,
          aug_scale_max=params.parser.copypaste.aug_scale_max,
          random_flip=params.parser.aug_rand_hflip,
          num_thing_classes=self.task_config.model.num_thing_classes)
    else:
      sample_fn = None

    parser = multiscale_maskconver_input.Parser(
        output_size=self.task_config.model.input_size[:2],
        min_level=self.task_config.model.min_level,
        max_level=self.task_config.model.max_level,
        fpn_low_range=params.parser.fpn_low_range,
        fpn_high_range=params.parser.fpn_high_range,
        dtype=params.dtype,
        aug_rand_hflip=params.parser.aug_rand_hflip,
        aug_scale_min=params.parser.aug_scale_min,
        aug_scale_max=params.parser.aug_scale_max,
        max_num_instances=params.parser.max_num_instances,
        segmentation_resize_eval_groundtruth=params.parser
        .segmentation_resize_eval_groundtruth,
        segmentation_groundtruth_padded_size=params.parser
        .segmentation_groundtruth_padded_size,
        segmentation_ignore_label=params.parser.segmentation_ignore_label,
        panoptic_ignore_label=params.parser.panoptic_ignore_label,
        num_panoptic_categories=self.task_config.model.num_classes,
        num_thing_categories=self.task_config.model.num_thing_classes,
        mask_target_level=params.parser.mask_target_level,
        level=self.task_config.model.level,
        gaussian_iou=params.parser.gaussaian_iou,
        aug_type=params.parser.aug_type,)

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        sample_fn=sample_fn.copypaste_fn(
            params.is_training) if sample_fn else None,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))
    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self,
                   outputs: Mapping[str, Any],
                   labels: Mapping[str, Any],
                   iteration: Any,
                   aux_losses: Optional[Any] = None,
                   step=None) -> Dict[str, tf.Tensor]:
    """Build Panoptic Mask R-CNN losses."""
    # pylint: disable=line-too-long
    loss_params = self._task_config.losses
    center_loss_fn = maskconver_losses.PenaltyReducedLogisticFocalLoss(
        alpha=loss_params.alpha, beta=loss_params.beta)

    true_flattened_ct_heatmap = labels['panoptic_heatmaps']
    true_flattened_ct_heatmap = tf.cast(true_flattened_ct_heatmap, tf.float32)

    pred_flattened_ct_heatmap = outputs['class_heatmaps']
    pred_flattened_ct_heatmap = tf.cast(pred_flattened_ct_heatmap, tf.float32)

    center_loss = center_loss_fn(
        target_tensor=true_flattened_ct_heatmap,
        prediction_tensor=pred_flattened_ct_heatmap,
        weights=1.0)

    replica_context = tf.distribute.get_replica_context()
    global_num_instances = replica_context.all_reduce(
        tf.distribute.ReduceOp.SUM, labels['num_instances'])
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    num_instances = tf.cast(global_num_instances, tf.float32) / tf.cast(num_replicas, tf.float32) + 1.0

    center_loss = tf.reduce_sum(center_loss) / num_instances

    gt_masks = labels['panoptic_masks']
    gt_mask_weights = labels['panoptic_mask_weights'][:, None, None, :] * tf.ones_like(gt_masks)
    panoptic_padding_mask = labels['panoptic_padding_mask'][:, :, :, None] * tf.ones_like(gt_masks)

    # gt_masks
    _, h, w, q = gt_masks.get_shape().as_list()
    predicted_masks = tf.cast(outputs['mask_proposal_logits'], tf.float32)
    predicted_masks = tf.image.resize(
        predicted_masks, tf.shape(gt_masks)[1:3], method='bilinear')

    mask_loss_fn = tf_keras.losses.BinaryCrossentropy(
        from_logits=True,
        label_smoothing=0.0,
        axis=-1,
        reduction=tf_keras.losses.Reduction.NONE,
        name='binary_crossentropy')

    mask_weights = tf.cast(gt_masks >= 0, tf.float32) * gt_mask_weights  * (
        1 - panoptic_padding_mask)  # b, h, w, # max inst
    mask_loss = mask_loss_fn(
        tf.expand_dims(gt_masks, -1),
        tf.expand_dims(predicted_masks, -1),
        sample_weight=tf.expand_dims(mask_weights, -1))

    mask_loss = tf.reshape(mask_loss, [-1, h * w, q])
    mask_loss = tf.reduce_sum(tf.reduce_mean(mask_loss, axis=1)) / num_instances

    # Dice loss
    masked_predictions = tf.sigmoid(predicted_masks) * tf.cast(
        gt_mask_weights > 0, tf.float32) * (1 - panoptic_padding_mask)
    masked_gt_masks = gt_masks * tf.cast(gt_mask_weights > 0, tf.float32) * (
        1 - panoptic_padding_mask)

    masked_predictions = tf.transpose(masked_predictions, [0, 3, 1, 2])
    masked_predictions = tf.reshape(masked_predictions, [-1, h, w, 1])
    masked_gt_masks = tf.transpose(masked_gt_masks, [0, 3, 1, 2])
    masked_gt_masks = tf.reshape(masked_gt_masks, [-1, h, w, 1])

    dice_loss_fn = volumeteric_segmentation_losses.SegmentationLossDiceScore(
        metric_type='adaptive', axis=(2, 3))
    dice_loss = dice_loss_fn(logits=masked_predictions, labels=masked_gt_masks)

    total_loss = center_loss + loss_params.mask_weight * (mask_loss + dice_loss)
    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    total_loss = loss_params.loss_weight * total_loss

    losses = {'total_loss': total_loss,
              'mask_loss': mask_loss,
              'center_loss': center_loss,
              'dice_loss': dice_loss,}
    return losses

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
          iteration=optimizer.iterations,
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
