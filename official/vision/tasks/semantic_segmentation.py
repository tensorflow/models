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

"""Image segmentation task definition."""
from typing import Any, List, Mapping, Optional, Tuple, Union

from absl import logging
import tensorflow as tf, tf_keras

from official.common import dataset_fn
from official.core import base_task
from official.core import task_factory
from official.vision.configs import semantic_segmentation as exp_cfg
from official.vision.dataloaders import input_reader
from official.vision.dataloaders import input_reader_factory
from official.vision.dataloaders import segmentation_input
from official.vision.dataloaders import tfds_factory
from official.vision.evaluation import segmentation_metrics
from official.vision.losses import segmentation_losses
from official.vision.modeling import factory
from official.vision.utils.object_detection import visualization_utils


@task_factory.register_task_cls(exp_cfg.SemanticSegmentationTask)
class SemanticSegmentationTask(base_task.Task):
  """A task for semantic segmentation."""

  def build_model(self):
    """Builds segmentation model."""
    input_specs = tf_keras.layers.InputSpec(shape=[None] +
                                            self.task_config.model.input_size)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (
        tf_keras.regularizers.l2(l2_weight_decay /
                                 2.0) if l2_weight_decay else None)

    model = factory.build_segmentation_model(
        input_specs=input_specs,
        model_config=self.task_config.model,
        l2_regularizer=l2_regularizer)
    # Builds the model
    dummy_inputs = tf_keras.Input(self.task_config.model.input_size)
    _ = model(dummy_inputs, training=False)
    return model

  def initialize(self, model: tf_keras.Model):
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
        ckpt_items.update(decoder=model.decoder)

      ckpt = tf.train.Checkpoint(**ckpt_items)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def build_inputs(self,
                   params: exp_cfg.DataConfig,
                   input_context: Optional[tf.distribute.InputContext] = None):
    """Builds classification input."""

    ignore_label = self.task_config.losses.ignore_label
    gt_is_matting_map = self.task_config.losses.gt_is_matting_map

    if params.tfds_name:
      decoder = tfds_factory.get_segmentation_decoder(params.tfds_name)
    else:
      decoder = segmentation_input.Decoder(
          image_feature=params.image_feature,
          additional_dense_features=params.additional_dense_features)

    parser = segmentation_input.Parser(
        output_size=params.output_size,
        crop_size=params.crop_size,
        ignore_label=ignore_label,
        resize_eval_groundtruth=params.resize_eval_groundtruth,
        gt_is_matting_map=gt_is_matting_map,
        groundtruth_padded_size=params.groundtruth_padded_size,
        aug_scale_min=params.aug_scale_min,
        aug_scale_max=params.aug_scale_max,
        aug_rand_hflip=params.aug_rand_hflip,
        preserve_aspect_ratio=params.preserve_aspect_ratio,
        dtype=params.dtype,
        image_feature=params.image_feature,
        additional_dense_features=params.additional_dense_features,
        centered_crop=params.centered_crop)

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        combine_fn=input_reader.create_combine_fn(params),
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self,
                   labels: Mapping[str, tf.Tensor],
                   model_outputs: Union[Mapping[str, tf.Tensor], tf.Tensor],
                   aux_losses: Optional[Any] = None):
    """Segmentation loss.

    Args:
      labels: labels.
      model_outputs: Output logits of the classifier.
      aux_losses: auxiliarly loss tensors, i.e. `losses` in keras.Model.

    Returns:
      The total loss tensor.
    """
    loss_params = self._task_config.losses
    segmentation_loss_fn = segmentation_losses.SegmentationLoss(
        loss_params.label_smoothing,
        loss_params.class_weights,
        loss_params.ignore_label,
        use_groundtruth_dimension=loss_params.use_groundtruth_dimension,
        use_binary_cross_entropy=loss_params.use_binary_cross_entropy,
        top_k_percent_pixels=loss_params.top_k_percent_pixels,
        gt_is_matting_map=loss_params.gt_is_matting_map)

    total_loss = segmentation_loss_fn(model_outputs['logits'], labels['masks'])

    if 'mask_scores' in model_outputs:
      mask_scoring_loss_fn = segmentation_losses.MaskScoringLoss(
          loss_params.ignore_label)
      total_loss += loss_params.mask_scoring_weight * mask_scoring_loss_fn(
          model_outputs['mask_scores'],
          model_outputs['logits'],
          labels['masks'])

    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    total_loss = loss_params.loss_weight * total_loss

    return total_loss

  def process_metrics(self, metrics, labels, model_outputs, **kwargs):
    """Process and update metrics.

    Called when using custom training loop API.

    Args:
      metrics: a nested structure of metrics objects. The return of function
        self.build_metrics.
      labels: a tensor or a nested structure of tensors.
      model_outputs: a tensor or a nested structure of tensors. For example,
        output of the keras model built by self.build_model.
      **kwargs: other args.
    """
    for metric in metrics:
      if 'mask_scores_mse' == metric.name:
        actual_mask_scores = segmentation_losses.get_actual_mask_scores(
            model_outputs['logits'], labels['masks'],
            self.task_config.losses.ignore_label)
        metric.update_state(actual_mask_scores, model_outputs['mask_scores'])
      else:
        metric.update_state(labels, model_outputs['logits'])

  def build_metrics(self, training: bool = True):
    """Gets streaming metrics for training/validation."""
    metrics = []
    self.iou_metric = None

    if training and self.task_config.evaluation.report_train_mean_iou:
      metrics.append(
          segmentation_metrics.MeanIoU(
              name='mean_iou',
              num_classes=self.task_config.model.num_classes,
              rescale_predictions=False,
              dtype=tf.float32))
      if self.task_config.model.get('mask_scoring_head'):
        metrics.append(
            tf_keras.metrics.MeanSquaredError(name='mask_scores_mse'))

    if not training:
      self.iou_metric = segmentation_metrics.PerClassIoU(
          name='per_class_iou',
          num_classes=self.task_config.model.num_classes,
          rescale_predictions=(
              not self.task_config.validation_data.resize_eval_groundtruth),
          dtype=tf.float32)
      if (self.task_config.validation_data.resize_eval_groundtruth and
          self.task_config.model.get('mask_scoring_head')):
        # Masks scores metric can only be computed if labels are scaled to match
        # preticted mask scores.
        metrics.append(
            tf_keras.metrics.MeanSquaredError(name='mask_scores_mse'))

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

    input_partition_dims = self.task_config.train_input_partition_dims
    if input_partition_dims:
      strategy = tf.distribute.get_strategy()
      features = strategy.experimental_split_to_logical_devices(
          features, input_partition_dims)

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      if isinstance(outputs, tf.Tensor):
        outputs = {'logits': outputs}
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      loss = self.build_losses(
          model_outputs=outputs, labels=labels, aux_losses=model.losses)
      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = loss / num_replicas

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient before apply_gradients when LossScaleOptimizer is
    # used.
    if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})

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

    input_partition_dims = self.task_config.eval_input_partition_dims
    if input_partition_dims:
      strategy = tf.distribute.get_strategy()
      features = strategy.experimental_split_to_logical_devices(
          features, input_partition_dims)

    outputs = self.inference_step(features, model)
    if isinstance(outputs, tf.Tensor):
      outputs = {'logits': outputs}
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

    if self.task_config.validation_data.resize_eval_groundtruth:
      loss = self.build_losses(
          model_outputs=outputs, labels=labels, aux_losses=model.losses)
    else:
      loss = 0

    logs = {self.loss: loss}

    if self.iou_metric is not None:
      self.iou_metric.update_state(labels, outputs['logits'])
    if metrics:
      self.process_metrics(metrics, labels, outputs)

    if (
        hasattr(self.task_config, 'allow_image_summary')
        and self.task_config.allow_image_summary
    ):
      logs.update(
          {'visualization': (tf.cast(features, dtype=tf.float32), outputs)}
      )

    return logs

  def inference_step(self, inputs: tf.Tensor, model: tf_keras.Model):
    """Performs the forward step."""
    return model(inputs, training=False)

  def aggregate_logs(self, state=None, step_outputs=None):
    if state is None and self.iou_metric is not None:
      self.iou_metric.reset_states()

    if 'visualization' in step_outputs:
      # Update segmentation state for writing summary if there are artifacts for
      # visualization.
      if state is None:
        state = {}
      state.update(visualization_utils.update_segmentation_state(step_outputs))

    if state is None:
      # Create an arbitrary state to indicate it's not the first step in the
      # following calls to this function.
      state = True

    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    logs = {}
    if self.iou_metric is not None:
      ious = self.iou_metric.result()
      # TODO(arashwan): support loading class name from a label map file.
      if self.task_config.evaluation.report_per_class_iou:
        for i, value in enumerate(ious.numpy()):
          logs.update({'iou/{}'.format(i): value})
      # Computes mean IoU
      logs.update({'mean_iou': tf.reduce_mean(ious)})

    # Add visualization for summary.
    if isinstance(aggregated_logs, dict) and 'image' in aggregated_logs:
      validation_outputs = visualization_utils.visualize_segmentation_outputs(
          logs=aggregated_logs, task_config=self.task_config
      )
      logs.update(validation_outputs)

    return logs
