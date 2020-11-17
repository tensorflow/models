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
"""Image segmentation task definition."""

from absl import logging
import tensorflow as tf

from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.vision.beta.configs import semantic_segmentation as exp_cfg
from official.vision.beta.dataloaders import segmentation_input
from official.vision.beta.evaluation import segmentation_metrics
from official.vision.beta.losses import segmentation_losses
from official.vision.beta.modeling import factory


@task_factory.register_task_cls(exp_cfg.SemanticSegmentationTask)
class SemanticSegmentationTask(base_task.Task):
  """A task for semantic segmentation."""

  def build_model(self):
    """Builds segmentation model."""
    input_specs = tf.keras.layers.InputSpec(
        shape=[None] + self.task_config.model.input_size)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf.keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = factory.build_segmentation_model(
        input_specs=input_specs,
        model_config=self.task_config.model,
        l2_regularizer=l2_regularizer)
    return model

  def initialize(self, model: tf.keras.Model):
    """Loads pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      return

    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    # Restoring checkpoint.
    if 'all' in self.task_config.init_checkpoint_modules:
      ckpt = tf.train.Checkpoint(**model.checkpoint_items)
      status = ckpt.restore(ckpt_dir_or_file)
      status.assert_consumed()
    else:
      ckpt_items = {}
      if 'backbone' in self.task_config.init_checkpoint_modules:
        ckpt_items.update(backbone=model.backbone)
      if 'decoder' in self.task_config.init_checkpoint_modules:
        ckpt_items.update(decoder=model.decoder)

      ckpt = tf.train.Checkpoint(**ckpt_items)
      status = ckpt.restore(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def build_inputs(self, params, input_context=None):
    """Builds classification input."""

    input_size = self.task_config.model.input_size
    ignore_label = self.task_config.losses.ignore_label

    decoder = segmentation_input.Decoder()
    parser = segmentation_input.Parser(
        output_size=input_size[:2],
        ignore_label=ignore_label,
        resize_eval_groundtruth=params.resize_eval_groundtruth,
        groundtruth_padded_size=params.groundtruth_padded_size,
        aug_scale_min=params.aug_scale_min,
        aug_scale_max=params.aug_scale_max,
        dtype=params.dtype)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self, labels, model_outputs, aux_losses=None):
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
        use_groundtruth_dimension=loss_params.use_groundtruth_dimension)

    total_loss = segmentation_loss_fn(model_outputs, labels['masks'])

    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    return total_loss

  def build_metrics(self, training=True):
    """Gets streaming metrics for training/validation."""
    metrics = []
    if training:
      # TODO(arashwan): make MeanIoU tpu friendly.
      if not isinstance(tf.distribute.get_strategy(),
                        tf.distribute.TPUStrategy):
        metrics.append(segmentation_metrics.MeanIoU(
            name='mean_iou',
            num_classes=self.task_config.model.num_classes,
            rescale_predictions=False))
    else:
      self.miou_metric = segmentation_metrics.MeanIoU(
          name='val_mean_iou',
          num_classes=self.task_config.model.num_classes,
          rescale_predictions=not self.task_config.validation_data
          .resize_eval_groundtruth)

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
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      loss = self.build_losses(
          model_outputs=outputs, labels=labels, aux_losses=model.losses)
      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = loss / num_replicas

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient before apply_gradients when LossScaleOptimizer is
    # used.
    if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})

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

    outputs = self.inference_step(features, model)
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

    if self.task_config.validation_data.resize_eval_groundtruth:
      loss = self.build_losses(model_outputs=outputs, labels=labels,
                               aux_losses=model.losses)
    else:
      loss = 0

    logs = {self.loss: loss}
    logs.update({self.miou_metric.name: (labels, outputs)})

    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})

    return logs

  def inference_step(self, inputs, model):
    """Performs the forward step."""
    return model(inputs, training=False)

  def aggregate_logs(self, state=None, step_outputs=None):
    if state is None:
      self.miou_metric.reset_states()
      state = self.miou_metric
    self.miou_metric.update_state(step_outputs[self.miou_metric.name][0],
                                  step_outputs[self.miou_metric.name][1])
    return state

  def reduce_aggregated_logs(self, aggregated_logs):
    return {self.miou_metric.name: self.miou_metric.result().numpy()}
