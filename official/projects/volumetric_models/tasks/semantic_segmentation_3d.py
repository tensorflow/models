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

"""Image segmentation task definition."""
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from absl import logging
import tensorflow as tf

from official.common import dataset_fn
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.projects.volumetric_models.configs import semantic_segmentation_3d as exp_cfg
from official.projects.volumetric_models.dataloaders import segmentation_input_3d
from official.projects.volumetric_models.evaluation import segmentation_metrics
from official.projects.volumetric_models.losses import segmentation_losses
from official.projects.volumetric_models.modeling import factory


@task_factory.register_task_cls(exp_cfg.SemanticSegmentation3DTask)
class SemanticSegmentation3DTask(base_task.Task):
  """A task for semantic segmentation."""

  def build_model(self) -> tf.keras.Model:
    """Builds segmentation model."""
    input_specs = tf.keras.layers.InputSpec(
        shape=[None] + self.task_config.model.input_size +
        [self.task_config.model.num_channels],
        dtype=self.task_config.train_data.dtype)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (
        tf.keras.regularizers.l2(l2_weight_decay /
                                 2.0) if l2_weight_decay else None)

    model = factory.build_segmentation_model_3d(
        input_specs=input_specs,
        model_config=self.task_config.model,
        l2_regularizer=l2_regularizer)

    # Create a dummy input and call model instance to initialize the model. This
    # is needed when launching multiple experiments using the same model
    # directory. Since there is already a trained model, forward pass will not
    # run and the model will never be built. This is only done when spatial
    # partitioning is not enabled; otherwise it will fail with OOM due to
    # extremely large input.
    if (not self.task_config.train_input_partition_dims) and (
        not self.task_config.eval_input_partition_dims):
      dummy_input = tf.random.uniform(shape=[1] + list(input_specs.shape[1:]))
      _ = model(dummy_input)

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

  def build_inputs(self, params, input_context=None) -> tf.data.Dataset:
    """Builds classification input."""
    decoder = segmentation_input_3d.Decoder(
        image_field_key=params.image_field_key,
        label_field_key=params.label_field_key)
    parser = segmentation_input_3d.Parser(
        input_size=params.input_size,
        num_classes=params.num_classes,
        num_channels=params.num_channels,
        image_field_key=params.image_field_key,
        label_field_key=params.label_field_key,
        dtype=params.dtype,
        label_dtype=params.label_dtype)

    reader = input_reader.InputReader(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self,
                   labels: tf.Tensor,
                   model_outputs: tf.Tensor,
                   aux_losses=None) -> tf.Tensor:
    """Segmentation loss.

    Args:
      labels: labels.
      model_outputs: Output logits of the classifier.
      aux_losses: auxiliarly loss tensors, i.e. `losses` in keras.Model.

    Returns:
      The total loss tensor.
    """
    segmentation_loss_fn = segmentation_losses.SegmentationLossDiceScore(
        metric_type='adaptive')

    total_loss = segmentation_loss_fn(model_outputs, labels)

    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    return total_loss

  def build_metrics(self,
                    training: bool = True) -> Sequence[tf.keras.metrics.Metric]:
    """Gets streaming metrics for training/validation."""
    metrics = []
    num_classes = self.task_config.model.num_classes
    if training:
      metrics.extend([
          tf.keras.metrics.CategoricalAccuracy(
              name='train_categorical_accuracy', dtype=tf.float32)
      ])
    else:
      self.metrics = [
          segmentation_metrics.DiceScore(
              num_classes=num_classes,
              metric_type='generalized',
              per_class_metric=self.task_config.evaluation
              .report_per_class_metric,
              name='val_generalized_dice',
              dtype=tf.float32)
      ]

    return metrics

  def train_step(
      self,
      inputs,
      model: tf.keras.Model,
      optimizer: tf.keras.optimizers.Optimizer,
      metrics: Optional[Sequence[tf.keras.metrics.Metric]] = None
  ) -> Dict[Any, Any]:
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
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

      outputs = outputs['logits']
      if self.task_config.model.head.output_logits:
        outputs = tf.nn.softmax(outputs)

      # Computes per-replica loss.
      loss = self.build_losses(
          labels=labels, model_outputs=outputs, aux_losses=model.losses)
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

    # Compute all metrics within strategy scope for training.
    if metrics:
      labels = tf.cast(labels, tf.float32)
      outputs = tf.cast(outputs, tf.float32)
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})

    return logs

  def validation_step(
      self,
      inputs,
      model: tf.keras.Model,
      metrics: Optional[Sequence[tf.keras.metrics.Metric]] = None
  ) -> Dict[Any, Any]:
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
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)
    outputs = outputs['logits']
    if self.task_config.model.head.output_logits:
      outputs = tf.nn.softmax(outputs)

    loss = self.build_losses(
        model_outputs=outputs, labels=labels, aux_losses=model.losses)
    logs = {self.loss: loss}

    # Compute dice score metrics on CPU.
    for metric in self.metrics:
      labels = tf.cast(labels, tf.float32)
      logits = tf.cast(outputs, tf.float32)
      logs.update({metric.name: (labels, logits)})

    return logs

  def inference_step(self, inputs, model: tf.keras.Model) -> tf.Tensor:
    """Performs the forward step."""
    return model(inputs, training=False)

  def aggregate_logs(
      self,
      state: Optional[Sequence[Union[segmentation_metrics.DiceScore,
                                     tf.keras.metrics.Metric]]] = None,
      step_outputs: Optional[Mapping[str, Any]] = None
  ) -> Sequence[tf.keras.metrics.Metric]:
    """Aggregates statistics to compute metrics over training.

    Args:
      state: A sequence of tf.keras.metrics.Metric objects. Each element records
        a metric.
      step_outputs: A dictionary of [metric_name, (labels, output)] from a step.

    Returns:
      An updated sequence of tf.keras.metrics.Metric objects.
    """
    if state is None:
      for metric in self.metrics:
        metric.reset_states()
      state = self.metrics

    for metric in self.metrics:
      labels = step_outputs[metric.name][0]
      predictions = step_outputs[metric.name][1]

      # If `step_output` is distributed, it contains a tuple of Tensors instead
      # of a single Tensor, so we need to concatenate them along the batch
      # dimension in this case to have a single Tensor.
      if isinstance(labels, tuple):
        labels = tf.concat(list(labels), axis=0)
      if isinstance(predictions, tuple):
        predictions = tf.concat(list(predictions), axis=0)

      labels = tf.cast(labels, tf.float32)
      predictions = tf.cast(predictions, tf.float32)
      metric.update_state(labels, predictions)
    return state

  def reduce_aggregated_logs(
      self,
      aggregated_logs: Optional[Mapping[str, Any]] = None,
      global_step: Optional[tf.Tensor] = None) -> Mapping[str, float]:
    """Reduces logs to obtain per-class metrics if needed.

    Args:
      aggregated_logs: An optional dictionary containing aggregated logs.
      global_step: An optional `tf.Tensor` of current global training steps.

    Returns:
      The reduced logs containing per-class metrics and overall metrics.

    Raises:
      ValueError: If `self.metrics` does not contain exactly 1 metric object.
    """
    result = {}
    if len(self.metrics) != 1:
      raise ValueError('Exact one metric must be present, but {0} are '
                       'present.'.format(len(self.metrics)))

    metric = self.metrics[0].result().numpy()
    if self.task_config.evaluation.report_per_class_metric:
      for i, metric_val in enumerate(metric):
        metric_name = self.metrics[0].name + '/class_{0}'.format(
            i - 1) if i > 0 else self.metrics[0].name
        result.update({metric_name: metric_val})
    else:
      result.update({self.metrics[0].name: metric})
    return result
