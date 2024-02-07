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

"""Spatiotemporal action localization task."""
from typing import Any, List, Optional, Mapping

from absl import logging
import tensorflow as tf, tf_keras

from official.core import task_factory
from official.projects.videoglue.configs import spatiotemporal_action_localization as exp_cfg
from official.projects.videoglue.datasets import dataset_factory
from official.projects.videoglue.evaluation import spatiotemporal_action_localization_evaluator as eval_util
from official.projects.videoglue.tools import checkpoint_loader
from official.vision.modeling import factory_3d
from official.vision.tasks import video_classification


@task_factory.register_task_cls(exp_cfg.SpatiotemporalActionLocalizationTask)
class SpatiotemporalActionLocalizationTask(
    video_classification.VideoClassificationTask):
  """Spatiotemporal action localization task."""

  def _is_multilabel(self):
    """Whether the dataset/task has multi-labels."""
    return True

  def build_model(self) -> tf_keras.Model:
    """Builds video model."""
    common_input_shape = [
        d1 if d1 == d2 else None
        for d1, d2 in zip(self.task_config.train_data.feature_shape,
                          self.task_config.validation_data.feature_shape)
    ]

    num_instances = self.task_config.train_data.num_instances
    input_specs_dict = {
        'image':
            tf_keras.layers.InputSpec(shape=[None] + common_input_shape),
        'instances_position':
            tf_keras.layers.InputSpec(shape=[None, num_instances, 4]),
    }

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf_keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = factory_3d.build_model(
        self.task_config.model.model_type,
        input_specs=input_specs_dict,
        model_config=self.task_config.model,
        num_classes=self.task_config.train_data.num_classes,
        l2_regularizer=l2_regularizer)

    if self.task_config.freeze_backbone:
      logging.info('Freezing model backbone.')
      model.backbone.trainable = False
    return model

  def build_inputs(
      self, params: exp_cfg.DataConfig, input_context: Any = None
  ) -> tf.data.Dataset:
    """Builds classification input."""
    augmentation_type = params.data_augmentation.type
    augmentation_params = params.data_augmentation.get().as_dict()
    dataset_config = {
        'is_training': params.is_training,
        'num_frames': params.feature_shape[0],
        'temporal_stride': params.temporal_stride,
        'num_instance_per_frame': params.num_instances,
        'min_resize': params.min_resize,
        'crop_size': params.feature_shape[1],
        'zero_centering_image': params.zero_centering_image,
        'color_augmentation': params.color_augmentation,
        'num_test_clips': params.num_test_clips,
        'augmentation_type': augmentation_type,
        'augmentation_params': augmentation_params,
        'one_hot_label': params.one_hot_label,
        'merge_multi_labels': params.merge_multi_labels,
        'import_detected_bboxes': params.import_detected_bboxes,
    }
    data_loader = dataset_factory.DataLoader(
        params=params, dataset_config=dataset_config)
    return data_loader(input_context=input_context)

  def initialize(self, model: tf_keras.Model):
    """Loads pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      return

    checkpoint_loader.get_checkpoint_loader(
        model=model,
        init_checkpoint=self.task_config.init_checkpoint,
        init_checkpoint_type=self.task_config.init_checkpoint_modules)

  def build_losses(self,
                   labels: Mapping[str, tf.Tensor],
                   model_outputs: tf.Tensor,
                   aux_losses: Optional[Any] = None):
    """Sparse categorical cross entropy loss.

    Args:
      labels: labels dictionary contains multi-hot "class_target" and
        "sample_weights".
      model_outputs: Output logits of the classifier.
      aux_losses: auxiliarly loss tensors, i.e. `losses` in keras.Model.

    Returns:
      The dictionary contains total loss tensors.
    """
    losses_config = self.task_config.losses

    # in shape [B, N]
    xent_loss_fn = tf_keras.losses.BinaryCrossentropy(
        reduction=tf_keras.losses.Reduction.NONE,
        from_logits=False,
        label_smoothing=losses_config.label_smoothing)
    class_targets = labels['class_target']
    sample_weight = tf.cast(labels['instances_mask'], tf.float32)
    model_loss = xent_loss_fn(y_true=class_targets,
                              y_pred=model_outputs,
                              sample_weight=sample_weight)
    if self._is_multilabel():
      # Re-scale the binary cross entropy loss with num_classes to make it
      # comparable with categorical cross entropy in scale. This enables to use
      # the same learning rate for different losses.
      model_loss *= self._get_num_classes()

    model_loss = tf.reduce_sum(model_loss) / (
        tf.reduce_sum(sample_weight) + 1e-6)

    if aux_losses:
      regularization_loss = tf.add_n(aux_losses)
    else:
      regularization_loss = 0.0

    total_loss = model_loss + regularization_loss
    logs = {
        'model_loss': model_loss,
        'regularization_loss': regularization_loss,
        # Include model predictions and corresponding boxes for the eval.
        'predictions': model_outputs,
        # Register total loss.
        self.loss: total_loss,
    }
    return logs

  def build_metrics(self, training: bool = True):
    """Gets streaming metrics for training/validation."""
    metrics = [
        tf_keras.metrics.AUC(curve='PR', multi_label=True, name='AUPR'),
        tf_keras.metrics.AUC(curve='ROC', multi_label=True, name='AUROC'),
    ]
    self.evaluator = eval_util.SpatiotemporalActionLocalizationEvaluator()
    return metrics

  def process_metrics(self, metrics: List[Any],
                      labels: Mapping[str, tf.Tensor],
                      model_outputs: tf.Tensor):
    """Processes and updates metrics.

    Called when using custom training loop API.

    Args:
      metrics: a nested structure of metrics objects. The return of function
        self.build_metrics.
      labels: a tensor or a nested structure of tensors.
      model_outputs: a tensor or a nested structure of tensors. For example,
        output of the keras model built by self.build_model.
    """
    num_classes = self.task_config.train_data.num_classes
    class_targets = tf.reshape(labels['class_target'], [-1, num_classes])
    model_outputs = tf.reshape(model_outputs, [-1, num_classes])
    sample_weight = tf.cast(tf.reshape(labels['instances_mask'], [-1]),
                            tf.float32)
    for metric in metrics:
      metric.update_state(
          y_true=class_targets,
          y_pred=model_outputs,
          sample_weight=sample_weight)

  def train_step(self,
                 inputs: Mapping[str, tf.Tensor],
                 model: tf_keras.Model,
                 optimizer: tf_keras.optimizers.Optimizer,
                 metrics: Optional[List[Any]] = None):
    """Does one forward and backward pass.

    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features = {
        'image': inputs['image'],
        'instances_position': inputs['instances_position'],
    }
    labels = {
        'class_target': inputs['label'],
        'instances_mask': inputs['instances_mask'],
    }
    return super().train_step(
        inputs=(features, labels),
        model=model,
        optimizer=optimizer,
        metrics=metrics)

  def validation_step(self,
                      inputs: Mapping[str, tf.Tensor],
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
    if self.task_config.validation_data.import_detected_bboxes:
      instances_position = inputs['detected_instances_position']
      instances_mask = inputs['detected_instances_mask']
      instances_score = inputs['detected_instances_score']
    else:
      instances_position = inputs['instances_position']
      instances_mask = inputs['instances_mask']
      instances_score = inputs['instances_score']

    features = {
        'image': inputs['image'],
        'instances_position': instances_position,
    }
    labels = {
        'class_target': inputs['label'],
        'instances_mask': instances_mask,
    }
    logs = super().validation_step(
        inputs=(features, labels), model=model, metrics=metrics)

    # Final action_prob should be multiplied by the box score.
    predictions = logs['predictions'] * instances_score[..., None]

    # Add predictions and labels for the next step eval_end().
    logs.update({
        'predictions': predictions,
        'instances_position': instances_position,
        'nonmerge_label': inputs['nonmerge_label'],
        'nonmerge_instances_position': inputs['nonmerge_instances_position'],
    })
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    """Aggregates logs."""
    if state is None:
      self.evaluator.reset_states()
      # Create an arbitrary state to indicate it's not the first step in the
      # following calls to this function.
      state = True
    self.evaluator.update_state(step_outputs)
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    """Reduces aggregated logs."""
    return self.evaluator.result()
