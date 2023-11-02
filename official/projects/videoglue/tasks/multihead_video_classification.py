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

"""HS Video Classification task."""
from typing import Any, List, Optional, Mapping

import tensorflow as tf, tf_keras

from official.core import task_factory
from official.projects.videoglue.configs import video_classification as exp_cfg
from official.projects.videoglue.datasets import dataset_factory
from official.projects.videoglue.tools import checkpoint_loader
from official.vision.tasks import video_classification


@task_factory.register_task_cls(exp_cfg.MultiHeadVideoClassificationTask)
class MultiHeadVideoClassificationTask(
    video_classification.VideoClassificationTask):
  """Internal video classification task."""

  def _is_multihead(self):
    """Reports the joint accuracy or not."""
    label_names = self.task_config.train_data.label_names
    is_multihead = isinstance(label_names, list) and len(label_names) > 1
    return is_multihead

  def _get_label_names(self):
    """Gets the label names."""
    if self._is_multihead():
      label_names = self.task_config.train_data.label_names
    else:
      label_names = [self.task_config.train_data.label_names]
    return label_names

  def build_inputs(self, params: exp_cfg.DataConfig, input_context=None):
    """Builds classification input."""
    augmentation_type = params.data_augmentation.type
    augmentation_params = params.data_augmentation.get().as_dict()

    randaug_params = None
    if params.randaug is not None:
      randaug_params = params.randaug.as_dict()

    autoaug_params = None
    if params.autoaug is not None:
      autoaug_params = params.autoaug.as_dict()

    mixup_cutmix_params = None
    if params.mixup_cutmix is not None:
      mixup_cutmix_params = params.mixup_cutmix.as_dict()

    dataset_config = {
        'is_training': params.is_training,
        'num_frames': params.feature_shape[0],
        'temporal_stride': params.temporal_stride,
        'sample_from_segments': params.sample_from_segments,
        'min_resize': params.min_resize,
        'crop_size': params.feature_shape[1],
        'zero_centering_image': params.zero_centering_image,
        'random_flip_image': params.random_flip_image,
        'num_test_clips': params.num_test_clips,
        'augmentation_type': augmentation_type,
        'augmentation_params': augmentation_params,
        'randaug_params': randaug_params,
        'autoaug_params': autoaug_params,
        'mixup_cutmix_params': mixup_cutmix_params,
        # TODO(lzyuan): Unify num_test_crops and multi_crop flags.
        'multi_crop': params.num_test_crops == 3,
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

  def build_metrics(self, training: bool = True):
    """Gets streaming metrics for training/validation."""
    metrics = []
    for label_name in self._get_label_names():
      if self._is_multilabel():
        metrics += [
            tf_keras.metrics.AUC(
                curve='ROC',
                multi_label=self._is_multilabel(),
                name=f'{label_name}/ROC-AUC'),
            tf_keras.metrics.RecallAtPrecision(
                precision=0.95, name=f'{label_name}/RecallAtPrecision95'),
            tf_keras.metrics.AUC(
                curve='PR',
                multi_label=self._is_multilabel(),
                name=f'{label_name}/PR-AUC'),
        ]
      else:
        metrics += [
            tf_keras.metrics.CategoricalAccuracy(
                name=f'{label_name}/accuracy'),
            tf_keras.metrics.TopKCategoricalAccuracy(
                k=1, name=f'{label_name}/top_1_accuracy'),
            tf_keras.metrics.TopKCategoricalAccuracy(
                k=5, name=f'{label_name}/top_5_accuracy')
        ]

    if self._is_multihead():
      metrics.append(
          tf_keras.metrics.Mean(name='label_joint/accuracy'))
    return metrics

  def process_metrics(self, metrics: List[Any],
                      labels: List[tf.Tensor],
                      model_outputs: List[tf.Tensor]):
    """Processes and updates metrics.

    Called when using custom training loop API.

    Args:
      metrics: a nested structure of metrics objects. The return of function
        self.build_metrics.
      labels: a nested structure of tensors contains labels.
      model_outputs: a list of output tensors. Assume the order is aligned with
        the label_names list.
    """
    for i, label_name in enumerate(self._get_label_names()):
      for metric in metrics:
        if label_name in metric.name:
          metric.update_state(labels[i], model_outputs[i])

    if self._is_multihead():

      def joint_accuracy_fn(y_true: List[tf.Tensor], y_pred: List[tf.Tensor]):
        """Calculates the joint accuracy of predictions."""
        hits = []
        for label, pred in zip(y_true, y_pred):
          label_id = tf.argmax(label, axis=-1)
          pred_id = tf.argmax(pred, axis=-1)
          hits.append(tf.equal(label_id, pred_id))
        hits = tf.math.reduce_all(tf.stack(hits, axis=-1), axis=-1)
        return tf.reduce_mean(tf.cast(hits, tf.float32))

      for metric in metrics:
        if 'joint' in metric.name:
          values = joint_accuracy_fn(labels, model_outputs)
          metric.update_state(values)

  def build_losses(self,
                   labels: List[tf.Tensor],
                   model_outputs: List[tf.Tensor],
                   aux_losses: Optional[Any] = None):
    """Builds losses."""

    all_losses = 0
    for model_output, label in zip(model_outputs, labels):
      losses = super().build_losses(model_outputs=model_output,
                                    labels=label,
                                    aux_losses=aux_losses)
      all_losses += losses[self.loss]
    return all_losses

  def train_step(self,
                 inputs: Mapping[str, Any],
                 model: tf_keras.Model,
                 optimizer: tf_keras.optimizers.Optimizer,
                 metrics: Optional[List[Any]] = None):
    """Does forward and backward pass.

    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features = inputs['image']
    labels = [inputs[k] for k in self._get_label_names()]

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      # tf_keras.Model eliminates the list if the outputs list len is 1.
      # Recover it here to be compatible with multihead settings.
      outputs = [outputs] if isinstance(outputs, tf.Tensor) else outputs
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

      if self._is_multilabel():
        outputs = tf.nest.map_structure(tf.math.sigmoid, outputs)
      else:
        outputs = tf.nest.map_structure(tf.math.softmax, outputs)

      all_losses = self.build_losses(model_outputs=outputs,
                                     labels=labels,
                                     aux_losses=model.losses)

      # Scale loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = all_losses / num_replicas

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(
          optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scale back gradient before apply_gradients when LossScaleOptimizer is
    # used.
    if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {self.loss: all_losses}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})
    return logs

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
    features = inputs['image']
    labels = [inputs[k] for k in self._get_label_names()]

    input_partition_dims = self.task_config.eval_input_partition_dims
    if input_partition_dims:
      strategy = tf.distribute.get_strategy()
      features['image'] = strategy.experimental_split_to_logical_devices(
          features['image'], input_partition_dims)

    outputs = self.inference_step(features, model)
    # tf_keras.Model eliminates the list if the outputs list len is 1.
    # Recover it here to be compatible with multihead settings.
    outputs = [outputs] if isinstance(outputs, tf.Tensor) else outputs
    # Casting output layer as float32 is necessary when mixed_precision is
    # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)
    all_losses = self.build_losses(model_outputs=outputs, labels=labels,
                                   aux_losses=model.losses)

    logs = {self.loss: all_losses}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})
    return logs
