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

"""Image classification task definition."""
from typing import Any, Optional, List, Tuple
from absl import logging
import tensorflow as tf

from official.common import dataset_fn
from official.core import base_task
from official.core import task_factory
from official.modeling import tf_utils
from official.vision.configs import image_classification as exp_cfg
from official.vision.dataloaders import classification_input
from official.vision.dataloaders import input_reader_factory
from official.vision.dataloaders import tfds_factory
from official.vision.modeling import factory
from official.vision.ops import augment


@task_factory.register_task_cls(exp_cfg.ImageClassificationTask)
class ImageClassificationTask(base_task.Task):
  """A task for image classification."""

  def build_model(self):
    """Builds classification model."""
    input_specs = tf.keras.layers.InputSpec(
        shape=[None] + self.task_config.model.input_size)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf.keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = factory.build_classification_model(
        input_specs=input_specs,
        model_config=self.task_config.model,
        l2_regularizer=l2_regularizer)

    if self.task_config.freeze_backbone:
      model.backbone.trainable = False
    return model

  def initialize(self, model: tf.keras.Model):
    """Loads pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      return

    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    # Restoring checkpoint.
    if self.task_config.init_checkpoint_modules == 'all':
      ckpt = tf.train.Checkpoint(model=model)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    elif self.task_config.init_checkpoint_modules == 'backbone':
      ckpt = tf.train.Checkpoint(backbone=model.backbone)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    else:
      raise ValueError(
          "Only 'all' or 'backbone' can be used to initialize the model.")

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def build_inputs(
      self,
      params: exp_cfg.DataConfig,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Builds classification input."""

    num_classes = self.task_config.model.num_classes
    input_size = self.task_config.model.input_size
    image_field_key = self.task_config.train_data.image_field_key
    label_field_key = self.task_config.train_data.label_field_key
    is_multilabel = self.task_config.train_data.is_multilabel

    if params.tfds_name:
      decoder = tfds_factory.get_classification_decoder(params.tfds_name)
    else:
      decoder = classification_input.Decoder(
          image_field_key=image_field_key, label_field_key=label_field_key,
          is_multilabel=is_multilabel)

    parser = classification_input.Parser(
        output_size=input_size[:2],
        num_classes=num_classes,
        image_field_key=image_field_key,
        label_field_key=label_field_key,
        decode_jpeg_only=params.decode_jpeg_only,
        aug_rand_hflip=params.aug_rand_hflip,
        aug_type=params.aug_type,
        color_jitter=params.color_jitter,
        random_erasing=params.random_erasing,
        is_multilabel=is_multilabel,
        dtype=params.dtype)

    postprocess_fn = None
    if params.mixup_and_cutmix:
      postprocess_fn = augment.MixupAndCutmix(
          mixup_alpha=params.mixup_and_cutmix.mixup_alpha,
          cutmix_alpha=params.mixup_and_cutmix.cutmix_alpha,
          prob=params.mixup_and_cutmix.prob,
          label_smoothing=params.mixup_and_cutmix.label_smoothing,
          num_classes=num_classes)

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training),
        postprocess_fn=postprocess_fn)

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self,
                   labels: tf.Tensor,
                   model_outputs: tf.Tensor,
                   aux_losses: Optional[Any] = None) -> tf.Tensor:
    """Builds sparse categorical cross entropy loss.

    Args:
      labels: Input groundtruth labels.
      model_outputs: Output logits of the classifier.
      aux_losses: The auxiliarly loss tensors, i.e. `losses` in tf.keras.Model.

    Returns:
      The total loss tensor.
    """
    losses_config = self.task_config.losses
    is_multilabel = self.task_config.train_data.is_multilabel

    if not is_multilabel:
      if losses_config.one_hot:
        total_loss = tf.keras.losses.categorical_crossentropy(
            labels,
            model_outputs,
            from_logits=True,
            label_smoothing=losses_config.label_smoothing)
      elif losses_config.soft_labels:
        total_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels, model_outputs)
      else:
        total_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, model_outputs, from_logits=True)
    else:
      # Multi-label weighted binary cross entropy loss.
      total_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=model_outputs)
      total_loss = tf.reduce_sum(total_loss, axis=-1)

    total_loss = tf_utils.safe_mean(total_loss)
    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    total_loss = losses_config.loss_weight * total_loss
    return total_loss

  def build_metrics(self,
                    training: bool = True) -> List[tf.keras.metrics.Metric]:
    """Gets streaming metrics for training/validation."""
    is_multilabel = self.task_config.train_data.is_multilabel
    if not is_multilabel:
      k = self.task_config.evaluation.top_k
      if (self.task_config.losses.one_hot or
          self.task_config.losses.soft_labels):
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=k, name='top_{}_accuracy'.format(k))]
        if hasattr(
            self.task_config.evaluation, 'precision_and_recall_thresholds'
        ) and self.task_config.evaluation.precision_and_recall_thresholds:
          thresholds = self.task_config.evaluation.precision_and_recall_thresholds
          # pylint:disable=g-complex-comprehension
          metrics += [
              tf.keras.metrics.Precision(
                  thresholds=th,
                  name='precision_at_threshold_{}'.format(th),
                  top_k=1) for th in thresholds
          ]
          metrics += [
              tf.keras.metrics.Recall(
                  thresholds=th,
                  name='recall_at_threshold_{}'.format(th),
                  top_k=1) for th in thresholds
          ]

          # Add per-class precision and recall.
          if hasattr(
              self.task_config.evaluation,
              'report_per_class_precision_and_recall'
          ) and self.task_config.evaluation.report_per_class_precision_and_recall:
            for class_id in range(self.task_config.model.num_classes):
              metrics += [
                  tf.keras.metrics.Precision(
                      thresholds=th,
                      class_id=class_id,
                      name=f'precision_at_threshold_{th}/{class_id}',
                      top_k=1) for th in thresholds
              ]
              metrics += [
                  tf.keras.metrics.Recall(
                      thresholds=th,
                      class_id=class_id,
                      name=f'recall_at_threshold_{th}/{class_id}',
                      top_k=1) for th in thresholds
              ]
              # pylint:enable=g-complex-comprehension
      else:
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=k, name='top_{}_accuracy'.format(k))]
    else:
      metrics = []
      # These metrics destablize the training if included in training. The jobs
      # fail due to OOM.
      # TODO(arashwan): Investigate adding following metric to train.
      if not training:
        metrics = [
            tf.keras.metrics.AUC(
                name='globalPR-AUC',
                curve='PR',
                multi_label=False,
                from_logits=True),
            tf.keras.metrics.AUC(
                name='meanPR-AUC',
                curve='PR',
                multi_label=True,
                num_labels=self.task_config.model.num_classes,
                from_logits=True),
        ]
    return metrics

  def train_step(self,
                 inputs: Tuple[Any, Any],
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 metrics: Optional[List[Any]] = None):
    """Does forward and backward.

    Args:
      inputs: A tuple of input tensors of (features, labels).
      model: A tf.keras.Model instance.
      optimizer: The optimizer for this training step.
      metrics: A nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features, labels = inputs
    is_multilabel = self.task_config.train_data.is_multilabel
    if self.task_config.losses.one_hot and not is_multilabel:
      labels = tf.one_hot(labels, self.task_config.model.num_classes)

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)

      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      loss = self.build_losses(
          model_outputs=outputs,
          labels=labels,
          aux_losses=model.losses)
      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = loss / num_replicas

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(
          optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient before apply_gradients when LossScaleOptimizer is
    # used.
    if isinstance(
        optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {self.loss: loss}

    # Convert logits to softmax for metric computation if needed.
    if hasattr(self.task_config.model,
               'output_softmax') and self.task_config.model.output_softmax:
      outputs = tf.nn.softmax(outputs, axis=-1)
    if metrics:
      self.process_metrics(metrics, labels, outputs)
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in model.metrics})
    return logs

  def validation_step(self,
                      inputs: Tuple[Any, Any],
                      model: tf.keras.Model,
                      metrics: Optional[List[Any]] = None):
    """Runs validatation step.

    Args:
      inputs: A tuple of input tensors of (features, labels).
      model: A tf.keras.Model instance.
      metrics: A nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features, labels = inputs
    one_hot = self.task_config.losses.one_hot
    soft_labels = self.task_config.losses.soft_labels
    is_multilabel = self.task_config.train_data.is_multilabel
    if (one_hot or soft_labels) and not is_multilabel:
      labels = tf.one_hot(labels, self.task_config.model.num_classes)

    outputs = self.inference_step(features, model)
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)
    loss = self.build_losses(
        model_outputs=outputs,
        labels=labels,
        aux_losses=model.losses)

    logs = {self.loss: loss}
    # Convert logits to softmax for metric computation if needed.
    if hasattr(self.task_config.model,
               'output_softmax') and self.task_config.model.output_softmax:
      outputs = tf.nn.softmax(outputs, axis=-1)
    if metrics:
      self.process_metrics(metrics, labels, outputs)
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in model.metrics})
    return logs

  def inference_step(self, inputs: tf.Tensor, model: tf.keras.Model):
    """Performs the forward step."""
    return model(inputs, training=False)
