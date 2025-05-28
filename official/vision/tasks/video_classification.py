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

"""Video classification task definition."""
from typing import Any, Optional, List, Tuple

from absl import logging
import tensorflow as tf, tf_keras
from official.core import base_task
from official.core import task_factory
from official.modeling import tf_utils
from official.vision.configs import video_classification as exp_cfg
from official.vision.dataloaders import input_reader_factory
from official.vision.dataloaders import video_input
from official.vision.modeling import factory_3d
from official.vision.ops import augment


@task_factory.register_task_cls(exp_cfg.VideoClassificationTask)
class VideoClassificationTask(base_task.Task):
  """A task for video classification."""

  def _get_num_classes(self):
    """Gets the number of classes."""
    return self.task_config.train_data.num_classes

  def _get_feature_shape(self):
    """Get the common feature shape for train and eval."""
    return [
        d1 if d1 == d2 else None
        for d1, d2 in zip(self.task_config.train_data.feature_shape,
                          self.task_config.validation_data.feature_shape)
    ]

  def _get_num_test_views(self):
    """Gets number of views for test."""
    num_test_clips = self.task_config.validation_data.num_test_clips
    num_test_crops = self.task_config.validation_data.num_test_crops
    num_test_views = num_test_clips * num_test_crops
    return num_test_views

  def _is_multilabel(self):
    """If the label is multi-labels."""
    return self.task_config.train_data.is_multilabel

  def build_model(self):
    """Builds video classification model."""
    common_input_shape = self._get_feature_shape()
    input_specs = tf_keras.layers.InputSpec(shape=[None] + common_input_shape)
    logging.info('Build model input %r', common_input_shape)

    l2_weight_decay = float(self.task_config.losses.l2_weight_decay)
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf_keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = factory_3d.build_model(
        self.task_config.model.model_type,
        input_specs=input_specs,
        model_config=self.task_config.model,
        num_classes=self._get_num_classes(),
        l2_regularizer=l2_regularizer)

    if self.task_config.freeze_backbone:
      logging.info('Freezing model backbone.')
      model.backbone.trainable = False
    return model

  def initialize(self, model: tf_keras.Model):
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

  def _get_dataset_fn(self, params):
    if params.file_type == 'tfrecord':
      return tf.data.TFRecordDataset
    else:
      raise ValueError('Unknown input file type {!r}'.format(params.file_type))

  def _get_decoder_fn(self, params):
    if params.tfds_name:
      decoder = video_input.VideoTfdsDecoder(
          image_key=params.image_field_key, label_key=params.label_field_key)
    else:
      decoder = video_input.Decoder(
          image_key=params.image_field_key, label_key=params.label_field_key)
    if self.task_config.train_data.output_audio:
      assert self.task_config.train_data.audio_feature, 'audio feature is empty'
      decoder.add_feature(self.task_config.train_data.audio_feature,
                          tf.io.VarLenFeature(dtype=tf.float32))
    return decoder.decode

  def build_inputs(self,
                   params: exp_cfg.DataConfig,
                   input_context: Optional[tf.distribute.InputContext] = None):
    """Builds classification input."""

    parser = video_input.Parser(
        input_params=params,
        image_key=params.image_field_key,
        label_key=params.label_field_key)
    postprocess_fn = video_input.PostBatchProcessor(params)
    if params.mixup_and_cutmix is not None:
      def mixup_and_cutmix(features, labels):
        augmenter = augment.MixupAndCutmix(
            mixup_alpha=params.mixup_and_cutmix.mixup_alpha,
            cutmix_alpha=params.mixup_and_cutmix.cutmix_alpha,
            prob=params.mixup_and_cutmix.prob,
            label_smoothing=params.mixup_and_cutmix.label_smoothing,
            num_classes=self._get_num_classes())
        features['image'], labels = augmenter(features['image'], labels)
        return features, labels
      postprocess_fn = mixup_and_cutmix

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=self._get_dataset_fn(params),
        decoder_fn=self._get_decoder_fn(params),
        parser_fn=parser.parse_fn(params.is_training),
        postprocess_fn=postprocess_fn)

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self,
                   labels: Any,
                   model_outputs: Any,
                   aux_losses: Optional[Any] = None):
    """Sparse categorical cross entropy loss.

    Args:
      labels: labels.
      model_outputs: Output logits of the classifier.
      aux_losses: auxiliarly loss tensors, i.e. `losses` in keras.Model.

    Returns:
      The total loss tensor.
    """
    all_losses = {}
    losses_config = self.task_config.losses
    total_loss = None
    if self._is_multilabel():
      entropy = -tf.reduce_mean(
          tf.reduce_sum(model_outputs * tf.math.log(model_outputs + 1e-8), -1))
      total_loss = tf_keras.losses.binary_crossentropy(
          labels, model_outputs, from_logits=False)
      all_losses.update({
          'class_loss': total_loss,
          'entropy': entropy,
      })
    else:
      if losses_config.one_hot:
        total_loss = tf_keras.losses.categorical_crossentropy(
            labels,
            model_outputs,
            from_logits=False,
            label_smoothing=losses_config.label_smoothing)
      else:
        total_loss = tf_keras.losses.sparse_categorical_crossentropy(
            labels, model_outputs, from_logits=False)

      total_loss = tf_utils.safe_mean(total_loss)
      all_losses.update({
          'class_loss': total_loss,
      })
    if aux_losses:
      all_losses.update({
          'reg_loss': aux_losses,
      })
      total_loss += tf.add_n(aux_losses)
    all_losses[self.loss] = total_loss

    return all_losses

  def build_metrics(self, training: bool = True):
    """Gets streaming metrics for training/validation."""
    if self.task_config.losses.one_hot:
      metrics = [
          tf_keras.metrics.CategoricalAccuracy(name='accuracy'),
          tf_keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_accuracy'),
          tf_keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
      ]
      if self._is_multilabel():
        metrics.append(
            tf_keras.metrics.AUC(
                curve='ROC', multi_label=self._is_multilabel(), name='ROC-AUC'))
        metrics.append(
            tf_keras.metrics.RecallAtPrecision(
                0.95, name='RecallAtPrecision95'))
        metrics.append(
            tf_keras.metrics.AUC(
                curve='PR', multi_label=self._is_multilabel(), name='PR-AUC'))
        if self.task_config.metrics.use_per_class_recall:
          for i in range(self._get_num_classes()):
            metrics.append(
                tf_keras.metrics.Recall(class_id=i, name=f'recall-{i}'))
    else:
      metrics = [
          tf_keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
          tf_keras.metrics.SparseTopKCategoricalAccuracy(
              k=1, name='top_1_accuracy'),
          tf_keras.metrics.SparseTopKCategoricalAccuracy(
              k=5, name='top_5_accuracy')
      ]
    return metrics

  def process_metrics(self, metrics: List[Any], labels: Any,
                      model_outputs: Any):
    """Process and update metrics.

    Called when using custom training loop API.

    Args:
      metrics: a nested structure of metrics objects. The return of function
        self.build_metrics.
      labels: a tensor or a nested structure of tensors.
      model_outputs: a tensor or a nested structure of tensors. For example,
        output of the keras model built by self.build_model.
    """
    for metric in metrics:
      metric.update_state(labels, model_outputs)

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
      features['image'] = strategy.experimental_split_to_logical_devices(
          features['image'], input_partition_dims)

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      if self._is_multilabel():
        outputs = tf.nest.map_structure(tf.math.sigmoid, outputs)
      else:
        outputs = tf.nest.map_structure(tf.math.softmax, outputs)
      all_losses = self.build_losses(
          model_outputs=outputs, labels=labels, aux_losses=model.losses)
      loss = all_losses[self.loss]
      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = loss / num_replicas

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(
          optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient before apply_gradients when LossScaleOptimizer is
    # used.
    if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = all_losses
    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in model.metrics})
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
      features['image'] = strategy.experimental_split_to_logical_devices(
          features['image'], input_partition_dims)

    outputs = self.inference_step(features, model)
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)
    logs = self.build_losses(model_outputs=outputs, labels=labels,
                             aux_losses=model.losses)

    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in model.metrics})
    return logs

  def inference_step(self, features: tf.Tensor, model: tf_keras.Model):
    """Performs the forward step."""
    outputs = model(features, training=False)
    if self._is_multilabel():
      outputs = tf.nest.map_structure(tf.math.sigmoid, outputs)
    else:
      outputs = tf.nest.map_structure(tf.math.softmax, outputs)
    num_test_views = self._get_num_test_views()
    if num_test_views > 1:
      # Averaging output probabilities across multiples views.
      outputs = tf.reshape(outputs, [-1, num_test_views, outputs.shape[-1]])
      outputs = tf.reduce_mean(outputs, axis=1)
    return outputs
