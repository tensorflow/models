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
"""Masked language task."""
import dataclasses
import tensorflow as tf

from official.core import base_task
from official.modeling.hyperparams import config_definitions as cfg
from official.nlp.configs import bert
from official.nlp.data import pretrain_dataloader
from official.nlp.modeling import losses as loss_lib


@dataclasses.dataclass
class MaskedLMConfig(cfg.TaskConfig):
  """The model config."""
  model: bert.BertPretrainerConfig = bert.BertPretrainerConfig(cls_heads=[
      bert.ClsHeadConfig(
          inner_dim=768, num_classes=2, dropout_rate=0.1, name='next_sentence')
  ])
  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()


@base_task.register_task_cls(MaskedLMConfig)
class MaskedLMTask(base_task.Task):
  """Mock task object for testing."""

  def build_model(self):
    return bert.instantiate_bertpretrainer_from_cfg(self.task_config.model)

  def build_losses(self,
                   labels,
                   model_outputs,
                   metrics,
                   aux_losses=None) -> tf.Tensor:
    metrics = dict([(metric.name, metric) for metric in metrics])
    lm_output = tf.nn.log_softmax(
        tf.cast(model_outputs['lm_output'], tf.float32), axis=-1)
    mlm_loss = loss_lib.weighted_sparse_categorical_crossentropy_loss(
        labels=labels['masked_lm_ids'],
        predictions=lm_output,
        weights=labels['masked_lm_weights'])
    metrics['lm_example_loss'].update_state(mlm_loss)
    if 'next_sentence_labels' in labels:
      sentence_labels = labels['next_sentence_labels']
      sentence_outputs = tf.cast(
          model_outputs['next_sentence'], dtype=tf.float32)
      sentence_loss = loss_lib.weighted_sparse_categorical_crossentropy_loss(
          labels=sentence_labels,
          predictions=tf.nn.log_softmax(sentence_outputs, axis=-1))
      metrics['next_sentence_loss'].update_state(sentence_loss)
      total_loss = mlm_loss + sentence_loss
    else:
      total_loss = mlm_loss

    if aux_losses:
      total_loss += tf.add_n(aux_losses)
    return total_loss

  def build_inputs(self, params, input_context=None):
    """Returns tf.data.Dataset for pretraining."""
    if params.input_path == 'dummy':
      def dummy_data(_):
        dummy_ids = tf.zeros((1, params.seq_length), dtype=tf.int32)
        dummy_lm = tf.zeros((1, params.max_predictions_per_seq), dtype=tf.int32)
        return dict(
            input_word_ids=dummy_ids,
            input_mask=dummy_ids,
            input_type_ids=dummy_ids,
            masked_lm_positions=dummy_lm,
            masked_lm_ids=dummy_lm,
            masked_lm_weights=tf.cast(dummy_lm, dtype=tf.float32),
            next_sentence_labels=tf.zeros((1, 1), dtype=tf.int32))

      dataset = tf.data.Dataset.range(1)
      dataset = dataset.repeat()
      dataset = dataset.map(
          dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    return pretrain_dataloader.BertPretrainDataLoader(params).load(
        input_context)

  def build_metrics(self, training=None):
    del training
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='masked_lm_accuracy'),
        tf.keras.metrics.Mean(name='lm_example_loss')
    ]
    # TODO(hongkuny): rethink how to manage metrics creation with heads.
    if self.task_config.train_data.use_next_sentence_label:
      metrics.append(
          tf.keras.metrics.SparseCategoricalAccuracy(
              name='next_sentence_accuracy'))
      metrics.append(tf.keras.metrics.Mean(name='next_sentence_loss'))
    return metrics

  def process_metrics(self, metrics, labels, model_outputs):
    metrics = dict([(metric.name, metric) for metric in metrics])
    if 'masked_lm_accuracy' in metrics:
      metrics['masked_lm_accuracy'].update_state(labels['masked_lm_ids'],
                                                 model_outputs['lm_output'],
                                                 labels['masked_lm_weights'])
    if 'next_sentence_accuracy' in metrics:
      metrics['next_sentence_accuracy'].update_state(
          labels['next_sentence_labels'], model_outputs['next_sentence'])

  def train_step(self, inputs, model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer, metrics):
    """Does forward and backward.

    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    with tf.GradientTape() as tape:
      outputs = model(inputs, training=True)
      # Computes per-replica loss.
      loss = self.build_losses(
          labels=inputs,
          model_outputs=outputs,
          metrics=metrics,
          aux_losses=model.losses)
      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      # TODO(b/154564893): enable loss scaling.
      # scaled_loss = loss / tf.distribute.get_strategy().num_replicas_in_sync
    tvars = model.trainable_variables
    grads = tape.gradient(loss, tvars)
    optimizer.apply_gradients(list(zip(grads, tvars)))
    self.process_metrics(metrics, inputs, outputs)
    return {self.loss: loss}

  def validation_step(self, inputs, model: tf.keras.Model, metrics):
    """Validatation step.

    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    outputs = self.inference_step(inputs, model)
    loss = self.build_losses(
        labels=inputs,
        model_outputs=outputs,
        metrics=metrics,
        aux_losses=model.losses)
    self.process_metrics(metrics, inputs, outputs)
    return {self.loss: loss}
