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

"""ELECTRA pretraining task (Joint Masked LM and Replaced Token Detection)."""

import dataclasses
import tensorflow as tf, tf_keras

from official.core import base_task
from official.core import config_definitions as cfg
from official.core import task_factory
from official.modeling import tf_utils
from official.nlp.configs import bert
from official.nlp.configs import electra
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader
from official.nlp.modeling import layers
from official.nlp.modeling import models


@dataclasses.dataclass
class ElectraPretrainConfig(cfg.TaskConfig):
  """The model config."""
  model: electra.ElectraPretrainerConfig = dataclasses.field(
      default_factory=lambda: electra.ElectraPretrainerConfig(  # pylint: disable=g-long-lambda
          cls_heads=[
              bert.ClsHeadConfig(
                  inner_dim=768,
                  num_classes=2,
                  dropout_rate=0.1,
                  name='next_sentence',
              )
          ]
      )
  )
  train_data: cfg.DataConfig = dataclasses.field(default_factory=cfg.DataConfig)
  validation_data: cfg.DataConfig = dataclasses.field(
      default_factory=cfg.DataConfig
  )


def _build_pretrainer(
    config: electra.ElectraPretrainerConfig) -> models.ElectraPretrainer:
  """Instantiates ElectraPretrainer from the config."""
  generator_encoder_cfg = config.generator_encoder
  discriminator_encoder_cfg = config.discriminator_encoder
  # Copy discriminator's embeddings to generator for easier model serialization.
  discriminator_network = encoders.build_encoder(discriminator_encoder_cfg)
  if config.tie_embeddings:
    embedding_layer = discriminator_network.get_embedding_layer()
    generator_network = encoders.build_encoder(
        generator_encoder_cfg, embedding_layer=embedding_layer)
  else:
    generator_network = encoders.build_encoder(generator_encoder_cfg)

  generator_encoder_cfg = generator_encoder_cfg.get()
  return models.ElectraPretrainer(
      generator_network=generator_network,
      discriminator_network=discriminator_network,
      vocab_size=generator_encoder_cfg.vocab_size,
      num_classes=config.num_classes,
      sequence_length=config.sequence_length,
      num_token_predictions=config.num_masked_tokens,
      mlm_activation=tf_utils.get_activation(
          generator_encoder_cfg.hidden_activation),
      mlm_initializer=tf_keras.initializers.TruncatedNormal(
          stddev=generator_encoder_cfg.initializer_range),
      classification_heads=[
          layers.ClassificationHead(**cfg.as_dict()) for cfg in config.cls_heads
      ],
      disallow_correct=config.disallow_correct)


@task_factory.register_task_cls(ElectraPretrainConfig)
class ElectraPretrainTask(base_task.Task):
  """ELECTRA Pretrain Task (Masked LM + Replaced Token Detection)."""

  def build_model(self):
    return _build_pretrainer(self.task_config.model)

  def build_losses(self,
                   labels,
                   model_outputs,
                   metrics,
                   aux_losses=None) -> tf.Tensor:
    metrics = dict([(metric.name, metric) for metric in metrics])

    # generator lm and (optional) nsp loss.
    lm_prediction_losses = tf_keras.losses.sparse_categorical_crossentropy(
        labels['masked_lm_ids'],
        tf.cast(model_outputs['lm_outputs'], tf.float32),
        from_logits=True)
    lm_label_weights = labels['masked_lm_weights']
    lm_numerator_loss = tf.reduce_sum(lm_prediction_losses * lm_label_weights)
    lm_denominator_loss = tf.reduce_sum(lm_label_weights)
    mlm_loss = tf.math.divide_no_nan(lm_numerator_loss, lm_denominator_loss)
    metrics['lm_example_loss'].update_state(mlm_loss)
    if 'next_sentence_labels' in labels:
      sentence_labels = labels['next_sentence_labels']
      sentence_outputs = tf.cast(
          model_outputs['sentence_outputs'], dtype=tf.float32)
      sentence_loss = tf_keras.losses.sparse_categorical_crossentropy(
          sentence_labels, sentence_outputs, from_logits=True)
      metrics['next_sentence_loss'].update_state(sentence_loss)
      total_loss = mlm_loss + sentence_loss
    else:
      total_loss = mlm_loss

    # discriminator replaced token detection (rtd) loss.
    rtd_logits = model_outputs['disc_logits']
    rtd_labels = tf.cast(model_outputs['disc_label'], tf.float32)
    input_mask = tf.cast(labels['input_mask'], tf.float32)
    rtd_ind_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=rtd_logits, labels=rtd_labels)
    rtd_numerator = tf.reduce_sum(input_mask * rtd_ind_loss)
    rtd_denominator = tf.reduce_sum(input_mask)
    rtd_loss = tf.math.divide_no_nan(rtd_numerator, rtd_denominator)
    metrics['discriminator_loss'].update_state(rtd_loss)
    total_loss = total_loss + \
        self.task_config.model.discriminator_loss_weight * rtd_loss

    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    metrics['total_loss'].update_state(total_loss)
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
        tf_keras.metrics.SparseCategoricalAccuracy(name='masked_lm_accuracy'),
        tf_keras.metrics.Mean(name='lm_example_loss'),
        tf_keras.metrics.SparseCategoricalAccuracy(
            name='discriminator_accuracy'),
    ]
    if self.task_config.train_data.use_next_sentence_label:
      metrics.append(
          tf_keras.metrics.SparseCategoricalAccuracy(
              name='next_sentence_accuracy'))
      metrics.append(tf_keras.metrics.Mean(name='next_sentence_loss'))

    metrics.append(tf_keras.metrics.Mean(name='discriminator_loss'))
    metrics.append(tf_keras.metrics.Mean(name='total_loss'))

    return metrics

  def process_metrics(self, metrics, labels, model_outputs):
    metrics = dict([(metric.name, metric) for metric in metrics])
    if 'masked_lm_accuracy' in metrics:
      metrics['masked_lm_accuracy'].update_state(labels['masked_lm_ids'],
                                                 model_outputs['lm_outputs'],
                                                 labels['masked_lm_weights'])
    if 'next_sentence_accuracy' in metrics:
      metrics['next_sentence_accuracy'].update_state(
          labels['next_sentence_labels'], model_outputs['sentence_outputs'])
    if 'discriminator_accuracy' in metrics:
      disc_logits_expanded = tf.expand_dims(model_outputs['disc_logits'], -1)
      discrim_full_logits = tf.concat(
          [-1.0 * disc_logits_expanded, disc_logits_expanded], -1)
      metrics['discriminator_accuracy'].update_state(
          model_outputs['disc_label'], discrim_full_logits,
          labels['input_mask'])

  def train_step(self, inputs, model: tf_keras.Model,
                 optimizer: tf_keras.optimizers.Optimizer, metrics):
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
      scaled_loss = loss / tf.distribute.get_strategy().num_replicas_in_sync
    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    optimizer.apply_gradients(list(zip(grads, tvars)))
    self.process_metrics(metrics, inputs, outputs)
    return {self.loss: loss}

  def validation_step(self, inputs, model: tf_keras.Model, metrics):
    """Validatation step.

    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    outputs = model(inputs, training=False)
    loss = self.build_losses(
        labels=inputs,
        model_outputs=outputs,
        metrics=metrics,
        aux_losses=model.losses)
    self.process_metrics(metrics, inputs, outputs)
    return {self.loss: loss}
