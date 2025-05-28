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

"""TEAMS pretraining task (Joint Masked LM, Replaced Token Detection and )."""

import dataclasses
import tensorflow as tf, tf_keras

from official.core import base_task
from official.core import config_definitions as cfg
from official.core import task_factory
from official.modeling import tf_utils
from official.nlp.data import pretrain_dataloader
from official.nlp.modeling import layers
from official.projects.teams import teams
from official.projects.teams import teams_pretrainer


@dataclasses.dataclass
class TeamsPretrainTaskConfig(cfg.TaskConfig):
  """The model config."""
  model: teams.TeamsPretrainerConfig = dataclasses.field(
      default_factory=teams.TeamsPretrainerConfig
  )
  train_data: cfg.DataConfig = dataclasses.field(default_factory=cfg.DataConfig)
  validation_data: cfg.DataConfig = dataclasses.field(
      default_factory=cfg.DataConfig
  )


def _get_generator_hidden_layers(discriminator_network, num_hidden_layers,
                                 num_shared_layers):
  if num_shared_layers <= 0:
    num_shared_layers = 0
    hidden_layers = []
  else:
    hidden_layers = discriminator_network.hidden_layers[:num_shared_layers]
  for _ in range(num_shared_layers, num_hidden_layers):
    hidden_layers.append(layers.Transformer)
  return hidden_layers


def _build_pretrainer(
    config: teams.TeamsPretrainerConfig) -> teams_pretrainer.TeamsPretrainer:
  """Instantiates TeamsPretrainer from the config."""
  generator_encoder_cfg = config.generator
  discriminator_encoder_cfg = config.discriminator
  discriminator_network = teams.get_encoder(discriminator_encoder_cfg)
  # Copy discriminator's embeddings to generator for easier model serialization.
  hidden_layers = _get_generator_hidden_layers(
      discriminator_network, generator_encoder_cfg.num_layers,
      config.num_shared_generator_hidden_layers)
  if config.tie_embeddings:
    generator_network = teams.get_encoder(
        generator_encoder_cfg,
        embedding_network=discriminator_network.embedding_network,
        hidden_layers=hidden_layers)
  else:
    generator_network = teams.get_encoder(
        generator_encoder_cfg, hidden_layers=hidden_layers)

  return teams_pretrainer.TeamsPretrainer(
      generator_network=generator_network,
      discriminator_mws_network=discriminator_network,
      num_discriminator_task_agnostic_layers=config
      .num_discriminator_task_agnostic_layers,
      vocab_size=generator_encoder_cfg.vocab_size,
      candidate_size=config.candidate_size,
      mlm_activation=tf_utils.get_activation(
          generator_encoder_cfg.hidden_activation),
      mlm_initializer=tf_keras.initializers.TruncatedNormal(
          stddev=generator_encoder_cfg.initializer_range))


@task_factory.register_task_cls(TeamsPretrainTaskConfig)
class TeamsPretrainTask(base_task.Task):
  """TEAMS Pretrain Task (Masked LM + RTD + MWS)."""

  def build_model(self):
    return _build_pretrainer(self.task_config.model)

  def build_losses(self,
                   labels,
                   model_outputs,
                   metrics,
                   aux_losses=None) -> tf.Tensor:
    with tf.name_scope('TeamsPretrainTask/losses'):
      metrics = dict([(metric.name, metric) for metric in metrics])

      # Generator MLM loss.
      lm_prediction_losses = tf_keras.losses.sparse_categorical_crossentropy(
          labels['masked_lm_ids'],
          tf.cast(model_outputs['lm_outputs'], tf.float32),
          from_logits=True)
      lm_label_weights = labels['masked_lm_weights']
      lm_numerator_loss = tf.reduce_sum(lm_prediction_losses * lm_label_weights)
      lm_denominator_loss = tf.reduce_sum(lm_label_weights)
      mlm_loss = tf.math.divide_no_nan(lm_numerator_loss, lm_denominator_loss)
      metrics['masked_lm_loss'].update_state(mlm_loss)
      weight = self.task_config.model.generator_loss_weight
      total_loss = weight * mlm_loss

      # Discriminator RTD loss.
      rtd_logits = model_outputs['disc_rtd_logits']
      rtd_labels = tf.cast(model_outputs['disc_rtd_label'], tf.float32)
      input_mask = tf.cast(labels['input_mask'], tf.float32)
      rtd_ind_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          logits=rtd_logits, labels=rtd_labels)
      rtd_numerator = tf.reduce_sum(input_mask * rtd_ind_loss)
      rtd_denominator = tf.reduce_sum(input_mask)
      rtd_loss = tf.math.divide_no_nan(rtd_numerator, rtd_denominator)
      metrics['replaced_token_detection_loss'].update_state(rtd_loss)
      weight = self.task_config.model.discriminator_rtd_loss_weight
      total_loss = total_loss + weight * rtd_loss

      # Discriminator MWS loss.
      mws_logits = model_outputs['disc_mws_logits']
      mws_labels = model_outputs['disc_mws_label']
      mws_loss = tf_keras.losses.sparse_categorical_crossentropy(
          mws_labels, mws_logits, from_logits=True)
      mws_numerator_loss = tf.reduce_sum(mws_loss * lm_label_weights)
      mws_denominator_loss = tf.reduce_sum(lm_label_weights)
      mws_loss = tf.math.divide_no_nan(mws_numerator_loss, mws_denominator_loss)
      metrics['multiword_selection_loss'].update_state(mws_loss)
      weight = self.task_config.model.discriminator_mws_loss_weight
      total_loss = total_loss + weight * mws_loss

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
            masked_lm_weights=tf.cast(dummy_lm, dtype=tf.float32))

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
        tf_keras.metrics.Mean(name='masked_lm_loss'),
        tf_keras.metrics.SparseCategoricalAccuracy(
            name='replaced_token_detection_accuracy'),
        tf_keras.metrics.Mean(name='replaced_token_detection_loss'),
        tf_keras.metrics.SparseCategoricalAccuracy(
            name='multiword_selection_accuracy'),
        tf_keras.metrics.Mean(name='multiword_selection_loss'),
        tf_keras.metrics.Mean(name='total_loss'),
    ]
    return metrics

  def process_metrics(self, metrics, labels, model_outputs):
    with tf.name_scope('TeamsPretrainTask/process_metrics'):
      metrics = dict([(metric.name, metric) for metric in metrics])
      if 'masked_lm_accuracy' in metrics:
        metrics['masked_lm_accuracy'].update_state(labels['masked_lm_ids'],
                                                   model_outputs['lm_outputs'],
                                                   labels['masked_lm_weights'])

      if 'replaced_token_detection_accuracy' in metrics:
        rtd_logits_expanded = tf.expand_dims(model_outputs['disc_rtd_logits'],
                                             -1)
        rtd_full_logits = tf.concat(
            [-1.0 * rtd_logits_expanded, rtd_logits_expanded], -1)
        metrics['replaced_token_detection_accuracy'].update_state(
            model_outputs['disc_rtd_label'], rtd_full_logits,
            labels['input_mask'])

      if 'multiword_selection_accuracy' in metrics:
        metrics['multiword_selection_accuracy'].update_state(
            model_outputs['disc_mws_label'], model_outputs['disc_mws_logits'],
            labels['masked_lm_weights'])

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
