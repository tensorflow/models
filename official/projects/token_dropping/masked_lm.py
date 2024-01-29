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

"""Masked language task."""

import dataclasses
from typing import Tuple
import tensorflow as tf

from official.core import task_factory
from official.nlp.tasks import masked_lm


@dataclasses.dataclass
class TokenDropMaskedLMConfig(masked_lm.MaskedLMConfig):
  """The model config."""
  pass


@task_factory.register_task_cls(TokenDropMaskedLMConfig)
class TokenDropMaskedLMTask(masked_lm.MaskedLMTask):
  """Task object for Mask language modeling."""

  def build_losses(self,
                   labels,
                   model_outputs,
                   metrics,
                   aux_losses=None) -> Tuple[tf.Tensor, tf.Tensor]:
    """Return the final loss, and the masked-lm loss."""
    with tf.name_scope('MaskedLMTask/losses'):
      metrics = dict([(metric.name, metric) for metric in metrics])
      lm_prediction_losses = tf.keras.losses.sparse_categorical_crossentropy(
          labels['masked_lm_ids'],
          tf.cast(model_outputs['mlm_logits'], tf.float32),
          from_logits=True)
      lm_label_weights = labels['masked_lm_weights']
      lm_numerator_loss = tf.reduce_sum(lm_prediction_losses *
                                        lm_label_weights)
      lm_denominator_loss = tf.reduce_sum(lm_label_weights)
      mlm_loss = tf.math.divide_no_nan(lm_numerator_loss, lm_denominator_loss)
      metrics['lm_example_loss'].update_state(mlm_loss)
      if 'next_sentence_labels' in labels:
        sentence_labels = labels['next_sentence_labels']
        sentence_outputs = tf.cast(
            model_outputs['next_sentence'], dtype=tf.float32)
        sentence_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                sentence_labels, sentence_outputs, from_logits=True))
        metrics['next_sentence_loss'].update_state(sentence_loss)
        total_loss = mlm_loss + sentence_loss
      else:
        total_loss = mlm_loss

      if aux_losses:
        total_loss += tf.add_n(aux_losses)
      return total_loss, lm_prediction_losses

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
      loss, lm_prediction_losses = self.build_losses(
          labels=inputs,
          model_outputs=outputs,
          metrics=metrics,
          aux_losses=model.losses)
      model.encoder_network.record_mlm_loss(
          mlm_ids=inputs['masked_lm_ids'],
          mlm_losses=lm_prediction_losses)
      if self.task_config.scale_loss:
        # Scales loss as the default gradients allreduce performs sum inside the
        # optimizer.
        scaled_loss = loss / tf.distribute.get_strategy().num_replicas_in_sync
    tvars = model.trainable_variables
    if self.task_config.scale_loss:
      grads = tape.gradient(scaled_loss, tvars)
    else:
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
    loss, _ = self.build_losses(
        labels=inputs,
        model_outputs=outputs,
        metrics=metrics,
        aux_losses=model.losses)
    self.process_metrics(metrics, inputs, outputs)
    return {self.loss: loss}
