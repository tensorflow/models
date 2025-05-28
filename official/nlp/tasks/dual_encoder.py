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

"""Dual encoder (retrieval) task."""
from typing import Mapping, Tuple
from absl import logging
import dataclasses
import tensorflow as tf, tf_keras

from official.core import base_task
from official.core import config_definitions as cfg
from official.core import task_factory
from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.nlp.configs import encoders
from official.nlp.data import data_loader_factory
from official.nlp.modeling import models
from official.nlp.tasks import utils


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """A dual encoder (retrieval) configuration."""
  # Normalize input embeddings if set to True.
  normalize: bool = True

  # Maximum input sequence length.
  max_sequence_length: int = 64

  # Parameters for training a dual encoder model with additive margin, see
  # https://www.ijcai.org/Proceedings/2019/0746.pdf for more details.
  logit_scale: float = 1
  logit_margin: float = 0
  bidirectional: bool = False

  # Defining k for calculating metrics recall@k.
  eval_top_k: Tuple[int, ...] = (1, 3, 10)

  encoder: encoders.EncoderConfig = dataclasses.field(
      default_factory=encoders.EncoderConfig
  )


@dataclasses.dataclass
class DualEncoderConfig(cfg.TaskConfig):
  """The model config."""
  # At most one of `init_checkpoint` and `hub_module_url` can
  # be specified.
  init_checkpoint: str = ''
  hub_module_url: str = ''
  # Defines the concrete model config at instantiation time.
  model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
  train_data: cfg.DataConfig = dataclasses.field(default_factory=cfg.DataConfig)
  validation_data: cfg.DataConfig = dataclasses.field(
      default_factory=cfg.DataConfig
  )


@task_factory.register_task_cls(DualEncoderConfig)
class DualEncoderTask(base_task.Task):
  """Task object for dual encoder."""

  def build_model(self):
    """Interface to build model. Refer to base_task.Task.build_model."""
    if self.task_config.hub_module_url and self.task_config.init_checkpoint:
      raise ValueError('At most one of `hub_module_url` and '
                       '`init_checkpoint` can be specified.')
    if self.task_config.hub_module_url:
      encoder_network = utils.get_encoder_from_hub(
          self.task_config.hub_module_url)
    else:
      encoder_network = encoders.build_encoder(self.task_config.model.encoder)

    # Currently, we only supports bert-style dual encoder.
    return models.DualEncoder(
        network=encoder_network,
        max_seq_length=self.task_config.model.max_sequence_length,
        normalize=self.task_config.model.normalize,
        logit_scale=self.task_config.model.logit_scale,
        logit_margin=self.task_config.model.logit_margin,
        output='logits')

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    """Interface to compute losses. Refer to base_task.Task.build_losses."""
    del labels

    left_logits = model_outputs['left_logits']
    right_logits = model_outputs['right_logits']

    batch_size = tf_utils.get_shape_list(left_logits, name='batch_size')[0]

    ranking_labels = tf.range(batch_size)

    loss = tf_utils.safe_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=ranking_labels,
            logits=left_logits))

    if self.task_config.model.bidirectional:
      right_rank_loss = tf_utils.safe_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=ranking_labels,
              logits=right_logits))

      loss += right_rank_loss
    return tf.reduce_mean(loss)

  def build_inputs(self, params, input_context=None) -> tf.data.Dataset:
    """Returns tf.data.Dataset for sentence_prediction task."""
    if params.input_path != 'dummy':
      return data_loader_factory.get_data_loader(params).load(input_context)

    def dummy_data(_):
      dummy_ids = tf.zeros((10, params.seq_length), dtype=tf.int32)
      x = dict(
          left_word_ids=dummy_ids,
          left_mask=dummy_ids,
          left_type_ids=dummy_ids,
          right_word_ids=dummy_ids,
          right_mask=dummy_ids,
          right_type_ids=dummy_ids)
      return x

    dataset = tf.data.Dataset.range(1)
    dataset = dataset.repeat()
    dataset = dataset.map(
        dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  def build_metrics(self, training=None):
    del training
    metrics = [tf_keras.metrics.Mean(name='batch_size_per_core')]
    for k in self.task_config.model.eval_top_k:
      metrics.append(tf_keras.metrics.SparseTopKCategoricalAccuracy(
          k=k, name=f'left_recall_at_{k}'))
      if self.task_config.model.bidirectional:
        metrics.append(tf_keras.metrics.SparseTopKCategoricalAccuracy(
            k=k, name=f'right_recall_at_{k}'))
    return metrics

  def process_metrics(self, metrics, labels, model_outputs):
    del labels

    metrics = dict([(metric.name, metric) for metric in metrics])

    left_logits = model_outputs['left_logits']
    right_logits = model_outputs['right_logits']
    batch_size = tf_utils.get_shape_list(
        left_logits, name='sequence_output_tensor')[0]

    ranking_labels = tf.range(batch_size)

    for k in self.task_config.model.eval_top_k:
      metrics[f'left_recall_at_{k}'].update_state(ranking_labels, left_logits)
      if self.task_config.model.bidirectional:
        metrics[f'right_recall_at_{k}'].update_state(ranking_labels,
                                                     right_logits)
    metrics['batch_size_per_core'].update_state(batch_size)

  def validation_step(self,
                      inputs,
                      model: tf_keras.Model,
                      metrics=None) -> Mapping[str, tf.Tensor]:
    outputs = model(inputs)
    loss = self.build_losses(
        labels=None, model_outputs=outputs, aux_losses=model.losses)
    logs = {self.loss: loss}

    if metrics:
      self.process_metrics(metrics, None, outputs)
      logs.update({m.name: m.result() for m in metrics})
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, None, outputs)
      logs.update({m.name: m.result() for m in model.metrics})

    return logs

  def initialize(self, model):
    """Load a pretrained checkpoint (if exists) and then train from iter 0."""
    ckpt_dir_or_file = self.task_config.init_checkpoint
    logging.info('Trying to load pretrained checkpoint from %s',
                 ckpt_dir_or_file)
    if ckpt_dir_or_file and tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
    if not ckpt_dir_or_file:
      logging.info('No checkpoint file found from %s. Will not load.',
                   ckpt_dir_or_file)
      return

    pretrain2finetune_mapping = {
        'encoder': model.checkpoint_items['encoder'],
    }

    ckpt = tf.train.Checkpoint(**pretrain2finetune_mapping)
    status = ckpt.read(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()
    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)
