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
"""Sentence prediction (classification) task."""
import logging
import dataclasses
import tensorflow as tf
import tensorflow_hub as hub

from official.core import base_task
from official.modeling.hyperparams import config_definitions as cfg
from official.nlp.configs import bert
from official.nlp.data import sentence_prediction_dataloader
from official.nlp.modeling import losses as loss_lib


@dataclasses.dataclass
class SentencePredictionConfig(cfg.TaskConfig):
  """The model config."""
  # At most one of `pretrain_checkpoint_dir` and `hub_module_url` can
  # be specified.
  pretrain_checkpoint_dir: str = ''
  hub_module_url: str = ''
  network: bert.BertPretrainerConfig = bert.BertPretrainerConfig(
      num_masked_tokens=0,
      cls_heads=[
          bert.ClsHeadConfig(
              inner_dim=768,
              num_classes=3,
              dropout_rate=0.1,
              name='sentence_prediction')
      ])
  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()


@base_task.register_task_cls(SentencePredictionConfig)
class SentencePredictionTask(base_task.Task):
  """Task object for sentence_prediction."""

  def __init__(self, params=cfg.TaskConfig):
    super(SentencePredictionTask, self).__init__(params)
    if params.hub_module_url and params.pretrain_checkpoint_dir:
      raise ValueError('At most one of `hub_module_url` and '
                       '`pretrain_checkpoint_dir` can be specified.')
    if params.hub_module_url:
      self._hub_module = hub.load(params.hub_module_url)
    else:
      self._hub_module = None

  def build_model(self):
    if self._hub_module:
      input_word_ids = tf.keras.layers.Input(
          shape=(None,), dtype=tf.int32, name='input_word_ids')
      input_mask = tf.keras.layers.Input(
          shape=(None,), dtype=tf.int32, name='input_mask')
      input_type_ids = tf.keras.layers.Input(
          shape=(None,), dtype=tf.int32, name='input_type_ids')
      bert_model = hub.KerasLayer(self._hub_module, trainable=True)
      pooled_output, sequence_output = bert_model(
          [input_word_ids, input_mask, input_type_ids])
      encoder_from_hub = tf.keras.Model(
          inputs=[input_word_ids, input_mask, input_type_ids],
          outputs=[sequence_output, pooled_output])
      return bert.instantiate_from_cfg(
          self.task_config.network, encoder_network=encoder_from_hub)
    else:
      return bert.instantiate_from_cfg(self.task_config.network)

  def build_losses(self, features, model_outputs, aux_losses=None) -> tf.Tensor:
    labels = features
    loss = loss_lib.weighted_sparse_categorical_crossentropy_loss(
        labels=labels,
        predictions=tf.nn.log_softmax(model_outputs['sentence_prediction'],
                                      axis=-1))

    if aux_losses:
      loss += tf.add_n(aux_losses)
    return loss

  def build_inputs(self, params, input_context=None):
    """Returns tf.data.Dataset for sentence_prediction task."""
    if params.input_path == 'dummy':
      def dummy_data(_):
        dummy_ids = tf.zeros((1, params.seq_length), dtype=tf.int32)
        x = dict(
            input_word_ids=dummy_ids,
            input_mask=dummy_ids,
            input_type_ids=dummy_ids)
        y = tf.ones((1, 1), dtype=tf.int32)
        return (x, y)

      dataset = tf.data.Dataset.range(1)
      dataset = dataset.repeat()
      dataset = dataset.map(
          dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    return sentence_prediction_dataloader.SentencePredictionDataLoader(
        params).load(input_context)

  def build_metrics(self, training=None):
    del training
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='cls_accuracy')
    ]
    return metrics

  def process_metrics(self, metrics, labels, outputs):
    for metric in metrics:
      metric.update_state(labels, outputs['sentence_prediction'])

  def process_compiled_metrics(self, compiled_metrics, labels, outputs):
    compiled_metrics.update_state(labels, outputs['sentence_prediction'])

  def initialize(self, model):
    """Load a pretrained checkpoint (if exists) and then train from iter 0."""
    pretrain_ckpt_dir = self.task_config.pretrain_checkpoint_dir
    if not pretrain_ckpt_dir:
      return

    pretrain2finetune_mapping = {
        'encoder':
            model.checkpoint_items['encoder'],
        'next_sentence.pooler_dense':
            model.checkpoint_items['sentence_prediction.pooler_dense'],
    }
    ckpt = tf.train.Checkpoint(**pretrain2finetune_mapping)
    latest_pretrain_ckpt = tf.train.latest_checkpoint(pretrain_ckpt_dir)
    if latest_pretrain_ckpt is None:
      raise FileNotFoundError(
          'Cannot find pretrain checkpoint under {}'.format(pretrain_ckpt_dir))
    status = ckpt.restore(latest_pretrain_ckpt)
    status.expect_partial().assert_existing_objects_matched()
    logging.info('finished loading pretrained checkpoint.')
