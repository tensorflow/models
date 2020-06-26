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
from absl import logging
import dataclasses
import numpy as np
from scipy import stats
from sklearn import metrics as sklearn_metrics
import tensorflow as tf
import tensorflow_hub as hub

from official.core import base_task
from official.modeling.hyperparams import config_definitions as cfg
from official.nlp.configs import bert
from official.nlp.data import sentence_prediction_dataloader
from official.nlp.tasks import utils


@dataclasses.dataclass
class SentencePredictionConfig(cfg.TaskConfig):
  """The model config."""
  # At most one of `init_checkpoint` and `hub_module_url` can
  # be specified.
  init_checkpoint: str = ''
  hub_module_url: str = ''
  metric_type: str = 'accuracy'
  model: bert.BertPretrainerConfig = bert.BertPretrainerConfig(
      num_masked_tokens=0,  # No masked language modeling head.
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
    if params.hub_module_url and params.init_checkpoint:
      raise ValueError('At most one of `hub_module_url` and '
                       '`pretrain_checkpoint_dir` can be specified.')
    if params.hub_module_url:
      self._hub_module = hub.load(params.hub_module_url)
    else:
      self._hub_module = None
    self.metric_type = params.metric_type

  def build_model(self):
    if self._hub_module:
      encoder_from_hub = utils.get_encoder_from_hub(self._hub_module)
      return bert.instantiate_bertpretrainer_from_cfg(
          self.task_config.model, encoder_network=encoder_from_hub)
    else:
      return bert.instantiate_bertpretrainer_from_cfg(self.task_config.model)

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels,
        tf.cast(model_outputs['sentence_prediction'], tf.float32),
        from_logits=True)

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
        y = tf.zeros((1, 1), dtype=tf.int32)
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
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='cls_accuracy')]
    return metrics

  def process_metrics(self, metrics, labels, model_outputs):
    for metric in metrics:
      metric.update_state(labels, model_outputs['sentence_prediction'])

  def process_compiled_metrics(self, compiled_metrics, labels, model_outputs):
    compiled_metrics.update_state(labels, model_outputs['sentence_prediction'])

  def validation_step(self, inputs, model: tf.keras.Model, metrics=None):
    if self.metric_type == 'accuracy':
      return super(SentencePredictionTask,
                   self).validation_step(inputs, model, metrics)
    features, labels = inputs
    outputs = self.inference_step(features, model)
    loss = self.build_losses(
        labels=labels, model_outputs=outputs, aux_losses=model.losses)
    if self.metric_type == 'matthews_corrcoef':
      return {
          self.loss:
              loss,
          'sentence_prediction':
              tf.expand_dims(
                  tf.math.argmax(outputs['sentence_prediction'], axis=1),
                  axis=0),
          'labels':
              labels,
      }
    if self.metric_type == 'pearson_spearman_corr':
      return {
          self.loss: loss,
          'sentence_prediction': outputs['sentence_prediction'],
          'labels': labels,
      }

  def aggregate_logs(self, state=None, step_outputs=None):
    if state is None:
      state = {'sentence_prediction': [], 'labels': []}
    state['sentence_prediction'].append(
        np.concatenate([v.numpy() for v in step_outputs['sentence_prediction']],
                       axis=0))
    state['labels'].append(
        np.concatenate([v.numpy() for v in step_outputs['labels']], axis=0))
    return state

  def reduce_aggregated_logs(self, aggregated_logs):
    if self.metric_type == 'matthews_corrcoef':
      preds = np.concatenate(aggregated_logs['sentence_prediction'], axis=0)
      labels = np.concatenate(aggregated_logs['labels'], axis=0)
      return {
          self.metric_type: sklearn_metrics.matthews_corrcoef(preds, labels)
      }
    if self.metric_type == 'pearson_spearman_corr':
      preds = np.concatenate(aggregated_logs['sentence_prediction'], axis=0)
      labels = np.concatenate(aggregated_logs['labels'], axis=0)
      pearson_corr = stats.pearsonr(preds, labels)[0]
      spearman_corr = stats.spearmanr(preds, labels)[0]
      corr_metric = (pearson_corr + spearman_corr) / 2
      return {self.metric_type: corr_metric}

  def initialize(self, model):
    """Load a pretrained checkpoint (if exists) and then train from iter 0."""
    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
    if not ckpt_dir_or_file:
      return

    pretrain2finetune_mapping = {
        'encoder':
            model.checkpoint_items['encoder'],
        'next_sentence.pooler_dense':
            model.checkpoint_items['sentence_prediction.pooler_dense'],
    }
    ckpt = tf.train.Checkpoint(**pretrain2finetune_mapping)
    status = ckpt.restore(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()
    logging.info('finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)
