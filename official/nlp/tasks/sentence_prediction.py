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
from typing import List, Union

from absl import logging
import dataclasses
import numpy as np
import orbit
from scipy import stats
from sklearn import metrics as sklearn_metrics
import tensorflow as tf
import tensorflow_hub as hub

from official.core import base_task
from official.core import task_factory
from official.modeling.hyperparams import base_config
from official.modeling.hyperparams import config_definitions as cfg
from official.nlp.configs import encoders
from official.nlp.data import data_loader_factory
from official.nlp.modeling import models
from official.nlp.tasks import utils


METRIC_TYPES = frozenset(
    ['accuracy', 'matthews_corrcoef', 'pearson_spearman_corr'])


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """A classifier/regressor configuration."""
  num_classes: int = 0
  use_encoder_pooler: bool = False
  encoder: encoders.TransformerEncoderConfig = (
      encoders.TransformerEncoderConfig())


@dataclasses.dataclass
class SentencePredictionConfig(cfg.TaskConfig):
  """The model config."""
  # At most one of `init_checkpoint` and `hub_module_url` can
  # be specified.
  init_checkpoint: str = ''
  init_cls_pooler: bool = False
  hub_module_url: str = ''
  metric_type: str = 'accuracy'
  # Defines the concrete model config at instantiation time.
  model: ModelConfig = ModelConfig()
  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()


@task_factory.register_task_cls(SentencePredictionConfig)
class SentencePredictionTask(base_task.Task):
  """Task object for sentence_prediction."""

  def __init__(self, params=cfg.TaskConfig, logging_dir=None):
    super(SentencePredictionTask, self).__init__(params, logging_dir)
    if params.hub_module_url and params.init_checkpoint:
      raise ValueError('At most one of `hub_module_url` and '
                       '`init_checkpoint` can be specified.')
    if params.hub_module_url:
      self._hub_module = hub.load(params.hub_module_url)
    else:
      self._hub_module = None

    if params.metric_type not in METRIC_TYPES:
      raise ValueError('Invalid metric_type: {}'.format(params.metric_type))
    self.metric_type = params.metric_type

  def build_model(self):
    if self._hub_module:
      encoder_network = utils.get_encoder_from_hub(self._hub_module)
    else:
      encoder_network = encoders.instantiate_encoder_from_cfg(
          self.task_config.model.encoder)

    # Currently, we only support bert-style sentence prediction finetuning.
    return models.BertClassifier(
        network=encoder_network,
        num_classes=self.task_config.model.num_classes,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=self.task_config.model.encoder.initializer_range),
        use_encoder_pooler=self.task_config.model.use_encoder_pooler)

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    if self.task_config.model.num_classes == 1:
      loss = tf.keras.losses.mean_squared_error(labels, model_outputs)
    else:
      loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, tf.cast(model_outputs, tf.float32), from_logits=True)

    if aux_losses:
      loss += tf.add_n(aux_losses)
    return tf.reduce_mean(loss)

  def build_inputs(self, params, input_context=None):
    """Returns tf.data.Dataset for sentence_prediction task."""
    if params.input_path == 'dummy':

      def dummy_data(_):
        dummy_ids = tf.zeros((1, params.seq_length), dtype=tf.int32)
        x = dict(
            input_word_ids=dummy_ids,
            input_mask=dummy_ids,
            input_type_ids=dummy_ids)

        if self.task_config.model.num_classes == 1:
          y = tf.zeros((1,), dtype=tf.float32)
        else:
          y = tf.zeros((1, 1), dtype=tf.int32)
        return x, y

      dataset = tf.data.Dataset.range(1)
      dataset = dataset.repeat()
      dataset = dataset.map(
          dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    return data_loader_factory.get_data_loader(params).load(input_context)

  def build_metrics(self, training=None):
    del training
    if self.task_config.model.num_classes == 1:
      metrics = [tf.keras.metrics.MeanSquaredError()]
    else:
      metrics = [
          tf.keras.metrics.SparseCategoricalAccuracy(name='cls_accuracy')]
    return metrics

  def process_metrics(self, metrics, labels, model_outputs):
    for metric in metrics:
      metric.update_state(labels, model_outputs)

  def process_compiled_metrics(self, compiled_metrics, labels, model_outputs):
    compiled_metrics.update_state(labels, model_outputs)

  def validation_step(self, inputs, model: tf.keras.Model, metrics=None):
    if self.metric_type == 'accuracy':
      return super(SentencePredictionTask,
                   self).validation_step(inputs, model, metrics)
    features, labels = inputs
    outputs = self.inference_step(features, model)
    loss = self.build_losses(
        labels=labels, model_outputs=outputs, aux_losses=model.losses)
    logs = {self.loss: loss}
    if self.metric_type == 'matthews_corrcoef':
      logs.update({
          'sentence_prediction':
              tf.expand_dims(tf.math.argmax(outputs, axis=1), axis=0),
          'labels':
              labels,
      })
    if self.metric_type == 'pearson_spearman_corr':
      logs.update({
          'sentence_prediction': outputs,
          'labels': labels,
      })
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    if self.metric_type == 'accuracy':
      return None
    if state is None:
      state = {'sentence_prediction': [], 'labels': []}
    # TODO(b/160712818): Add support for concatenating partial batches.
    state['sentence_prediction'].append(
        np.concatenate([v.numpy() for v in step_outputs['sentence_prediction']],
                       axis=0))
    state['labels'].append(
        np.concatenate([v.numpy() for v in step_outputs['labels']], axis=0))
    return state

  def reduce_aggregated_logs(self, aggregated_logs):
    if self.metric_type == 'accuracy':
      return None
    elif self.metric_type == 'matthews_corrcoef':
      preds = np.concatenate(aggregated_logs['sentence_prediction'], axis=0)
      preds = np.reshape(preds, -1)
      labels = np.concatenate(aggregated_logs['labels'], axis=0)
      labels = np.reshape(labels, -1)
      return {
          self.metric_type: sklearn_metrics.matthews_corrcoef(preds, labels)
      }
    elif self.metric_type == 'pearson_spearman_corr':
      preds = np.concatenate(aggregated_logs['sentence_prediction'], axis=0)
      preds = np.reshape(preds, -1)
      labels = np.concatenate(aggregated_logs['labels'], axis=0)
      labels = np.reshape(labels, -1)
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
        'encoder': model.checkpoint_items['encoder'],
    }
    # TODO(b/160251903): Investigate why no pooler dense improves finetuning
    # accuracies.
    if self.task_config.init_cls_pooler:
      pretrain2finetune_mapping[
          'next_sentence.pooler_dense'] = model.checkpoint_items[
              'sentence_prediction.pooler_dense']
    ckpt = tf.train.Checkpoint(**pretrain2finetune_mapping)
    status = ckpt.read(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()
    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)


def predict(task: SentencePredictionTask, params: cfg.DataConfig,
            model: tf.keras.Model) -> List[Union[int, float]]:
  """Predicts on the input data.

  Args:
    task: A `SentencePredictionTask` object.
    params: A `cfg.DataConfig` object.
    model: A keras.Model.

  Returns:
    A list of predictions with length of `num_examples`. For regression task,
      each element in the list is the predicted score; for classification task,
      each element is the predicted class id.
  """
  is_regression = task.task_config.model.num_classes == 1

  @tf.function
  def predict_step(iterator):
    """Predicts on distributed devices."""

    def _replicated_step(inputs):
      """Replicated prediction calculation."""
      x, _ = inputs
      outputs = task.inference_step(x, model)
      if is_regression:
        return outputs
      else:
        return tf.argmax(outputs, axis=-1)

    outputs = tf.distribute.get_strategy().run(
        _replicated_step, args=(next(iterator),))
    return tf.nest.map_structure(
        tf.distribute.get_strategy().experimental_local_results, outputs)

  def reduce_fn(state, outputs):
    """Concatenates model's outputs."""
    for per_replica_batch_predictions in outputs:
      state.extend(per_replica_batch_predictions)
    return state

  loop_fn = orbit.utils.create_loop_fn(predict_step)
  dataset = orbit.utils.make_distributed_dataset(tf.distribute.get_strategy(),
                                                 task.build_inputs, params)
  # Set `num_steps` to -1 to exhaust the dataset.
  predictions = loop_fn(
      iter(dataset), num_steps=-1, state=[], reduce_fn=reduce_fn)
  return predictions
