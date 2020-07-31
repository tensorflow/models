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
"""Question answering task."""
import collections
import json
import os
from absl import logging
import dataclasses
import tensorflow as tf
import tensorflow_hub as hub

from official.core import base_task
from official.core import task_factory
from official.modeling.hyperparams import base_config
from official.modeling.hyperparams import config_definitions as cfg
from official.nlp.bert import squad_evaluate_v1_1
from official.nlp.bert import squad_evaluate_v2_0
from official.nlp.bert import tokenization
from official.nlp.configs import encoders
from official.nlp.data import data_loader_factory
from official.nlp.data import squad_lib as squad_lib_wp
from official.nlp.data import squad_lib_sp
from official.nlp.modeling import models
from official.nlp.tasks import utils


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """A base span labeler configuration."""
  encoder: encoders.TransformerEncoderConfig = (
      encoders.TransformerEncoderConfig())


@dataclasses.dataclass
class QuestionAnsweringConfig(cfg.TaskConfig):
  """The model config."""
  # At most one of `init_checkpoint` and `hub_module_url` can be specified.
  init_checkpoint: str = ''
  hub_module_url: str = ''
  n_best_size: int = 20
  max_answer_length: int = 30
  null_score_diff_threshold: float = 0.0
  model: ModelConfig = ModelConfig()
  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()


@task_factory.register_task_cls(QuestionAnsweringConfig)
class QuestionAnsweringTask(base_task.Task):
  """Task object for question answering."""

  def __init__(self, params=cfg.TaskConfig, logging_dir=None):
    super(QuestionAnsweringTask, self).__init__(params, logging_dir)
    if params.hub_module_url and params.init_checkpoint:
      raise ValueError('At most one of `hub_module_url` and '
                       '`init_checkpoint` can be specified.')
    if params.hub_module_url:
      self._hub_module = hub.load(params.hub_module_url)
    else:
      self._hub_module = None

    if params.validation_data.tokenization == 'WordPiece':
      self.squad_lib = squad_lib_wp
    elif params.validation_data.tokenization == 'SentencePiece':
      self.squad_lib = squad_lib_sp
    else:
      raise ValueError('Unsupported tokenization method: {}'.format(
          params.validation_data.tokenization))

    if params.validation_data.input_path:
      self._tf_record_input_path, self._eval_examples, self._eval_features = (
          self._preprocess_eval_data(params.validation_data))

  def build_model(self):
    if self._hub_module:
      encoder_network = utils.get_encoder_from_hub(self._hub_module)
    else:
      encoder_network = encoders.instantiate_encoder_from_cfg(
          self.task_config.model.encoder)
    # Currently, we only supports bert-style question answering finetuning.
    return models.BertSpanLabeler(
        network=encoder_network,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=self.task_config.model.encoder.initializer_range))

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    start_positions = labels['start_positions']
    end_positions = labels['end_positions']
    start_logits, end_logits = model_outputs

    start_loss = tf.keras.losses.sparse_categorical_crossentropy(
        start_positions,
        tf.cast(start_logits, dtype=tf.float32),
        from_logits=True)
    end_loss = tf.keras.losses.sparse_categorical_crossentropy(
        end_positions,
        tf.cast(end_logits, dtype=tf.float32),
        from_logits=True)

    loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2
    return loss

  def _preprocess_eval_data(self, params):
    eval_examples = self.squad_lib.read_squad_examples(
        input_file=params.input_path,
        is_training=False,
        version_2_with_negative=params.version_2_with_negative)

    temp_file_path = params.input_preprocessed_data_path or self.logging_dir
    if not temp_file_path:
      raise ValueError('You must specify a temporary directory, either in '
                       'params.input_preprocessed_data_path or logging_dir to '
                       'store intermediate evaluation TFRecord data.')
    eval_writer = self.squad_lib.FeatureWriter(
        filename=os.path.join(temp_file_path, 'eval.tf_record'),
        is_training=False)
    eval_features = []

    def _append_feature(feature, is_padding):
      if not is_padding:
        eval_features.append(feature)
      eval_writer.process_feature(feature)

    kwargs = dict(
        examples=eval_examples,
        tokenizer=tokenization.FullTokenizer(
            vocab_file=params.vocab_file,
            do_lower_case=params.do_lower_case),
        max_seq_length=params.seq_length,
        doc_stride=params.doc_stride,
        max_query_length=params.query_length,
        is_training=False,
        output_fn=_append_feature,
        batch_size=params.global_batch_size)
    if params.tokenization == 'SentencePiece':
      # squad_lib_sp requires one more argument 'do_lower_case'.
      kwargs['do_lower_case'] = params.do_lower_case

    eval_dataset_size = self.squad_lib.convert_examples_to_features(**kwargs)
    eval_writer.close()

    logging.info('***** Evaluation input stats *****')
    logging.info('  Num orig examples = %d', len(eval_examples))
    logging.info('  Num split examples = %d', len(eval_features))
    logging.info('  Batch size = %d', params.global_batch_size)
    logging.info('  Dataset size = %d', eval_dataset_size)

    return eval_writer.filename, eval_examples, eval_features

  def build_inputs(self, params, input_context=None):
    """Returns tf.data.Dataset for sentence_prediction task."""
    if params.input_path == 'dummy':
      # Dummy training data for unit test.
      def dummy_data(_):
        dummy_ids = tf.zeros((1, params.seq_length), dtype=tf.int32)
        x = dict(
            input_word_ids=dummy_ids,
            input_mask=dummy_ids,
            input_type_ids=dummy_ids)
        y = dict(
            start_positions=tf.constant(0, dtype=tf.int32),
            end_positions=tf.constant(1, dtype=tf.int32))
        return (x, y)

      dataset = tf.data.Dataset.range(1)
      dataset = dataset.repeat()
      dataset = dataset.map(
          dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    if params.is_training:
      dataloader_params = params
    else:
      input_path = self._tf_record_input_path
      dataloader_params = params.replace(input_path=input_path)

    return data_loader_factory.get_data_loader(
        dataloader_params).load(input_context)

  def build_metrics(self, training=None):
    del training
    # TODO(lehou): a list of metrics doesn't work the same as in compile/fit.
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(
            name='start_position_accuracy'),
        tf.keras.metrics.SparseCategoricalAccuracy(
            name='end_position_accuracy'),
    ]
    return metrics

  def process_metrics(self, metrics, labels, model_outputs):
    metrics = dict([(metric.name, metric) for metric in metrics])
    start_logits, end_logits = model_outputs
    metrics['start_position_accuracy'].update_state(
        labels['start_positions'], start_logits)
    metrics['end_position_accuracy'].update_state(
        labels['end_positions'], end_logits)

  def process_compiled_metrics(self, compiled_metrics, labels, model_outputs):
    start_logits, end_logits = model_outputs
    compiled_metrics.update_state(
        y_true=labels,  # labels has keys 'start_positions' and 'end_positions'.
        y_pred={'start_positions': start_logits, 'end_positions': end_logits})

  def validation_step(self, inputs, model: tf.keras.Model, metrics=None):
    features, _ = inputs
    unique_ids = features.pop('unique_ids')
    model_outputs = self.inference_step(features, model)
    start_logits, end_logits = model_outputs
    logs = {
        self.loss: 0.0,  # TODO(lehou): compute the real validation loss.
        'unique_ids': unique_ids,
        'start_logits': start_logits,
        'end_logits': end_logits,
    }
    return logs

  raw_aggregated_result = collections.namedtuple(
      'RawResult', ['unique_id', 'start_logits', 'end_logits'])

  def aggregate_logs(self, state=None, step_outputs=None):
    assert step_outputs is not None, 'Got no logs from self.validation_step.'
    if state is None:
      state = []

    for unique_ids, start_logits, end_logits in zip(
        step_outputs['unique_ids'],
        step_outputs['start_logits'],
        step_outputs['end_logits']):
      u_ids, s_logits, e_logits = (
          unique_ids.numpy(), start_logits.numpy(), end_logits.numpy())
      if u_ids.size == 1:
        u_ids = [u_ids]
        s_logits = [s_logits]
        e_logits = [e_logits]
      for values in zip(u_ids, s_logits, e_logits):
        state.append(self.raw_aggregated_result(
            unique_id=values[0],
            start_logits=values[1].tolist(),
            end_logits=values[2].tolist()))
    return state

  def reduce_aggregated_logs(self, aggregated_logs):
    all_predictions, _, scores_diff = (
        self.squad_lib.postprocess_output(
            self._eval_examples,
            self._eval_features,
            aggregated_logs,
            self.task_config.n_best_size,
            self.task_config.max_answer_length,
            self.task_config.validation_data.do_lower_case,
            version_2_with_negative=(
                self.task_config.validation_data.version_2_with_negative),
            null_score_diff_threshold=(
                self.task_config.null_score_diff_threshold),
            verbose=False))

    with tf.io.gfile.GFile(
        self.task_config.validation_data.input_path, 'r') as reader:
      dataset_json = json.load(reader)
      pred_dataset = dataset_json['data']
    if self.task_config.validation_data.version_2_with_negative:
      eval_metrics = squad_evaluate_v2_0.evaluate(
          pred_dataset, all_predictions, scores_diff)
      # Filter out useless metrics, such as start_position_accuracy that
      # we did not actually compute.
      eval_metrics = {
          'exact_match': eval_metrics['final_exact'],
          'exact_match_threshold': eval_metrics['final_exact_thresh'],
          'final_f1': eval_metrics['final_f1'] / 100.0,  # scale back to [0, 1].
          'f1_threshold': eval_metrics['final_f1_thresh'],
          'has_answer_exact_match': eval_metrics['HasAns_exact'],
          'has_answer_f1': eval_metrics['HasAns_f1']}
    else:
      eval_metrics = squad_evaluate_v1_1.evaluate(pred_dataset, all_predictions)
      # Filter out useless metrics, such as start_position_accuracy that
      # we did not actually compute.
      eval_metrics = {'exact_match': eval_metrics['exact_match'],
                      'final_f1': eval_metrics['final_f1']}
    return eval_metrics
