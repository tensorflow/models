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

"""Question answering task."""
import dataclasses
import functools
import json
import os
from typing import List, Optional

from absl import logging
import orbit
import tensorflow as tf

from official.core import base_task
from official.core import config_definitions as cfg
from official.core import task_factory
from official.modeling.hyperparams import base_config
from official.nlp.configs import encoders
from official.nlp.data import data_loader_factory
from official.nlp.data import squad_lib as squad_lib_wp
from official.nlp.data import squad_lib_sp
from official.nlp.modeling import models
from official.nlp.tasks import utils
from official.nlp.tools import squad_evaluate_v1_1
from official.nlp.tools import squad_evaluate_v2_0
from official.nlp.tools import tokenization


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """A base span labeler configuration."""
  encoder: encoders.EncoderConfig = encoders.EncoderConfig()


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


@dataclasses.dataclass
class RawAggregatedResult:
  """Raw representation for SQuAD predictions."""
  unique_id: int
  start_logits: List[float]
  end_logits: List[float]
  start_indexes: Optional[List[int]] = None
  end_indexes: Optional[List[int]] = None
  class_logits: Optional[float] = None


@task_factory.register_task_cls(QuestionAnsweringConfig)
class QuestionAnsweringTask(base_task.Task):
  """Task object for question answering."""

  def __init__(self, params: cfg.TaskConfig, logging_dir=None, name=None):
    super().__init__(params, logging_dir, name=name)

    if params.validation_data is None:
      return

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

  def set_preprocessed_eval_input_path(self, eval_input_path):
    """Sets the path to the preprocessed eval data."""
    self._tf_record_input_path = eval_input_path

  def build_model(self):
    if self.task_config.hub_module_url and self.task_config.init_checkpoint:
      raise ValueError('At most one of `hub_module_url` and '
                       '`init_checkpoint` can be specified.')
    if self.task_config.hub_module_url:
      encoder_network = utils.get_encoder_from_hub(
          self.task_config.hub_module_url)
    else:
      encoder_network = encoders.build_encoder(self.task_config.model.encoder)
    encoder_cfg = self.task_config.model.encoder.get()
    return models.BertSpanLabeler(
        network=encoder_network,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range))

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    start_positions = labels['start_positions']
    end_positions = labels['end_positions']
    start_logits, end_logits = model_outputs

    start_loss = tf.keras.losses.sparse_categorical_crossentropy(
        start_positions,
        tf.cast(start_logits, dtype=tf.float32),
        from_logits=True)
    end_loss = tf.keras.losses.sparse_categorical_crossentropy(
        end_positions, tf.cast(end_logits, dtype=tf.float32), from_logits=True)

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

    # XLNet preprocesses SQuAD examples in a P, Q, class order whereas
    # BERT preprocesses in a class, Q, P order.
    xlnet_ordering = self.task_config.model.encoder.type == 'xlnet'
    kwargs = dict(
        examples=eval_examples,
        max_seq_length=params.seq_length,
        doc_stride=params.doc_stride,
        max_query_length=params.query_length,
        is_training=False,
        output_fn=_append_feature,
        batch_size=params.global_batch_size,
        xlnet_format=xlnet_ordering)

    if params.tokenization == 'SentencePiece':
      # squad_lib_sp requires one more argument 'do_lower_case'.
      kwargs['do_lower_case'] = params.do_lower_case
      kwargs['tokenizer'] = tokenization.FullSentencePieceTokenizer(
          sp_model_file=params.vocab_file)
    elif params.tokenization == 'WordPiece':
      kwargs['tokenizer'] = tokenization.FullTokenizer(
          vocab_file=params.vocab_file, do_lower_case=params.do_lower_case)
    else:
      raise ValueError('Unexpected tokenization: %s' % params.tokenization)

    eval_dataset_size = self.squad_lib.convert_examples_to_features(**kwargs)
    eval_writer.close()

    logging.info('***** Evaluation input stats *****')
    logging.info('  Num orig examples = %d', len(eval_examples))
    logging.info('  Num split examples = %d', len(eval_features))
    logging.info('  Batch size = %d', params.global_batch_size)
    logging.info('  Dataset size = %d', eval_dataset_size)

    return eval_writer.filename, eval_examples, eval_features

  def _dummy_data(self, params, _):
    """Returns dummy data."""
    dummy_ids = tf.zeros((1, params.seq_length), dtype=tf.int32)
    x = dict(
        input_word_ids=dummy_ids,
        input_mask=dummy_ids,
        input_type_ids=dummy_ids)
    y = dict(
        start_positions=tf.constant(0, dtype=tf.int32),
        end_positions=tf.constant(1, dtype=tf.int32),
        is_impossible=tf.constant(0, dtype=tf.int32))
    return x, y

  def build_inputs(self, params, input_context=None):
    """Returns tf.data.Dataset for sentence_prediction task."""
    if params.input_path == 'dummy':
      dataset = tf.data.Dataset.range(1)
      dataset = dataset.repeat()
      dummy_data = functools.partial(self._dummy_data, params)
      dataset = dataset.map(
          dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    if params.is_training:
      dataloader_params = params
    else:
      input_path = self._tf_record_input_path
      dataloader_params = params.replace(input_path=input_path)

    return data_loader_factory.get_data_loader(dataloader_params).load(
        input_context)

  def build_metrics(self, training=None):
    if not training:
      # We cannot compute start/end_position_accuracy because start/end_position
      # labels are not available in the validation dataset (b/173794928).
      return []
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
    metrics['start_position_accuracy'].update_state(labels['start_positions'],
                                                    start_logits)
    metrics['end_position_accuracy'].update_state(labels['end_positions'],
                                                  end_logits)

  def process_compiled_metrics(self, compiled_metrics, labels, model_outputs):
    start_logits, end_logits = model_outputs
    compiled_metrics.update_state(
        y_true=labels,  # labels has keys 'start_positions' and 'end_positions'.
        y_pred={
            'start_positions': start_logits,
            'end_positions': end_logits
        })

  def validation_step(self, inputs, model: tf.keras.Model, metrics=None):
    features, _ = inputs
    unique_ids = features.pop('unique_ids')
    model_outputs = self.inference_step(features, model)
    start_logits, end_logits = model_outputs
    # We cannot compute validation_loss here, because start/end_position
    # labels are not available in the validation dataset (b/173794928).
    logs = {
        'unique_ids': unique_ids,
        'start_logits': start_logits,
        'end_logits': end_logits,
    }
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    assert step_outputs is not None, 'Got no logs from self.validation_step.'
    if state is None:
      state = []

    for outputs in zip(step_outputs['unique_ids'],
                       step_outputs['start_logits'],
                       step_outputs['end_logits']):
      numpy_values = [
          output.numpy() for output in outputs if output is not None]

      for values in zip(*numpy_values):
        state.append(RawAggregatedResult(
            unique_id=values[0],
            start_logits=values[1],
            end_logits=values[2]))
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
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
            xlnet_format=self.task_config.validation_data.xlnet_format,
            verbose=False))

    with tf.io.gfile.GFile(self.task_config.validation_data.input_path,
                           'r') as reader:
      dataset_json = json.load(reader)
      pred_dataset = dataset_json['data']
    if self.task_config.validation_data.version_2_with_negative:
      eval_metrics = squad_evaluate_v2_0.evaluate(pred_dataset, all_predictions,
                                                  scores_diff)
      eval_metrics = {
          'exact_match': eval_metrics['final_exact'],
          'exact_match_threshold': eval_metrics['final_exact_thresh'],
          'final_f1': eval_metrics['final_f1'] / 100.0,  # scale back to [0, 1].
          'f1_threshold': eval_metrics['final_f1_thresh'],
          'has_answer_exact_match': eval_metrics['HasAns_exact'],
          'has_answer_f1': eval_metrics['HasAns_f1']
      }
    else:
      eval_metrics = squad_evaluate_v1_1.evaluate(pred_dataset, all_predictions)
      eval_metrics = {
          'exact_match': eval_metrics['exact_match'],
          'final_f1': eval_metrics['final_f1']
      }
    return eval_metrics


@dataclasses.dataclass
class XLNetQuestionAnsweringConfig(QuestionAnsweringConfig):
  """The config for the XLNet variation of QuestionAnswering."""
  pass


@task_factory.register_task_cls(XLNetQuestionAnsweringConfig)
class XLNetQuestionAnsweringTask(QuestionAnsweringTask):
  """XLNet variant of the Question Answering Task.

  The main differences include:
    - The encoder is an `XLNetBase` class.
    - The `SpanLabeling` head is an instance of `XLNetSpanLabeling` which
      predicts start/end positions and impossibility score. During inference,
      it predicts the top N scores and indexes.
  """

  def build_model(self):
    if self.task_config.hub_module_url and self.task_config.init_checkpoint:
      raise ValueError('At most one of `hub_module_url` and '
                       '`init_checkpoint` can be specified.')
    if self.task_config.hub_module_url:
      encoder_network = utils.get_encoder_from_hub(
          self.task_config.hub_module_url)
    else:
      encoder_network = encoders.build_encoder(self.task_config.model.encoder)
    encoder_cfg = self.task_config.model.encoder.get()
    return models.XLNetSpanLabeler(
        network=encoder_network,
        start_n_top=self.task_config.n_best_size,
        end_n_top=self.task_config.n_best_size,
        initializer=tf.keras.initializers.RandomNormal(
            stddev=encoder_cfg.initializer_range))

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    start_positions = labels['start_positions']
    end_positions = labels['end_positions']
    is_impossible = labels['is_impossible']
    is_impossible = tf.cast(tf.reshape(is_impossible, [-1]), tf.float32)

    start_logits = model_outputs['start_logits']
    end_logits = model_outputs['end_logits']
    class_logits = model_outputs['class_logits']

    start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        start_positions, start_logits)
    end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        end_positions, end_logits)
    is_impossible_loss = tf.keras.losses.binary_crossentropy(
        is_impossible, class_logits, from_logits=True)

    loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2
    loss += tf.reduce_mean(is_impossible_loss) / 2
    return loss

  def process_metrics(self, metrics, labels, model_outputs):
    metrics = dict([(metric.name, metric) for metric in metrics])
    start_logits = model_outputs['start_logits']
    end_logits = model_outputs['end_logits']
    metrics['start_position_accuracy'].update_state(labels['start_positions'],
                                                    start_logits)
    metrics['end_position_accuracy'].update_state(labels['end_positions'],
                                                  end_logits)

  def process_compiled_metrics(self, compiled_metrics, labels, model_outputs):
    start_logits = model_outputs['start_logits']
    end_logits = model_outputs['end_logits']
    compiled_metrics.update_state(
        y_true=labels,  # labels has keys 'start_positions' and 'end_positions'.
        y_pred={
            'start_positions': start_logits,
            'end_positions': end_logits,
        })

  def _dummy_data(self, params, _):
    """Returns dummy data."""
    dummy_ids = tf.zeros((1, params.seq_length), dtype=tf.int32)
    zero = tf.constant(0, dtype=tf.int32)
    x = dict(
        input_word_ids=dummy_ids,
        input_mask=dummy_ids,
        input_type_ids=dummy_ids,
        class_index=zero,
        is_impossible=zero,
        paragraph_mask=dummy_ids,
        start_positions=tf.zeros((1), dtype=tf.int32))
    y = dict(
        start_positions=tf.zeros((1), dtype=tf.int32),
        end_positions=tf.ones((1), dtype=tf.int32),
        is_impossible=zero)
    return x, y

  def validation_step(self, inputs, model: tf.keras.Model, metrics=None):
    features, _ = inputs
    unique_ids = features.pop('unique_ids')
    model_outputs = self.inference_step(features, model)
    start_top_predictions = model_outputs['start_top_predictions']
    end_top_predictions = model_outputs['end_top_predictions']
    start_indexes = model_outputs['start_top_index']
    end_indexes = model_outputs['end_top_index']
    class_logits = model_outputs['class_logits']

    logs = {
        'unique_ids': unique_ids,
        'start_top_predictions': start_top_predictions,
        'end_top_predictions': end_top_predictions,
        'start_indexes': start_indexes,
        'end_indexes': end_indexes,
        'class_logits': class_logits,
    }
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    assert step_outputs is not None, 'Got no logs from self.validation_step.'
    if state is None:
      state = []

    for outputs in zip(step_outputs['unique_ids'],
                       step_outputs['start_top_predictions'],
                       step_outputs['end_top_predictions'],
                       step_outputs['start_indexes'],
                       step_outputs['end_indexes'],
                       step_outputs['class_logits']):
      numpy_values = [
          output.numpy() for output in outputs]

      for (unique_id, start_top_predictions, end_top_predictions, start_indexes,
           end_indexes, class_logits) in zip(*numpy_values):
        state.append(RawAggregatedResult(
            unique_id=unique_id,
            start_logits=start_top_predictions.tolist(),
            end_logits=end_top_predictions.tolist(),
            start_indexes=start_indexes.tolist(),
            end_indexes=end_indexes.tolist(),
            class_logits=class_logits))
    return state


def predict(task: QuestionAnsweringTask, params: cfg.DataConfig,
            model: tf.keras.Model):
  """Predicts on the input data.

  Args:
    task: A `QuestionAnsweringTask` object.
    params: A `cfg.DataConfig` object.
    model: A keras.Model.

  Returns:
    A tuple of `all_predictions`, `all_nbest` and `scores_diff`, which
      are dict and can be written to json files including prediction json file,
      nbest json file and null_odds json file.
  """
  tf_record_input_path, eval_examples, eval_features = (
      task._preprocess_eval_data(params))  # pylint: disable=protected-access

  # `tf_record_input_path` will overwrite `params.input_path`,
  # when `task.buid_inputs()` is called.
  task.set_preprocessed_eval_input_path(tf_record_input_path)

  def predict_step(inputs):
    """Replicated prediction calculation."""
    return task.validation_step(inputs, model)

  dataset = orbit.utils.make_distributed_dataset(tf.distribute.get_strategy(),
                                                 task.build_inputs, params)
  aggregated_outputs = utils.predict(predict_step, task.aggregate_logs, dataset)

  all_predictions, all_nbest, scores_diff = (
      task.squad_lib.postprocess_output(
          eval_examples,
          eval_features,
          aggregated_outputs,
          task.task_config.n_best_size,
          task.task_config.max_answer_length,
          task.task_config.validation_data.do_lower_case,
          version_2_with_negative=(params.version_2_with_negative),
          null_score_diff_threshold=task.task_config.null_score_diff_threshold,
          xlnet_format=task.task_config.validation_data.xlnet_format,
          verbose=False))
  return all_predictions, all_nbest, scores_diff
