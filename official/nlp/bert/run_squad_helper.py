# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Library for running BERT family models on SQuAD 1.1/2.0 in TF 2.x."""

import collections
import json
import os

from absl import flags
from absl import logging
import tensorflow as tf
from official.modeling import performance
from official.nlp import optimization
from official.nlp.bert import bert_models
from official.nlp.bert import common_flags
from official.nlp.bert import input_pipeline
from official.nlp.bert import model_saving_utils
from official.nlp.bert import model_training_utils
from official.nlp.bert import squad_evaluate_v1_1
from official.nlp.bert import squad_evaluate_v2_0
from official.nlp.data import squad_lib_sp
from official.utils.misc import keras_utils


def define_common_squad_flags():
  """Defines common flags used by SQuAD tasks."""
  flags.DEFINE_enum(
      'mode', 'train_and_eval', [
          'train_and_eval', 'train_and_predict', 'train', 'eval', 'predict',
          'export_only'
      ], 'One of {"train_and_eval", "train_and_predict", '
      '"train", "eval", "predict", "export_only"}. '
      '`train_and_eval`: train & predict to json files & compute eval metrics. '
      '`train_and_predict`: train & predict to json files. '
      '`train`: only trains the model. '
      '`eval`: predict answers from squad json file & compute eval metrics. '
      '`predict`: predict answers from the squad json file. '
      '`export_only`: will take the latest checkpoint inside '
      'model_dir and export a `SavedModel`.')
  flags.DEFINE_string('train_data_path', '',
                      'Training data path with train tfrecords.')
  flags.DEFINE_string(
      'input_meta_data_path', None,
      'Path to file that contains meta data about input '
      'to be used for training and evaluation.')
  # Model training specific flags.
  flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')
  # Predict processing related.
  flags.DEFINE_string(
      'predict_file', None, 'SQuAD prediction json file path. '
      '`predict` mode supports multiple files: one can use '
      'wildcard to specify multiple files and it can also be '
      'multiple file patterns separated by comma. Note that '
      '`eval` mode only supports a single predict file.')
  flags.DEFINE_bool(
      'do_lower_case', True,
      'Whether to lower case the input text. Should be True for uncased '
      'models and False for cased models.')
  flags.DEFINE_float(
      'null_score_diff_threshold', 0.0,
      'If null_score - best_non_null is greater than the threshold, '
      'predict null. This is only used for SQuAD v2.')
  flags.DEFINE_bool(
      'verbose_logging', False,
      'If true, all of the warnings related to data processing will be '
      'printed. A number of warnings are expected for a normal SQuAD '
      'evaluation.')
  flags.DEFINE_integer('predict_batch_size', 8,
                       'Total batch size for prediction.')
  flags.DEFINE_integer(
      'n_best_size', 20,
      'The total number of n-best predictions to generate in the '
      'nbest_predictions.json output file.')
  flags.DEFINE_integer(
      'max_answer_length', 30,
      'The maximum length of an answer that can be generated. This is needed '
      'because the start and end predictions are not conditioned on one '
      'another.')

  common_flags.define_common_bert_flags()


FLAGS = flags.FLAGS


def squad_loss_fn(start_positions, end_positions, start_logits, end_logits):
  """Returns sparse categorical crossentropy for start/end logits."""
  start_loss = tf.keras.losses.sparse_categorical_crossentropy(
      start_positions, start_logits, from_logits=True)
  end_loss = tf.keras.losses.sparse_categorical_crossentropy(
      end_positions, end_logits, from_logits=True)

  total_loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2
  return total_loss


def get_loss_fn():
  """Gets a loss function for squad task."""

  def _loss_fn(labels, model_outputs):
    start_positions = labels['start_positions']
    end_positions = labels['end_positions']
    start_logits, end_logits = model_outputs
    return squad_loss_fn(start_positions, end_positions, start_logits,
                         end_logits)

  return _loss_fn


RawResult = collections.namedtuple('RawResult',
                                   ['unique_id', 'start_logits', 'end_logits'])


def get_raw_results(predictions):
  """Converts multi-replica predictions to RawResult."""
  for unique_ids, start_logits, end_logits in zip(predictions['unique_ids'],
                                                  predictions['start_logits'],
                                                  predictions['end_logits']):
    for values in zip(unique_ids.numpy(), start_logits.numpy(),
                      end_logits.numpy()):
      yield RawResult(
          unique_id=values[0],
          start_logits=values[1].tolist(),
          end_logits=values[2].tolist())


def get_dataset_fn(input_file_pattern, max_seq_length, global_batch_size,
                   is_training):
  """Gets a closure to create a dataset.."""

  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    batch_size = ctx.get_per_replica_batch_size(
        global_batch_size) if ctx else global_batch_size
    dataset = input_pipeline.create_squad_dataset(
        input_file_pattern,
        max_seq_length,
        batch_size,
        is_training=is_training,
        input_pipeline_context=ctx)
    return dataset

  return _dataset_fn


def get_squad_model_to_predict(strategy, bert_config, checkpoint_path,
                               input_meta_data):
  """Gets a squad model to make predictions."""
  with strategy.scope():
    # Prediction always uses float32, even if training uses mixed precision.
    tf.keras.mixed_precision.set_global_policy('float32')
    squad_model, _ = bert_models.squad_model(
        bert_config,
        input_meta_data['max_seq_length'],
        hub_module_url=FLAGS.hub_module_url)

  if checkpoint_path is None:
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
  logging.info('Restoring checkpoints from %s', checkpoint_path)
  checkpoint = tf.train.Checkpoint(model=squad_model)
  checkpoint.restore(checkpoint_path).expect_partial()
  return squad_model


def predict_squad_customized(strategy, input_meta_data, predict_tfrecord_path,
                             num_steps, squad_model):
  """Make predictions using a Bert-based squad model."""
  predict_dataset_fn = get_dataset_fn(
      predict_tfrecord_path,
      input_meta_data['max_seq_length'],
      FLAGS.predict_batch_size,
      is_training=False)
  predict_iterator = iter(
      strategy.distribute_datasets_from_function(predict_dataset_fn))

  @tf.function
  def predict_step(iterator):
    """Predicts on distributed devices."""

    def _replicated_step(inputs):
      """Replicated prediction calculation."""
      x, _ = inputs
      unique_ids = x.pop('unique_ids')
      start_logits, end_logits = squad_model(x, training=False)
      return dict(
          unique_ids=unique_ids,
          start_logits=start_logits,
          end_logits=end_logits)

    outputs = strategy.run(_replicated_step, args=(next(iterator),))
    return tf.nest.map_structure(strategy.experimental_local_results, outputs)

  all_results = []
  for _ in range(num_steps):
    predictions = predict_step(predict_iterator)
    for result in get_raw_results(predictions):
      all_results.append(result)
    if len(all_results) % 100 == 0:
      logging.info('Made predictions for %d records.', len(all_results))
  return all_results


def train_squad(strategy,
                input_meta_data,
                bert_config,
                custom_callbacks=None,
                run_eagerly=False,
                init_checkpoint=None,
                sub_model_export_name=None):
  """Run bert squad training."""
  if strategy:
    logging.info('Training using customized training loop with distribution'
                 ' strategy.')
  # Enables XLA in Session Config. Should not be set for TPU.
  keras_utils.set_session_config(FLAGS.enable_xla)
  performance.set_mixed_precision_policy(common_flags.dtype(),
                                         use_experimental_api=False)

  epochs = FLAGS.num_train_epochs
  num_train_examples = input_meta_data['train_data_size']
  max_seq_length = input_meta_data['max_seq_length']
  steps_per_epoch = int(num_train_examples / FLAGS.train_batch_size)
  warmup_steps = int(epochs * num_train_examples * 0.1 / FLAGS.train_batch_size)
  train_input_fn = get_dataset_fn(
      FLAGS.train_data_path,
      max_seq_length,
      FLAGS.train_batch_size,
      is_training=True)

  def _get_squad_model():
    """Get Squad model and optimizer."""
    squad_model, core_model = bert_models.squad_model(
        bert_config,
        max_seq_length,
        hub_module_url=FLAGS.hub_module_url,
        hub_module_trainable=FLAGS.hub_module_trainable)
    optimizer = optimization.create_optimizer(FLAGS.learning_rate,
                                              steps_per_epoch * epochs,
                                              warmup_steps, FLAGS.end_lr,
                                              FLAGS.optimizer_type)

    squad_model.optimizer = performance.configure_optimizer(
        optimizer,
        use_float16=common_flags.use_float16(),
        use_graph_rewrite=common_flags.use_graph_rewrite(),
        use_experimental_api=False)
    return squad_model, core_model

  # Only when explicit_allreduce = True, post_allreduce_callbacks and
  # allreduce_bytes_per_pack will take effect. optimizer.apply_gradients() no
  # longer implicitly allreduce gradients, users manually allreduce gradient and
  # pass the allreduced grads_and_vars to apply_gradients().
  # With explicit_allreduce = True, clip_by_global_norm is moved to after
  # allreduce.
  model_training_utils.run_customized_training_loop(
      strategy=strategy,
      model_fn=_get_squad_model,
      loss_fn=get_loss_fn(),
      model_dir=FLAGS.model_dir,
      steps_per_epoch=steps_per_epoch,
      steps_per_loop=FLAGS.steps_per_loop,
      epochs=epochs,
      train_input_fn=train_input_fn,
      init_checkpoint=init_checkpoint or FLAGS.init_checkpoint,
      sub_model_export_name=sub_model_export_name,
      run_eagerly=run_eagerly,
      custom_callbacks=custom_callbacks,
      explicit_allreduce=FLAGS.explicit_allreduce,
      pre_allreduce_callbacks=[
          model_training_utils.clip_by_global_norm_callback
      ],
      allreduce_bytes_per_pack=FLAGS.allreduce_bytes_per_pack)


def prediction_output_squad(strategy, input_meta_data, tokenizer, squad_lib,
                            predict_file, squad_model):
  """Makes predictions for a squad dataset."""
  doc_stride = input_meta_data['doc_stride']
  max_query_length = input_meta_data['max_query_length']
  # Whether data should be in Ver 2.0 format.
  version_2_with_negative = input_meta_data.get('version_2_with_negative',
                                                False)
  eval_examples = squad_lib.read_squad_examples(
      input_file=predict_file,
      is_training=False,
      version_2_with_negative=version_2_with_negative)

  eval_writer = squad_lib.FeatureWriter(
      filename=os.path.join(FLAGS.model_dir, 'eval.tf_record'),
      is_training=False)
  eval_features = []

  def _append_feature(feature, is_padding):
    if not is_padding:
      eval_features.append(feature)
    eval_writer.process_feature(feature)

  # TPU requires a fixed batch size for all batches, therefore the number
  # of examples must be a multiple of the batch size, or else examples
  # will get dropped. So we pad with fake examples which are ignored
  # later on.
  kwargs = dict(
      examples=eval_examples,
      tokenizer=tokenizer,
      max_seq_length=input_meta_data['max_seq_length'],
      doc_stride=doc_stride,
      max_query_length=max_query_length,
      is_training=False,
      output_fn=_append_feature,
      batch_size=FLAGS.predict_batch_size)

  # squad_lib_sp requires one more argument 'do_lower_case'.
  if squad_lib == squad_lib_sp:
    kwargs['do_lower_case'] = FLAGS.do_lower_case
  dataset_size = squad_lib.convert_examples_to_features(**kwargs)
  eval_writer.close()

  logging.info('***** Running predictions *****')
  logging.info('  Num orig examples = %d', len(eval_examples))
  logging.info('  Num split examples = %d', len(eval_features))
  logging.info('  Batch size = %d', FLAGS.predict_batch_size)

  num_steps = int(dataset_size / FLAGS.predict_batch_size)
  all_results = predict_squad_customized(strategy, input_meta_data,
                                         eval_writer.filename, num_steps,
                                         squad_model)

  all_predictions, all_nbest_json, scores_diff_json = (
      squad_lib.postprocess_output(
          eval_examples,
          eval_features,
          all_results,
          FLAGS.n_best_size,
          FLAGS.max_answer_length,
          FLAGS.do_lower_case,
          version_2_with_negative=version_2_with_negative,
          null_score_diff_threshold=FLAGS.null_score_diff_threshold,
          verbose=FLAGS.verbose_logging))

  return all_predictions, all_nbest_json, scores_diff_json


def dump_to_files(all_predictions,
                  all_nbest_json,
                  scores_diff_json,
                  squad_lib,
                  version_2_with_negative,
                  file_prefix=''):
  """Save output to json files."""
  output_prediction_file = os.path.join(FLAGS.model_dir,
                                        '%spredictions.json' % file_prefix)
  output_nbest_file = os.path.join(FLAGS.model_dir,
                                   '%snbest_predictions.json' % file_prefix)
  output_null_log_odds_file = os.path.join(FLAGS.model_dir, file_prefix,
                                           '%snull_odds.json' % file_prefix)
  logging.info('Writing predictions to: %s', (output_prediction_file))
  logging.info('Writing nbest to: %s', (output_nbest_file))

  squad_lib.write_to_json_files(all_predictions, output_prediction_file)
  squad_lib.write_to_json_files(all_nbest_json, output_nbest_file)
  if version_2_with_negative:
    squad_lib.write_to_json_files(scores_diff_json, output_null_log_odds_file)


def _get_matched_files(input_path):
  """Returns all files that matches the input_path."""
  input_patterns = input_path.strip().split(',')
  all_matched_files = []
  for input_pattern in input_patterns:
    input_pattern = input_pattern.strip()
    if not input_pattern:
      continue
    matched_files = tf.io.gfile.glob(input_pattern)
    if not matched_files:
      raise ValueError('%s does not match any files.' % input_pattern)
    else:
      all_matched_files.extend(matched_files)
  return sorted(all_matched_files)


def predict_squad(strategy,
                  input_meta_data,
                  tokenizer,
                  bert_config,
                  squad_lib,
                  init_checkpoint=None):
  """Get prediction results and evaluate them to hard drive."""
  if init_checkpoint is None:
    init_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)

  all_predict_files = _get_matched_files(FLAGS.predict_file)
  squad_model = get_squad_model_to_predict(strategy, bert_config,
                                           init_checkpoint, input_meta_data)
  for idx, predict_file in enumerate(all_predict_files):
    all_predictions, all_nbest_json, scores_diff_json = prediction_output_squad(
        strategy, input_meta_data, tokenizer, squad_lib, predict_file,
        squad_model)
    if len(all_predict_files) == 1:
      file_prefix = ''
    else:
      # if predict_file is /path/xquad.ar.json, the `file_prefix` may be
      # "xquad.ar-0-"
      file_prefix = '%s-' % os.path.splitext(
          os.path.basename(all_predict_files[idx]))[0]
    dump_to_files(all_predictions, all_nbest_json, scores_diff_json, squad_lib,
                  input_meta_data.get('version_2_with_negative', False),
                  file_prefix)


def eval_squad(strategy,
               input_meta_data,
               tokenizer,
               bert_config,
               squad_lib,
               init_checkpoint=None):
  """Get prediction results and evaluate them against ground truth."""
  if init_checkpoint is None:
    init_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)

  all_predict_files = _get_matched_files(FLAGS.predict_file)
  if len(all_predict_files) != 1:
    raise ValueError('`eval_squad` only supports one predict file, '
                     'but got %s' % all_predict_files)

  squad_model = get_squad_model_to_predict(strategy, bert_config,
                                           init_checkpoint, input_meta_data)
  all_predictions, all_nbest_json, scores_diff_json = prediction_output_squad(
      strategy, input_meta_data, tokenizer, squad_lib, all_predict_files[0],
      squad_model)
  dump_to_files(all_predictions, all_nbest_json, scores_diff_json, squad_lib,
                input_meta_data.get('version_2_with_negative', False))

  with tf.io.gfile.GFile(FLAGS.predict_file, 'r') as reader:
    dataset_json = json.load(reader)
    pred_dataset = dataset_json['data']
  if input_meta_data.get('version_2_with_negative', False):
    eval_metrics = squad_evaluate_v2_0.evaluate(pred_dataset, all_predictions,
                                                scores_diff_json)
  else:
    eval_metrics = squad_evaluate_v1_1.evaluate(pred_dataset, all_predictions)
  return eval_metrics


def export_squad(model_export_path, input_meta_data, bert_config):
  """Exports a trained model as a `SavedModel` for inference.

  Args:
    model_export_path: a string specifying the path to the SavedModel directory.
    input_meta_data: dictionary containing meta data about input and model.
    bert_config: Bert configuration file to define core bert layers.

  Raises:
    Export path is not specified, got an empty string or None.
  """
  if not model_export_path:
    raise ValueError('Export path is not specified: %s' % model_export_path)
  # Export uses float32 for now, even if training uses mixed precision.
  tf.keras.mixed_precision.set_global_policy('float32')
  squad_model, _ = bert_models.squad_model(bert_config,
                                           input_meta_data['max_seq_length'])
  model_saving_utils.export_bert_model(
      model_export_path, model=squad_model, checkpoint_dir=FLAGS.model_dir)
