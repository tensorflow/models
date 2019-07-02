# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Run BERT on SQuAD 1.1 and SQuAD 2.0 in tf2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

# Import BERT model libraries.
from official.bert import bert_models
from official.bert import common_flags
from official.bert import input_pipeline
from official.bert import model_training_utils
from official.bert import modeling
from official.bert import optimization
from official.bert import squad_lib
from official.bert import tokenization
from official.bert import tpu_lib

flags.DEFINE_bool('do_train', False, 'Whether to run training.')
flags.DEFINE_bool('do_predict', False, 'Whether to run eval on the dev set.')
flags.DEFINE_string('train_data_path', '',
                    'Training data path with train tfrecords.')
flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')
# Model training specific flags.
flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')
# Predict processing related.
flags.DEFINE_string('predict_file', None,
                    'Prediction data path with train tfrecords.')
flags.DEFINE_string('vocab_file', None,
                    'The vocabulary file that the BERT model was trained on.')
flags.DEFINE_bool(
    'do_lower_case', True,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_bool(
    'verbose_logging', False,
    'If true, all of the warnings related to data processing will be printed. '
    'A number of warnings are expected for a normal SQuAD evaluation.')
flags.DEFINE_integer('predict_batch_size', 8,
                     'Total batch size for prediction.')
flags.DEFINE_integer(
    'n_best_size', 20,
    'The total number of n-best predictions to generate in the '
    'nbest_predictions.json output file.')
flags.DEFINE_integer(
    'max_answer_length', 30,
    'The maximum length of an answer that can be generated. This is needed '
    'because the start and end predictions are not conditioned on one another.')

common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS


def squad_loss_fn(start_positions,
                  end_positions,
                  start_logits,
                  end_logits,
                  loss_scale=1.0):
  """Returns sparse categorical crossentropy for start/end logits."""
  start_loss = tf.keras.backend.sparse_categorical_crossentropy(
      start_positions, start_logits, from_logits=True)
  end_loss = tf.keras.backend.sparse_categorical_crossentropy(
      end_positions, end_logits, from_logits=True)

  total_loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2
  total_loss *= loss_scale
  return total_loss


def get_loss_fn(loss_scale=1.0):
  """Gets a loss function for squad task."""

  def _loss_fn(labels, model_outputs):
    start_positions = labels['start_positions']
    end_positions = labels['end_positions']
    _, start_logits, end_logits = model_outputs
    return squad_loss_fn(
        start_positions,
        end_positions,
        start_logits,
        end_logits,
        loss_scale=loss_scale)

  return _loss_fn


def get_raw_results(predictions):
  """Converts multi-replica predictions to RawResult."""
  for unique_ids, start_logits, end_logits in zip(predictions['unique_ids'],
                                                  predictions['start_logits'],
                                                  predictions['end_logits']):
    for values in zip(unique_ids.numpy(), start_logits.numpy(),
                      end_logits.numpy()):
      yield squad_lib.RawResult(
          unique_id=values[0],
          start_logits=values[1].tolist(),
          end_logits=values[2].tolist())


def predict_squad_customized(strategy, input_meta_data, bert_config,
                             predict_tfrecord_path, num_steps):
  """Make predictions using a Bert-based squad model."""
  primary_cpu_task = '/job:worker' if FLAGS.tpu else ''

  with tf.device(primary_cpu_task):
    predict_dataset = input_pipeline.create_squad_dataset(
        predict_tfrecord_path,
        input_meta_data['max_seq_length'],
        FLAGS.predict_batch_size,
        is_training=False)
    predict_iterator = iter(
        strategy.experimental_distribute_dataset(predict_dataset))

    with strategy.scope():
      squad_model, _ = bert_models.squad_model(
          bert_config, input_meta_data['max_seq_length'], float_type=tf.float32)

    checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    logging.info('Restoring checkpoints from %s', checkpoint_path)
    checkpoint = tf.train.Checkpoint(model=squad_model)
    checkpoint.restore(checkpoint_path)

    @tf.function
    def predict_step(iterator):
      """Predicts on distributed devices."""

      def _replicated_step(inputs):
        """Replicated prediction calculation."""
        x, _ = inputs
        unique_ids, start_logits, end_logits = squad_model(x, training=False)
        return dict(
            unique_ids=unique_ids,
            start_logits=start_logits,
            end_logits=end_logits)

      outputs = strategy.experimental_run_v2(
          _replicated_step, args=(next(iterator),))
      return tf.nest.map_structure(strategy.unwrap, outputs)

    all_results = []
    for _ in range(num_steps):
      predictions = predict_step(predict_iterator)
      for result in get_raw_results(predictions):
        all_results.append(result)
      if len(all_results) % 100 == 0:
        logging.info('Made predictions for %d records.', len(all_results))
    return all_results


def train_squad(strategy, input_meta_data, custom_callbacks=None):
  """Run bert squad training."""
  if not strategy:
    raise ValueError('Distribution strategy cannot be None.')

  logging.info('Training using customized training loop with distribution'
               ' strategy.')

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  epochs = FLAGS.num_train_epochs
  num_train_examples = input_meta_data['train_data_size']
  max_seq_length = input_meta_data['max_seq_length']
  steps_per_epoch = int(num_train_examples / FLAGS.train_batch_size)
  warmup_steps = int(epochs * num_train_examples * 0.1 / FLAGS.train_batch_size)
  train_input_fn = functools.partial(
      input_pipeline.create_squad_dataset,
      FLAGS.train_data_path,
      max_seq_length,
      FLAGS.train_batch_size,
      is_training=True)

  def _get_squad_model():
    squad_model, core_model = bert_models.squad_model(
        bert_config, max_seq_length, float_type=tf.float32)
    squad_model.optimizer = optimization.create_optimizer(
        FLAGS.learning_rate, steps_per_epoch * epochs, warmup_steps)
    return squad_model, core_model

  # The original BERT model does not scale the loss by
  # 1/num_replicas_in_sync. It could be an accident. So, in order to use
  # the same hyper parameter, we do the same thing here by keeping each
  # replica loss as it is.
  loss_fn = get_loss_fn(loss_scale=1.0)
  use_remote_tpu = (FLAGS.strategy_type == 'tpu' and FLAGS.tpu)

  model_training_utils.run_customized_training_loop(
      strategy=strategy,
      model_fn=_get_squad_model,
      loss_fn=loss_fn,
      model_dir=FLAGS.model_dir,
      steps_per_epoch=steps_per_epoch,
      steps_per_loop=FLAGS.steps_per_loop,
      epochs=epochs,
      train_input_fn=train_input_fn,
      init_checkpoint=FLAGS.init_checkpoint,
      use_remote_tpu=use_remote_tpu,
      custom_callbacks=custom_callbacks)


def predict_squad(strategy, input_meta_data):
  """Makes predictions for a squad dataset."""
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  doc_stride = input_meta_data['doc_stride']
  max_query_length = input_meta_data['max_query_length']
  # Whether data should be in Ver 2.0 format.
  version_2_with_negative = input_meta_data.get('version_2_with_negative',
                                                False)
  eval_examples = squad_lib.read_squad_examples(
      input_file=FLAGS.predict_file,
      is_training=False,
      version_2_with_negative=version_2_with_negative)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

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
  dataset_size = squad_lib.convert_examples_to_features(
      examples=eval_examples,
      tokenizer=tokenizer,
      max_seq_length=input_meta_data['max_seq_length'],
      doc_stride=doc_stride,
      max_query_length=max_query_length,
      is_training=False,
      output_fn=_append_feature,
      batch_size=FLAGS.predict_batch_size)
  eval_writer.close()

  logging.info('***** Running predictions *****')
  logging.info('  Num orig examples = %d', len(eval_examples))
  logging.info('  Num split examples = %d', len(eval_features))
  logging.info('  Batch size = %d', FLAGS.predict_batch_size)

  num_steps = int(dataset_size / FLAGS.predict_batch_size)
  all_results = predict_squad_customized(strategy, input_meta_data, bert_config,
                                         eval_writer.filename, num_steps)

  output_prediction_file = os.path.join(FLAGS.model_dir, 'predictions.json')
  output_nbest_file = os.path.join(FLAGS.model_dir, 'nbest_predictions.json')
  output_null_log_odds_file = os.path.join(FLAGS.model_dir, 'null_odds.json')

  squad_lib.write_predictions(
      eval_examples,
      eval_features,
      all_results,
      FLAGS.n_best_size,
      FLAGS.max_answer_length,
      FLAGS.do_lower_case,
      output_prediction_file,
      output_nbest_file,
      output_null_log_odds_file,
      verbose=FLAGS.verbose_logging)


def main(_):
  # Users should always run this script under TF 2.x
  assert tf.version.VERSION.startswith('2.')

  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  strategy = None
  if FLAGS.strategy_type == 'mirror':
    strategy = tf.distribute.MirroredStrategy()
  elif FLAGS.strategy_type == 'tpu':
    # Initialize TPU System.
    cluster_resolver = tpu_lib.tpu_initialize(FLAGS.tpu)
    strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
  else:
    raise ValueError('The distribution strategy type is not supported: %s' %
                     FLAGS.strategy_type)
  if FLAGS.do_train:
    train_squad(strategy, input_meta_data)
  if FLAGS.do_predict:
    predict_squad(strategy, input_meta_data)


if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
