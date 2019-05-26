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
"""BERT classification finetuning runner in tf2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import math

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from official.bert import bert_models
from official.bert import input_pipeline
from official.bert import model_saving_utils
from official.bert import model_training_utils
from official.bert import modeling
from official.bert import optimization

flags.DEFINE_enum(
    'mode', 'train_and_eval', ['train_and_eval', 'export_only'],
    'One of {"train_and_eval", "export_only"}. `train_and_eval`: '
    'trains the model and evaluates in the meantime. '
    '`export_only`: will take the latest checkpoint inside '
    'model_dir and export a `SavedModel`.')
flags.DEFINE_string('bert_config_file', None,
                    'Bert configuration file to define core bert layers.')
flags.DEFINE_string(
    'model_dir', None,
    ('The directory where the model weights and training/evaluation summaries '
     'are stored. If not specified, save to /tmp/bert20/.'))
flags.DEFINE_string('tpu', '', 'TPU address to connect to.')
flags.DEFINE_string('train_data_path', None,
                    'Path to training data for BERT classifier.')
flags.DEFINE_string('eval_data_path', None,
                    'Path to evaluation data for BERT classifier.')
flags.DEFINE_string(
    'init_checkpoint', None,
    'Initial checkpoint (usually from a pre-trained BERT model).')
flags.DEFINE_string(
    'model_export_path', None,
    'Path to the directory, where trainined model will be '
    'exported.')
flags.DEFINE_enum(
    'strategy_type',
    'mirror',
    ['tpu', 'mirror'],
    'Distribution Strategy type to use for training. `tpu` uses '
    'TPUStrategy for running on TPUs, `mirror` uses GPUs with '
    'single host.')
# Model training specific flags.
flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')
flags.DEFINE_integer('train_batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('eval_batch_size', 8, 'Batch size for evaluation.')
flags.DEFINE_integer('num_train_epochs', 3,
                     'Total number of training epochs to perform.')
flags.DEFINE_integer('steps_per_run', 200,
                     'Number of steps running on TPU devices.')
flags.DEFINE_float('learning_rate', 5e-5, 'The initial learning rate for Adam.')

FLAGS = flags.FLAGS


def get_loss_fn(num_classes, loss_scale=1.0):
  """Gets the classification loss function."""

  def classification_loss_fn(labels, logits):
    """Classification loss."""
    labels = tf.squeeze(labels)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(
        tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(
        tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    loss *= loss_scale
    return loss

  return classification_loss_fn


def run_customized_training(strategy,
                            bert_config,
                            input_meta_data,
                            model_dir,
                            epochs,
                            steps_per_epoch,
                            eval_steps,
                            warmup_steps,
                            initial_lr,
                            init_checkpoint,
                            use_remote_tpu=False):
  """Run BERT classifier training using low-level API."""
  max_seq_length = input_meta_data['max_seq_length']
  num_classes = input_meta_data['num_labels']

  train_input_fn = functools.partial(
      input_pipeline.create_classifier_dataset,
      FLAGS.train_data_path,
      seq_length=max_seq_length,
      batch_size=FLAGS.train_batch_size)
  eval_input_fn = functools.partial(
      input_pipeline.create_classifier_dataset,
      FLAGS.eval_data_path,
      seq_length=max_seq_length,
      batch_size=FLAGS.eval_batch_size,
      is_training=False,
      drop_remainder=False)

  def _get_classifier_model():
    classifier_model, core_model = (
        bert_models.classifier_model(bert_config, tf.float32, num_classes,
                                     max_seq_length))
    classifier_model.optimizer = optimization.create_optimizer(
        initial_lr, steps_per_epoch * epochs, warmup_steps)
    return classifier_model, core_model

  loss_fn = get_loss_fn(num_classes, loss_scale=1.0)

  # Defines evaluation metrics function, which will create metrics in the
  # correct device and strategy scope.
  def metric_fn():
    return tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)

  return model_training_utils.run_customized_training_loop(
      strategy=strategy,
      model_fn=_get_classifier_model,
      loss_fn=loss_fn,
      model_dir=model_dir,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      eval_steps=eval_steps,
      init_checkpoint=init_checkpoint,
      metric_fn=metric_fn,
      use_remote_tpu=use_remote_tpu)


def export_classifier(model_export_path, input_meta_data):
  """Exports a trained model as a `SavedModel` for inference.

  Args:
    model_export_path: a string specifying the path to the SavedModel directory.
    input_meta_data: dictionary containing meta data about input and model.

  Raises:
    Export path is not specified, got an empty string or None.
  """
  if not model_export_path:
    raise ValueError('Export path is not specified: %s' % model_export_path)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  def _model_fn():
    return bert_models.classifier_model(bert_config, tf.float32,
                                        input_meta_data['num_labels'],
                                        input_meta_data['max_seq_length'])[0]

  model_saving_utils.export_bert_model(
      model_export_path, model_fn=_model_fn, checkpoint_dir=FLAGS.model_dir)


def run_bert(strategy, input_meta_data):
  """Run BERT training."""
  if FLAGS.mode == 'export_only':
    export_classifier(FLAGS.model_export_path, input_meta_data)
    return

  if FLAGS.mode != 'train_and_eval':
    raise ValueError('Unsupported mode is specified: %s' % FLAGS.mode)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  epochs = FLAGS.num_train_epochs
  train_data_size = input_meta_data['train_data_size']
  steps_per_epoch = int(train_data_size / FLAGS.train_batch_size)
  warmup_steps = int(epochs * train_data_size * 0.1 / FLAGS.train_batch_size)
  eval_steps = int(
      math.ceil(input_meta_data['eval_data_size'] / FLAGS.eval_batch_size))

  if not strategy:
    raise ValueError('Distribution strategy has not been specified.')
  # Runs customized training loop.
  logging.info('Training using customized training loop TF 2.0 with distrubuted'
               'strategy.')
  use_remote_tpu = (FLAGS.strategy_type == 'tpu' and FLAGS.tpu)
  trained_model = run_customized_training(
      strategy,
      bert_config,
      input_meta_data,
      FLAGS.model_dir,
      epochs,
      steps_per_epoch,
      eval_steps,
      warmup_steps,
      FLAGS.learning_rate,
      FLAGS.init_checkpoint,
      use_remote_tpu=use_remote_tpu)

  if FLAGS.model_export_path:
    model_saving_utils.export_bert_model(
        FLAGS.model_export_path, model=trained_model)


def main(_):
  # Users should always run this script under TF 2.x
  assert tf.version.VERSION.startswith('2.')
  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  if not FLAGS.model_dir:
    FLAGS.model_dir = '/tmp/bert20/'

  strategy = None
  if FLAGS.strategy_type == 'mirror':
    strategy = tf.distribute.MirroredStrategy()
  elif FLAGS.strategy_type == 'tpu':
    logging.info('Use TPU at %s', FLAGS.tpu if FLAGS.tpu is not None else '')
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_host(cluster_resolver.master())  # pylint: disable=line-too-long
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.experimental.TPUStrategy(
        cluster_resolver, steps_per_run=FLAGS.steps_per_run)

  run_bert(strategy, input_meta_data)


if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('input_meta_data_path')
  app.run(main)
