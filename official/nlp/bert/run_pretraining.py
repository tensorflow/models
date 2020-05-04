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
"""Run masked LM/next sentence pre-training for BERT in TF 2.x."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf
from official.modeling import performance
from official.nlp import optimization
from official.nlp.bert import bert_models
from official.nlp.bert import common_flags
from official.nlp.bert import configs
from official.nlp.bert import input_pipeline
from official.nlp.bert import model_training_utils
from official.utils.misc import distribution_utils


flags.DEFINE_string('input_files', None,
                    'File path to retrieve training data for pre-training.')
# Model training specific flags.
flags.DEFINE_integer(
    'max_seq_length', 128,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')
flags.DEFINE_integer('max_predictions_per_seq', 20,
                     'Maximum predictions per sequence_output.')
flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')
flags.DEFINE_integer('num_steps_per_epoch', 1000,
                     'Total number of training steps to run per epoch.')
flags.DEFINE_float('warmup_steps', 10000,
                   'Warmup steps for Adam weight decay optimizer.')
flags.DEFINE_bool('use_next_sentence_label', True,
                  'Whether to use next sentence label to compute final loss.')

common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS


def get_pretrain_dataset_fn(input_file_pattern, seq_length,
                            max_predictions_per_seq, global_batch_size,
                            use_next_sentence_label=True):
  """Returns input dataset from input file string."""
  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    input_patterns = input_file_pattern.split(',')
    batch_size = ctx.get_per_replica_batch_size(global_batch_size)
    train_dataset = input_pipeline.create_pretrain_dataset(
        input_patterns,
        seq_length,
        max_predictions_per_seq,
        batch_size,
        is_training=True,
        input_pipeline_context=ctx,
        use_next_sentence_label=use_next_sentence_label)
    return train_dataset

  return _dataset_fn


def get_loss_fn():
  """Returns loss function for BERT pretraining."""

  def _bert_pretrain_loss_fn(unused_labels, losses, **unused_args):
    return tf.reduce_mean(losses)

  return _bert_pretrain_loss_fn


def run_customized_training(strategy,
                            bert_config,
                            init_checkpoint,
                            max_seq_length,
                            max_predictions_per_seq,
                            model_dir,
                            steps_per_epoch,
                            steps_per_loop,
                            epochs,
                            initial_lr,
                            warmup_steps,
                            end_lr,
                            optimizer_type,
                            input_files,
                            train_batch_size,
                            use_next_sentence_label=True,
                            custom_callbacks=None):
  """Run BERT pretrain model training using low-level API."""

  train_input_fn = get_pretrain_dataset_fn(input_files, max_seq_length,
                                           max_predictions_per_seq,
                                           train_batch_size,
                                           use_next_sentence_label)

  def _get_pretrain_model():
    """Gets a pretraining model."""
    pretrain_model, core_model = bert_models.pretrain_model(
        bert_config, max_seq_length, max_predictions_per_seq,
        use_next_sentence_label=use_next_sentence_label)
    optimizer = optimization.create_optimizer(
        initial_lr, steps_per_epoch * epochs, warmup_steps,
        end_lr, optimizer_type)
    pretrain_model.optimizer = performance.configure_optimizer(
        optimizer,
        use_float16=common_flags.use_float16(),
        use_graph_rewrite=common_flags.use_graph_rewrite())
    return pretrain_model, core_model

  trained_model = model_training_utils.run_customized_training_loop(
      strategy=strategy,
      model_fn=_get_pretrain_model,
      loss_fn=get_loss_fn(),
      scale_loss=FLAGS.scale_loss,
      model_dir=model_dir,
      init_checkpoint=init_checkpoint,
      train_input_fn=train_input_fn,
      steps_per_epoch=steps_per_epoch,
      steps_per_loop=steps_per_loop,
      epochs=epochs,
      sub_model_export_name='pretrained/bert_model',
      custom_callbacks=custom_callbacks)

  return trained_model


def run_bert_pretrain(strategy, custom_callbacks=None):
  """Runs BERT pre-training."""

  bert_config = configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  if not strategy:
    raise ValueError('Distribution strategy is not specified.')

  # Runs customized training loop.
  logging.info('Training using customized training loop TF 2.0 with distributed'
               'strategy.')

  performance.set_mixed_precision_policy(common_flags.dtype())

  return run_customized_training(
      strategy,
      bert_config,
      FLAGS.init_checkpoint,  # Used to initialize only the BERT submodel.
      FLAGS.max_seq_length,
      FLAGS.max_predictions_per_seq,
      FLAGS.model_dir,
      FLAGS.num_steps_per_epoch,
      FLAGS.steps_per_loop,
      FLAGS.num_train_epochs,
      FLAGS.learning_rate,
      FLAGS.warmup_steps,
      FLAGS.end_lr,
      FLAGS.optimizer_type,
      FLAGS.input_files,
      FLAGS.train_batch_size,
      FLAGS.use_next_sentence_label,
      custom_callbacks=custom_callbacks)


def main(_):
  # Users should always run this script under TF 2.x
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  if not FLAGS.model_dir:
    FLAGS.model_dir = '/tmp/bert20/'
  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      tpu_address=FLAGS.tpu)
  if strategy:
    print('***** Number of cores used : ', strategy.num_replicas_in_sync)

  run_bert_pretrain(strategy)


if __name__ == '__main__':
  app.run(main)
