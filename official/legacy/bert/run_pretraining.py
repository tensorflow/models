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

"""Run masked LM/next sentence pre-training for BERT in TF 2.x."""

# Import libraries
from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf
from official.common import distribute_utils
from official.legacy.bert import bert_models
from official.legacy.bert import common_flags
from official.legacy.bert import configs
from official.legacy.bert import input_pipeline
from official.legacy.bert import model_training_utils
from official.modeling import performance
from official.nlp import optimization


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
flags.DEFINE_bool('train_summary_interval', 0, 'Step interval for training '
                  'summaries. If the value is a negative number, '
                  'then training summaries are not enabled.')

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
                            train_summary_interval=0,
                            custom_callbacks=None,
                            explicit_allreduce=False,
                            pre_allreduce_callbacks=None,
                            post_allreduce_callbacks=None,
                            allreduce_bytes_per_pack=0):
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
        use_float16=common_flags.use_float16())
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
      explicit_allreduce=explicit_allreduce,
      pre_allreduce_callbacks=pre_allreduce_callbacks,
      post_allreduce_callbacks=post_allreduce_callbacks,
      allreduce_bytes_per_pack=allreduce_bytes_per_pack,
      train_summary_interval=train_summary_interval,
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

  # Only when explicit_allreduce = True, post_allreduce_callbacks and
  # allreduce_bytes_per_pack will take effect. optimizer.apply_gradients() no
  # longer implicitly allreduce gradients, users manually allreduce gradient and
  # pass the allreduced grads_and_vars to apply_gradients().
  # With explicit_allreduce = True, clip_by_global_norm is moved to after
  # allreduce.
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
      FLAGS.train_summary_interval,
      custom_callbacks=custom_callbacks,
      explicit_allreduce=FLAGS.explicit_allreduce,
      pre_allreduce_callbacks=[
          model_training_utils.clip_by_global_norm_callback
      ],
      allreduce_bytes_per_pack=FLAGS.allreduce_bytes_per_pack)


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  if not FLAGS.model_dir:
    FLAGS.model_dir = '/tmp/bert20/'
  # Configures cluster spec for multi-worker distribution strategy.
  if FLAGS.num_gpus > 0:
    _ = distribute_utils.configure_cluster(FLAGS.worker_hosts, FLAGS.task_index)
  strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      all_reduce_alg=FLAGS.all_reduce_alg,
      tpu_address=FLAGS.tpu)
  if strategy:
    print('***** Number of cores used : ', strategy.num_replicas_in_sync)

  run_bert_pretrain(strategy)


if __name__ == '__main__':
  app.run(main)
