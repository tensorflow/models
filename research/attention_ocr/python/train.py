# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Script to train the Attention OCR model.

A simple usage example:
python train.py
"""
import collections
import logging
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import app
from tensorflow.python.platform import flags
from tensorflow.contrib.tfprof import model_analyzer

import data_provider
import common_flags

FLAGS = flags.FLAGS
common_flags.define()

# yapf: disable
flags.DEFINE_integer('task', 0,
                     'The Task ID. This value is used when training with '
                     'multiple workers to identify each worker.')

flags.DEFINE_integer('ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then'
                     ' the parameters are handled locally by the worker.')

flags.DEFINE_integer('save_summaries_secs', 60,
                     'The frequency with which summaries are saved, in '
                     'seconds.')

flags.DEFINE_integer('save_interval_secs', 600,
                     'Frequency in seconds of saving the model.')

flags.DEFINE_integer('max_number_of_steps', int(1e10),
                     'The maximum number of gradient steps.')

flags.DEFINE_string('checkpoint_inception', '',
                    'Checkpoint to recover inception weights from.')

flags.DEFINE_float('clip_gradient_norm', 2.0,
                   'If greater than 0 then the gradients would be clipped by '
                   'it.')

flags.DEFINE_bool('sync_replicas', False,
                  'If True will synchronize replicas during training.')

flags.DEFINE_integer('replicas_to_aggregate', 1,
                     'The number of gradients updates before updating params.')

flags.DEFINE_integer('total_num_replicas', 1,
                     'Total number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_boolean('reset_train_dir', False,
                     'If true will delete all files in the train_log_dir')

flags.DEFINE_boolean('show_graph_stats', False,
                     'Output model size stats to stderr.')
# yapf: enable

TrainingHParams = collections.namedtuple('TrainingHParams', [
    'learning_rate',
    'optimizer',
    'momentum',
    'use_augment_input',
])


def get_training_hparams():
  return TrainingHParams(
      learning_rate=FLAGS.learning_rate,
      optimizer=FLAGS.optimizer,
      momentum=FLAGS.momentum,
      use_augment_input=FLAGS.use_augment_input)


def create_optimizer(hparams):
  """Creates optimized based on the specified flags."""
  if hparams.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        hparams.learning_rate, momentum=hparams.momentum)
  elif hparams.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(hparams.learning_rate)
  elif hparams.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(hparams.learning_rate)
  elif hparams.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(hparams.learning_rate)
  elif hparams.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        hparams.learning_rate, momentum=hparams.momentum)
  return optimizer


def train(loss, init_fn, hparams):
  """Wraps slim.learning.train to run a training loop.

  Args:
    loss: a loss tensor
    init_fn: A callable to be executed after all other initialization is done.
    hparams: a model hyper parameters
  """
  optimizer = create_optimizer(hparams)

  if FLAGS.sync_replicas:
    replica_id = tf.constant(FLAGS.task, tf.int32, shape=())
    optimizer = tf.LegacySyncReplicasOptimizer(
        opt=optimizer,
        replicas_to_aggregate=FLAGS.replicas_to_aggregate,
        replica_id=replica_id,
        total_num_replicas=FLAGS.total_num_replicas)
    sync_optimizer = optimizer
    startup_delay_steps = 0
  else:
    startup_delay_steps = 0
    sync_optimizer = None

  train_op = slim.learning.create_train_op(
      loss,
      optimizer,
      summarize_gradients=True,
      clip_gradient_norm=FLAGS.clip_gradient_norm)

  slim.learning.train(
      train_op=train_op,
      logdir=FLAGS.train_log_dir,
      graph=loss.graph,
      master=FLAGS.master,
      is_chief=(FLAGS.task == 0),
      number_of_steps=FLAGS.max_number_of_steps,
      save_summaries_secs=FLAGS.save_summaries_secs,
      save_interval_secs=FLAGS.save_interval_secs,
      startup_delay_steps=startup_delay_steps,
      sync_optimizer=sync_optimizer,
      init_fn=init_fn)


def prepare_training_dir():
  if not tf.gfile.Exists(FLAGS.train_log_dir):
    logging.info('Create a new training directory %s', FLAGS.train_log_dir)
    tf.gfile.MakeDirs(FLAGS.train_log_dir)
  else:
    if FLAGS.reset_train_dir:
      logging.info('Reset the training directory %s', FLAGS.train_log_dir)
      tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
      tf.gfile.MakeDirs(FLAGS.train_log_dir)
    else:
      logging.info('Use already existing training directory %s',
                   FLAGS.train_log_dir)


def calculate_graph_metrics():
  param_stats = model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  return param_stats.total_parameters


def main(_):
  prepare_training_dir()

  dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
  model = common_flags.create_model(dataset.num_char_classes,
                                    dataset.max_sequence_length,
                                    dataset.num_of_views, dataset.null_code)
  hparams = get_training_hparams()

  # If ps_tasks is zero, the local device is used. When using multiple
  # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
  # across the different devices.
  device_setter = tf.train.replica_device_setter(
      FLAGS.ps_tasks, merge_devices=True)
  with tf.device(device_setter):
    data = data_provider.get_data(
        dataset,
        FLAGS.batch_size,
        augment=hparams.use_augment_input,
        central_crop_size=common_flags.get_crop_size())
    endpoints = model.create_base(data.images, data.labels_one_hot)
    total_loss = model.create_loss(data, endpoints)
    model.create_summaries(data, endpoints, dataset.charset, is_training=True)
    init_fn = model.create_init_fn_to_restore(FLAGS.checkpoint,
                                              FLAGS.checkpoint_inception)
    if FLAGS.show_graph_stats:
      logging.info('Total number of weights in the graph: %s',
                   calculate_graph_metrics())
    train(total_loss, init_fn, hparams)


if __name__ == '__main__':
  app.run()
