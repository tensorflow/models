# Copyright 2017 Google Inc.
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

r"""Trains the classification/pose baselines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

# Dependency imports

import tensorflow as tf

from domain_adaptation.datasets import dataset_factory
from domain_adaptation.pixel_domain_adaptation import pixelda_preprocess
from domain_adaptation.pixel_domain_adaptation import pixelda_task_towers

flags = tf.app.flags
FLAGS = flags.FLAGS

slim = tf.contrib.slim

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('task', 0, 'The task ID.')

flags.DEFINE_integer('num_ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')

flags.DEFINE_integer('batch_size', 32, 'The number of samples per batch.')

flags.DEFINE_string('dataset_name', None, 'The name of the dataset.')

flags.DEFINE_string('dataset_dir', None,
                    'The directory where the data is stored.')

flags.DEFINE_string('split_name', None, 'The name of the train/test split.')

flags.DEFINE_float('learning_rate', 0.001, 'The initial learning rate.')

flags.DEFINE_integer(
    'learning_rate_decay_steps', 20000,
    'The frequency, in steps, at which the learning rate is decayed.')

flags.DEFINE_float('learning_rate_decay_factor',
                   0.95,
                   'The factor with which the learning rate is decayed.')

flags.DEFINE_float('adam_beta1', 0.5, 'The beta1 value for the AdamOptimizer')

flags.DEFINE_float('weight_decay', 1e-5,
                   'The L2 coefficient on the model weights.')

flags.DEFINE_string(
    'logdir', None, 'The location of the logs and checkpoints.')

flags.DEFINE_integer('save_interval_secs', 600,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summaries_secs', 600,
                     'How often, in seconds, we compute the summaries.')

flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

flags.DEFINE_float(
    'moving_average_decay', 0.9999,
    'The amount of decay to use for moving averages.')


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  hparams = tf.contrib.training.HParams()
  hparams.weight_decay_task_classifier = FLAGS.weight_decay

  if FLAGS.dataset_name in ['mnist', 'mnist_m', 'usps']:
    hparams.task_tower = 'mnist'
  else:
    raise ValueError('Unknown dataset %s' % FLAGS.dataset_name)

  with tf.Graph().as_default():
    with tf.device(
        tf.train.replica_device_setter(FLAGS.num_ps_tasks, merge_devices=True)):
      dataset = dataset_factory.get_dataset(FLAGS.dataset_name,
                                            FLAGS.split_name, FLAGS.dataset_dir)
      num_classes = dataset.num_classes

      preprocess_fn = partial(pixelda_preprocess.preprocess_classification,
                              is_training=True)

      images, labels = dataset_factory.provide_batch(
          FLAGS.dataset_name,
          FLAGS.split_name,
          dataset_dir=FLAGS.dataset_dir,
          num_readers=FLAGS.num_readers,
          batch_size=FLAGS.batch_size,
          num_preprocessing_threads=FLAGS.num_readers)
      # preprocess_fn=preprocess_fn)

      # Define the model
      logits, _ = pixelda_task_towers.add_task_specific_model(
          images, hparams, num_classes=num_classes, is_training=True)

      # Define the losses
      if 'classes' in labels:
        one_hot_labels = labels['classes']
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=one_hot_labels, logits=logits)
        tf.summary.scalar('losses/Classification_Loss', loss)
      else:
        raise ValueError('Only support classification for now.')

      total_loss = tf.losses.get_total_loss()
      tf.summary.scalar('losses/Total_Loss', total_loss)

      # Setup the moving averages
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, slim.get_or_create_global_step())
      tf.add_to_collection(
          tf.GraphKeys.UPDATE_OPS,
          variable_averages.apply(moving_average_variables))

      # Specify the optimization scheme:
      learning_rate = tf.train.exponential_decay(
          FLAGS.learning_rate,
          slim.get_or_create_global_step(),
          FLAGS.learning_rate_decay_steps,
          FLAGS.learning_rate_decay_factor,
          staircase=True)

      optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.adam_beta1)

      train_op = slim.learning.create_train_op(total_loss, optimizer)

      slim.learning.train(
          train_op,
          FLAGS.logdir,
          master=FLAGS.master,
          is_chief=(FLAGS.task == 0),
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
  tf.app.run()
