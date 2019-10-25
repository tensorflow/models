# Copyright 2018 Google Inc. All Rights Reserved.
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

"""Program which runs evaluation of Imagenet 64x64 and TinyImagenet models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import tensorflow as tf

import adversarial_attack
import model_lib
from datasets import dataset_factory

FLAGS = flags.FLAGS


flags.DEFINE_string('train_dir', None,
                    'Training directory. If specified then this program '
                    'runs in continuous evaluation mode.')

flags.DEFINE_string('checkpoint_path', None,
                    'Path to the file with checkpoint. If specified then '
                    'this program evaluates only provided checkpoint one time.')

flags.DEFINE_string('output_file', None,
                    'Name of output file. Used only in single evaluation mode.')

flags.DEFINE_string('eval_name', 'default', 'Name for eval subdirectory.')

flags.DEFINE_string('master', '', 'Tensorflow master.')

flags.DEFINE_string('model_name', 'resnet_v2_50', 'Name of the model.')

flags.DEFINE_string('adv_method', 'clean',
                    'Method which is used to generate adversarial examples.')

flags.DEFINE_string('dataset', 'imagenet',
                    'Dataset: "tiny_imagenet" or "imagenet".')

flags.DEFINE_integer('dataset_image_size', 64,
                     'Size of the images in the dataset.')

flags.DEFINE_string('hparams', '', 'Hyper parameters.')

flags.DEFINE_string('split_name', 'validation', 'Name of the split.')

flags.DEFINE_float('moving_average_decay', 0.9999,
                   'The decay to use for the moving average.')

flags.DEFINE_integer('eval_interval_secs', 120,
                     'The frequency, in seconds, with which evaluation is run.')

flags.DEFINE_integer(
    'num_examples', -1,
    'If positive - maximum number of example to use for evaluation.')

flags.DEFINE_bool('eval_once', False,
                  'If true then evaluate model only once.')

flags.DEFINE_string('trainable_scopes', None,
                    'If set then it defines list of variable scopes for '
                    'trainable variables.')


def main(_):
  if not FLAGS.train_dir and not FLAGS.checkpoint_path:
    print('Either --train_dir or --checkpoint_path flags has to be provided.')
  if FLAGS.train_dir and FLAGS.checkpoint_path:
    print('Only one of --train_dir or --checkpoint_path should be provided.')
  params = model_lib.default_hparams()
  params.parse(FLAGS.hparams)
  tf.logging.info('User provided hparams: %s', FLAGS.hparams)
  tf.logging.info('All hyper parameters: %s', params)
  batch_size = params.eval_batch_size
  graph = tf.Graph()
  with graph.as_default():
    # dataset
    dataset, num_examples, num_classes, bounds = dataset_factory.get_dataset(
        FLAGS.dataset,
        FLAGS.split_name,
        batch_size,
        FLAGS.dataset_image_size,
        is_training=False)
    dataset_iterator = dataset.make_one_shot_iterator()
    images, labels = dataset_iterator.get_next()
    if FLAGS.num_examples > 0:
      num_examples = min(num_examples, FLAGS.num_examples)

    # setup model
    global_step = tf.train.get_or_create_global_step()
    model_fn_two_args = model_lib.get_model(FLAGS.model_name, num_classes)
    model_fn = lambda x: model_fn_two_args(x, is_training=False)
    if not FLAGS.adv_method or FLAGS.adv_method == 'clean':
      logits = model_fn(images)
    else:
      adv_examples = adversarial_attack.generate_adversarial_examples(
          images, bounds, model_fn, FLAGS.adv_method)
      logits = model_fn(adv_examples)

    # update trainable variables if fine tuning is used
    model_lib.filter_trainable_variables(FLAGS.trainable_scopes)

    # Setup the moving averages
    if FLAGS.moving_average_decay and (FLAGS.moving_average_decay > 0):
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          tf.contrib.framework.get_model_variables())
      variables_to_restore[global_step.op.name] = global_step
    else:
      variables_to_restore = tf.contrib.framework.get_variables_to_restore()

    # Setup evaluation metric
    with tf.name_scope('Eval'):
      names_to_values, names_to_updates = (
          tf.contrib.metrics.aggregate_metric_map({
              'Accuracy': tf.metrics.accuracy(labels, tf.argmax(logits, 1)),
              'Top5': tf.metrics.recall_at_k(tf.to_int64(labels), logits, 5)
          }))

      for name, value in names_to_values.iteritems():
        tf.summary.scalar(name, value)

    # Run evaluation
    num_batches = int(num_examples / batch_size)
    if FLAGS.train_dir:
      output_dir = os.path.join(FLAGS.train_dir, FLAGS.eval_name)
      if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
      tf.contrib.training.evaluate_repeatedly(
          FLAGS.train_dir,
          master=FLAGS.master,
          scaffold=tf.train.Scaffold(
              saver=tf.train.Saver(variables_to_restore)),
          eval_ops=names_to_updates.values(),
          eval_interval_secs=FLAGS.eval_interval_secs,
          hooks=[
              tf.contrib.training.StopAfterNEvalsHook(num_batches),
              tf.contrib.training.SummaryAtEndHook(output_dir),
              tf.train.LoggingTensorHook(names_to_values, at_end=True),
          ],
          max_number_of_evaluations=1 if FLAGS.eval_once else None)
    else:
      result = tf.contrib.training.evaluate_once(
          FLAGS.checkpoint_path,
          master=FLAGS.master,
          scaffold=tf.train.Scaffold(
              saver=tf.train.Saver(variables_to_restore)),
          eval_ops=names_to_updates.values(),
          final_ops=names_to_values,
          hooks=[
              tf.contrib.training.StopAfterNEvalsHook(num_batches),
              tf.train.LoggingTensorHook(names_to_values, at_end=True),
          ])
      if FLAGS.output_file:
        with tf.gfile.Open(FLAGS.output_file, 'a') as f:
          f.write('%s,%.3f,%.3f\n'
                  % (FLAGS.eval_name, result['Accuracy'], result['Top5']))


if __name__ == '__main__':
  app.run(main)
