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

r"""Evals the classification/pose baselines."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import math

# Dependency imports

import tensorflow as tf

from domain_adaptation.datasets import dataset_factory
from domain_adaptation.pixel_domain_adaptation import pixelda_preprocess
from domain_adaptation.pixel_domain_adaptation import pixelda_task_towers

flags = tf.app.flags
FLAGS = flags.FLAGS

slim = tf.contrib.slim

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_string(
    'checkpoint_dir', None, 'The location of the checkpoint files.')

flags.DEFINE_string(
    'eval_dir', None, 'The directory where evaluation logs are written.')

flags.DEFINE_integer('batch_size', 32, 'The number of samples per batch.')

flags.DEFINE_string('dataset_name', None, 'The name of the dataset.')

flags.DEFINE_string('dataset_dir', None,
                    'The directory where the data is stored.')

flags.DEFINE_string('split_name', None, 'The name of the train/test split.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  hparams = tf.contrib.training.HParams()
  hparams.weight_decay_task_classifier = 0.0

  if FLAGS.dataset_name in ['mnist', 'mnist_m', 'usps']:
    hparams.task_tower = 'mnist'
  else:
    raise ValueError('Unknown dataset %s' % FLAGS.dataset_name)

  if not tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.MakeDirs(FLAGS.eval_dir)

  with tf.Graph().as_default():
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.split_name,
                                          FLAGS.dataset_dir)
    num_classes = dataset.num_classes
    num_samples = dataset.num_samples

    preprocess_fn = partial(pixelda_preprocess.preprocess_classification,
                            is_training=False)

    images, labels = dataset_factory.provide_batch(
        FLAGS.dataset_name,
        FLAGS.split_name,
        dataset_dir=FLAGS.dataset_dir,
        num_readers=FLAGS.num_readers,
        batch_size=FLAGS.batch_size,
        num_preprocessing_threads=FLAGS.num_readers)

    # Define the model
    logits, _ = pixelda_task_towers.add_task_specific_model(
        images, hparams, num_classes=num_classes, is_training=True)

    #####################
    # Define the losses #
    #####################
    if 'classes' in labels:
      one_hot_labels = labels['classes']
      loss = tf.losses.softmax_cross_entropy(
          onehot_labels=one_hot_labels, logits=logits)
      tf.summary.scalar('losses/Classification_Loss', loss)
    else:
      raise ValueError('Only support classification for now.')

    total_loss = tf.losses.get_total_loss()

    predictions = tf.reshape(tf.argmax(logits, 1), shape=[-1])
    class_labels = tf.argmax(labels['classes'], 1)

    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
        'Mean_Loss':
            tf.contrib.metrics.streaming_mean(total_loss),
        'Accuracy':
            tf.contrib.metrics.streaming_accuracy(predictions,
                                                  tf.reshape(
                                                      class_labels,
                                                      shape=[-1])),
        'Recall_at_5':
            tf.contrib.metrics.streaming_recall_at_k(logits, class_labels, 5),
    })

    tf.summary.histogram('outputs/Predictions', predictions)
    tf.summary.histogram('outputs/Ground_Truth', class_labels)

    for name, value in metrics_to_values.iteritems():
      tf.summary.scalar(name, value)

    num_batches = int(math.ceil(num_samples / float(FLAGS.batch_size)))

    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=metrics_to_updates.values(),
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
  tf.app.run()
