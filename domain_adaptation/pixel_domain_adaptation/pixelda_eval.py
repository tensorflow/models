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

r"""Evaluates the PIXELDA model.

-- Compiles the model for CPU.
$ bazel build -c opt third_party/tensorflow_models/domain_adaptation/pixel_domain_adaptation:pixelda_eval

-- Compile the model for GPU.
$ bazel build -c opt --copt=-mavx --config=cuda \
    third_party/tensorflow_models/domain_adaptation/pixel_domain_adaptation:pixelda_eval

-- Runs the training.
$ ./bazel-bin/third_party/tensorflow_models/domain_adaptation/pixel_domain_adaptation/pixelda_eval \
    --source_dataset=mnist \
    --target_dataset=mnist_m \
    --dataset_dir=/tmp/datasets/ \
    --alsologtostderr

-- Visualize the results.
$ bash learning/brain/tensorboard/tensorboard.sh \
    --port 2222 --logdir=/tmp/pixelda/
"""
from functools import partial
import math

# Dependency imports

import tensorflow as tf

from domain_adaptation.datasets import dataset_factory
from domain_adaptation.pixel_domain_adaptation import pixelda_model
from domain_adaptation.pixel_domain_adaptation import pixelda_preprocess
from domain_adaptation.pixel_domain_adaptation import pixelda_utils
from domain_adaptation.pixel_domain_adaptation import pixelda_losses
from domain_adaptation.pixel_domain_adaptation.hparams import create_hparams

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')

flags.DEFINE_string('checkpoint_dir', '/tmp/pixelda/',
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', '/tmp/pixelda/',
                    'Directory where the results are saved to.')

flags.DEFINE_integer('eval_interval_secs', 60,
                     'The frequency, in seconds, with which evaluation is run.')

flags.DEFINE_string('target_split_name', 'test',
                    'The name of the train/test split.')
flags.DEFINE_string('source_split_name', 'train', 'Split for source dataset.'
                    ' Defaults to train.')

flags.DEFINE_string('source_dataset', 'mnist',
                    'The name of the source dataset.')

flags.DEFINE_string('target_dataset', 'mnist_m',
                    'The name of the target dataset.')

flags.DEFINE_string(
    'dataset_dir',
    '',  # None,
    'The directory where the datasets can be found.')

flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

flags.DEFINE_integer('num_preprocessing_threads', 4,
                     'The number of threads used to create the batches.')

# HParams

flags.DEFINE_string('hparams', '', 'Comma separated hyperparameter values')


def run_eval(run_dir, checkpoint_dir, hparams):
  """Runs the eval loop.

  Args:
    run_dir: The directory where eval specific logs are placed
    checkpoint_dir: The directory where the checkpoints are stored
    hparams: The hyperparameters struct.

  Raises:
    ValueError: if hparams.arch is not recognized.
  """
  for checkpoint_path in slim.evaluation.checkpoints_iterator(
      checkpoint_dir, FLAGS.eval_interval_secs):
    with tf.Graph().as_default():
      #########################
      # Preprocess the inputs #
      #########################
      target_dataset = dataset_factory.get_dataset(
          FLAGS.target_dataset,
          split_name=FLAGS.target_split_name,
          dataset_dir=FLAGS.dataset_dir)
      target_images, target_labels = dataset_factory.provide_batch(
          FLAGS.target_dataset, FLAGS.target_split_name, FLAGS.dataset_dir,
          FLAGS.num_readers, hparams.batch_size,
          FLAGS.num_preprocessing_threads)
      num_target_classes = target_dataset.num_classes
      target_labels['class'] = tf.argmax(target_labels['classes'], 1)
      del target_labels['classes']

      if hparams.arch not in ['dcgan']:
        source_dataset = dataset_factory.get_dataset(
            FLAGS.source_dataset,
            split_name=FLAGS.source_split_name,
            dataset_dir=FLAGS.dataset_dir)
        num_source_classes = source_dataset.num_classes
        source_images, source_labels = dataset_factory.provide_batch(
            FLAGS.source_dataset, FLAGS.source_split_name, FLAGS.dataset_dir,
            FLAGS.num_readers, hparams.batch_size,
            FLAGS.num_preprocessing_threads)
        source_labels['class'] = tf.argmax(source_labels['classes'], 1)
        del source_labels['classes']
        if num_source_classes != num_target_classes:
          raise ValueError(
              'Input and output datasets must have same number of classes')
      else:
        source_images = None
        source_labels = None

      ####################
      # Define the model #
      ####################
      end_points = pixelda_model.create_model(
          hparams,
          target_images,
          source_images=source_images,
          source_labels=source_labels,
          is_training=False,
          num_classes=num_target_classes)

      #######################
      # Metrics & Summaries #
      #######################
      names_to_values, names_to_updates = create_metrics(end_points,
                                                         source_labels,
                                                         target_labels, hparams)
      pixelda_utils.summarize_model(end_points)
      pixelda_utils.summarize_transferred_grid(
          end_points['transferred_images'], source_images, name='Transferred')
      if 'source_images_recon' in end_points:
        pixelda_utils.summarize_transferred_grid(
            end_points['source_images_recon'],
            source_images,
            name='Source Reconstruction')
      pixelda_utils.summarize_images(target_images, 'Target')

      for name, value in names_to_values.iteritems():
        tf.summary.scalar(name, value)

      # Use the entire split by default
      num_examples = target_dataset.num_samples

      num_batches = math.ceil(num_examples / float(hparams.batch_size))
      global_step = slim.get_or_create_global_step()

      result = slim.evaluation.evaluate_once(
          master=FLAGS.master,
          checkpoint_path=checkpoint_path,
          logdir=run_dir,
          num_evals=num_batches,
          eval_op=names_to_updates.values(),
          final_op=names_to_values)


def to_degrees(log_quaternion_loss):
  """Converts a log quaternion distance to an angle.

  Args:
    log_quaternion_loss: The log quaternion distance between two
      unit quaternions (or a batch of pairs of quaternions).

  Returns:
    The angle in degrees of the implied angle-axis representation.
  """
  return tf.acos(-(tf.exp(log_quaternion_loss) - 1)) * 2 * 180 / math.pi


def create_metrics(end_points, source_labels, target_labels, hparams):
  """Create metrics for the model.

  Args:
    end_points: A dictionary of end point name to tensor
    source_labels: Labels for source images. batch_size x 1
    target_labels: Labels for target images. batch_size x 1
    hparams: The hyperparameters struct.

  Returns:
    Tuple of (names_to_values, names_to_updates), dictionaries that map a metric
    name to its value and update op, respectively

  """
  ###########################################
  # Evaluate the Domain Prediction Accuracy #
  ###########################################
  batch_size = hparams.batch_size
  names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      ('eval/Domain_Accuracy-Transferred'):
          tf.contrib.metrics.streaming_accuracy(
              tf.to_int32(
                  tf.round(tf.sigmoid(end_points[
                      'transferred_domain_logits']))),
              tf.zeros(batch_size, dtype=tf.int32)),
      ('eval/Domain_Accuracy-Target'):
          tf.contrib.metrics.streaming_accuracy(
              tf.to_int32(
                  tf.round(tf.sigmoid(end_points['target_domain_logits']))),
              tf.ones(batch_size, dtype=tf.int32))
  })

  ################################
  # Evaluate the task classifier #
  ################################
  if 'source_task_logits' in end_points:
    metric_name = 'eval/Task_Accuracy-Source'
    names_to_values[metric_name], names_to_updates[
        metric_name] = tf.contrib.metrics.streaming_accuracy(
            tf.argmax(end_points['source_task_logits'], 1),
            source_labels['class'])

  if 'transferred_task_logits' in end_points:
    metric_name = 'eval/Task_Accuracy-Transferred'
    names_to_values[metric_name], names_to_updates[
        metric_name] = tf.contrib.metrics.streaming_accuracy(
            tf.argmax(end_points['transferred_task_logits'], 1),
            source_labels['class'])

  if 'target_task_logits' in end_points:
    metric_name = 'eval/Task_Accuracy-Target'
    names_to_values[metric_name], names_to_updates[
        metric_name] = tf.contrib.metrics.streaming_accuracy(
            tf.argmax(end_points['target_task_logits'], 1),
            target_labels['class'])

  ##########################################################################
  # Pose data-specific losses.
  ##########################################################################
  if 'quaternion' in source_labels.keys():
    params = {}
    params['use_logging'] = False
    params['batch_size'] = batch_size

    angle_loss_source = to_degrees(
        pixelda_losses.log_quaternion_loss_batch(end_points[
            'source_quaternion'], source_labels['quaternion'], params))
    angle_loss_transferred = to_degrees(
        pixelda_losses.log_quaternion_loss_batch(end_points[
            'transferred_quaternion'], source_labels['quaternion'], params))
    angle_loss_target = to_degrees(
        pixelda_losses.log_quaternion_loss_batch(end_points[
            'target_quaternion'], target_labels['quaternion'], params))

    metric_name = 'eval/Angle_Loss-Source'
    names_to_values[metric_name], names_to_updates[
        metric_name] = slim.metrics.mean(angle_loss_source)

    metric_name = 'eval/Angle_Loss-Transferred'
    names_to_values[metric_name], names_to_updates[
        metric_name] = slim.metrics.mean(angle_loss_transferred)

    metric_name = 'eval/Angle_Loss-Target'
    names_to_values[metric_name], names_to_updates[
        metric_name] = slim.metrics.mean(angle_loss_target)

  return names_to_values, names_to_updates


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  hparams = create_hparams(FLAGS.hparams)
  run_eval(
      run_dir=FLAGS.eval_dir,
      checkpoint_dir=FLAGS.checkpoint_dir,
      hparams=hparams)


if __name__ == '__main__':
  tf.app.run()
