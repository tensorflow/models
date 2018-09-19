# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Defines training scheme for neural networks for Seq2Species prediction.

Defines and runs the loop for training a (optionally) depthwise separable
convolutional model for predicting taxonomic labels from short reads of DNA.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import flags
import numpy as np
import tensorflow as tf
from google.protobuf import text_format

import build_model
import configuration
import input as seq2species_input
from protos import seq2label_pb2
import seq2label_utils

# Define non-tunable parameters.
flags.DEFINE_integer('num_filters', 1, 'Number of filters for conv model')
flags.DEFINE_string('hparams', '',
                    'Comma-separated list of name=value hyperparameter '
                    "pairs ('hp1=value1,hp2=value2'). Unspecified "
                    'hyperparameters will be filled with defaults.')
flags.DEFINE_integer('batch_size', 512, 'Size of batches during training.')
flags.DEFINE_integer('min_train_steps', 1000,
                     'Minimum number of training steps to run.')
flags.DEFINE_float('max_task_loss', 10.0,
                   "Terminate trial if task loss doesn't fall below this "
                   'within --min_train_steps.')
flags.DEFINE_integer('n_print_progress_every', 1000,
                     'Print training progress every '
                     '--n_print_progress_every global steps.')
flags.DEFINE_list('targets', ['species'],
                  'Names of taxonomic ranks to use as training targets.')
flags.DEFINE_float(
    'noise_rate', 0.0, 'Rate [0.0, 1.0] at which to inject '
    'base-flipping noise into input read sequences.')

# Define paths to logs and data.
flags.DEFINE_list(
    'train_files', [], 'Full paths to the TFRecords containing the '
    'training examples.')
flags.DEFINE_string(
    'metadata_path', '', 'Full path of the text proto containing configuration '
    'information about the set of training examples.')
flags.DEFINE_string('logdir', '/tmp/seq2species',
                    'Directory to which to write logs.')

# Define supervisor/checkpointing options.
flags.DEFINE_integer('task', 0, 'Task ID of the replica running the training.')
flags.DEFINE_string('master', '', 'Name of the TF master to use.')
flags.DEFINE_integer(
    'save_model_secs', 900, 'Rate at which to save model parameters. '
    'Set to 0 to disable checkpointing.')
flags.DEFINE_integer('recovery_wait_secs', 30,
                     'Wait to recover model from checkpoint '
                     'before timing out.')
flags.DEFINE_integer('save_summaries_secs', 900,
                     'Rate at which to save Tensorboard summaries.')
flags.DEFINE_integer('ps_tasks', 0,
                     'Number of tasks in the ps job; 0 if no ps is used.')

FLAGS = flags.FLAGS
RANDOM_SEED = 42


def wait_until(time_sec):
  """Stalls execution until a given time.

  Args:
    time_sec: time, in seconds, until which to loop idly.
  """
  while time.time() < time_sec:
    pass


def update_measures(measures, new_measures, loss_val, max_loss=None):
  """Updates tracking of experimental measures and infeasibilty.

  Args:
    measures: dict; mapping from measure name to measure value.
    new_measures: dict; mapping from measure name to new measure values.
    loss_val: float; value of loss metric by which to determine fesibility.
    max_loss: float; maximum value at which to consider the loss feasible.

  Side Effects:
    Updates the given mapping of measures and values based on the current
    experimental metrics stored in new_measures, and determines current
    feasibility of the experiment based on the provided loss value.
  """
  max_loss = max_loss if max_loss else np.finfo('f').max
  measures['is_infeasible'] = (
      loss_val >= max_loss or not np.isfinite(loss_val))
  measures.update(new_measures)


def run_training(model, hparams, training_dataset, logdir, batch_size):
  """Trains the given model on random mini-batches of reads.

  Args:
    model: ConvolutionalNet instance containing the model graph and operations.
    hparams: tf.contrib.training.Hparams object containing the model's
      hyperparamters; see configuration.py for hyperparameter definitions.
    training_dataset: an `InputDataset` that can feed labelled examples.
    logdir: string; full path of directory to which to save checkpoints.
    batch_size: integer batch size.

  Yields:
    Tuple comprising a dictionary of experimental measures and the save path
    for train checkpoints and summaries.
  """
  input_params = dict(batch_size=batch_size)
  features, labels = training_dataset.input_fn(input_params)
  model.build_graph(features, labels, tf.estimator.ModeKeys.TRAIN, batch_size)

  is_chief = FLAGS.task == 0
  scaffold = tf.train.Scaffold(
      saver=tf.train.Saver(
          tf.global_variables(),
          max_to_keep=5,
          keep_checkpoint_every_n_hours=1.0),
      init_op=tf.global_variables_initializer(),
      summary_op=model.summary_op)
  with tf.train.MonitoredTrainingSession(
      master=FLAGS.master,
      checkpoint_dir=logdir,
      is_chief=is_chief,
      scaffold=scaffold,
      save_summaries_secs=FLAGS.save_summaries_secs,
      save_checkpoint_secs=FLAGS.save_model_secs,
      max_wait_secs=FLAGS.recovery_wait_secs) as sess:
    global_step = sess.run(model.global_step)
    print('Initialized model at global step ', global_step)
    init_time = time.time()
    measures = {'is_infeasible': False}

    if is_chief:
      model_info = seq2label_utils.construct_seq2label_model_info(
          hparams, 'conv', FLAGS.targets, FLAGS.metadata_path, FLAGS.batch_size,
          FLAGS.num_filters, FLAGS.noise_rate)
      write_message(model_info, os.path.join(logdir, 'model_info.pbtxt'))

    ops = [
        model.accuracy, model.weighted_accuracy, model.total_loss,
        model.global_step, model.train_op
    ]

    while not sess.should_stop() and global_step < hparams.train_steps:
      accuracy, weighted_accuracy, loss, global_step, _ = sess.run(ops)

      def gather_measures():
        """Updates the measures dictionary from this batch."""
        new_measures = {'train_loss': loss, 'global_step': global_step}
        for target in FLAGS.targets:
          new_measures.update({
              ('train_accuracy/%s' % target): accuracy[target],
              ('train_weighted_accuracy/%s' % target): weighted_accuracy[target]
          })
        update_measures(
            measures, new_measures, loss, max_loss=FLAGS.max_task_loss)

      # Periodically track measures according to current mini-batch performance.

      # Log a message.
      if global_step % FLAGS.n_print_progress_every == 0:
        log_message = ('\tstep: %d (%d sec), loss: %f' %
                       (global_step, time.time() - init_time, loss))
        for target in FLAGS.targets:
          log_message += (', accuracy/%s: %f ' % (target, accuracy[target]))
          log_message += (', weighted_accuracy/%s: %f ' %
                          (target, weighted_accuracy[target]))
        print(log_message)

        # Gather new measures and update the measures dictionary.
        gather_measures()
        yield measures, scaffold.saver.last_checkpoints[-1]

      # Check for additional stopping criteria.
      if not np.isfinite(loss) or (loss >= FLAGS.max_task_loss and
                                   global_step > FLAGS.min_train_steps):
        break

    # Always yield once at the end.
    gather_measures()
    yield measures, scaffold.saver.last_checkpoints[-1]


def write_message(message, filename):
  """Writes contents of the given message to the given filename as a text proto.

  Args:
    message: the proto message to save.
    filename: full path of file to which to save the text proto.

  Side Effects:
    Outputs a text proto file to the given filename.
  """
  message_string = text_format.MessageToString(message)
  with tf.gfile.GFile(filename, 'w') as f:
    f.write(message_string)


def write_measures(measures, checkpoint_file, init_time):
  """Writes performance measures to file.

  Args:
    measures: dict; mapping from measure name to measure value.
    checkpoint_file: string; full save path for checkpoints and summaries.
    init_time: int; start time for work on the current experiment.

  Side Effects:
    Writes given dictionary of performance measures for the current experiment
    to a 'measures.pbtxt' file in the checkpoint directory.
  """
  # Save experiment measures.
  print('global_step: ', measures['global_step'])
  experiment_measures = seq2label_pb2.Seq2LabelExperimentMeasures(
      checkpoint_path=checkpoint_file,
      steps=measures['global_step'],
      experiment_infeasible=measures['is_infeasible'],
      wall_time=time.time() - init_time)  # Inaccurate for restarts.
  for name, value in measures.iteritems():
    if name not in ['is_infeasible', 'global_step']:
      experiment_measures.measures.add(name=name, value=value)
  measures_file = os.path.join(
      os.path.dirname(checkpoint_file), 'measures.pbtxt')
  write_message(experiment_measures, measures_file)
  print('Wrote ', measures_file,
        ' containing the following experiment measures:\n', experiment_measures)


def main(unused_argv):
  dataset_info = seq2species_input.load_dataset_info(FLAGS.metadata_path)

  init_time = time.time()

  # Determine model hyperparameters.
  hparams = configuration.parse_hparams(FLAGS.hparams, FLAGS.num_filters)
  print('Current Hyperparameters:')
  for hp_name, hp_val in hparams.values().items():
    print('\t', hp_name, ': ', hp_val)

  # Initialize the model graph.
  print('Constructing TensorFlow Graph.')
  tf.reset_default_graph()

  input_dataset = seq2species_input.InputDataset.from_tfrecord_files(
      FLAGS.train_files,
      'train',
      FLAGS.targets,
      dataset_info,
      noise_rate=FLAGS.noise_rate,
      random_seed=RANDOM_SEED)

  with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
    model = build_model.ConvolutionalNet(
        hparams, dataset_info, targets=FLAGS.targets)

  # Run the experiment.
  measures, checkpoint_file = None, None
  print('Starting model training.')
  for cur_measures, cur_file in run_training(
      model, hparams, input_dataset, FLAGS.logdir, batch_size=FLAGS.batch_size):
    measures, checkpoint_file = cur_measures, cur_file

  # Save experiment results.
  write_measures(measures, checkpoint_file, init_time)


if __name__ == '__main__':
  tf.app.run(main)
