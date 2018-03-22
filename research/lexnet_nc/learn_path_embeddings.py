#!/usr/bin/env python
# Copyright 2017, 2018 Google, Inc. All Rights Reserved.
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

"""Trains the LexNET path-based model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import lexnet_common
import path_model
from sklearn import metrics
import tensorflow as tf

tf.flags.DEFINE_string(
    'dataset_dir', 'datasets',
    'Dataset base directory')

tf.flags.DEFINE_string(
    'dataset',
    'tratz/fine_grained',
    'Subdirectory containing the corpus directories: '
    'subdirectory of dataset_dir')

tf.flags.DEFINE_string(
    'corpus', 'random/wiki_gigawords',
    'Subdirectory containing the corpus and split: '
    'subdirectory of dataset_dir/dataset')

tf.flags.DEFINE_string(
    'embeddings_base_path', 'embeddings',
    'Embeddings base directory')

tf.flags.DEFINE_string(
    'logdir', 'logdir',
    'Directory of model output files')

FLAGS = tf.flags.FLAGS


def main(_):
  # Pick up any one-off hyper-parameters.
  hparams = path_model.PathBasedModel.default_hparams()

  # Set the number of classes
  classes_filename = os.path.join(
      FLAGS.dataset_dir, FLAGS.dataset, 'classes.txt')

  with open(classes_filename) as f_in:
    classes = f_in.read().splitlines()

  hparams.num_classes = len(classes)
  print('Model will predict into %d classes' % hparams.num_classes)

  # Get the datasets
  train_set, val_set, test_set = (
      os.path.join(
          FLAGS.dataset_dir, FLAGS.dataset, FLAGS.corpus,
          filename + '.tfrecs.gz')
      for filename in ['train', 'val', 'test'])

  print('Running with hyper-parameters: {}'.format(hparams))

  # Load the instances
  print('Loading instances...')
  opts = tf.python_io.TFRecordOptions(
      compression_type=tf.python_io.TFRecordCompressionType.GZIP)
  train_instances = list(tf.python_io.tf_record_iterator(train_set, opts))
  val_instances = list(tf.python_io.tf_record_iterator(val_set, opts))
  test_instances = list(tf.python_io.tf_record_iterator(test_set, opts))

  # Load the word embeddings
  print('Loading word embeddings...')
  lemma_embeddings = lexnet_common.load_word_embeddings(
      FLAGS.embeddings_base_path, hparams.lemma_embeddings_file)

  # Define the graph and the model
  with tf.Graph().as_default():
    with tf.variable_scope('lexnet'):
      options = tf.python_io.TFRecordOptions(
          compression_type=tf.python_io.TFRecordCompressionType.GZIP)
      reader = tf.TFRecordReader(options=options)
      _, train_instance = reader.read(
          tf.train.string_input_producer([train_set]))
      shuffled_train_instance = tf.train.shuffle_batch(
          [train_instance],
          batch_size=1,
          num_threads=1,
          capacity=len(train_instances),
          min_after_dequeue=100,
      )[0]

      train_model = path_model.PathBasedModel(
          hparams, lemma_embeddings, shuffled_train_instance)

    with tf.variable_scope('lexnet', reuse=True):
      val_instance = tf.placeholder(dtype=tf.string)
      val_model = path_model.PathBasedModel(
          hparams, lemma_embeddings, val_instance)

    # Initialize a session and start training
    logdir = (
        '{logdir}/results/{dataset}/path/{corpus}/supervisor.logdir'.format(
            logdir=FLAGS.logdir, dataset=FLAGS.dataset, corpus=FLAGS.corpus))

    best_model_saver = tf.train.Saver()
    f1_t = tf.placeholder(tf.float32)
    best_f1_t = tf.Variable(0.0, trainable=False, name='best_f1')
    assign_best_f1_op = tf.assign(best_f1_t, f1_t)

    supervisor = tf.train.Supervisor(
        logdir=logdir,
        global_step=train_model.global_step)

    with supervisor.managed_session() as session:
      # Load the labels
      print('Loading labels...')
      val_labels = train_model.load_labels(session, val_instances)

      save_path = '{logdir}/results/{dataset}/path/{corpus}/'.format(
          logdir=FLAGS.logdir,
          dataset=FLAGS.dataset,
          corpus=FLAGS.corpus)

      # Train the model
      print('Training the model...')

      while True:
        step = session.run(train_model.global_step)
        epoch = (step + len(train_instances) - 1) // len(train_instances)
        if epoch > hparams.num_epochs:
          break

        print('Starting epoch %d (step %d)...' % (1 + epoch, step))

        epoch_loss = train_model.run_one_epoch(session, len(train_instances))

        best_f1 = session.run(best_f1_t)
        f1 = epoch_completed(val_model, session, epoch, epoch_loss,
                             val_instances, val_labels, best_model_saver,
                             save_path, best_f1)

        if f1 > best_f1:
          session.run(assign_best_f1_op, {f1_t: f1})

        if f1 < best_f1 - 0.08:
          tf.logging.fino('Stopping training after %d epochs.\n' % epoch)
          break

      # Print the best performance on the validation set
      best_f1 = session.run(best_f1_t)
      print('Best performance on the validation set: F1=%.3f' % best_f1)

      # Save the path embeddings
      print('Computing the path embeddings...')
      instances = train_instances + val_instances + test_instances
      path_index, path_vectors = path_model.compute_path_embeddings(
          val_model, session, instances)
      path_emb_dir = '{dir}/path_embeddings/{dataset}/{corpus}/'.format(
          dir=FLAGS.embeddings_base_path,
          dataset=FLAGS.dataset,
          corpus=FLAGS.corpus)

      if not os.path.exists(path_emb_dir):
        os.makedirs(path_emb_dir)

      path_model.save_path_embeddings(
          val_model, path_vectors, path_index, path_emb_dir)


def epoch_completed(model, session, epoch, epoch_loss,
                    val_instances, val_labels, saver, save_path, best_f1):
  """Runs every time an epoch completes.

  Print the performance on the validation set, and update the saved model if
  its performance is better on the previous ones. If the performance dropped,
  tell the training to stop.

  Args:
    model: The currently trained path-based model.
    session: The current TensorFlow session.
    epoch: The epoch number.
    epoch_loss: The current epoch loss.
    val_instances: The validation set instances (evaluation between epochs).
    val_labels: The validation set labels (for evaluation between epochs).
    saver: tf.Saver object
    save_path: Where to save the model.
    best_f1: the best F1 achieved so far.

  Returns:
    The F1 achieved on the training set.
  """
  # Evaluate on the validation set
  val_pred = model.predict(session, val_instances)
  precision, recall, f1, _ = metrics.precision_recall_fscore_support(
      val_labels, val_pred, average='weighted')
  print(
      'Epoch: %d/%d, Loss: %f, validation set: P: %.3f, R: %.3f, F1: %.3f\n' % (
          epoch + 1, model.hparams.num_epochs, epoch_loss,
          precision, recall, f1))

  if f1 > best_f1:
    print('Saving model in: %s' % (save_path + 'best.ckpt'))
    saver.save(session, save_path + 'best.ckpt')
    print('Model saved in file: %s' % (save_path + 'best.ckpt'))

  return f1


if __name__ == '__main__':
  tf.app.run(main)
