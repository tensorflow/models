#!/usr/bin/env python
#
# Copyright 2016 Google Inc. All Rights Reserved.
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

"""Submatrix-wise Vector Embedding Learner.

Implementation of SwiVel algorithm described at:
http://arxiv.org/abs/1602.02215

This program expects an input directory that contains the following files.

  row_vocab.txt, col_vocab.txt

    The row an column vocabulary files.  Each file should contain one token per
    line; these will be used to generate a tab-separate file containing the
    trained embeddings.

  row_sums.txt, col_sum.txt

    The matrix row and column marginal sums.  Each file should contain one
    decimal floating point number per line which corresponds to the marginal
    count of the matrix for that row or column.

  shards.recs

    A file containing the sub-matrix shards, stored as TFRecords.  Each shard is
    expected to be a serialzed tf.Example protocol buffer with the following
    properties:

      global_row: the global row indicies contained in the shard
      global_col: the global column indicies contained in the shard
      sparse_local_row, sparse_local_col, sparse_value: three parallel arrays
      that are a sparse representation of the submatrix counts.

It will generate embeddings, training from the input directory for the specified
number of epochs.  When complete, it will output the trained vectors to a
tab-separated file that contains one line per embedding.  Row and column
embeddings are stored in separate files.

"""

import argparse
import glob
import math
import os
import sys
import time
import threading

import numpy as np
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('input_base_path', '/tmp/swivel_data',
                    'Directory containing input shards, vocabularies, '
                    'and marginals.')
flags.DEFINE_string('output_base_path', '/tmp/swivel_data',
                    'Path where to write the trained embeddings.')
flags.DEFINE_integer('embedding_size', 300, 'Size of the embeddings')
flags.DEFINE_boolean('trainable_bias', False, 'Biases are trainable')
flags.DEFINE_integer('submatrix_rows', 4096, 'Rows in each training submatrix. '
                     'This must match the training data.')
flags.DEFINE_integer('submatrix_cols', 4096, 'Rows in each training submatrix. '
                     'This must match the training data.')
flags.DEFINE_float('loss_multiplier', 1.0 / 4096,
                   'constant multiplier on loss.')
flags.DEFINE_float('confidence_exponent', 0.5,
                   'Exponent for l2 confidence function')
flags.DEFINE_float('confidence_scale', 0.25, 'Scale for l2 confidence function')
flags.DEFINE_float('confidence_base', 0.1, 'Base for l2 confidence function')
flags.DEFINE_float('learning_rate', 1.0, 'Initial learning rate')
flags.DEFINE_integer('num_concurrent_steps', 2,
                     'Number of threads to train with')
flags.DEFINE_float('num_epochs', 40, 'Number epochs to train for')
flags.DEFINE_float('per_process_gpu_memory_fraction', 0.25,
                   'Fraction of GPU memory to use')

FLAGS = flags.FLAGS


def embeddings_with_init(vocab_size, embedding_dim, name):
  """Creates and initializes the embedding tensors."""
  return tf.get_variable(name=name,
                         shape=[vocab_size, embedding_dim],
                         initializer=tf.random_normal_initializer(
                             stddev=math.sqrt(1.0 / embedding_dim)))


def count_matrix_input(filenames, submatrix_rows, submatrix_cols):
  """Reads submatrix shards from disk."""
  filename_queue = tf.train.string_input_producer(filenames)
  reader = tf.WholeFileReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'global_row': tf.FixedLenFeature([submatrix_rows], dtype=tf.int64),
          'global_col': tf.FixedLenFeature([submatrix_cols], dtype=tf.int64),
          'sparse_local_row': tf.VarLenFeature(dtype=tf.int64),
          'sparse_local_col': tf.VarLenFeature(dtype=tf.int64),
          'sparse_value': tf.VarLenFeature(dtype=tf.float32)
      })

  global_row = features['global_row']
  global_col = features['global_col']

  sparse_local_row = features['sparse_local_row'].values
  sparse_local_col = features['sparse_local_col'].values
  sparse_count = features['sparse_value'].values

  sparse_indices = tf.concat(1, [tf.expand_dims(sparse_local_row, 1),
                                 tf.expand_dims(sparse_local_col, 1)])
  count = tf.sparse_to_dense(sparse_indices, [submatrix_rows, submatrix_cols],
                             sparse_count)

  queued_global_row, queued_global_col, queued_count = tf.train.batch(
      [global_row, global_col, count],
      batch_size=1,
      num_threads=4,
      capacity=32)

  queued_global_row = tf.reshape(queued_global_row, [submatrix_rows])
  queued_global_col = tf.reshape(queued_global_col, [submatrix_cols])
  queued_count = tf.reshape(queued_count, [submatrix_rows, submatrix_cols])

  return queued_global_row, queued_global_col, queued_count


def read_marginals_file(filename):
  """Reads text file with one number per line to an array."""
  with open(filename) as lines:
    return [float(line) for line in lines]


def write_embedding_tensor_to_disk(vocab_path, output_path, sess, embedding):
  """Writes tensor to output_path as tsv"""
  # Fetch the embedding values from the model
  embeddings = sess.run(embedding)

  with open(output_path, 'w') as out_f:
    with open(vocab_path) as vocab_f:
      for index, word in enumerate(vocab_f):
        word = word.strip()
        embedding = embeddings[index]
        out_f.write(word + '\t' + '\t'.join([str(x) for x in embedding]) + '\n')


def write_embeddings_to_disk(config, model, sess):
  """Writes row and column embeddings disk"""
  # Row Embedding
  row_vocab_path = config.input_base_path + '/row_vocab.txt'
  row_embedding_output_path = config.output_base_path + '/row_embedding.tsv'
  print 'Writing row embeddings to:', row_embedding_output_path
  write_embedding_tensor_to_disk(row_vocab_path, row_embedding_output_path,
                                 sess, model.row_embedding)

  # Column Embedding
  col_vocab_path = config.input_base_path + '/col_vocab.txt'
  col_embedding_output_path = config.output_base_path + '/col_embedding.tsv'
  print 'Writing column embeddings to:', col_embedding_output_path
  write_embedding_tensor_to_disk(col_vocab_path, col_embedding_output_path,
                                 sess, model.col_embedding)


class SwivelModel(object):
  """Small class to gather needed pieces from a Graph being built."""

  def __init__(self, config):
    """Construct graph for dmc."""
    self._config = config

    # Create paths to input data files
    print 'Reading model from:', config.input_base_path
    count_matrix_files = glob.glob(config.input_base_path + '/shard-*.pb')
    row_sums_path = config.input_base_path + '/row_sums.txt'
    col_sums_path = config.input_base_path + '/col_sums.txt'

    # Read marginals
    row_sums = read_marginals_file(row_sums_path)
    col_sums = read_marginals_file(col_sums_path)

    self.n_rows = len(row_sums)
    self.n_cols = len(col_sums)
    print 'Matrix dim: (%d,%d) SubMatrix dim: (%d,%d) ' % (
        self.n_rows, self.n_cols, config.submatrix_rows, config.submatrix_cols)
    self.n_submatrices = (self.n_rows * self.n_cols /
                          (config.submatrix_rows * config.submatrix_cols))
    print 'n_submatrices: %d' % (self.n_submatrices)

    # ===== CREATE VARIABLES ======

    with tf.device('/cpu:0'):
      # embeddings
      self.row_embedding = embeddings_with_init(
          embedding_dim=config.embedding_size,
          vocab_size=self.n_rows,
          name='row_embedding')
      self.col_embedding = embeddings_with_init(
          embedding_dim=config.embedding_size,
          vocab_size=self.n_cols,
          name='col_embedding')
      tf.histogram_summary('row_emb', self.row_embedding)
      tf.histogram_summary('col_emb', self.col_embedding)

      matrix_log_sum = math.log(np.sum(row_sums) + 1)
      row_bias_init = [math.log(x + 1) for x in row_sums]
      col_bias_init = [math.log(x + 1) for x in col_sums]
      self.row_bias = tf.Variable(row_bias_init,
                                  trainable=config.trainable_bias)
      self.col_bias = tf.Variable(col_bias_init,
                                  trainable=config.trainable_bias)
      tf.histogram_summary('row_bias', self.row_bias)
      tf.histogram_summary('col_bias', self.col_bias)

    # ===== CREATE GRAPH =====

    # Get input
    with tf.device('/cpu:0'):
      global_row, global_col, count = count_matrix_input(
          count_matrix_files, config.submatrix_rows, config.submatrix_cols)

      # Fetch embeddings.
      selected_row_embedding = tf.nn.embedding_lookup(self.row_embedding,
                                                      global_row)
      selected_col_embedding = tf.nn.embedding_lookup(self.col_embedding,
                                                      global_col)

      # Fetch biases.
      selected_row_bias = tf.nn.embedding_lookup([self.row_bias], global_row)
      selected_col_bias = tf.nn.embedding_lookup([self.col_bias], global_col)

    # Multiply the row and column embeddings to generate predictions.
    predictions = tf.matmul(
        selected_row_embedding, selected_col_embedding, transpose_b=True)

    # These binary masks separate zero from non-zero values.
    count_is_nonzero = tf.to_float(tf.cast(count, tf.bool))
    count_is_zero = 1 - tf.to_float(tf.cast(count, tf.bool))

    objectives = count_is_nonzero * tf.log(count + 1e-30)
    objectives -= tf.reshape(selected_row_bias, [config.submatrix_rows, 1])
    objectives -= selected_col_bias
    objectives += matrix_log_sum

    err = predictions - objectives

    # The confidence function scales the L2 loss based on the raw co-occurrence
    # count.
    l2_confidence = (config.confidence_base + config.confidence_scale * tf.pow(
        count, config.confidence_exponent))

    l2_loss = config.loss_multiplier * tf.reduce_sum(
        0.5 * l2_confidence * err * err * count_is_nonzero)

    sigmoid_loss = config.loss_multiplier * tf.reduce_sum(
        tf.nn.softplus(err) * count_is_zero)

    self.loss = l2_loss + sigmoid_loss

    tf.scalar_summary("l2_loss", l2_loss)
    tf.scalar_summary("sigmoid_loss", sigmoid_loss)
    tf.scalar_summary("loss", self.loss)

    # Add optimizer.
    self.global_step = tf.Variable(0, name='global_step')
    opt = tf.train.AdagradOptimizer(config.learning_rate)
    self.train_op = opt.minimize(self.loss, global_step=self.global_step)
    self.saver = tf.train.Saver(sharded=True)


def main(_):
  # Create the output path.  If this fails, it really ought to fail
  # now. :)
  if not os.path.isdir(FLAGS.output_base_path):
    os.makedirs(FLAGS.output_base_path)

  # Create and run model
  with tf.Graph().as_default():
    model = SwivelModel(FLAGS)

    # Create a session for running Ops on the Graph.
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Run the Op to initialize the variables.
    sess.run(tf.initialize_all_variables())

    # Start feeding input
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Calculate how many steps each thread should run
    n_total_steps = int(FLAGS.num_epochs * model.n_rows * model.n_cols) / (
        FLAGS.submatrix_rows * FLAGS.submatrix_cols)
    n_steps_per_thread = n_total_steps / FLAGS.num_concurrent_steps
    n_submatrices_to_train = model.n_submatrices * FLAGS.num_epochs
    t0 = [time.time()]

    def TrainingFn():
      for _ in range(n_steps_per_thread):
        _, global_step = sess.run([model.train_op, model.global_step])
        n_steps_between_status_updates = 100
        if (global_step % n_steps_between_status_updates) == 0:
          elapsed = float(time.time() - t0[0])
          print '%d/%d submatrices trained (%.1f%%), %.1f submatrices/sec' % (
              global_step, n_submatrices_to_train,
              100.0 * global_step / n_submatrices_to_train,
              n_steps_between_status_updates / elapsed)
          t0[0] = time.time()

    # Start training threads
    train_threads = []
    for _ in range(FLAGS.num_concurrent_steps):
      t = threading.Thread(target=TrainingFn)
      train_threads.append(t)
      t.start()

    # Wait for threads to finish.
    for t in train_threads:
      t.join()

    coord.request_stop()
    coord.join(threads)

    # Write out vectors
    write_embeddings_to_disk(FLAGS, model, sess)

    #Shutdown
    sess.close()


if __name__ == '__main__':
  tf.app.run()
